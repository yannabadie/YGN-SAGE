//! Embedded RustPython interpreter compiled to wasm32-wasip1, loaded by
//! wasmtime with deny-by-default WASI capabilities.
//!
//! This is the Wasm-based Python execution sandbox that backs the
//! `execute_raw` path when `SAGE_UNSAFE_RAW_EXEC=1` AND this feature
//! is compiled in. Unlike the subprocess fallback — which gives the
//! code full host access and is only unlocked via a separate
//! `SAGE_UNSAFE_UNSANDBOXED=1` gate — the Wasm path enforces:
//!
//! * No filesystem: `WasiCtxBuilder` has NO `preopened_dir()`.
//! * No network: WASI-preview1 doesn't expose sockets at all.
//! * No subprocess / fork: WASI-preview1 has no `proc_exec`.
//! * No env inheritance: the WasiCtxBuilder starts with empty env;
//!   secrets in the host env (API keys, DB URLs) are invisible.
//! * No clock leak: we don't call `inherit_stdio()` — stdout and
//!   stderr are captured into `MemoryOutputPipe`s, not the host's.
//! * Bounded runtime: wasmtime fuel caps CPU, our own deadline caps
//!   wall-clock.
//!
//! The RustPython binary itself is embedded via `include_bytes!` at
//! a path written by `sage-core/build.rs`. When the sandbox feature
//! is enabled but the runtime bytes weren't built / shipped,
//! `WasmPythonExecutor::new()` returns an error and callers fall
//! back through the normal deny-path (subprocess gate if the
//! separate unsandboxed opt-in is set, else fail-closed).
//!
//! Build recipe (local dev):
//! ```text
//! rustup target add wasm32-wasip1
//! git clone https://github.com/RustPython/RustPython external/rustpython
//! cd external/rustpython
//! CARGO_TARGET_DIR=../rustpython-wasm-target \
//!   cargo build --release --target wasm32-wasip1 --features freeze-stdlib
//! ```
//! The resulting `.wasm` is picked up by `build.rs`; sage-core then
//! `include_bytes!`s it into the library.

#![cfg(all(feature = "sandbox", feature = "cranelift"))]

use std::hash::{Hash, Hasher};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use sha2::{Digest, Sha256};
use wasmtime::{Config, Engine, Linker, Module, Store, StoreLimits, StoreLimitsBuilder};
use wasmtime_wasi::p1::{self, WasiP1Ctx};
use wasmtime_wasi::p2::pipe::MemoryOutputPipe;
use wasmtime_wasi::{I32Exit, WasiCtxBuilder};

/// Memory cap for a single sandbox call. wasm32 allows up to 4 GiB
/// linear memory by default — catastrophic for a shared host on a
/// DoS probe like `x = [0] * (10 ** 9)`. 256 MiB is generous for a
/// single CPython-equivalent REPL script and low enough that even a
/// dozen concurrent sandbox calls stay well under host pressure.
const SANDBOX_MAX_MEMORY_BYTES: usize = 256 * 1024 * 1024;

/// Stdout/stderr cap per call. Sized to match the red-team plan's
/// ENG-4 assertion (truncation cap) while staying small enough to
/// avoid memory pressure when many sandbox calls run concurrently.
const SANDBOX_PIPE_CAPACITY: usize = 64 * 1024;

use super::subprocess::ExecResult;

/// Bytes of the RustPython wasm32-wasip1 build. Emitted by
/// `sage-core/build.rs` into OUT_DIR — see that file for the source
/// path. Zero-length when the runtime wasn't built; runtime-detected
/// via `RUSTPYTHON_WASM.is_empty()`.
const RUSTPYTHON_WASM: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/rustpython.wasm"));

/// Result of constructing a `WasmPythonExecutor`.
#[derive(Debug)]
pub enum WasmPythonInitError {
    /// sage-core was built with the `sandbox` feature but
    /// `build.rs` couldn't find a compiled rustpython.wasm.
    BytesMissing,
    /// wasmtime rejected the module (shouldn't happen with a
    /// cleanly-built RustPython wasm; surfaces clearly if a stale
    /// or corrupted build slips through).
    Wasmtime(String),
}

impl std::fmt::Display for WasmPythonInitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BytesMissing => write!(
                f,
                "RustPython wasm runtime bytes are empty — build.rs did not \
                 find a .wasm artifact. Build it with the recipe in \
                 sage-core/src/sandbox/wasm_python.rs's module docstring."
            ),
            Self::Wasmtime(e) => write!(f, "wasmtime module load failed: {}", e),
        }
    }
}

impl std::error::Error for WasmPythonInitError {}

/// Per-call store state. `wasi` holds the WASI-p1 capability
/// context, `limits` is the memory cap enforced by wasmtime on
/// every linear-memory grow. Both live in the Store so they can
/// be read from inside the wasm guest at syscall / grow time.
struct StoreState {
    wasi: WasiP1Ctx,
    limits: StoreLimits,
}

/// Embedded-RustPython executor backed by wasmtime.
///
/// Construction is expensive (compiles the .wasm via cranelift) so
/// keep a single instance and call `execute()` repeatedly. Each
/// `execute()` creates a fresh `Store` + `WasiCtx` so there is no
/// cross-call state leak.
pub struct WasmPythonExecutor {
    engine: Engine,
    module: Arc<Module>,
    /// Monotonic counter: each `execute()` call claims the next
    /// value as its epoch deadline. wasmtime `set_epoch_deadline`
    /// is ABSOLUTE (interrupt when engine.epoch >= deadline), and
    /// the shared engine's epoch grows across calls because each
    /// watchdog thread bumps it once after its timeout. Using a
    /// per-call monotonic deadline keeps every new call starting
    /// at `engine.epoch + 1`, so a prior call's watchdog firing
    /// can't interrupt a subsequent call. Initialised at 1 so
    /// deadline for the first call is 1 (engine starts at 0).
    next_deadline: AtomicU64,
}

/// Test-only signal: `true` if the most recent `WasmPythonExecutor::new()`
/// (or `new_with_overrides()`) call loaded the module via `Module::deserialize`
/// from a cached .cwasm artifact; `false` if it fell back to the slow
/// `Module::new` cranelift compilation path. Used by the cache_tests
/// module to deterministically assert warm-hit behaviour without relying
/// on wall-clock timing.
#[cfg(test)]
pub(crate) static LAST_NEW_USED_CACHE: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Override options for `WasmPythonExecutor::new_with_overrides`. Public
/// to the crate's test module so tests can set the cache dir and disable
/// flag explicitly without touching process-wide environment variables
/// (which would race across parallel `cargo test` threads).
#[cfg(test)]
#[derive(Default, Clone)]
pub(crate) struct CacheOverrides {
    /// Absolute path to the cache directory. `None` means "resolve from
    /// `SAGE_WASM_CACHE_DIR` / `$HOME/.sage/wasm_python_cache/`".
    pub cache_dir: Option<PathBuf>,
    /// If `true`, skip the cache entirely (always compile, never read
    /// or write). Equivalent to `SAGE_WASM_CACHE_DISABLE=1`.
    pub disabled: bool,
}

impl WasmPythonExecutor {
    /// Build + cache the Python interpreter module. Safe to hold in
    /// a long-lived struct (the engine is thread-safe and cheap to
    /// share via Arc).
    ///
    /// On a cold cache this compiles ~37 MB of RustPython wasm via
    /// cranelift (~30 s wallclock). Subsequent calls with a warm cache
    /// deserialize a precompiled `.cwasm` in ~1 s. The cache lives under
    /// `$SAGE_WASM_CACHE_DIR` (if set) or platform-default
    /// `$HOME/.sage/wasm_python_cache/` (unix) /
    /// `$USERPROFILE\.sage\wasm_python_cache\` (Windows). Set
    /// `SAGE_WASM_CACHE_DISABLE=1` to opt out entirely (useful when
    /// debugging a wasmtime upgrade).
    pub fn new() -> Result<Self, WasmPythonInitError> {
        let disabled = std::env::var_os("SAGE_WASM_CACHE_DISABLE")
            .map(|v| v != "0" && !v.is_empty())
            .unwrap_or(false);
        let cache_dir = std::env::var_os("SAGE_WASM_CACHE_DIR").map(PathBuf::from);
        Self::new_inner(cache_dir.as_deref(), disabled)
    }

    /// Crate-private constructor used by tests to pass explicit cache
    /// options so we don't race on process-wide env vars under
    /// `cargo test`'s default multi-thread runner.
    #[cfg(test)]
    pub(crate) fn new_with_overrides(o: &CacheOverrides) -> Result<Self, WasmPythonInitError> {
        Self::new_inner(o.cache_dir.as_deref(), o.disabled)
    }

    fn new_inner(
        cache_dir_override: Option<&Path>,
        disabled: bool,
    ) -> Result<Self, WasmPythonInitError> {
        if RUSTPYTHON_WASM.is_empty() {
            return Err(WasmPythonInitError::BytesMissing);
        }
        let engine = Self::build_engine()?;

        #[cfg(test)]
        LAST_NEW_USED_CACHE.store(false, Ordering::SeqCst);

        if disabled {
            let module = Module::new(&engine, RUSTPYTHON_WASM)
                .map_err(|e| WasmPythonInitError::Wasmtime(e.to_string()))?;
            return Ok(Self::from_module(engine, module));
        }

        // Resolve cache dir (override -> env -> platform default). Any
        // failure to *resolve* a path is non-fatal: log and fall through
        // to a cacheless compile. Only the actual JIT step is fatal.
        let cache_dir = cache_dir_override
            .map(PathBuf::from)
            .or_else(default_cache_dir);
        let cache_path = cache_dir
            .as_deref()
            .map(|d| d.join(format!("{}.cwasm", cache_key(&engine))));

        // 1. Try a warm read.
        if let Some(path) = cache_path.as_deref() {
            match std::fs::read(path) {
                Ok(bytes) => match load_cached_module(&engine, &bytes) {
                    Ok(module) => {
                        #[cfg(test)]
                        LAST_NEW_USED_CACHE.store(true, Ordering::SeqCst);
                        return Ok(Self::from_module(engine, module));
                    }
                    Err(e) => {
                        // Stale / corrupt: drop it and fall through to
                        // fresh compile. A failed unlink is fine — the
                        // atomic rename at the end overwrites anyway.
                        tracing::warn!(
                            target: "sage_core::sandbox",
                            ?path,
                            error = %e,
                            "wasm cache deserialize failed; recompiling"
                        );
                        let _ = std::fs::remove_file(path);
                    }
                },
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                    // Cold miss — expected on first run.
                }
                Err(e) => {
                    tracing::warn!(
                        target: "sage_core::sandbox",
                        ?path,
                        error = %e,
                        "wasm cache read failed; recompiling"
                    );
                }
            }
        }

        // 2. Slow path: compile from source bytes.
        let module = Module::new(&engine, RUSTPYTHON_WASM)
            .map_err(|e| WasmPythonInitError::Wasmtime(e.to_string()))?;

        // 3. Best-effort cache write (non-fatal on failure).
        if let Some(path) = cache_path.as_deref() {
            if let Err(e) = write_cache_atomic(&module, path) {
                tracing::warn!(
                    target: "sage_core::sandbox",
                    ?path,
                    error = %e,
                    "wasm cache write failed; continuing without cache"
                );
            }
        }

        Ok(Self::from_module(engine, module))
    }

    fn build_engine() -> Result<Engine, WasmPythonInitError> {
        let mut config = Config::new();
        // Epoch-based deadline is what we use for per-call timeouts;
        // fuel would be more deterministic but epoch is lighter.
        config.epoch_interruption(true);
        // Component Model is off — rustpython.wasm is a plain
        // wasip1 module, not a Component-Model component.
        config.wasm_component_model(false);
        // NOTE: every knob set on this Config contributes to the engine's
        // precompile-compatibility hash. If you tweak any option above,
        // the next `new()` invocation will compute a different cache key
        // and silently re-compile — no stale .cwasm will be loaded. That
        // is the whole point of keying on `precompile_compatibility_hash`.
        Engine::new(&config).map_err(|e| WasmPythonInitError::Wasmtime(e.to_string()))
    }

    fn from_module(engine: Engine, module: Module) -> Self {
        Self {
            engine,
            module: Arc::new(module),
            next_deadline: AtomicU64::new(1),
        }
    }

    /// Is the wasm runtime bytes present? True only when sage-core
    /// was built with the sandbox feature AND a compiled
    /// rustpython.wasm was available at build time.
    pub fn is_available() -> bool {
        !RUSTPYTHON_WASM.is_empty()
    }

    /// Execute a chunk of Python code inside the sandbox.
    ///
    /// Passes `rustpython -c <wrapped_code>` as argv to the embedded
    /// interpreter. `args_json` is made available to user code as
    /// the variable `args` (parsed JSON). Stdout/stderr are captured
    /// (capped at 64 KiB each) and returned in the `ExecResult`. The
    /// call is interrupted after `timeout_secs` wall-clock using the
    /// wasmtime epoch interrupt.
    pub fn execute(&self, code: &str, args_json: &str, timeout_secs: u64) -> ExecResult {
        let start = Instant::now();
        let stdout = MemoryOutputPipe::new(SANDBOX_PIPE_CAPACITY);
        let stderr = MemoryOutputPipe::new(SANDBOX_PIPE_CAPACITY);

        let wrapped = Self::wrap_code(code, args_json);

        let mut builder = WasiCtxBuilder::new();
        builder.stdout(stdout.clone()).stderr(stderr.clone());
        // Args: `rustpython -c "<code>"` — matches how someone
        // would invoke the CLI directly. RustPython honours the -c
        // flag the same as CPython.
        builder.arg("rustpython");
        builder.arg("-c");
        builder.arg(&wrapped);
        // Deliberately NOT called: inherit_env(), inherit_stdio(),
        // preopened_dir(), preopened_stdio(). No inheritance means
        // the component sees an empty env, an empty filesystem
        // view, and only our stdout/stderr pipes.

        let wasi: WasiP1Ctx = builder.build_p1();

        // StoreState holds both the WASI context and the memory
        // limiter. wasmtime requires the limiter to be stored in the
        // Store's data so it can read it during memory-grow calls.
        let state = StoreState {
            wasi,
            limits: StoreLimitsBuilder::new()
                .memory_size(SANDBOX_MAX_MEMORY_BYTES)
                .build(),
        };
        // Claim a fresh monotonic deadline. The watchdog will bump
        // engine.epoch exactly once after `timeout_secs`, so our
        // deadline must be > current engine.epoch. We track this
        // via AtomicU64 — previous calls' watchdogs have driven
        // engine.epoch up by exactly (deadline_val - 1).
        let deadline_val = self.next_deadline.fetch_add(1, Ordering::SeqCst);
        let mut store: Store<StoreState> = Store::new(&self.engine, state);
        store.limiter(|s| &mut s.limits);
        store.set_epoch_deadline(deadline_val);

        // Spawn a watchdog thread that bumps the epoch after the
        // timeout. This is how wasmtime cancels long-running Wasm.
        // Detached on purpose — if it fires after the call returned
        // cleanly, the epoch just ticks past this call's deadline,
        // which is harmless because next_deadline has moved on.
        let watchdog_engine = self.engine.clone();
        let deadline = Duration::from_secs(timeout_secs);
        let timer = std::thread::spawn(move || {
            std::thread::sleep(deadline);
            watchdog_engine.increment_epoch();
        });

        let mut linker: Linker<StoreState> = Linker::new(&self.engine);
        if let Err(e) = p1::add_to_linker_sync(&mut linker, |s: &mut StoreState| &mut s.wasi) {
            return mk_error_result(format!("linker wire-up failed: {}", e), &start);
        }

        let instance = match linker.instantiate(&mut store, &self.module) {
            Ok(i) => i,
            Err(e) => return mk_error_result(format!("instantiate failed: {}", e), &start),
        };

        let func = match instance.get_typed_func::<(), ()>(&mut store, "_start") {
            Ok(f) => f,
            Err(e) => return mk_error_result(format!("missing _start export: {}", e), &start),
        };

        let call_result = func.call(&mut store, ());
        // Cancel the watchdog timer — the wasm either finished or
        // was already interrupted. If the thread is still sleeping
        // we don't need to wake it; it'll bump the epoch harmlessly
        // after we've discarded the store.
        drop(timer); // detach

        let duration_ms = start.elapsed().as_millis() as u64;
        let stdout_text = pipe_to_string(&stdout);
        let stderr_text = pipe_to_string(&stderr);

        match call_result {
            Ok(()) => ExecResult {
                stdout: stdout_text,
                stderr: stderr_text,
                exit_code: 0,
                timed_out: false,
                duration_ms,
            },
            Err(trap) => {
                // wasmtime maps Python's sys.exit(N) to an I32Exit
                // trap — surface the actual exit code.
                let (exit_code, timed_out) = if let Some(I32Exit(code)) =
                    trap.downcast_ref::<I32Exit>()
                {
                    (*code, false)
                } else if start.elapsed() >= deadline {
                    (-1, true)
                } else {
                    (-1, false)
                };
                let err_msg = if timed_out {
                    format!("[WASM TIMEOUT after {}s]\n{}", timeout_secs, stderr_text)
                } else {
                    format!("{}\n[wasm trap: {}]", stderr_text, trap)
                };
                ExecResult {
                    stdout: stdout_text,
                    stderr: err_msg,
                    exit_code,
                    timed_out,
                    duration_ms,
                }
            }
        }
    }
}

impl WasmPythonExecutor {
    /// Prepend a tiny boilerplate that makes `args_json` available as
    /// the Python variable `args` (dict). args_json is hex-encoded
    /// into the source so no escaping ambiguity can sneak in — the
    /// payload can contain any bytes including quotes, newlines and
    /// backslashes.
    fn wrap_code(code: &str, args_json: &str) -> String {
        let payload = if args_json.trim().is_empty() {
            "{}"
        } else {
            args_json
        };
        let hex: String = payload
            .as_bytes()
            .iter()
            .map(|b| format!("{:02x}", b))
            .collect();
        // Match the subprocess-path contract: `json`, `sys`, and
        // `args` are available to user code as top-level names. Tool
        // authors expect them pre-imported since that was the
        // behaviour before the §5 sandbox flip. `codecs` stays
        // underscore-prefixed because it's only used internally to
        // decode the hex-escaped args payload.
        format!(
            "import json\nimport sys\nimport codecs as _sage_codecs\n\
             args = json.loads(_sage_codecs.decode(\"{}\", \"hex\").decode(\"utf-8\"))\n\
             {}",
            hex, code
        )
    }
}

/// Resolve the platform-default cache directory. Prefers `$HOME`
/// (unix) / `$USERPROFILE` (Windows). Returns `None` when neither is
/// set — in that case caching is silently disabled for the call.
fn default_cache_dir() -> Option<PathBuf> {
    // Order matters: HOME is set on both Unix and the MSYS2 shell we
    // dev-in; USERPROFILE is the native Windows fallback.
    let base = std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(PathBuf::from)?;
    Some(base.join(".sage").join("wasm_python_cache"))
}

/// Compute a stable cache key for the current engine + embedded wasm.
///
/// The key encodes three things:
///
/// * A hardcoded `sage-wasm-python-v1` prefix so we can bump the cache
///   format in the future without risking a stale-key collision.
/// * `Engine::precompile_compatibility_hash` — wasmtime's own promise
///   that binaries serialized with one engine are loadable by another
///   if (and only if) this hash matches. Encodes wasmtime version,
///   every `Config` knob we set, target triple, CPU features, etc.
/// * A SHA-256 of the `RUSTPYTHON_WASM` bytes — so rebuilding the
///   embedded interpreter invalidates the cache even if the engine
///   config is unchanged.
///
/// Returns a lowercase hex string suitable for filenames on all
/// supported hosts.
fn cache_key(engine: &Engine) -> String {
    // Fold Engine::precompile_compatibility_hash (an opaque `impl Hash`)
    // through a std DefaultHasher into a u64, then feed that u64 into
    // our stable SHA-256 alongside the other inputs. The DefaultHasher
    // itself is not guaranteed stable across rustc versions — which is
    // fine: if the DefaultHasher flips, the cache key changes and the
    // cache is silently rebuilt. We just need the *same rustc build*
    // of sage-core to produce a stable key, which DefaultHasher does.
    let mut engine_hasher = std::collections::hash_map::DefaultHasher::new();
    engine.precompile_compatibility_hash().hash(&mut engine_hasher);
    let engine_digest = engine_hasher.finish();

    let mut sha = Sha256::new();
    sha.update(b"sage-wasm-python-v1|");
    sha.update(engine_digest.to_le_bytes());
    sha.update(b"|");
    let mut wasm_sha = Sha256::new();
    wasm_sha.update(RUSTPYTHON_WASM);
    sha.update(wasm_sha.finalize());

    let digest = sha.finalize();
    let mut hex = String::with_capacity(digest.len() * 2);
    for b in digest.iter() {
        hex.push_str(&format!("{:02x}", b));
    }
    hex
}

/// Load a precompiled `.cwasm` into a `Module`. Thin wrapper around
/// `Module::deserialize` so the one `unsafe` call has a narrow
/// `#[allow(unsafe_code)]` scope and the SAFETY invariant is
/// documented in exactly one place.
fn load_cached_module(engine: &Engine, bytes: &[u8]) -> wasmtime::Result<Module> {
    // SAFETY: These bytes were read from a cache directory this
    // process owns — either the path pointed to by
    // `SAGE_WASM_CACHE_DIR` (set explicitly by the operator or the
    // test harness) or `$HOME/.sage/wasm_python_cache/`. The cache
    // key binds both the wasmtime engine's precompile-compatibility
    // hash AND a SHA-256 of the embedded wasm bytes; a mismatch
    // triggers a deserialize-error that we handle above by deleting
    // the file and recompiling. wasmtime's documented contract
    // (`Module::deserialize`): "bytes previously produced by
    // Module::serialize are always safe to pass in". A malicious
    // actor who can write arbitrary bytes into our cache directory
    // already has code-execution on the host, so forged-bytes is out
    // of scope.
    #[allow(unsafe_code)]
    unsafe {
        Module::deserialize(engine, bytes)
    }
}

/// Serialize `module` and atomically install the bytes at `path`.
///
/// The write uses the tempfile + rename idiom so a process crash
/// mid-write can never leave a truncated `.cwasm` in place. On a
/// cross-process race two writers may both produce a tempfile and
/// race to rename; the loser's rename either succeeds (clobbering the
/// winner's byte-for-byte identical bytes) or fails and leaves the
/// winner's file — both fine.
fn write_cache_atomic(module: &Module, path: &Path) -> std::io::Result<()> {
    let serialized = module
        .serialize()
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let tmp = path.with_extension(format!("cwasm.tmp.{}", std::process::id()));
    {
        let mut f = std::fs::File::create(&tmp)?;
        f.write_all(&serialized)?;
        f.sync_all()?;
    }
    match std::fs::rename(&tmp, path) {
        Ok(()) => Ok(()),
        Err(e) => {
            // Cleanup the tempfile on failure; ignore cleanup error.
            let _ = std::fs::remove_file(&tmp);
            Err(e)
        }
    }
}

fn mk_error_result(msg: String, start: &Instant) -> ExecResult {
    ExecResult {
        stdout: String::new(),
        stderr: format!("[WASM EXECUTOR ERROR] {}", msg),
        exit_code: -1,
        timed_out: false,
        duration_ms: start.elapsed().as_millis() as u64,
    }
}

fn pipe_to_string(pipe: &MemoryOutputPipe) -> String {
    let contents = pipe.contents();
    String::from_utf8_lossy(&contents).into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Skip the test if the embedded wasm bytes are missing (fresh
    /// clone without a RustPython build). CI / dev should build the
    /// wasm once and cache it; until then these tests are inert.
    fn skip_if_no_runtime() -> Option<WasmPythonExecutor> {
        if !WasmPythonExecutor::is_available() {
            eprintln!(
                "test skipped: rustpython.wasm not built (expected at \
                 external/rustpython-wasm-target/wasm32-wasip1/release/rustpython.wasm)"
            );
            return None;
        }
        Some(WasmPythonExecutor::new().expect("wasm executor should load"))
    }

    #[test]
    fn test_wasm_runtime_availability_is_binary() {
        // Either the bytes are there (non-empty) OR they're zero-
        // length (fresh build). Anything else is a build-pipeline
        // bug.
        let len = RUSTPYTHON_WASM.len();
        assert!(len == 0 || len > 1_000_000, "unexpected runtime size: {}", len);
    }

    #[test]
    fn test_execute_prints_hello() {
        let Some(exec) = skip_if_no_runtime() else {
            return;
        };
        let r = exec.execute(r#"print("hello from wasm")"#, "", 10);
        assert_eq!(r.exit_code, 0, "stderr: {}", r.stderr);
        assert!(r.stdout.contains("hello from wasm"));
    }

    #[test]
    fn test_execute_propagates_sys_exit_code() {
        let Some(exec) = skip_if_no_runtime() else {
            return;
        };
        let r = exec.execute("import sys; sys.exit(42)", "", 10);
        assert_eq!(r.exit_code, 42, "wasm should propagate sys.exit(42)");
    }

    #[test]
    fn test_execute_timeout_fires() {
        let Some(exec) = skip_if_no_runtime() else {
            return;
        };
        let r = exec.execute("while True: pass", "", 2);
        assert!(r.timed_out, "expected timeout; got exit_code={}", r.exit_code);
        assert!(r.duration_ms >= 1500, "timed out too early: {}ms", r.duration_ms);
    }

    #[test]
    fn test_args_json_round_trip() {
        let Some(exec) = skip_if_no_runtime() else {
            return;
        };
        let r = exec.execute(
            r#"print(args["x"], args["msg"])"#,
            r#"{"x": 42, "msg": "hi \"quoted\""}"#,
            5,
        );
        assert_eq!(r.exit_code, 0, "stderr: {}", r.stderr);
        assert!(r.stdout.contains("42"), "got: {}", r.stdout);
        assert!(r.stdout.contains("hi \"quoted\""), "got: {}", r.stdout);
    }

    #[test]
    fn test_sandbox_denies_filesystem_read() {
        let Some(exec) = skip_if_no_runtime() else {
            return;
        };
        let r = exec.execute(
            r#"
try:
    with open("/etc/passwd") as f:
        print("LEAK:", f.read(20))
except Exception as e:
    print("DENIED:", type(e).__name__)
"#,
            "",
            5,
        );
        assert!(!r.stdout.contains("LEAK:"), "filesystem read LEAKED: {}", r.stdout);
        assert!(
            r.stdout.contains("DENIED:") || r.exit_code != 0,
            "expected OSError / PermissionError; got: {}",
            r.stdout
        );
    }

    #[test]
    fn test_sandbox_denies_env_var_read() {
        let Some(exec) = skip_if_no_runtime() else {
            return;
        };
        // HOST sets a sentinel. The sandbox must NOT see it.
        std::env::set_var("WASM_REDTEAM_SENTINEL", "sk-do-not-leak");
        let r = exec.execute(
            r#"
import os
v = os.environ.get("WASM_REDTEAM_SENTINEL", "<absent>")
print("env:", v)
"#,
            "",
            5,
        );
        std::env::remove_var("WASM_REDTEAM_SENTINEL");
        assert!(
            !r.stdout.contains("sk-do-not-leak"),
            "host secret LEAKED into wasm sandbox: {}",
            r.stdout
        );
        assert!(
            r.stdout.contains("<absent>") || r.exit_code == 0,
            "expected empty env inside sandbox; got: {}",
            r.stdout
        );
    }
}

/// Tests for the precompiled-wasm cache (`SAGE_WASM_CACHE_DIR` /
/// `SAGE_WASM_CACHE_DISABLE`). Each test uses an isolated `tempdir` so
/// the parallel `cargo test` runner doesn't let tests stomp one
/// another. Env vars are DELIBERATELY NOT TOUCHED by these tests —
/// they exercise `new_with_overrides()` directly to keep the whole
/// suite race-free.
#[cfg(test)]
mod cache_tests {
    use super::*;
    use std::sync::atomic::Ordering;
    use std::sync::Mutex;

    /// Process-wide mutex that serialises every cache-test's
    /// `new_with_overrides` → `LAST_NEW_USED_CACHE` read pair. The flag
    /// is a single AtomicBool; without this lock, a parallel test's
    /// flip could race between our flip and our assertion.
    /// `parking_lot`-style poisoning on panic is not a concern — we
    /// just need serialisation, not integrity.
    static CACHE_TEST_LOCK: Mutex<()> = Mutex::new(());

    /// Returns `None` when the rustpython.wasm bytes aren't embedded
    /// (fresh worktree, no build). Matches the `skip_if_no_runtime`
    /// helper on the sibling tests mod.
    fn skip_if_no_runtime_emit() {
        if !WasmPythonExecutor::is_available() {
            eprintln!(
                "cache test skipped: rustpython.wasm not built (expected \
                 at external/rustpython-wasm-target/wasm32-wasip1/release/rustpython.wasm)"
            );
        }
    }

    fn list_cwasm_files(dir: &std::path::Path) -> Vec<std::path::PathBuf> {
        let Ok(rd) = std::fs::read_dir(dir) else {
            return Vec::new();
        };
        rd.filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("cwasm"))
            .collect()
    }

    #[test]
    fn cold_miss_compiles_and_writes_cache() {
        if !WasmPythonExecutor::is_available() {
            skip_if_no_runtime_emit();
            return;
        }
        let _guard = CACHE_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let tmp = tempfile::tempdir().expect("mk tempdir");
        assert!(
            list_cwasm_files(tmp.path()).is_empty(),
            "tempdir should be empty before the cold call"
        );
        let opts = CacheOverrides {
            cache_dir: Some(tmp.path().to_path_buf()),
            disabled: false,
        };
        let _exec = WasmPythonExecutor::new_with_overrides(&opts)
            .expect("cold new should succeed");
        assert!(
            !LAST_NEW_USED_CACHE.load(Ordering::SeqCst),
            "cold miss should NOT load from cache"
        );
        let files = list_cwasm_files(tmp.path());
        assert_eq!(
            files.len(),
            1,
            "expected exactly one .cwasm after cold miss; got {:?}",
            files
        );
        let meta = std::fs::metadata(&files[0]).expect("stat cwasm");
        assert!(meta.len() > 0, "cwasm file should be non-empty");
    }

    #[test]
    fn warm_hit_deserializes_instead_of_compiling() {
        if !WasmPythonExecutor::is_available() {
            skip_if_no_runtime_emit();
            return;
        }
        let _guard = CACHE_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let tmp = tempfile::tempdir().expect("mk tempdir");
        let opts = CacheOverrides {
            cache_dir: Some(tmp.path().to_path_buf()),
            disabled: false,
        };

        // Cold — populate the cache.
        let _a = WasmPythonExecutor::new_with_overrides(&opts)
            .expect("cold new should succeed");
        assert!(
            !LAST_NEW_USED_CACHE.load(Ordering::SeqCst),
            "first call should be a cold miss"
        );

        // Warm — deserialize path.
        let _b = WasmPythonExecutor::new_with_overrides(&opts)
            .expect("warm new should succeed");
        assert!(
            LAST_NEW_USED_CACHE.load(Ordering::SeqCst),
            "second call should load the cached .cwasm"
        );
    }

    #[test]
    fn corrupt_cache_is_self_healing() {
        if !WasmPythonExecutor::is_available() {
            skip_if_no_runtime_emit();
            return;
        }
        let _guard = CACHE_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let tmp = tempfile::tempdir().expect("mk tempdir");

        // Compute the expected cache filename by building an engine
        // the same way `new_inner` does.
        let engine = WasmPythonExecutor::build_engine().expect("build engine");
        let key = cache_key(&engine);
        let cache_path = tmp.path().join(format!("{}.cwasm", key));

        // Plant 128 bytes of nonsense at the cache path.
        std::fs::create_dir_all(tmp.path()).unwrap();
        let junk: Vec<u8> = (0..128).map(|i| (i as u8).wrapping_mul(17)).collect();
        std::fs::write(&cache_path, &junk).expect("write junk");
        assert_eq!(
            std::fs::metadata(&cache_path).unwrap().len(),
            128,
            "corrupt plant is 128 bytes"
        );

        let opts = CacheOverrides {
            cache_dir: Some(tmp.path().to_path_buf()),
            disabled: false,
        };
        let exec = WasmPythonExecutor::new_with_overrides(&opts)
            .expect("new should self-heal past a corrupt cache");
        // The deserialize failed → we took the recompile path.
        assert!(
            !LAST_NEW_USED_CACHE.load(Ordering::SeqCst),
            "corrupt cache should NOT be reported as a cache hit"
        );

        // The junk file should have been replaced by a real .cwasm
        // sized > 128 bytes.
        let meta = std::fs::metadata(&cache_path).expect("cache should be replaced");
        assert!(
            meta.len() > 128,
            "replaced cache file should be larger than the 128-byte junk (got {})",
            meta.len()
        );

        // Verify the healed executor actually works end-to-end.
        let r = exec.execute(r#"print("healed")"#, "", 10);
        assert_eq!(r.exit_code, 0, "stderr: {}", r.stderr);
        assert!(r.stdout.contains("healed"));
    }

    #[test]
    fn disabled_flag_bypasses_cache() {
        if !WasmPythonExecutor::is_available() {
            skip_if_no_runtime_emit();
            return;
        }
        let _guard = CACHE_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let tmp = tempfile::tempdir().expect("mk tempdir");
        let opts = CacheOverrides {
            cache_dir: Some(tmp.path().to_path_buf()),
            disabled: true,
        };
        let _a = WasmPythonExecutor::new_with_overrides(&opts)
            .expect("disabled new should succeed");
        let _b = WasmPythonExecutor::new_with_overrides(&opts)
            .expect("second disabled new should succeed");
        assert!(
            !LAST_NEW_USED_CACHE.load(Ordering::SeqCst),
            "disabled path should never report a cache hit"
        );
        assert!(
            list_cwasm_files(tmp.path()).is_empty(),
            "SAGE_WASM_CACHE_DISABLE should leave the cache dir untouched"
        );
    }

    #[test]
    fn cached_executor_runs_python_after_warm_hit() {
        if !WasmPythonExecutor::is_available() {
            skip_if_no_runtime_emit();
            return;
        }
        let _guard = CACHE_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let tmp = tempfile::tempdir().expect("mk tempdir");
        let opts = CacheOverrides {
            cache_dir: Some(tmp.path().to_path_buf()),
            disabled: false,
        };

        // Cold → populate.
        let _ = WasmPythonExecutor::new_with_overrides(&opts)
            .expect("cold new should succeed");
        // Warm → deserialize.
        let exec = WasmPythonExecutor::new_with_overrides(&opts)
            .expect("warm new should succeed");
        assert!(
            LAST_NEW_USED_CACHE.load(Ordering::SeqCst),
            "expected this call to be served from cache"
        );
        let r = exec.execute("print(40 + 2)", "{}", 10);
        assert_eq!(r.exit_code, 0, "stderr: {}", r.stderr);
        assert!(r.stdout.contains("42"), "got: {}", r.stdout);
    }
}
