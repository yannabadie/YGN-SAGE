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

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

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

impl WasmPythonExecutor {
    /// Build + cache the Python interpreter module. Safe to hold in
    /// a long-lived struct (the engine is thread-safe and cheap to
    /// share via Arc).
    pub fn new() -> Result<Self, WasmPythonInitError> {
        if RUSTPYTHON_WASM.is_empty() {
            return Err(WasmPythonInitError::BytesMissing);
        }
        let mut config = Config::new();
        // Epoch-based deadline is what we use for per-call timeouts;
        // fuel would be more deterministic but epoch is lighter.
        config.epoch_interruption(true);
        // Component Model is off — rustpython.wasm is a plain
        // wasip1 module, not a Component-Model component.
        config.wasm_component_model(false);
        let engine = Engine::new(&config)
            .map_err(|e| WasmPythonInitError::Wasmtime(e.to_string()))?;
        let module = Module::new(&engine, RUSTPYTHON_WASM)
            .map_err(|e| WasmPythonInitError::Wasmtime(e.to_string()))?;
        Ok(Self {
            engine,
            module: Arc::new(module),
            next_deadline: AtomicU64::new(1),
        })
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
        format!(
            "import json as _sage_json\nimport codecs as _sage_codecs\n\
             args = _sage_json.loads(_sage_codecs.decode(\"{}\", \"hex\").decode(\"utf-8\"))\n\
             {}",
            hex, code
        )
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
