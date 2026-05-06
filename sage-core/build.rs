//! Build script — wires the RustPython wasm runtime into OUT_DIR so
//! `sandbox::wasm_python` can `include_bytes!` it.
//!
//! When the `sandbox` feature is enabled we look for a compiled
//! `rustpython.wasm` at a well-known path relative to the
//! repository. If it exists, we copy it into OUT_DIR under a stable
//! filename. If it doesn't, the default behaviour is to emit an empty
//! placeholder so `include_bytes!` compiles — the runtime detects the
//! zero-length bytes and short-circuits via
//! `WasmPythonInitError::BytesMissing`. This keeps fresh clones
//! buildable (a `cargo build` with no prior RustPython compile still
//! succeeds) at the cost of a runtime-only sandbox-missing error.
//!
//! Release / CI builds that need the sandbox to be present can opt
//! into a compile-time check by setting `SAGE_REQUIRE_WASM=1` in the
//! build environment. With that flag set, a missing `rustpython.wasm`
//! is a hard `panic!` (build fails loudly) instead of a silent
//! placeholder. This addresses the ALIRE audit's "assert non-empty
//! sandbox artifact in CI" recommendation without breaking fresh-
//! clone developer workflows.
//!
//! Build recipe for the source `.wasm` (one-time, cached in
//! `external/`):
//!
//! ```text
//! rustup target add wasm32-wasip1
//! git clone https://github.com/RustPython/RustPython external/rustpython
//! cd external/rustpython
//! CARGO_TARGET_DIR=../rustpython-wasm-target \
//!   cargo build --release --target wasm32-wasip1 --features freeze-stdlib
//! ```

use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

fn main() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));
    let out_wasm = out_dir.join("rustpython.wasm");

    // Cycle-13 B Q1 (cgpro post-push 2026-05-06 NEXT_BLOCK_ID=G):
    // expose `sage_core.__commit_sha__` / `__build_timestamp__` /
    // `__build_profile__` so Python can detect a stale Rust binary
    // (the operationally-observed gap that wasted operator cycles
    // until commit `32d39bdf` shipped the regression test).
    //
    // `git rev-parse HEAD` runs at build time relative to the
    // sage-core manifest dir; an override env var
    // `SAGE_CORE_COMMIT_SHA_OVERRIDE` lets CI inject a known SHA
    // when building outside a git checkout (PyPI wheel build, etc).
    // Falls back to "unknown" gracefully when git is absent.
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("manifest"));
    let commit_sha = env::var("SAGE_CORE_COMMIT_SHA_OVERRIDE")
        .ok()
        .filter(|s| !s.is_empty())
        .or_else(|| {
            Command::new("git")
                .args(["rev-parse", "HEAD"])
                .current_dir(&manifest_dir)
                .output()
                .ok()
                .filter(|o| o.status.success())
                .and_then(|o| {
                    let raw = String::from_utf8_lossy(&o.stdout).trim().to_string();
                    if raw.is_empty() {
                        None
                    } else {
                        Some(raw)
                    }
                })
        })
        .unwrap_or_else(|| "unknown".to_string());
    println!("cargo:rustc-env=SAGE_CORE_BUILD_COMMIT_SHA={commit_sha}");

    let build_ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs().to_string())
        .unwrap_or_else(|_| "0".to_string());
    println!("cargo:rustc-env=SAGE_CORE_BUILD_TIMESTAMP={build_ts}");

    let profile = env::var("PROFILE").unwrap_or_else(|_| "unknown".to_string());
    println!("cargo:rustc-env=SAGE_CORE_BUILD_PROFILE={profile}");

    // Re-run build.rs when HEAD moves so the embedded SHA stays
    // current. Per cgpro HARD_STOP 2026-05-06 (conv
    // `cgpro_pi_mono_pivot_20260505`): hard-coding `../.git/HEAD` was
    // wrong for two edge cases:
    //   (1) PyPI / sdist source builds where .git is absent (Cargo
    //       treats a non-existent rerun-trigger as "rebuild every
    //       time", which destroys cache discipline).
    //   (2) Git worktrees where the .git is a file pointing at the
    //       real git-dir, not a directory at the conventional path.
    //
    // Fix: use `git rev-parse --git-path <sub>` to resolve each path
    // through git's own logic (handles worktrees + non-default
    // git-dir layouts), and emit `cargo:rerun-if-changed` ONLY for
    // paths that actually exist at build-script-run time.
    fn git_path(cwd: &PathBuf, sub: &str) -> Option<PathBuf> {
        Command::new("git")
            .args(["rev-parse", "--git-path", sub])
            .current_dir(cwd)
            .output()
            .ok()
            .filter(|o| o.status.success())
            .and_then(|o| {
                let raw = String::from_utf8_lossy(&o.stdout).trim().to_string();
                if raw.is_empty() {
                    None
                } else {
                    Some(PathBuf::from(raw))
                }
            })
    }

    for sub in ["HEAD", "refs/heads", "packed-refs"] {
        if let Some(path) = git_path(&manifest_dir, sub) {
            // git rev-parse --git-path returns paths relative to cwd
            // (the manifest dir we passed in). Resolve to absolute
            // before existence-check so the rerun trigger points at
            // the right file regardless of cargo's cwd at build time.
            let absolute = if path.is_absolute() {
                path.clone()
            } else {
                manifest_dir.join(&path)
            };
            if absolute.exists() {
                println!("cargo:rerun-if-changed={}", absolute.display());
            }
        }
    }
    println!("cargo:rerun-if-env-changed=SAGE_CORE_COMMIT_SHA_OVERRIDE");

    // Always emit a placeholder so include_bytes! compiles regardless
    // of the sandbox feature state. sage-core/src/sandbox/wasm_python.rs
    // is already cfg-gated on the sandbox feature; this placeholder is
    // for the include_bytes! macro path on any build.
    let mut wrote_placeholder = true;

    if env::var("CARGO_FEATURE_SANDBOX").is_ok() {
        // Candidate path for a locally-built RustPython wasm.
        // sage-core lives at <repo>/sage-core; the wasm artifact is
        // at <repo>/external/rustpython-wasm-target/wasm32-wasip1/release/rustpython.wasm.
        let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("manifest"));
        let candidate = manifest_dir
            .parent()
            .expect("parent of sage-core")
            .join("external/rustpython-wasm-target/wasm32-wasip1/release/rustpython.wasm");

        if candidate.exists() {
            fs::copy(&candidate, &out_wasm).expect("copy rustpython.wasm to OUT_DIR");
            println!("cargo:rerun-if-changed={}", candidate.display());
            wrote_placeholder = false;
            println!(
                "cargo:warning=sage-core: embedded rustpython.wasm from {}",
                candidate.display()
            );
        } else if env::var_os("SAGE_REQUIRE_WASM")
            .map(|v| v != "0" && !v.is_empty())
            .unwrap_or(false)
        {
            // Strict mode: the operator declared this build must ship
            // a real sandbox artifact. Fail the build loudly instead
            // of silently producing a placeholder-backed binary that
            // runtime-fails the first time someone tries to run
            // Python in the sandbox.
            panic!(
                "SAGE_REQUIRE_WASM=1 but rustpython.wasm not found at {} — \
                 build it with the recipe in sage-core/src/sandbox/wasm_python.rs's \
                 module docstring, or unset SAGE_REQUIRE_WASM to fall back to the \
                 placeholder + runtime-fail-close behaviour.",
                candidate.display()
            );
        } else {
            println!(
                "cargo:warning=sage-core: sandbox feature enabled but rustpython.wasm not found at {} — the Wasm Python sandbox will report WasmPythonInitError::BytesMissing at runtime. Set SAGE_REQUIRE_WASM=1 to turn this into a build-time error. Build the wasm with the recipe in sage-core/src/sandbox/wasm_python.rs's module docstring.",
                candidate.display()
            );
        }
        // Let cargo know to re-run build.rs if the flag changes.
        println!("cargo:rerun-if-env-changed=SAGE_REQUIRE_WASM");
    }

    if wrote_placeholder {
        // Empty placeholder; wasm_python::RUSTPYTHON_WASM.is_empty()
        // is the runtime check callers use.
        fs::write(&out_wasm, b"").expect("write placeholder rustpython.wasm");
    }

    println!("cargo:rerun-if-changed=build.rs");
}
