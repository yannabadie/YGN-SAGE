//! Build script — wires the RustPython wasm runtime into OUT_DIR so
//! `sandbox::wasm_python` can `include_bytes!` it.
//!
//! When the `sandbox` feature is enabled we look for a compiled
//! `rustpython.wasm` at a well-known path relative to the
//! repository. If it exists, we copy it into OUT_DIR under a stable
//! filename. If it doesn't, we emit an empty placeholder so
//! `include_bytes!` compiles — the runtime detects the zero-length
//! bytes and short-circuits via `WasmPythonInitError::BytesMissing`.
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

fn main() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));
    let out_wasm = out_dir.join("rustpython.wasm");

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
        } else {
            println!(
                "cargo:warning=sage-core: sandbox feature enabled but rustpython.wasm not found at {} — the Wasm Python sandbox will report WasmPythonInitError::BytesMissing at runtime. Build it with the recipe in sage-core/src/sandbox/wasm_python.rs's module docstring.",
                candidate.display()
            );
        }
    }

    if wrote_placeholder {
        // Empty placeholder; wasm_python::RUSTPYTHON_WASM.is_empty()
        // is the runtime check callers use.
        fs::write(&out_wasm, b"").expect("write placeholder rustpython.wasm");
    }

    println!("cargo:rerun-if-changed=build.rs");
}
