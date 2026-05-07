#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if let Ok(code) = std::str::from_utf8(data) {
        let _ = sage_core::sandbox::validator::validate_python_code(code);
    }
});
