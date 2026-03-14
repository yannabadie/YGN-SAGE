#![no_main]
use libfuzzer_sys::fuzz_target;
use sage_core::verification::smt::SmtVerifier;

fuzz_target!(|data: &[u8]| {
    if let Ok(s) = std::str::from_utf8(data) {
        let verifier = SmtVerifier::new();
        let _ = verifier.verify_invariant(s, "x > 0");
        let _ = verifier.verify_invariant("x > 0", s);
        let _ = verifier.verify_arithmetic_expr(s, 0, 100);
    }
});
