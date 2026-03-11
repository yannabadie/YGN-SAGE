//! Integration tests for RustEmbedder (ONNX).
//!
//! These tests are #[ignore] by default because the test binary links PyO3,
//! which causes a deadlock on Windows when Python isn't initialized before
//! PyO3's static constructors run. Run via Python instead:
//!   maturin develop --features onnx && python -c "from sage_core import RustEmbedder; ..."
//!
//! To run these on Linux CI (where PyO3 init works):
//!   cargo test --features onnx --test test_embedder -- --ignored

#[cfg(feature = "onnx")]
mod onnx_embedder_tests {
    use sage_core::memory::embedder::RustEmbedder;

    fn model_path() -> String {
        let base = std::env::var("CARGO_MANIFEST_DIR").unwrap();
        format!("{}/models/model.onnx", base)
    }

    fn tokenizer_path() -> String {
        let base = std::env::var("CARGO_MANIFEST_DIR").unwrap();
        format!("{}/models/tokenizer.json", base)
    }

    fn skip_if_no_model() -> bool {
        !std::path::Path::new(&model_path()).exists()
    }

    #[test]
    #[ignore = "deadlocks on Windows — PyO3 static init before main(). Run via Python or Linux CI."]
    fn test_embed_single() {
        pyo3::prepare_freethreaded_python();
        if skip_if_no_model() {
            return;
        }
        pyo3::Python::with_gil(|py| {
            let mut emb = RustEmbedder::new(py, model_path(), tokenizer_path()).unwrap();
            let vec = emb.embed("Hello world").unwrap();
            assert_eq!(vec.len(), 384);
            let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!((norm - 1.0).abs() < 0.01);
        });
    }

    #[test]
    #[ignore = "deadlocks on Windows — PyO3 static init before main(). Run via Python or Linux CI."]
    fn test_embed_batch() {
        pyo3::prepare_freethreaded_python();
        if skip_if_no_model() {
            return;
        }
        pyo3::Python::with_gil(|py| {
            let mut emb = RustEmbedder::new(py, model_path(), tokenizer_path()).unwrap();
            let vecs = emb
                .embed_batch(vec!["Hello".into(), "World".into(), "Rust is fast".into()])
                .unwrap();
            assert_eq!(vecs.len(), 3);
            assert!(vecs.iter().all(|v| v.len() == 384));
        });
    }

    #[test]
    #[ignore = "deadlocks on Windows — PyO3 static init before main(). Run via Python or Linux CI."]
    fn test_embed_deterministic() {
        pyo3::prepare_freethreaded_python();
        if skip_if_no_model() {
            return;
        }
        pyo3::Python::with_gil(|py| {
            let mut emb = RustEmbedder::new(py, model_path(), tokenizer_path()).unwrap();
            let v1 = emb.embed("deterministic test").unwrap();
            let v2 = emb.embed("deterministic test").unwrap();
            assert_eq!(v1, v2);
        });
    }

    #[test]
    #[ignore = "deadlocks on Windows — PyO3 static init before main(). Run via Python or Linux CI."]
    fn test_embed_empty_batch() {
        pyo3::prepare_freethreaded_python();
        if skip_if_no_model() {
            return;
        }
        pyo3::Python::with_gil(|py| {
            let mut emb = RustEmbedder::new(py, model_path(), tokenizer_path()).unwrap();
            let vecs = emb.embed_batch(vec![]).unwrap();
            assert!(vecs.is_empty());
        });
    }

    #[test]
    #[ignore = "deadlocks on Windows — PyO3 static init before main(). Run via Python or Linux CI."]
    fn test_similar_texts_closer() {
        pyo3::prepare_freethreaded_python();
        if skip_if_no_model() {
            return;
        }
        pyo3::Python::with_gil(|py| {
            let mut emb = RustEmbedder::new(py, model_path(), tokenizer_path()).unwrap();
            let cat = emb.embed("I love cats").unwrap();
            let dog = emb.embed("I love dogs").unwrap();
            let code = emb.embed("fn main() { println!(\"hello\"); }").unwrap();

            let sim_cat_dog: f32 = cat.iter().zip(&dog).map(|(a, b)| a * b).sum();
            let sim_cat_code: f32 = cat.iter().zip(&code).map(|(a, b)| a * b).sum();

            assert!(
                sim_cat_dog > sim_cat_code,
                "Expected cat-dog ({:.3}) > cat-code ({:.3})",
                sim_cat_dog,
                sim_cat_code
            );
        });
    }
}
