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
    fn test_embed_single() {
        if skip_if_no_model() { return; }
        let mut emb = RustEmbedder::new(model_path(), tokenizer_path()).unwrap();
        let vec = emb.embed("Hello world").unwrap();
        assert_eq!(vec.len(), 384);
        let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_embed_batch() {
        if skip_if_no_model() { return; }
        let mut emb = RustEmbedder::new(model_path(), tokenizer_path()).unwrap();
        let vecs = emb.embed_batch(vec![
            "Hello".into(), "World".into(), "Rust is fast".into()
        ]).unwrap();
        assert_eq!(vecs.len(), 3);
        assert!(vecs.iter().all(|v| v.len() == 384));
    }

    #[test]
    fn test_embed_deterministic() {
        if skip_if_no_model() { return; }
        let mut emb = RustEmbedder::new(model_path(), tokenizer_path()).unwrap();
        let v1 = emb.embed("deterministic test").unwrap();
        let v2 = emb.embed("deterministic test").unwrap();
        assert_eq!(v1, v2);
    }

    #[test]
    fn test_embed_empty_batch() {
        if skip_if_no_model() { return; }
        let mut emb = RustEmbedder::new(model_path(), tokenizer_path()).unwrap();
        let vecs = emb.embed_batch(vec![]).unwrap();
        assert!(vecs.is_empty());
    }

    #[test]
    fn test_similar_texts_closer() {
        if skip_if_no_model() { return; }
        let mut emb = RustEmbedder::new(model_path(), tokenizer_path()).unwrap();
        let cat = emb.embed("I love cats").unwrap();
        let dog = emb.embed("I love dogs").unwrap();
        let code = emb.embed("fn main() { println!(\"hello\"); }").unwrap();

        let sim_cat_dog: f32 = cat.iter().zip(&dog).map(|(a, b)| a * b).sum();
        let sim_cat_code: f32 = cat.iter().zip(&code).map(|(a, b)| a * b).sum();

        assert!(sim_cat_dog > sim_cat_code,
            "Expected cat-dog ({:.3}) > cat-code ({:.3})", sim_cat_dog, sim_cat_code);
    }
}
