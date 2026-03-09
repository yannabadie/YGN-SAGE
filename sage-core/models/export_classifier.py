"""Export routellm/bert classifier to ONNX for Rust inference.

Usage:
    pip install optimum[onnxruntime] transformers
    python sage-core/models/export_classifier.py

Outputs:
    sage-core/models/classifier/model.onnx
    sage-core/models/classifier/tokenizer.json
"""
import sys
from pathlib import Path


def main():
    try:
        from optimum.onnxruntime import ORTModelForSequenceClassification
        from transformers import AutoTokenizer
    except ImportError:
        print("Install: pip install optimum[onnxruntime] transformers")
        sys.exit(1)

    model_name = "routellm/bert"
    out_dir = Path(__file__).parent / "classifier"
    out_dir.mkdir(exist_ok=True)

    print(f"Exporting {model_name} to ONNX...")
    model = ORTModelForSequenceClassification.from_pretrained(
        model_name, export=True
    )
    model.save_pretrained(out_dir)

    print("Exporting tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(out_dir)
    # Save the fast tokenizer JSON for the Rust `tokenizers` crate
    if hasattr(tokenizer, "backend_tokenizer"):
        tokenizer.backend_tokenizer.save(str(out_dir / "tokenizer.json"))

    print(
        f"Done. Model: {out_dir / 'model.onnx'}, "
        f"Tokenizer: {out_dir / 'tokenizer.json'}"
    )

    # Optional: dynamic int8 quantization for faster CPU inference
    try:
        from optimum.onnxruntime import ORTQuantizer
        from optimum.onnxruntime.configuration import AutoQuantizationConfig

        print("Quantizing to int8...")
        quantizer = ORTQuantizer.from_pretrained(out_dir)
        qconfig = AutoQuantizationConfig.avx512_vnni(is_static=False)
        quantizer.quantize(save_dir=out_dir, quantization_config=qconfig)
        print(f"Quantized: {out_dir / 'model_quantized.onnx'}")
    except Exception as e:
        print(f"Quantization skipped: {e}")


if __name__ == "__main__":
    main()
