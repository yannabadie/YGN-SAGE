"""Patch existing benchmark JSON files to add correct model metadata."""
import json
import glob
import sys

MODEL_INFO = {
    "model": "gemini-2.5-flash",
    "provider": "GoogleProvider",
    "tier": "budget",
}


def patch_file(path: str) -> bool:
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict) and data.get("model") == "unknown":
        data.update(MODEL_INFO)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"  Patched: {path}")
        return True
    return False


if __name__ == "__main__":
    search_dir = sys.argv[1] if len(sys.argv) > 1 else "docs/benchmarks"
    files = glob.glob(f"{search_dir}/*.json")
    patched = sum(patch_file(f) for f in files)
    print(f"\nPatched {patched}/{len(files)} files.")
