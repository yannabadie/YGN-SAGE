"""Upload extracted NotebookLM sources to ExoCortex FileSearch store."""
import asyncio
import os
import sys
from pathlib import Path

EXPORT_DIR = Path.home() / ".sage" / "notebooklm_export"


async def main():
    api_key = os.environ.get("GOOGLE_API_KEY")
    store = os.environ.get("SAGE_EXOCORTEX_STORE")
    if not api_key or not store:
        print("ERROR: Set GOOGLE_API_KEY and SAGE_EXOCORTEX_STORE")
        sys.exit(1)

    sys.path.insert(0, str(Path(__file__).parent.parent / "sage-python" / "src"))
    from sage.memory.remote_rag import ExoCortex

    exo = ExoCortex()
    uploaded = 0
    errors = 0

    # Collect all markdown files from all notebook subdirs
    files = sorted(EXPORT_DIR.rglob("*.md"))
    print(f"Found {len(files)} files to upload")

    for f in files:
        notebook = f.parent.name
        display = f"{notebook}: {f.stem[:60]}"
        try:
            await exo.upload(str(f), display)
            uploaded += 1
            if uploaded % 20 == 0:
                print(f"  Uploaded {uploaded}/{len(files)}...")
        except Exception as e:
            errors += 1
            if "already been terminated" not in str(e):
                print(f"  ERROR ({f.name[:40]}): {e}", file=sys.stderr)

    print(f"\nDone: {uploaded} uploaded, {errors} errors")


if __name__ == "__main__":
    asyncio.run(main())
