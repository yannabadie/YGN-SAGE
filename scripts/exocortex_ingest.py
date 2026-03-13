#!/usr/bin/env python3
"""Ingest papers into ExoCortex File Search store.

Supports:
  - Local PDF files
  - arXiv IDs (auto-downloads PDF)
  - Batch mode via --manifest JSON file

Usage:
  python scripts/exocortex_ingest.py --pdf path/to/paper.pdf --title "Paper Title"
  python scripts/exocortex_ingest.py --arxiv 2406.18665 --title "RouteLLM"
  python scripts/exocortex_ingest.py --manifest scripts/missing_papers.json
  python scripts/exocortex_ingest.py --list                    # List all docs in store
  python scripts/exocortex_ingest.py --delete DOC_RESOURCE_NAME  # Delete a document

Env: GOOGLE_API_KEY required. SSL verify disabled for corporate proxy.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*urllib3.*")
warnings.filterwarnings("ignore", message=".*Unverified HTTPS.*")

STORE_ID = "fileSearchStores/ygnsageresearch-wii7kwkqozrd"
DOWNLOAD_DIR = Path(__file__).resolve().parent.parent / "Researches"


def _get_client():
    """Create a google-genai client with SSL bypass."""
    import httpx
    from google import genai

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("ERROR: GOOGLE_API_KEY not set")
        sys.exit(1)

    client = genai.Client(
        api_key=api_key, http_options={"api_version": "v1beta"}
    )
    client._api_client._httpx_client = httpx.Client(verify=False, timeout=300)
    return client


def download_arxiv(arxiv_id: str, dest_dir: Path | None = None) -> Path:
    """Download a paper from arXiv by ID. Returns local PDF path."""
    import httpx

    dest_dir = dest_dir or DOWNLOAD_DIR
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Normalize ID (strip version if present for download, keep for filename)
    clean_id = arxiv_id.strip()
    # arXiv PDF URL
    url = f"https://arxiv.org/pdf/{clean_id}"

    # Determine filename
    pdf_name = f"{clean_id.replace('/', '_')}.pdf"
    dest_path = dest_dir / pdf_name

    if dest_path.exists():
        print(f"  Already downloaded: {dest_path}")
        return dest_path

    print(f"  Downloading {url} ...")
    with httpx.Client(verify=False, timeout=120, follow_redirects=True) as http:
        resp = http.get(url)
        resp.raise_for_status()

        if "application/pdf" not in resp.headers.get("content-type", ""):
            # arXiv sometimes returns HTML for invalid IDs
            raise ValueError(
                f"Expected PDF, got {resp.headers.get('content-type')}. "
                f"Check arXiv ID: {arxiv_id}"
            )

        dest_path.write_bytes(resp.content)
        size_mb = len(resp.content) / (1024 * 1024)
        print(f"  Saved: {dest_path} ({size_mb:.1f} MB)")

    return dest_path


def upload_to_store(
    client, pdf_path: Path, display_name: str, store_id: str = STORE_ID
) -> bool:
    """Upload a PDF to the ExoCortex File Search store. Returns True on success."""
    print(f"  Uploading {pdf_path.name} as '{display_name}' ...")

    try:
        operation = client.file_search_stores.upload_to_file_search_store(
            file=str(pdf_path),
            file_search_store_name=store_id,
            config={"display_name": display_name, "mimeType": "application/pdf"},
        )

        # Poll until done
        attempts = 0
        while not operation.done:
            time.sleep(3)
            operation = client.operations.get(operation)
            attempts += 1
            if attempts > 60:  # 3 min timeout
                print(f"  TIMEOUT waiting for upload of {pdf_path.name}")
                return False
            if attempts % 5 == 0:
                print(f"  ... still processing ({attempts * 3}s)")

        print(f"  Upload complete: {display_name}")
        return True

    except Exception as e:
        print(f"  ERROR uploading {pdf_path.name}: {e}")
        return False


def verify_upload(client, title_fragment: str, store_id: str = STORE_ID) -> bool:
    """Verify a paper is searchable by querying the store."""
    from google.genai import types

    tools = [
        types.Tool(
            file_search=types.FileSearch(file_search_store_names=[store_id])
        )
    ]

    result = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=f"Find the paper titled '{title_fragment}'. Return only the exact title if found, or 'NOT FOUND'.",
        config=types.GenerateContentConfig(tools=tools, temperature=0.0),
    )
    text = result.text or ""
    found = "NOT FOUND" not in text.upper()
    if found:
        print(f"  Verified: '{title_fragment}' is searchable")
    else:
        print(f"  WARNING: '{title_fragment}' not found in search (may need indexing time)")
    return found


def list_documents(client, store_id: str = STORE_ID) -> list[dict]:
    """List all documents in the store."""
    docs = list(client.file_search_stores.documents.list(parent=store_id))
    return [
        {
            "name": doc.name,
            "display_name": doc.display_name or "(unnamed)",
            "size_bytes": getattr(doc, "size_bytes", None),
        }
        for doc in docs
    ]


def delete_document(client, doc_name: str) -> bool:
    """Delete a document from the store (force=True to delete with chunks)."""
    from google.genai import types

    try:
        client.file_search_stores.documents.delete(
            name=doc_name,
            config=types.DeleteDocumentConfig(force=True),
        )
        print(f"  Deleted: {doc_name}")
        return True
    except Exception as e:
        print(f"  ERROR deleting {doc_name}: {e}")
        return False


def ingest_paper(
    client,
    *,
    arxiv_id: str | None = None,
    pdf_path: str | None = None,
    title: str,
    verify: bool = True,
) -> bool:
    """Ingest a single paper (download if arXiv, then upload)."""
    print(f"\n--- Ingesting: {title} ---")

    # Get/download the PDF
    if arxiv_id:
        try:
            local_pdf = download_arxiv(arxiv_id)
        except Exception as e:
            print(f"  FAILED to download arXiv {arxiv_id}: {e}")
            return False
    elif pdf_path:
        local_pdf = Path(pdf_path)
        if not local_pdf.exists():
            print(f"  ERROR: File not found: {pdf_path}")
            return False
    else:
        print("  ERROR: Must provide --arxiv or --pdf")
        return False

    # Check file size (File Search has limits)
    size_mb = local_pdf.stat().st_size / (1024 * 1024)
    if size_mb > 50:
        print(f"  WARNING: File is {size_mb:.1f} MB - may exceed upload limits")

    # Build display name
    display = title
    if arxiv_id:
        display = f"{title} [arXiv:{arxiv_id}]"

    # Upload
    ok = upload_to_store(client, local_pdf, display)
    if not ok:
        return False

    # Verify (optional, costs an API call)
    if verify:
        time.sleep(5)  # Give indexing a moment
        verify_upload(client, title)

    return True


def batch_ingest(client, manifest_path: str, verify: bool = True) -> dict:
    """Ingest papers from a JSON manifest file.

    Manifest format: [{"arxiv_id": "...", "title": "..."}, {"pdf": "...", "title": "..."}]
    """
    manifest = json.loads(Path(manifest_path).read_text())
    results = {"success": 0, "failed": 0, "skipped": 0, "papers": []}

    for entry in manifest:
        title = entry.get("title", "Unknown")
        arxiv_id = entry.get("arxiv_id")
        pdf = entry.get("pdf")

        ok = ingest_paper(
            client,
            arxiv_id=arxiv_id,
            pdf_path=pdf,
            title=title,
            verify=verify,
        )
        status = "success" if ok else "failed"
        results[status] += 1
        results["papers"].append({"title": title, "status": status})

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Ingest papers into ExoCortex File Search store"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--arxiv", help="arXiv paper ID (e.g., 2406.18665)")
    group.add_argument("--pdf", help="Path to local PDF file")
    group.add_argument("--manifest", help="JSON manifest for batch ingest")
    group.add_argument("--list", action="store_true", help="List all documents in store")
    group.add_argument("--delete", help="Delete a document by resource name")
    group.add_argument("--count", action="store_true", help="Count documents in store")

    parser.add_argument("--title", help="Paper title (required for --arxiv/--pdf)")
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip post-upload verification query",
    )
    parser.add_argument(
        "--store",
        default=STORE_ID,
        help=f"File Search store ID (default: {STORE_ID})",
    )

    args = parser.parse_args()
    client = _get_client()

    if args.list:
        docs = list_documents(client, args.store)
        print(f"Total documents: {len(docs)}\n")
        for i, d in enumerate(docs, 1):
            size = f" ({d['size_bytes']} bytes)" if d["size_bytes"] else ""
            print(f"{i:4d}. {d['display_name']}{size}")
        return

    if args.count:
        docs = list_documents(client, args.store)
        print(f"Documents in store: {len(docs)}")
        return

    if args.delete:
        delete_document(client, args.delete)
        return

    if args.manifest:
        results = batch_ingest(client, args.manifest, verify=not args.no_verify)
        print(f"\n=== Batch Results ===")
        print(f"Success: {results['success']}")
        print(f"Failed:  {results['failed']}")
        for p in results["papers"]:
            status_icon = "OK" if p["status"] == "success" else "FAIL"
            print(f"  [{status_icon}] {p['title']}")
        return

    # Single paper
    if not args.title and not args.arxiv:
        parser.error("--title is required for --pdf")

    title = args.title or f"arXiv {args.arxiv}"
    ok = ingest_paper(
        client,
        arxiv_id=args.arxiv,
        pdf_path=args.pdf,
        title=title,
        verify=not args.no_verify,
    )
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
