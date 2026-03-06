"""CLI entry point for ``python -m discover.pipeline``."""
from __future__ import annotations

import argparse
import asyncio
import logging
from datetime import date

from discover.pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="YGN-SAGE Knowledge Pipeline")
    parser.add_argument(
        "--mode",
        choices=["nightly", "on-demand", "migrate", "watch"],
        default="nightly",
    )
    parser.add_argument(
        "--query",
        type=str,
        help="Search query (on-demand mode)",
    )
    parser.add_argument(
        "--domains",
        type=str,
        nargs="*",
        help="Restrict to specific domains",
    )
    parser.add_argument(
        "--since",
        type=str,
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    # --- Watch mode: detect new unprofiled models ---
    if args.mode == "watch":
        from discover.model_watcher import ModelWatcher

        watcher = ModelWatcher()
        report_text = asyncio.run(watcher.report())
        print(report_text)
        return

    since = date.fromisoformat(args.since) if args.since else None
    report = asyncio.run(
        run_pipeline(
            mode=args.mode,
            query=args.query,
            since=since,
            domains=args.domains,
        )
    )
    print(
        f"Pipeline complete: discovered={report.discovered}, "
        f"curated={report.curated}, ingested={report.ingested}"
    )


if __name__ == "__main__":
    main()
