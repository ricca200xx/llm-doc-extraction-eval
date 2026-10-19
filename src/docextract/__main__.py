"""CLI: ``docextract run --model M --prompt v2`` then ``docextract report``."""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(prog="docextract")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="extract fields with one model and prompt version")
    run.add_argument("--model", required=True, help="LiteLLM model string")
    run.add_argument("--prompt", choices=["v1", "v2"], required=True)
    run.add_argument("--n", type=int, default=50, help="first N test receipts")
    run.add_argument("--max-tokens", type=int, default=512)
    sub.add_parser("report", help="score all runs, write results/report.md and chart.png")
    args = parser.parse_args()

    if args.command == "run":
        from .data import load_docs
        from .extract import run as run_extraction

        if args.n < 1:
            parser.error("--n must be at least 1")
        run_extraction(args.model, args.prompt, load_docs(args.n), args.max_tokens)
    else:
        from .report import main as report_main

        report_main()


if __name__ == "__main__":
    main()
