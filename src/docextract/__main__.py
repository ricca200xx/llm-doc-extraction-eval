"""CLI: ``docextract run --model M --prompt v2`` then ``docextract report``."""

from __future__ import annotations

import argparse
from collections.abc import Callable

PROMPTS = ("v1", "v2")
TEST_SPLIT_SIZE = 100  # receipts in the CORD-v2 test split


def _int_between(low: int, high: int) -> Callable[[str], int]:
    def parse(text: str) -> int:
        try:
            value = int(text)
        except ValueError:
            raise argparse.ArgumentTypeError(f"expected an integer, got {text!r}") from None
        if not low <= value <= high:
            raise argparse.ArgumentTypeError(f"must be between {low} and {high}, got {value}")
        return value

    return parse


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="docextract",
        description="Evaluate vision-LLM receipt extraction on CORD-v2.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="extract fields with one model and one prompt version")
    run.add_argument(
        "--model", required=True, help="LiteLLM model string, e.g. ollama_chat/qwen2.5vl:3b"
    )
    run.add_argument("--prompt", choices=PROMPTS, required=True, help="prompt version")
    run.add_argument(
        "--n",
        type=_int_between(1, TEST_SPLIT_SIZE),
        default=50,
        help=f"number of receipts to use (1-{TEST_SPLIT_SIZE}, default 50)",
    )
    run.add_argument(
        "--start",
        type=_int_between(0, TEST_SPLIT_SIZE - 1),
        default=0,
        help="index of the first receipt in the test split (default 0)",
    )
    run.add_argument(
        "--max-tokens",
        type=_int_between(16, 8192),
        default=512,
        help="output token limit per call (default 512)",
    )
    show = sub.add_parser("prompt", help="print the exact messages a prompt version sends")
    show.add_argument("version", choices=PROMPTS)
    sub.add_parser("report", help="score all runs, write results/report.md and chart.png")
    args = parser.parse_args()

    if args.command == "run":
        from .data import load_docs
        from .extract import run as run_extraction

        if args.start + args.n > TEST_SPLIT_SIZE:
            parser.error(f"--start + --n must be at most {TEST_SPLIT_SIZE}")
        docs = load_docs(args.n, args.start)
        run_extraction(args.model, args.prompt, docs, args.max_tokens)
    elif args.command == "prompt":
        from .prompts import build_messages

        for message in build_messages(args.version, "<base64 JPEG>"):
            content = message["content"]
            parts = content if isinstance(content, list) else [{"type": "text", "text": content}]
            print(f"[{message['role']}]")
            for part in parts:
                print(part["text"] if part["type"] == "text" else "<image>")
            print()
    else:
        from .report import main as report_main

        report_main()


if __name__ == "__main__":
    main()
