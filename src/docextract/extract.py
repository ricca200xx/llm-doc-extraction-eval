"""Run one (model, prompt) pair over the documents through LiteLLM.

Any LiteLLM model string works: ``ollama_chat/qwen2.5vl:3b`` locally,
``ollama_chat/gemma4:31b-cloud`` on Ollama Cloud, ``openai/gpt-4.1-mini``,
``anthropic/claude-haiku-4-5`` or ``gemini/gemini-2.5-flash`` with the matching
API key in the environment. Each output is appended to a JSONL file as soon as
it arrives, so an interrupted run resumes where it stopped.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Iterable
from pathlib import Path

from .data import Doc
from .prompts import build_messages
from .repair import repair_json
from .schema import Receipt

RAW_DIR = Path("results/raw")


def run_path(model: str, prompt: str) -> Path:
    return RAW_DIR / f"{re.sub(r'[^A-Za-z0-9.-]+', '_', model)}__{prompt}.jsonl"


def parse_output(raw: str) -> tuple[dict, str]:
    """Raw text -> (schema-shaped prediction, parse status). Unparseable means all null."""
    obj, status = repair_json(raw)
    return Receipt.model_validate(obj or {}).model_dump(), status


def _call(model: str, messages: list[dict], max_tokens: int) -> dict:
    import litellm  # slow import; keep it out of the scoring path and the tests

    resp = litellm.completion(
        model=model,
        messages=messages,
        temperature=0,
        max_tokens=max_tokens,
        num_retries=2,
        timeout=300,
    )
    choice = resp.choices[0]
    usage = getattr(resp, "usage", None)
    return {
        "raw": choice.message.content or "",
        "finish_reason": choice.finish_reason,
        "completion_tokens": getattr(usage, "completion_tokens", None),
    }


def read_records(path: Path) -> list[dict]:
    """All records of a run file, skipping a partial line left by a killed run."""
    records = []
    for line in path.read_text().splitlines() if path.exists() else []:
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def _completed(path: Path) -> set[str]:
    return {r["doc_id"] for r in read_records(path) if r["error"] is None}


def run(model: str, prompt: str, docs: Iterable[Doc], max_tokens: int = 512) -> Path:
    path = run_path(model, prompt)
    path.parent.mkdir(parents=True, exist_ok=True)
    done = _completed(path)
    todo = [d for d in docs if d.doc_id not in done]
    print(f"{model} / {prompt}: {len(done)} done, {len(todo)} to go -> {path}", flush=True)
    with path.open("a") as out:
        if path.stat().st_size and not path.read_text().endswith("\n"):
            out.write("\n")  # never glue a new record onto a partial line
        for i, doc in enumerate(todo, 1):
            start = time.perf_counter()
            messages = build_messages(prompt, doc.image_b64)
            try:
                result, error = _call(model, messages, max_tokens), None
            except Exception as exc:  # provider errors vary; record and keep going
                result, error = (
                    {"raw": "", "finish_reason": None, "completion_tokens": None},
                    (f"{type(exc).__name__}: {exc}"),
                )
            pred, status = parse_output(result["raw"])
            record = {
                "doc_id": doc.doc_id,
                "model": model,
                "prompt": prompt,
                "gold": doc.gold.model_dump(),
                "pred": pred,
                "parse_status": status if error is None else "error",
                "error": error,
                "latency_s": round(time.perf_counter() - start, 2),
                **result,
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            print(
                f"  [{i}/{len(todo)}] {doc.doc_id} {record['parse_status']} "
                f"{record['latency_s']}s" + (f"  {error[:120]}" if error else ""),
                flush=True,
            )
    return path
