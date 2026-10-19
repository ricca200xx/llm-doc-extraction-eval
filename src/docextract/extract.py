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
from .prompts import build_messages, prompt_fingerprint
from .repair import repair_json
from .schema import Receipt

RAW_DIR = Path("results/raw")
TEMPERATURE = 0
RETRIES = 2  # LiteLLM retries on transient provider errors (needs tenacity)
TIMEOUT_S = 300
MAX_CONSECUTIVE_ERRORS = 3  # a wrong model name or key fails every call: stop early
_NO_RESULT = {"raw": "", "finish_reason": None, "completion_tokens": None}


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
        temperature=TEMPERATURE,
        max_tokens=max_tokens,
        num_retries=RETRIES,
        timeout=TIMEOUT_S,
    )
    if not resp.choices:
        raise RuntimeError("provider returned a response with no choices")
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


def _completed(path: Path, prompt: str, max_tokens: int) -> set[str]:
    """Doc ids already done; refuses to mix results from a different prompt text or limit."""
    records = read_records(path)
    current = {"prompt_sha": prompt_fingerprint(prompt), "max_tokens": max_tokens}
    for key, value in current.items():
        # Records written before these keys existed carry no value and are accepted.
        seen = {r[key] for r in records if r.get(key) is not None} - {value}
        if seen:
            raise SystemExit(
                f"{path} holds results with {key}={sorted(seen)}, but this run uses {value}. "
                "Delete or rename that file to start a fresh run."
            )
    return {r["doc_id"] for r in records if r.get("error") is None}


def _end_with_newline(path: Path) -> None:
    """Never glue a new record onto a partial line left by a killed run."""
    if path.exists() and path.stat().st_size and not path.read_text().endswith("\n"):
        with path.open("a") as out:
            out.write("\n")


def _extract_one(model: str, prompt: str, doc: Doc, max_tokens: int) -> dict:
    start = time.perf_counter()
    try:
        result, error = _call(model, build_messages(prompt, doc.image_b64), max_tokens), None
    except Exception as exc:  # provider errors vary; record them and let the caller decide
        result, error = _NO_RESULT, f"{type(exc).__name__}: {exc}"
    pred, status = parse_output(result["raw"])
    return {
        "doc_id": doc.doc_id,
        "model": model,
        "prompt": prompt,
        "prompt_sha": prompt_fingerprint(prompt),
        "max_tokens": max_tokens,
        "gold": doc.gold.model_dump(),
        "pred": pred,
        "parse_status": status if error is None else "error",
        "error": error,
        "latency_s": round(time.perf_counter() - start, 2),
        **result,
    }


def run(model: str, prompt: str, docs: Iterable[Doc], max_tokens: int = 512) -> Path:
    path = run_path(model, prompt)
    path.parent.mkdir(parents=True, exist_ok=True)
    _end_with_newline(path)
    done = _completed(path, prompt, max_tokens)
    todo = [d for d in docs if d.doc_id not in done]
    print(f"{model} / {prompt}: {len(done)} done, {len(todo)} to go -> {path}", flush=True)
    consecutive_errors = 0
    with path.open("a") as out:
        for i, doc in enumerate(todo, 1):
            record = _extract_one(model, prompt, doc, max_tokens)
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            error = record["error"]
            print(
                f"  [{i}/{len(todo)}] {doc.doc_id} {record['parse_status']} "
                f"{record['latency_s']}s" + (f"  {error[:120]}" if error else ""),
                flush=True,
            )
            consecutive_errors = consecutive_errors + 1 if error else 0
            if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                raise SystemExit(
                    f"Stopped after {consecutive_errors} failed calls in a row. Check the model "
                    f"name and the provider's API key. Last error: {error[:300]}"
                )
    return path
