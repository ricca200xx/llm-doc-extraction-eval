"""Recover a JSON object from raw LLM output.

Handles the failure modes seen in practice: markdown code fences, prose around
the object, trailing commas, and output truncated by the token limit (open
strings, open brackets, a dangling key or a half-written member).
"""

from __future__ import annotations

import json
import re

_FENCE = re.compile(r"```(?:json)?\s*(.*?)(?:```|$)", re.DOTALL | re.IGNORECASE)
_TRAILING_COMMA = re.compile(r",(\s*[}\]])")
_CLOSER = {"{": "}", "[": "]"}


def repair_json(text: str) -> tuple[dict | None, str]:
    """Return ``(obj, status)`` where status is ``ok``, ``repaired`` or ``failed``."""
    text = (text or "").strip()
    obj = _loads_object(text)
    if obj is not None:
        return obj, "ok"

    fence = _FENCE.search(text)
    if fence:
        text = fence.group(1)
    for brace in re.finditer(r"\{", text):  # prose may contain a stray "{" before the object
        obj = _complete(text[brace.start() :])
        if obj is not None:
            return obj, "repaired"
    return None, "failed"


def _loads_lenient(text: str) -> dict | None:
    """Parse, retrying once without trailing commas (only if the plain parse fails,
    so a ", }" inside a string value is never touched on valid input)."""
    return _loads_object(text) or _loads_object(_TRAILING_COMMA.sub(r"\1", text))


def _loads_object(text: str) -> dict | None:
    try:
        obj = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None
    return obj if isinstance(obj, dict) else None


def _complete(fragment: str) -> dict | None:
    """Parse ``fragment`` (starting at ``{``), closing it if it was cut short.

    Scans once, remembering the bracket stack at every comma outside a string.
    If the object closes, prose after it is dropped. If it never closes, the
    tail is kept only when it ends on a finished string or container; otherwise
    (open string, dangling key, a number that may itself be cut short) the last
    member is dropped by cutting at the previous comma. A truncated value is
    never kept, so truncation shows up as a miss, not as a wrong value.
    """
    stack: list[str] = []
    commas: list[tuple[int, tuple[str, ...]]] = []
    in_str = esc = False
    for i, ch in enumerate(fragment):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch in _CLOSER:
            stack.append(_CLOSER[ch])
        elif ch in "}]":
            if stack:
                stack.pop()
            if not stack:
                return _loads_lenient(fragment[: i + 1])
        elif ch == ",":
            commas.append((i, tuple(stack)))

    candidates = []
    tail = fragment.rstrip()
    if not in_str and tail.endswith(('"', "}", "]")):
        candidates.append((tail, tuple(stack)))
    candidates += [(fragment[:pos], st) for pos, st in reversed(commas)]
    for body, open_stack in candidates:
        obj = _loads_lenient(body + "".join(reversed(open_stack)))
        if obj is not None:
            return obj
    return None
