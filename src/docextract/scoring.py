"""Strict per-field exact match.

Each (document, field) pair lands in exactly one cell:

    gold null,  pred null   -> correct abstention (counts for accuracy only)
    gold null,  pred value  -> hallucination: FP
    gold value, pred null   -> miss: FN
    gold value, pred same   -> TP
    gold value, pred other  -> wrong value: FP and FN

A document whose call failed or whose output could not be parsed has an empty
prediction, so its gold values are already FN; the ``failed`` mask also takes away
the credit it would get for "abstaining", making all its cells wrong for accuracy.

The only normalisation is number formatting, so ``Rp 60.000``, ``60,000`` and
``60000.00`` are the same amount. Any different digit is a wrong value.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

# One or two digits after the last separator are decimals; three are a thousands
# group (CORD receipts are mostly Indonesian: "60.000" is sixty thousand).
_DECIMALS = re.compile(r"[.,](\d{1,2})$")


def canonical_amount(value: str | None) -> str | None:
    if value is None or not re.search(r"\d", value):
        return None
    text = re.sub(r"\D+$", "", value)  # drop trailing currency, "%", ")" ...
    decimals = _DECIMALS.search(text)
    cents = decimals.group(1).rstrip("0") if decimals else ""
    if decimals:
        text = text[: decimals.start()]
    whole = re.sub(r"\D", "", text).lstrip("0") or "0"
    return f"{whole}.{cents}" if cents else whole


@dataclass(frozen=True)
class Scores:
    """Boolean (n_docs, n_fields) matrices; rows can be resampled for the bootstrap."""

    fields: tuple[str, ...]
    tp: np.ndarray
    fp: np.ndarray
    fn: np.ndarray
    gold_null: np.ndarray
    correct: np.ndarray

    @property
    def n_docs(self) -> int:
        return self.tp.shape[0]


def score(
    golds: Sequence[dict],
    preds: Sequence[dict],
    fields: Sequence[str],
    failed: Sequence[bool] | None = None,
) -> Scores:
    """``failed[i]`` marks a document with no usable output (API error, unparseable text)."""
    if len(golds) != len(preds):
        raise ValueError(f"{len(golds)} gold records but {len(preds)} predictions")
    no_output = np.zeros(len(golds), dtype=bool) if failed is None else np.asarray(failed, bool)
    g = np.array([[canonical_amount(d.get(f)) for f in fields] for d in golds], dtype=object)
    p = np.array([[canonical_amount(d.get(f)) for f in fields] for d in preds], dtype=object)
    g_null, p_null = g == None, p == None  # noqa: E711  (element-wise on object arrays)
    same = (g == p) & ~g_null
    wrong = ~g_null & ~p_null & ~same
    return Scores(
        fields=tuple(fields),
        tp=same,
        fp=(g_null & ~p_null) | wrong,
        fn=(~g_null & p_null) | wrong,
        gold_null=g_null,
        correct=(same | (g_null & p_null)) & ~no_output[:, None],
    )


def _rows(s: Scores, idx: np.ndarray | None) -> np.ndarray:
    return np.arange(s.n_docs) if idx is None else idx


def field_accuracy(s: Scores, idx: np.ndarray | None = None) -> np.ndarray:
    return s.correct[_rows(s, idx)].mean(axis=0)


def field_f1(s: Scores, idx: np.ndarray | None = None) -> np.ndarray:
    """F1 = 2TP / (2TP + FP + FN); NaN for a field with no gold and no predictions."""
    r = _rows(s, idx)
    tp, fp, fn = (m[r].sum(axis=0) for m in (s.tp, s.fp, s.fn))
    denom = 2 * tp + fp + fn
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denom > 0, 2 * tp / denom, np.nan)


def macro_accuracy(s: Scores, idx: np.ndarray | None = None) -> float:
    return float(field_accuracy(s, idx).mean())


def macro_f1(s: Scores, idx: np.ndarray | None = None) -> float:
    f1 = field_f1(s, idx)
    return float(np.nanmean(f1)) if np.any(~np.isnan(f1)) else float("nan")


def hallucination_rate(s: Scores, idx: np.ndarray | None = None) -> float:
    """Share of null-gold cells where the model still produced a value."""
    r = _rows(s, idx)
    null_cells = s.gold_null[r].sum()
    return float((s.fp[r] & s.gold_null[r]).sum() / null_cells) if null_cells else float("nan")
