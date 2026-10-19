"""Uncertainty and significance for small evaluation sets.

Everything resamples or pairs at the *document* level, because fields of the
same receipt are not independent.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from math import comb

import numpy as np

Statistic = Callable[[np.ndarray], float]


def bootstrap_ci(
    stat: Statistic, n_docs: int, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05
) -> tuple[float, float]:
    """Percentile bootstrap CI of ``stat(doc_indices)`` over documents."""
    idx = np.random.default_rng(seed).integers(0, n_docs, size=(n_boot, n_docs))
    values = np.array([stat(row) for row in idx])
    if np.isnan(values).all():  # e.g. F1 when no resample has a gold value or a prediction
        return float("nan"), float("nan")
    lo, hi = np.nanpercentile(values, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def paired_bootstrap_diff(
    stat_a: Statistic,
    stat_b: Statistic,
    n_docs: int,
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """``(diff, lo, hi)`` for ``stat_b - stat_a``, resampling the same docs for both."""
    all_docs = np.arange(n_docs)
    diff = stat_b(all_docs) - stat_a(all_docs)
    lo, hi = bootstrap_ci(lambda i: stat_b(i) - stat_a(i), n_docs, n_boot, seed, alpha)
    return float(diff), lo, hi


def mcnemar_exact(correct_a: Sequence[bool], correct_b: Sequence[bool]) -> float:
    """Two-sided exact McNemar p-value on paired per-document correctness."""
    a = np.asarray(correct_a, dtype=bool)
    b = np.asarray(correct_b, dtype=bool)
    only_a = int(np.sum(a & ~b))
    only_b = int(np.sum(~a & b))
    n = only_a + only_b
    if n == 0:
        return 1.0
    tail = sum(comb(n, k) for k in range(min(only_a, only_b) + 1)) / 2**n
    return min(1.0, 2 * tail)


def holm(pvalues: Sequence[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values, returned in the input order."""
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvalues[i]))
        adjusted[i] = running
    return adjusted
