"""Score every run in results/raw and write report.md, summary.json and chart.png."""

from __future__ import annotations

import json
from collections import Counter
from functools import partial
from pathlib import Path

import numpy as np

from .extract import read_records
from .schema import FIELDS
from .scoring import (
    Scores,
    field_accuracy,
    field_f1,
    hallucination_rate,
    macro_accuracy,
    macro_f1,
    score,
)
from .stats import bootstrap_ci, holm, mcnemar_exact, paired_bootstrap_diff

RESULTS = Path("results")
N_BOOT = 2000


def load_runs(raw_dir: Path = RESULTS / "raw") -> dict[tuple[str, str], list[dict]]:
    """(model, prompt) -> records sorted by doc_id; the last record per doc wins."""
    runs: dict[tuple[str, str], dict[str, dict]] = {}
    for path in sorted(raw_dir.glob("*.jsonl")):
        for r in read_records(path):
            runs.setdefault((r["model"], r["prompt"]), {})[r["doc_id"]] = r
    return {key: [docs[d] for d in sorted(docs)] for key, docs in runs.items()}


def _scores(records: list[dict]) -> Scores:
    return score([r["gold"] for r in records], [r["pred"] for r in records], FIELDS)


def summarize(records: list[dict]) -> dict:
    s = _scores(records)
    acc, f1 = partial(macro_accuracy, s), partial(macro_f1, s)
    latencies = [r["latency_s"] for r in records if r["error"] is None]
    return {
        "n_docs": s.n_docs,
        "macro_accuracy": acc(),
        "macro_accuracy_ci": bootstrap_ci(acc, s.n_docs, N_BOOT),
        "macro_f1": f1(),
        "macro_f1_ci": bootstrap_ci(f1, s.n_docs, N_BOOT),
        "hallucination_rate": hallucination_rate(s),
        "field_accuracy": dict(zip(FIELDS, field_accuracy(s).tolist(), strict=True)),
        "field_f1": dict(zip(FIELDS, field_f1(s).tolist(), strict=True)),
        "parse_status": dict(Counter(r["parse_status"] for r in records)),
        "truncated": sum(r.get("finish_reason") == "length" for r in records),
        "median_latency_s": float(np.median(latencies)) if latencies else None,
    }


def compare(a: list[dict], b: list[dict]) -> dict:
    """Paired v1 -> v2 comparison on the documents both runs scored."""
    common = {r["doc_id"] for r in a} & {r["doc_id"] for r in b}
    sa = _scores([r for r in a if r["doc_id"] in common])
    sb = _scores([r for r in b if r["doc_id"] in common])
    n = len(common)
    raw_p = [mcnemar_exact(sa.correct[:, j], sb.correct[:, j]) for j in range(len(FIELDS))]
    acc = paired_bootstrap_diff(partial(macro_accuracy, sa), partial(macro_accuracy, sb), n, N_BOOT)
    f1 = paired_bootstrap_diff(partial(macro_f1, sa), partial(macro_f1, sb), n, N_BOOT)
    delta = field_accuracy(sb) - field_accuracy(sa)
    return {
        "n_docs": n,
        "macro_accuracy_diff": acc,
        "macro_f1_diff": f1,
        "fields": {
            f: {"accuracy_diff": float(d), "p": p, "p_holm": ph}
            for f, d, p, ph in zip(FIELDS, delta, raw_p, holm(raw_p), strict=True)
        },
    }


def _pct(x: float) -> str:
    return "n/a" if x is None or np.isnan(x) else f"{100 * x:.1f}"


def _num(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.1f}"


def _p(p: float) -> str:
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def _ci(ci: tuple[float, float]) -> str:
    return f"[{_pct(ci[0])}, {_pct(ci[1])}]"


def to_markdown(summaries: dict, comparisons: dict) -> str:
    lines = [
        "| Model | Prompt | Docs | Macro accuracy % [95% CI] | Macro F1 % [95% CI] "
        "| Hallucination rate % | Parse ok / repaired / failed | Median latency s |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for (model, prompt), s in summaries.items():
        ps = s["parse_status"]
        lines.append(
            f"| `{model}` | {prompt} | {s['n_docs']} "
            f"| {_pct(s['macro_accuracy'])} {_ci(s['macro_accuracy_ci'])} "
            f"| {_pct(s['macro_f1'])} {_ci(s['macro_f1_ci'])} "
            f"| {_pct(s['hallucination_rate'])} "
            f"| {ps.get('ok', 0)} / {ps.get('repaired', 0)} / "
            f"{ps.get('failed', 0) + ps.get('error', 0)} "
            f"| {_num(s['median_latency_s'])} |"
        )
    lines += ["", "Per-field accuracy % (exact match, nulls included):", ""]
    keys = list(summaries)
    lines.append("| Field | " + " | ".join(f"{m.split('/')[-1]} {p}" for m, p in keys) + " |")
    lines.append("|---|" + "---:|" * len(keys))
    for f in FIELDS:
        cells = " | ".join(_pct(summaries[k]["field_accuracy"][f]) for k in keys)
        lines.append(f"| {f} | {cells} |")
    for model, c in comparisons.items():
        d_acc, d_f1 = c["macro_accuracy_diff"], c["macro_f1_diff"]
        lines += [
            "",
            f"Paired comparison v1 -> v2 for `{model}` ({c['n_docs']} docs): "
            f"macro accuracy {100 * d_acc[0]:+.1f} pts [{100 * d_acc[1]:+.1f}, "
            f"{100 * d_acc[2]:+.1f}], macro F1 {100 * d_f1[0]:+.1f} pts "
            f"[{100 * d_f1[1]:+.1f}, {100 * d_f1[2]:+.1f}].",
            "",
            "| Field | Δ accuracy pts | McNemar p | Holm-adjusted p |",
            "|---|---:|---:|---:|",
        ]
        for f, r in c["fields"].items():
            lines.append(
                f"| {f} | {100 * r['accuracy_diff']:+.1f} | {_p(r['p'])} | {_p(r['p_holm'])} |"
            )
    return "\n".join(lines) + "\n"


def plot(summaries: dict, path: Path) -> None:
    """Dumbbell per field (v1 vs v2 accuracy), one panel per model; macro row has 95% CIs."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = sorted({m for m, _ in summaries})
    colors = {"v1": "#2a78d6", "v2": "#eb6834"}  # categorical slots 1 and 2
    surface, ink, muted, grid = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
    rows = ["macro", *FIELDS]

    def value(s: dict, row: str) -> float:
        return 100 * (s["macro_accuracy"] if row == "macro" else s["field_accuracy"][row])

    lowest = min(value(s, r) for s in summaries.values() for r in rows)
    x_min = max(0, 10 * int((lowest - 5) // 10))
    n_docs = max(s["n_docs"] for s in summaries.values())

    fig, axes = plt.subplots(
        1, len(models), figsize=(4.6 * len(models), 4.6), sharey=True, squeeze=False
    )
    fig.patch.set_facecolor(surface)
    for ax, model in zip(axes[0], models, strict=True):
        ax.set_facecolor(surface)
        runs = {p: summaries[(model, p)] for p in ("v1", "v2") if (model, p) in summaries}
        for y, row in enumerate(rows):
            xs = {p: value(s, row) for p, s in runs.items()}
            if len(xs) == 2:
                ax.plot([xs["v1"], xs["v2"]], [y, y], color=grid, lw=2, zorder=1)
            for prompt, x in xs.items():
                if row == "macro":  # bootstrap CI as a thin whisker, offset per prompt
                    lo, hi = (100 * v for v in runs[prompt]["macro_accuracy_ci"])
                    dy = -0.18 if prompt == "v1" else 0.18
                    ax.plot([lo, hi], [y + dy, y + dy], color=colors[prompt], lw=1.5, zorder=1)
                ax.scatter(
                    x,
                    y,
                    s=64,
                    color=colors[prompt],
                    edgecolor=surface,
                    linewidth=2,
                    zorder=2,
                    clip_on=False,
                )
        ax.axhline(0.5, color=grid, lw=0.8)
        ax.set_title(model.split("/")[-1], color=ink, fontsize=11, loc="left")
        ax.set_xlim(x_min, 100)
        ax.set_yticks(range(len(rows)), [r.replace("_price", "") for r in rows], color=muted)
        for label in ax.get_yticklabels()[:1]:  # empty on shared-y panels after the first
            label.set_fontweight("bold")
        ax.grid(axis="x", color=grid, lw=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(colors=muted, length=0)
        ax.set_xlabel("exact-match accuracy %", color=muted)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(grid)
    axes[0][0].invert_yaxis()  # once: the y axis is shared, a second call would undo it
    handles = [
        plt.Line2D([], [], marker="o", ls="", color=c, markersize=8, label=f"prompt {p}")
        for p, c in colors.items()
    ]
    fig.suptitle(
        f"Prompt v1 vs v2 on {n_docs} CORD-v2 receipts", color=ink, x=0.01, ha="left", fontsize=12
    )
    fig.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.0, 0.95),
        ncol=2,
        frameon=False,
        labelcolor=muted,
        handletextpad=0.2,
        columnspacing=1.2,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.89))
    fig.savefig(path, dpi=160, facecolor=surface)
    plt.close(fig)


def main() -> None:
    runs = load_runs()
    if not runs:
        raise SystemExit("No runs in results/raw. Run `docextract run ...` first.")
    summaries = {key: summarize(records) for key, records in sorted(runs.items())}
    comparisons = {
        model: compare(runs[(model, "v1")], runs[(model, "v2")])
        for model in sorted({m for m, _ in runs})
        if (model, "v1") in runs and (model, "v2") in runs
    }
    md = to_markdown(summaries, comparisons)
    (RESULTS / "report.md").write_text(md)
    (RESULTS / "summary.json").write_text(
        json.dumps(
            {
                "runs": {f"{m}|{p}": s for (m, p), s in summaries.items()},
                "comparisons": comparisons,
            },
            indent=2,
        )
    )
    plot(summaries, RESULTS / "chart.png")
    print(md)
