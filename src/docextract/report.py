"""Score every run in results/raw and write report.md, summary.json and chart.png."""

from __future__ import annotations

import json
from collections import Counter
from functools import partial
from pathlib import Path

import numpy as np

from .extract import read_records
from .repair import repair_json
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


def gold_by_doc(runs: dict[tuple[str, str], list[dict]]) -> dict[str, dict]:
    """One gold record per document, after checking that every run agrees on it."""
    gold: dict[str, dict] = {}
    for records in runs.values():
        for r in records:
            if gold.setdefault(r["doc_id"], r["gold"]) != r["gold"]:
                raise SystemExit(
                    f"Runs disagree on the gold labels of {r['doc_id']}. They were made with "
                    "different loader versions; re-run the older ones."
                )
    return dict(sorted(gold.items()))


def _failed(record: dict) -> bool:
    return record.get("error") is not None or record["parse_status"] == "failed"


def _scores(records: list[dict]) -> Scores:
    return score(
        [r["gold"] for r in records],
        [r["pred"] for r in records],
        FIELDS,
        failed=[_failed(r) for r in records],
    )


def _metrics(s: Scores) -> dict:
    acc = partial(macro_accuracy, s)
    f1 = partial(macro_f1, s)
    hallucination = partial(hallucination_rate, s)
    return {
        "n_docs": s.n_docs,
        "macro_accuracy": acc(),
        "macro_accuracy_ci": bootstrap_ci(acc, s.n_docs, N_BOOT),
        "macro_f1": f1(),
        "macro_f1_ci": bootstrap_ci(f1, s.n_docs, N_BOOT),
        "hallucination_rate": hallucination(),
        "hallucination_rate_ci": bootstrap_ci(hallucination, s.n_docs, N_BOOT),
        "field_accuracy": dict(zip(FIELDS, field_accuracy(s).tolist(), strict=True)),
        "field_f1": dict(zip(FIELDS, field_f1(s).tolist(), strict=True)),
    }


def _has_unknown_keys(raw: str) -> bool:
    obj, _ = repair_json(raw)
    return bool(set(obj or {}) - set(FIELDS))


def summarize(records: list[dict]) -> dict:
    latencies = [r["latency_s"] for r in records if r.get("error") is None]
    return {
        **_metrics(_scores(records)),
        "parse_status": dict(Counter(r["parse_status"] for r in records)),
        "truncated": sum(r.get("finish_reason") == "length" for r in records),
        "unknown_keys": sum(_has_unknown_keys(r["raw"]) for r in records),
        "median_latency_s": float(np.median(latencies)) if latencies else None,
    }


def baseline(gold: dict[str, dict]) -> dict:
    """Scores of a 'model' that answers null for every field: the floor to beat."""
    golds = list(gold.values())
    return _metrics(score(golds, [{}] * len(golds), FIELDS))


def compare(a: list[dict], b: list[dict]) -> dict:
    """Paired v1 -> v2 comparison on the documents both runs scored."""
    common = {r["doc_id"] for r in a} & {r["doc_id"] for r in b}
    if not common:
        raise ValueError("the two runs have no document in common")
    sa = _scores([r for r in a if r["doc_id"] in common])
    sb = _scores([r for r in b if r["doc_id"] in common])
    n = len(common)
    raw_p = [mcnemar_exact(sa.correct[:, j], sb.correct[:, j]) for j in range(len(FIELDS))]
    acc_diff = paired_bootstrap_diff(
        partial(macro_accuracy, sa), partial(macro_accuracy, sb), n, N_BOOT
    )
    f1_diff = paired_bootstrap_diff(partial(macro_f1, sa), partial(macro_f1, sb), n, N_BOOT)
    delta = field_accuracy(sb) - field_accuracy(sa)
    return {
        "n_docs": n,
        "macro_accuracy_diff": acc_diff,
        "macro_f1_diff": f1_diff,
        "fields": {
            f: {"accuracy_diff": float(d), "p": p, "p_holm": ph}
            for f, d, p, ph in zip(FIELDS, delta, raw_p, holm(raw_p), strict=True)
        },
    }


def _fmt_pct(x: float | None) -> str:
    return "n/a" if x is None or np.isnan(x) else f"{100 * x:.1f}"


def _fmt_ci(ci: tuple[float, float]) -> str:
    return f"[{_fmt_pct(ci[0])}, {_fmt_pct(ci[1])}]"


def _fmt_num(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.1f}"


def _fmt_p(p: float) -> str:
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def _metric_cells(m: dict) -> str:
    return (
        f"| {_fmt_pct(m['macro_accuracy'])} {_fmt_ci(m['macro_accuracy_ci'])} "
        f"| {_fmt_pct(m['macro_f1'])} {_fmt_ci(m['macro_f1_ci'])} "
        f"| {_fmt_pct(m['hallucination_rate'])} {_fmt_ci(m['hallucination_rate_ci'])} "
    )


def to_markdown(summaries: dict, comparisons: dict, floor: dict) -> str:
    lines = [
        "| Model | Prompt | Docs | Macro accuracy % [95% CI] | Macro F1 % [95% CI] "
        "| Hallucination rate % [95% CI] | Parse ok / repaired / failed | Median latency s |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
        f"| *always `null` (baseline)* | – | {floor['n_docs']} {_metric_cells(floor)}| – | – |",
    ]
    for (model, prompt), s in summaries.items():
        ps = s["parse_status"]
        lines.append(
            f"| `{model}` | {prompt} | {s['n_docs']} {_metric_cells(s)}"
            f"| {ps.get('ok', 0)} / {ps.get('repaired', 0)} / "
            f"{ps.get('failed', 0) + ps.get('error', 0)} "
            f"| {_fmt_num(s['median_latency_s'])} |"
        )
    unknown = {k: s["unknown_keys"] for k, s in summaries.items() if s["unknown_keys"]}
    lines += [
        "",
        "No output used a key outside the schema."
        if not unknown
        else "Outputs with keys outside the schema (ignored by the scorer): "
        + ", ".join(f"`{m}` {p}: {n}" for (m, p), n in unknown.items())
        + ".",
        "",
        "Per-field accuracy % (exact match, nulls included):",
        "",
    ]
    keys = list(summaries)
    header = " | ".join(f"{m.split('/')[-1]} {p}" for m, p in keys)
    lines.append(f"| Field | always null | {header} |")
    lines.append("|---|---:|" + "---:|" * len(keys))
    for f in FIELDS:
        cells = " | ".join(_fmt_pct(summaries[k]["field_accuracy"][f]) for k in keys)
        lines.append(f"| {f} | {_fmt_pct(floor['field_accuracy'][f])} | {cells} |")
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
                f"| {f} | {100 * r['accuracy_diff']:+.1f} "
                f"| {_fmt_p(r['p'])} | {_fmt_p(r['p_holm'])} |"
            )
    return "\n".join(lines) + "\n"


def plot(summaries: dict, floor: dict, path: Path) -> None:
    """Dumbbell per field (v1 vs v2 accuracy), one panel per model; macro row has 95% CIs.

    A grey tick on every row marks the always-null baseline for that field.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = sorted({m for m, _ in summaries})
    colors = {"v1": "#2a78d6", "v2": "#eb6834"}  # categorical slots 1 and 2
    surface, ink, muted, grid = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
    baseline_ink = "#9a9890"
    rows = ["macro", *FIELDS]

    def value(s: dict, row: str) -> float:
        return 100 * (s["macro_accuracy"] if row == "macro" else s["field_accuracy"][row])

    lowest = min(value(s, r) for s in [*summaries.values(), floor] for r in rows)
    x_min = max(0, 10 * int((lowest - 5) // 10))  # next gridline below the lowest mark
    n_docs = max(s["n_docs"] for s in summaries.values())

    fig, axes = plt.subplots(
        1, len(models), figsize=(4.6 * len(models), 4.6), sharey=True, squeeze=False
    )
    fig.patch.set_facecolor(surface)
    for ax, model in zip(axes[0], models, strict=True):
        ax.set_facecolor(surface)
        runs = {p: summaries[(model, p)] for p in ("v1", "v2") if (model, p) in summaries}
        for y, row in enumerate(rows):
            ax.plot(value(floor, row), y, marker="|", markersize=14, color=baseline_ink, zorder=1)
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
    ] + [
        plt.Line2D(
            [], [], marker="|", ls="", color=baseline_ink, markersize=12, label="always null"
        )
    ]
    fig.suptitle(
        f"Prompt v1 vs v2 on {n_docs} CORD-v2 receipts", color=ink, x=0.01, ha="left", fontsize=12
    )
    fig.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.0, 0.95),
        ncol=3,
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
        raise SystemExit(
            f"No runs in {Path.cwd() / RESULTS / 'raw'}. Run `docextract run ...` first, "
            "from the repository root."
        )
    gold = gold_by_doc(runs)
    # Baseline on the documents every run scored, so it compares like with like.
    shared = set.intersection(*({r["doc_id"] for r in recs} for recs in runs.values()))
    floor = baseline({d: g for d, g in gold.items() if d in shared} or gold)
    summaries = {key: summarize(records) for key, records in sorted(runs.items())}
    comparisons = {
        model: compare(runs[(model, "v1")], runs[(model, "v2")])
        for model in sorted({m for m, _ in runs})
        if (model, "v1") in runs and (model, "v2") in runs
    }
    md = to_markdown(summaries, comparisons, floor)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "report.md").write_text(md)
    (RESULTS / "summary.json").write_text(
        json.dumps(
            {
                "baseline_always_null": floor,
                "runs": {f"{m}|{p}": s for (m, p), s in summaries.items()},
                "comparisons": comparisons,
            },
            indent=2,
        )
    )
    plot(summaries, floor, RESULTS / "chart.png")
    print(md)
