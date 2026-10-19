import json

import pytest

from docextract.report import baseline, compare, gold_by_doc, load_runs, summarize
from docextract.schema import FIELDS
from docextract.scoring import macro_accuracy, score

EMPTY = dict.fromkeys(FIELDS)


def record(doc_id, gold=None, pred=None, prompt="v1", **extra):
    return {
        "doc_id": doc_id,
        "model": "m",
        "prompt": prompt,
        "gold": {**EMPTY, **(gold or {})},
        "pred": {**EMPTY, **(pred or {})},
        "parse_status": "ok",
        "error": None,
        "latency_s": 1.0,
        "raw": "{}",
        "finish_reason": "stop",
        **extra,
    }


def test_load_runs_last_record_wins_sorted_and_skips_partial_line(tmp_path):
    lines = [
        json.dumps(record("d2")),
        json.dumps(record("d1", error="Timeout", parse_status="error")),
        json.dumps(record("d1", pred={"total_price": "5"})),
        '{"doc_id": "d3", "mod',  # left by a killed run
    ]
    (tmp_path / "m__v1.jsonl").write_text("\n".join(lines))
    runs = load_runs(tmp_path)
    assert list(runs) == [("m", "v1")]
    assert [r["doc_id"] for r in runs[("m", "v1")]] == ["d1", "d2"]
    assert runs[("m", "v1")][0]["pred"]["total_price"] == "5"


def test_gold_by_doc_rejects_runs_that_disagree():
    runs = {
        ("m", "v1"): [record("d1", gold={"total_price": "5"})],
        ("m", "v2"): [record("d1", gold={"total_price": "6"})],
    }
    with pytest.raises(SystemExit):
        gold_by_doc(runs)


def test_failed_documents_get_no_credit_for_abstaining():
    ok = record("d1")
    failed = record("d2", parse_status="failed")
    errored = record("d3", parse_status="error", error="AuthenticationError: bad key")
    s = summarize([ok, failed, errored])
    # d1 abstains correctly on all 8 null fields; d2 and d3 earn nothing.
    assert s["macro_accuracy"] == pytest.approx(1 / 3)
    assert s["parse_status"] == {"ok": 1, "failed": 1, "error": 1}


def test_summarize_counts_truncation_and_unknown_keys():
    recs = [
        record("d1", finish_reason="length"),
        record("d2", raw='{"total": "5"}'),
    ]
    s = summarize(recs)
    assert s["truncated"] == 1
    assert s["unknown_keys"] == 1


def test_baseline_is_the_share_of_null_cells():
    gold = {"d1": {**EMPTY, "total_price": "5"}, "d2": EMPTY}
    floor = baseline(gold)
    assert floor["macro_accuracy"] == pytest.approx(15 / 16)
    assert floor["macro_f1"] == 0.0
    assert floor["hallucination_rate"] == 0.0


def test_compare_identical_runs_is_zero_and_ignores_unpaired_docs():
    a = [record("d1", gold={"total_price": "5"}, pred={"total_price": "5"}), record("d2")]
    b = [*a, record("d3", gold={"total_price": "7"}, prompt="v2")]
    c = compare(a, b)
    assert c["n_docs"] == 2
    assert c["macro_accuracy_diff"] == (0.0, 0.0, 0.0)
    assert all(f["p"] == 1.0 and f["accuracy_diff"] == 0.0 for f in c["fields"].values())


def test_compare_without_common_documents_fails_clearly():
    with pytest.raises(ValueError, match="no document in common"):
        compare([record("d1")], [record("d2")])


def test_score_failed_mask_only_changes_accuracy():
    golds = [{"a": None}, {"a": "1"}]
    preds = [{"a": None}, {"a": None}]
    plain = score(golds, preds, ["a"])
    masked = score(golds, preds, ["a"], failed=[True, False])
    assert macro_accuracy(plain) == 0.5
    assert macro_accuracy(masked) == 0.0
    assert (plain.fn == masked.fn).all() and (plain.fp == masked.fp).all()
