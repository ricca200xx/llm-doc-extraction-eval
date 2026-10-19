import math

import numpy as np
import pytest

from docextract.data import gold_from_cord
from docextract.schema import Receipt
from docextract.scoring import (
    canonical_amount,
    field_f1,
    hallucination_rate,
    macro_accuracy,
    macro_f1,
    score,
)
from docextract.stats import bootstrap_ci, holm, mcnemar_exact, paired_bootstrap_diff


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("60.000", "60000"),
        ("Rp 60,000", "60000"),
        ("60,000.00", "60000"),
        ("0.00", "0"),
        ("(0)", "0"),
        ("-19,400", "19400"),
        ("28.182", "28182"),
        ("365000.0", "365000"),
        ("3.50", "3.5"),
        ("3,5", "3.5"),
        ("[145,900]", "145900"),
        ("60,000.00 IDR", "60000"),
        ("n/a", None),
        ("", None),
        (None, None),
    ],
)
def test_canonical_amount(raw, expected):
    assert canonical_amount(raw) == expected


def test_every_outcome_cell():
    fields = ["a"]
    golds = [{"a": None}, {"a": None}, {"a": "10"}, {"a": "10"}, {"a": "10"}]
    preds = [{"a": None}, {"a": "10"}, {"a": None}, {"a": "10.00"}, {"a": "11"}]
    s = score(golds, preds, fields)
    # abstain, hallucination, miss, match, wrong value
    assert s.correct[:, 0].tolist() == [True, False, False, True, False]
    assert s.tp[:, 0].tolist() == [False, False, False, True, False]
    assert s.fp[:, 0].tolist() == [False, True, False, False, True]
    assert s.fn[:, 0].tolist() == [False, False, True, False, True]
    assert macro_accuracy(s) == pytest.approx(2 / 5)
    # F1 = 2TP / (2TP + FP + FN) = 2 / (2 + 2 + 2)
    assert macro_f1(s) == pytest.approx(1 / 3)
    assert hallucination_rate(s) == pytest.approx(1 / 2)


def test_hallucination_costs_f1_even_when_everything_else_is_right():
    golds = [{"a": "5"}, {"a": None}]
    honest = score(golds, [{"a": "5"}, {"a": None}], ["a"])
    hallucinating = score(golds, [{"a": "5"}, {"a": "5"}], ["a"])
    assert macro_f1(honest) == 1.0
    assert macro_f1(hallucinating) == pytest.approx(2 / 3)


def test_f1_undefined_field_is_excluded_from_macro():
    golds = [{"a": "1", "b": None}, {"a": "2", "b": None}]
    preds = [{"a": "1", "b": None}, {"a": "3", "b": None}]
    s = score(golds, preds, ["a", "b"])
    assert math.isnan(field_f1(s)[1])
    assert macro_f1(s) == pytest.approx(field_f1(s)[0])
    assert macro_accuracy(s) == pytest.approx((0.5 + 1.0) / 2)


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        score([{"a": "1"}], [], ["a"])


def test_gold_from_cord_maps_nested_keys_and_lists():
    gold = gold_from_cord(
        {
            "menu": [{"nm": "tea", "price": "10"}],
            "sub_total": {"subtotal_price": ["50.000", "55.000"], "tax_price": "5.000"},
            "total": {"total_price": "55.000", "cashprice": "100.000"},
        }
    )
    assert gold == Receipt(
        subtotal_price="50.000", tax_price="5.000", total_price="55.000", cash_price="100.000"
    )


def test_mcnemar_exact():
    a = [True] * 10
    assert mcnemar_exact(a, a) == 1.0
    # 6 discordant pairs all one way: p = 2 * 0.5**6
    assert mcnemar_exact([False] * 6 + [True] * 4, a) == pytest.approx(2 / 64)
    # symmetric discordance gives p = 1
    assert mcnemar_exact([True, False], [False, True]) == 1.0


def test_holm():
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert holm([0.5, 0.9]) == pytest.approx([1.0, 1.0])


def test_bootstrap_ci_brackets_the_estimate():
    values = np.random.default_rng(1).random(200)
    lo, hi = bootstrap_ci(lambda idx: float(values[idx].mean()), len(values), n_boot=500)
    assert lo < values.mean() < hi
    assert hi - lo < 0.2


def test_paired_bootstrap_diff_sign():
    a = np.zeros(50)
    b = np.ones(50)
    diff, lo, hi = paired_bootstrap_diff(
        lambda i: float(a[i].mean()), lambda i: float(b[i].mean()), 50, n_boot=200
    )
    assert (diff, lo, hi) == (1.0, 1.0, 1.0)
