import pytest

from docextract.extract import parse_output
from docextract.repair import repair_json


def test_clean_json_is_ok():
    assert repair_json('{"a": "1", "b": null}') == ({"a": "1", "b": None}, "ok")


@pytest.mark.parametrize(
    "raw, expected",
    [
        ('```json\n{"a": "1"}\n```', {"a": "1"}),
        ('Here you go:\n{"a": "1"}\nHope this helps!', {"a": "1"}),
        ('{"a": "1", "b": "2",}', {"a": "1", "b": "2"}),
        ('```json\n{"a": "1", "b": "2"', {"a": "1", "b": "2"}),  # fence never closed
        ('Result: {"a": "x, }", "b": "2"} done', {"a": "x, }", "b": "2"}),  # brackets in strings
        ('{"a": "say \\"hi\\"", "b": "2"', {"a": 'say "hi"', "b": "2"}),
        ('Here is {the} JSON: {"a": "1"}', {"a": "1"}),  # stray brace in prose
    ],
)
def test_repairs(raw, expected):
    assert repair_json(raw) == (expected, "repaired")


@pytest.mark.parametrize(
    "raw, expected",
    [
        # Cut inside a string value: the half-written value is dropped, never kept.
        ('{"total": "60.000", "tax": "5.4', {"total": "60.000"}),
        # Cut after a number: it may be partial ("60" of "60000"), so it is dropped.
        ('{"tax": "5.455", "total": 60', {"tax": "5.455"}),
        # Dangling key, dangling colon, dangling comma.
        ('{"total": "60.000", "tax"', {"total": "60.000"}),
        ('{"total": "60.000", "tax":', {"total": "60.000"}),
        ('{"total": "60.000",', {"total": "60.000"}),
        # A finished string at the cut point is complete and kept.
        ('{"total": "60.000", "tax": "5.455"', {"total": "60.000", "tax": "5.455"}),
        # Nested containers are closed in the right order.
        ('{"items": [{"nm": "tea"}, {"nm": "co', {"items": [{"nm": "tea"}]}),
    ],
)
def test_truncated_output(raw, expected):
    assert repair_json(raw) == (expected, "repaired")


@pytest.mark.parametrize("raw", ["", "I cannot read this receipt.", "[1, 2]", '{"total": "6'])
def test_unrecoverable(raw):
    assert repair_json(raw) == (None, "failed")


def test_parse_output_fills_schema_and_coerces_numbers():
    pred, status = parse_output('{"total_price": 60000, "made_up_key": "x"}')
    assert status == "ok"
    assert pred["total_price"] == "60000"
    assert pred["tax_price"] is None
    assert "made_up_key" not in pred


def test_parse_output_failure_is_all_null():
    pred, status = parse_output("no json here")
    assert status == "failed"
    assert set(pred.values()) == {None}
