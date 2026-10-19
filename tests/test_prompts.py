import pytest

from docextract.data import gold_from_cord, load_docs
from docextract.prompts import build_messages, prompt_fingerprint
from docextract.schema import FIELDS, Receipt


def test_v1_is_one_user_message_with_text_and_image():
    (msg,) = build_messages("v1", "QUJD")
    assert msg["role"] == "user"
    text, image = msg["content"]
    assert all(f in text["text"] for f in FIELDS)
    assert image["image_url"]["url"] == "data:image/jpeg;base64,QUJD"


def test_v2_adds_system_rules_and_every_field_definition():
    system, user = build_messages("v2", "QUJD")
    assert system["role"] == "system" and "null" in system["content"]
    user_text = user["content"][0]["text"]
    for name, info in Receipt.model_fields.items():
        assert f'"{name}": {info.description}' in user_text


def test_unknown_prompt_version_is_rejected():
    with pytest.raises(ValueError, match="unknown prompt version"):
        build_messages("v3", "QUJD")


def test_fingerprint_is_stable_and_tells_versions_apart():
    assert prompt_fingerprint("v1") == prompt_fingerprint("v1")
    assert prompt_fingerprint("v1") != prompt_fingerprint("v2")
    assert len(prompt_fingerprint("v2")) == 12


def test_gold_from_cord_handles_missing_sections_and_keys():
    assert gold_from_cord({}) == Receipt()
    assert gold_from_cord({"total": None, "sub_total": {"etc": "1"}}) == Receipt()
    assert gold_from_cord({"total": {"cashprice": 5000}}).cash_price == "5000"


@pytest.mark.parametrize("n, start", [(0, 0), (101, 0), (10, 95), (5, -1)])
def test_load_docs_rejects_ranges_outside_the_split(n, start, tmp_path):
    # Fails before touching the (missing) dataset file.
    with pytest.raises(ValueError):
        load_docs(n, start, path=tmp_path / "missing.parquet")
