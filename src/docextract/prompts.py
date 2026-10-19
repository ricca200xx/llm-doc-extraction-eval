"""Two prompt versions, compared head to head on the same documents.

v1 is the prompt most people write first. v2 changes three things: it states
what each field means, it makes ``null`` the explicit answer for anything not
printed, and it forbids computing a value from other numbers.
"""

from __future__ import annotations

from .schema import FIELDS, Receipt

_V1 = (
    "Extract the following fields from this receipt and return them as JSON: "
    + ", ".join(FIELDS)
    + "."
)

_V2_SYSTEM = """You read receipt images and return one JSON object, nothing else.

Rules:
- Use exactly the keys listed below. No extra keys, no markdown, no comments.
- Copy each amount as it is printed on the receipt.
- If a field is not printed on the receipt, its value is null.
- Never compute, estimate or infer a value from other numbers, and never reuse
  one printed number for two different fields unless the receipt prints it twice."""


def _field_list() -> str:
    return "\n".join(
        f'- "{name}": {info.description}' for name, info in Receipt.model_fields.items()
    )


def build_messages(version: str, image_b64: str) -> list[dict]:
    image = {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
    if version == "v1":
        return [{"role": "user", "content": [{"type": "text", "text": _V1}, image]}]
    if version == "v2":
        user = f"Fields:\n{_field_list()}\n\nReturn the JSON object for this receipt."
        return [
            {"role": "system", "content": _V2_SYSTEM},
            {"role": "user", "content": [{"type": "text", "text": user}, image]},
        ]
    raise ValueError(f"unknown prompt version: {version!r} (expected 'v1' or 'v2')")
