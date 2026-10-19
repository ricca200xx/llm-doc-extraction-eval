"""Load CORD-v2 test receipts (CC-BY-4.0) from a pinned Hugging Face revision."""

from __future__ import annotations

import base64
import io
import json
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image

from .schema import CORD_KEYS, Receipt

REVISION = "7f0115a4b758a71d6473b8d085751692da2fef98"
TEST_URL = (
    f"https://huggingface.co/datasets/naver-clova-ix/cord-v2/resolve/{REVISION}"
    "/data/test-00000-of-00001-9c204eb3f4e11791.parquet"
)
DATA_PATH = Path("data/cord-v2-test.parquet")
MAX_SIDE = 1280  # px; keeps local VLM latency sane without losing small print


@dataclass(frozen=True)
class Doc:
    doc_id: str
    image_b64: str  # JPEG
    gold: Receipt


def gold_from_cord(gt_parse: dict) -> Receipt:
    values = {}
    for field, (section, key) in CORD_KEYS.items():
        block = gt_parse.get(section)
        value = block.get(key) if isinstance(block, dict) else None
        # ponytail: a handful of receipts list two subtotals; we keep the first.
        values[field] = value[0] if isinstance(value, list) else value
    return Receipt(**values)


def _encode(image_bytes: bytes) -> str:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img.thumbnail((MAX_SIDE, MAX_SIDE))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()


def load_docs(n: int, path: Path = DATA_PATH) -> list[Doc]:
    """First ``n`` receipts of the CORD-v2 test split (downloaded once, ~230 MB)."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading CORD-v2 test split to {path} ...")
        urllib.request.urlretrieve(TEST_URL, path)
    rows = pq.read_table(path, columns=["image", "ground_truth"]).slice(0, n).to_pylist()
    return [
        Doc(
            doc_id=f"cord-test-{i:03d}",
            image_b64=_encode(row["image"]["bytes"]),
            gold=gold_from_cord(json.loads(row["ground_truth"])["gt_parse"]),
        )
        for i, row in enumerate(rows)
    ]
