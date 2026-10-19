"""Load CORD-v2 test receipts (CC-BY-4.0) from a pinned Hugging Face revision."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from .schema import CORD_KEYS, Receipt

REVISION = "7f0115a4b758a71d6473b8d085751692da2fef98"
TEST_URL = (
    f"https://huggingface.co/datasets/naver-clova-ix/cord-v2/resolve/{REVISION}"
    "/data/test-00000-of-00001-9c204eb3f4e11791.parquet"
)
SHA256 = "51c65f1788faff392abe2a0b55b023eb23e9be551c509138eaa3a832514224e7"  # LFS oid at REVISION
TEST_SPLIT_SIZE = 100
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
        # A handful of receipts list two subtotals; the first one is kept.
        values[field] = value[0] if isinstance(value, list) else value
    return Receipt(**values)


def _encode(image_bytes: bytes) -> str:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img.thumbnail((MAX_SIDE, MAX_SIDE))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()


def _download(path: Path) -> None:
    """Download to a temporary file, check its SHA-256, and only then move it into place."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    print(f"Downloading CORD-v2 test split to {path} ...", flush=True)
    digest = hashlib.sha256()
    try:
        with urllib.request.urlopen(TEST_URL, timeout=60) as resp, tmp.open("wb") as out:
            while chunk := resp.read(1 << 20):
                digest.update(chunk)
                out.write(chunk)
    except OSError as exc:
        tmp.unlink(missing_ok=True)
        raise SystemExit(
            f"Could not download CORD-v2: {exc}. Check the connection and retry."
        ) from exc
    if digest.hexdigest() != SHA256:
        tmp.unlink(missing_ok=True)
        raise SystemExit("The downloaded file does not match the expected SHA-256; discarded it.")
    tmp.replace(path)


def load_docs(n: int, start: int = 0, path: Path = DATA_PATH) -> list[Doc]:
    """Receipts ``start`` .. ``start + n - 1`` of the CORD-v2 test split (~230 MB, cached)."""
    if n < 1 or start < 0 or start + n > TEST_SPLIT_SIZE:
        raise ValueError(
            f"start={start}, n={n} is outside the {TEST_SPLIT_SIZE} receipts of the test split"
        )
    if not path.exists():
        _download(path)
    try:
        table = pq.read_table(path, columns=["image", "ground_truth"])
    except (pa.ArrowInvalid, OSError) as exc:
        raise SystemExit(
            f"Cannot read {path} ({exc}). Delete it and run again to re-download."
        ) from exc
    rows = table.slice(start, n).to_pylist()
    return [
        Doc(
            doc_id=f"cord-test-{i:03d}",  # absolute index, so runs over other ranges still pair
            image_b64=_encode(row["image"]["bytes"]),
            gold=gold_from_cord(json.loads(row["ground_truth"])["gt_parse"]),
        )
        for i, row in enumerate(rows, start)
    ]
