from pathlib import Path

import pytest

from contextforge.documents import ingest, split_text
from contextforge.store import load_index, save_index


def test_chunking_overlap():
    assert split_text("one two three four five", 3, 1) == [
        "one two three", "three four five", "five",
    ]


def test_invalid_chunk_parameters():
    with pytest.raises(ValueError):
        split_text("abc", 2, 2)


def test_deterministic_ingestion_and_roundtrip(tmp_path: Path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "sample.md").write_text("a short useful document", encoding="utf-8")
    first = ingest(docs)
    assert first == ingest(docs)
    assert first[0].source == "sample.md"
    index = tmp_path / "index.json"
    save_index(first, index)
    assert load_index(index) == first


def test_tampered_index_rejected(tmp_path: Path):
    import json
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "sample.txt").write_text("original", encoding="utf-8")
    index = tmp_path / "index.json"
    save_index(ingest(docs), index)
    payload = json.loads(index.read_text())
    payload["chunks"][0]["text"] = "tampered"
    index.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="digest"):
        load_index(index)
