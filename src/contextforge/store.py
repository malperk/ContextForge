"""Portable JSON index with integrity checks; never unpickle user documents."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .documents import Chunk

SCHEMA_VERSION = 1


def save_index(chunks: list[Chunk], path: Path) -> None:
    payload = {"schema": SCHEMA_VERSION, "chunks": [c.to_dict() for c in chunks]}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_index(path: Path) -> list[Chunk]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA_VERSION:
        raise ValueError("Unsupported index schema")
    chunks = [Chunk(**entry) for entry in payload["chunks"]]
    if len({c.id for c in chunks}) != len(chunks):
        raise ValueError("Duplicate chunk identifiers")
    for chunk in chunks:
        if hashlib.sha256(chunk.text.encode("utf-8")).hexdigest() != chunk.sha256:
            raise ValueError(f"Content digest mismatch: {chunk.id}")
    return chunks
