"""Deterministic document ingestion and chunk provenance."""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class Chunk:
    id: str
    source: str
    page: int | None
    text: str
    sha256: str

    def to_dict(self) -> dict:
        return asdict(self)


SUPPORTED = {".txt", ".md", ".pdf"}


def split_text(text: str, chunk_words: int = 160, overlap_words: int = 32) -> list[str]:
    if chunk_words < 1 or not 0 <= overlap_words < chunk_words:
        raise ValueError("Require chunk_words > 0 and 0 <= overlap_words < chunk_words")
    words = text.split()
    step = chunk_words - overlap_words
    return [" ".join(words[i:i + chunk_words]) for i in range(0, len(words), step)
            if words[i:i + chunk_words]]


def _read_pages(path: Path) -> list[tuple[int | None, str]]:
    if path.suffix.lower() == ".pdf":
        try:
            import fitz
        except ImportError as exc:
            raise RuntimeError("PDF ingestion requires: pip install '.[pdf]'") from exc
        with fitz.open(path) as doc:
            return [(i + 1, page.get_text("text")) for i, page in enumerate(doc)]
    return [(None, path.read_text(encoding="utf-8"))]


def ingest(root: Path, chunk_words: int = 160, overlap_words: int = 32) -> list[Chunk]:
    root = root.resolve()
    if not root.is_dir():
        raise ValueError(f"Not a directory: {root}")
    output: list[Chunk] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED or path.is_symlink():
            continue
        source = path.relative_to(root).as_posix()
        for page, raw in _read_pages(path):
            normalised = re.sub(r"\s+", " ", raw).strip()
            if not normalised:
                continue
            for index, part in enumerate(split_text(normalised, chunk_words, overlap_words)):
                digest = hashlib.sha256(part.encode("utf-8")).hexdigest()
                marker = f"p{page}" if page is not None else "doc"
                output.append(Chunk(f"{source}::{marker}::{index}", source, page, part, digest))
    return output
