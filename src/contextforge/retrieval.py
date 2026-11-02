"""TF-IDF, dense retrieval and reciprocal-rank fusion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .documents import Chunk

Mode = Literal["lexical", "dense", "hybrid"]


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float
    rank: int

    def to_dict(self) -> dict:
        return {
            "id": self.chunk.id, "source": self.chunk.source, "page": self.chunk.page,
            "text": self.chunk.text, "score": round(self.score, 6), "rank": self.rank,
        }


class Retriever:
    def __init__(self, chunks: list[Chunk], mode: Mode = "lexical",
                 model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        if mode not in ("lexical", "dense", "hybrid"):
            raise ValueError("Unknown retrieval mode")
        self.chunks = chunks
        self.mode = mode
        self.vectorizer = None
        self.lexical_matrix = None
        self.dense_model = None
        self.dense_matrix = None
        if not chunks:
            return
        if mode in ("lexical", "hybrid"):
            self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
            try:
                self.lexical_matrix = self.vectorizer.fit_transform([c.text for c in chunks])
            except ValueError as exc:
                raise ValueError("No searchable terms in indexed text") from exc
        if mode in ("dense", "hybrid"):
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError("Dense retrieval requires: pip install '.[semantic]'") from exc
            self.dense_model = SentenceTransformer(model_name)
            self.dense_matrix = np.asarray(
                self.dense_model.encode([c.text for c in chunks], normalize_embeddings=True),
                dtype=np.float32,
            )

    @staticmethod
    def _rank(scores: np.ndarray) -> list[int]:
        return [int(i) for i in np.argsort(-scores, kind="stable") if scores[i] > 0]

    def search(self, query: str, top_k: int = 5, rrf_k: int = 60) -> list[Hit]:
        if top_k < 1 or rrf_k < 1:
            raise ValueError("top_k and rrf_k must be positive")
        if not self.chunks or not query.strip():
            return []
        rankings: list[list[int]] = []
        raw_scores: dict[int, float] = {}
        if self.vectorizer is not None:
            vector = self.vectorizer.transform([query])
            scores = (self.lexical_matrix @ vector.T).toarray().ravel()
            ranked = self._rank(scores)
            rankings.append(ranked)
            if self.mode == "lexical":
                raw_scores = {i: float(scores[i]) for i in ranked}
        if self.dense_model is not None:
            embedding = np.asarray(
                self.dense_model.encode([query], normalize_embeddings=True)[0], dtype=np.float32
            )
            scores = self.dense_matrix @ embedding
            ranked = self._rank(scores)
            rankings.append(ranked)
            if self.mode == "dense":
                raw_scores = {i: float(scores[i]) for i in ranked}
        if self.mode == "hybrid":
            for ranking in rankings:
                for rank, i in enumerate(ranking, 1):
                    raw_scores[i] = raw_scores.get(i, 0.0) + 1.0 / (rrf_k + rank)
        ranked_ids = sorted(raw_scores, key=lambda i: (-raw_scores[i], i))[:top_k]
        return [Hit(self.chunks[i], raw_scores[i], rank)
                for rank, i in enumerate(ranked_ids, 1)]
