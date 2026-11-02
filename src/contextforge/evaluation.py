"""Label-based retrieval evaluation; not a generative faithfulness score."""

from __future__ import annotations

import json
from pathlib import Path

from .retrieval import Retriever


def evaluate(retriever: Retriever, dataset: Path, k: int = 5) -> dict:
    if k < 1:
        raise ValueError("k must be positive")
    records = json.loads(dataset.read_text(encoding="utf-8"))
    if not isinstance(records, list) or not records:
        raise ValueError("Expected a non-empty list of evaluation queries")
    valid_ids = {chunk.id for chunk in retriever.chunks}
    rows = []
    for record in records:
        query = record["query"]
        relevant = set(record["relevant_chunk_ids"])
        if not relevant or not relevant <= valid_ids:
            raise ValueError(f"Missing/invalid ground truth chunk IDs for {query!r}")
        ids = [hit.chunk.id for hit in retriever.search(query, top_k=k)]
        found = relevant.intersection(ids)
        first = next((i for i, item in enumerate(ids, 1) if item in relevant), None)
        rows.append({
            "query": query, "retrieved": ids, "relevant": sorted(relevant),
            "recall_at_k": len(found) / len(relevant),
            "hit_at_k": float(bool(found)),
            "reciprocal_rank": 1 / first if first else 0.0,
        })
    n = len(rows)
    return {
        "count": n, "k": k,
        "recall_at_k": round(sum(r["recall_at_k"] for r in rows) / n, 4),
        "hit_at_k": round(sum(r["hit_at_k"] for r in rows) / n, 4),
        "mrr_at_k": round(sum(r["reciprocal_rank"] for r in rows) / n, 4),
        "queries": rows,
    }
