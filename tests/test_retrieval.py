from pathlib import Path

import pytest

from contextforge.answer import ABSTENTION, answer
from contextforge.documents import Chunk, ingest
from contextforge.evaluation import evaluate
from contextforge.retrieval import Retriever


def test_retrieval_and_provenance(tmp_path: Path):
    (tmp_path / "alpha.md").write_text("Citation source references.", encoding="utf-8")
    (tmp_path / "beta.md").write_text("Order book market microstructure.", encoding="utf-8")
    retriever = Retriever(ingest(tmp_path))
    hits = retriever.search("citation source")
    assert hits[0].chunk.source == "alpha.md"
    assert hits[0].rank == 1


def test_unknown_query_abstains(tmp_path: Path):
    (tmp_path / "source.md").write_text("Citation source references.", encoding="utf-8")
    result = answer("zebras and aquariums", Retriever(ingest(tmp_path)))
    assert result.abstained
    assert result.answer == ABSTENTION
    assert result.sources == []


def test_answer_extractive_has_references(tmp_path: Path):
    (tmp_path / "source.md").write_text("Citation source references.", encoding="utf-8")
    result = answer("citation source", Retriever(ingest(tmp_path)))
    assert not result.abstained
    assert result.sources[0]["source"] == "source.md"
    assert "[1]" in result.answer


def test_empty_and_invalid_retrieval():
    assert Retriever([]).search("anything") == []
    with pytest.raises(ValueError):
        Retriever([]).search("anything", top_k=0)


def test_eval_dataset(tmp_path: Path):
    import json
    (tmp_path / "source.md").write_text("Citation source references.", encoding="utf-8")
    chunks = ingest(tmp_path)
    labels = tmp_path.parent / "labels.json"
    labels.write_text(json.dumps([
        {"query": "citation source", "relevant_chunk_ids": [chunks[0].id]}
    ]))
    results = evaluate(Retriever(chunks), labels, k=1)
    assert results["recall_at_k"] == 1.0
    assert results["mrr_at_k"] == 1.0
