# ContextForge

**A reproducible, evidence-grounded Retrieval-Augmented Generation (RAG) research project.**

ContextForge indexes local Markdown, text and optionally PDF documents, retrieves candidate evidence with lexical, dense or hybrid search, and returns source-linked answers. A deterministic evaluation harness measures retrieval quality rather than relying on a polished demo alone.

## Features

- Ingest `.md`, `.txt`, and optional `.pdf` with page-aware provenance and deterministic chunk IDs.
- Lexical TF-IDF retrieval, optional Sentence Transformers embeddings, and hybrid reciprocal rank fusion (RRF).
- Source references, explicit abstention when lexical retrieval has no sufficiently relevant match.
- Extractive mode with **no API key**; optional OpenAI generation with instruction-injection-resistant prompt boundaries.
- Evaluation: Recall@K, Hit@K, MRR@K on labelled queries.
- Unit tests and GitHub Actions CI.

**Research honesty:** The included example dataset is tiny and intended as a smoke test, **not** evidence of generalisable retrieval performance. No guarantee is made that generation is faithful just because supporting excerpts are returned. The lexical abstention threshold is not calibrated for dense/hybrid scores.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest
contextforge index examples/docs
contextforge search "How are citations formatted?"
contextforge ask "How are citations formatted?"
contextforge eval examples/eval/queries.json --k 3
```

For semantic search:

```bash
pip install -e '.[semantic]'
contextforge search "How do I attribute a factual statement?" --mode dense
contextforge search "How do I attribute a factual statement?" --mode hybrid
```

For optional LLM-based synthesis:

```bash
pip install -e '.[llm]'
export OPENAI_API_KEY="your-key"
contextforge ask "How are citations formatted?" --generator openai
```

For PDFs: `pip install -e '.[pdf]'`. PDFs containing scanned images require OCR, which is not included.

## Design

```text
documents -> extract -> deterministic chunks -> index.json
                                               |
question -> lexical / dense / hybrid retrieval -+
                                               |
                      top-K evidence -> answer + source/page references
                                               |
                             labelled queries -> Recall@K / Hit@K / MRR@K
```

Lexical search is the default, so the repository can be tested without downloading models or providing an API key. Dense and hybrid modes load a pretrained sentence-embedding model on demand. Hybrid uses reciprocal rank fusion, not raw score addition.

## CLI

```bash
contextforge index DIRECTORY [--output artifacts/index.json] [--chunk-words 160] [--overlap-words 32]
contextforge search QUESTION [--index artifacts/index.json] [--mode lexical|dense|hybrid] [--k 5]
contextforge ask QUESTION [--generator extractive|openai] [--llm-model gpt-4.1-mini] [--min-score 0.05]
contextforge eval DATASET [--mode lexical|dense|hybrid] [--k 5]
```

Outputs are machine-readable JSON. Retrieval evaluation labels refer to stable chunk IDs; if you change chunking parameters, you must regenerate and review labels.

## Evaluation and limitations

See [experiment plan](docs/EXPERIMENT_PLAN.md) for a larger benchmark protocol. Relevant questions include: when does hybrid improve lexical retrieval, how sensitive are results to chunk size and overlap, and when should the system abstain?

Current limitations:
- This is a **research starter**, not a production multi-tenant document service.
- Index storage is JSON, and retrieval vectors are recomputed on startup.
- Chunking is word-based, not semantic or layout aware.
- No adversarial prompt injection guarantee; retrieved material remains untrusted.
- Citation formatting is prompted, not independently verified against each factual claim.
- No calibrated answer-faithfulness metric or production-scale dataset yet.

Contributions and careful reproducibility are more valuable here than claiming impressive results from the sample.

MIT licence.
