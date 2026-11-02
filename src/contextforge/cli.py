"""Command-line interface for indexing, querying and evaluating."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .answer import answer
from .documents import ingest
from .evaluation import evaluate
from .retrieval import Retriever
from .store import load_index, save_index


def parser() -> argparse.ArgumentParser:
    cli = argparse.ArgumentParser(prog="contextforge")
    subs = cli.add_subparsers(dest="command", required=True)
    create = subs.add_parser("index", help="Ingest .txt, .md and optional .pdf files")
    create.add_argument("directory", type=Path)
    create.add_argument("--output", type=Path, default=Path("artifacts/index.json"))
    create.add_argument("--chunk-words", type=int, default=160)
    create.add_argument("--overlap-words", type=int, default=32)
    for command in ("search", "ask", "eval"):
        cmd = subs.add_parser(command)
        cmd.add_argument("--index", type=Path, default=Path("artifacts/index.json"))
        cmd.add_argument("--mode", choices=["lexical", "dense", "hybrid"], default="lexical")
        cmd.add_argument("--k", type=int, default=5)
        cmd.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
        if command in ("search", "ask"):
            cmd.add_argument("question")
        else:
            cmd.add_argument("dataset", type=Path)
        if command == "ask":
            cmd.add_argument("--generator", choices=["extractive", "openai"], default="extractive")
            cmd.add_argument("--llm-model", default="gpt-4.1-mini")
            cmd.add_argument("--min-score", type=float, default=0.05)
    return cli


def main() -> None:
    args = parser().parse_args()
    if args.command == "index":
        chunks = ingest(args.directory, args.chunk_words, args.overlap_words)
        save_index(chunks, args.output)
        result = {
            "indexed_chunks": len(chunks),
            "unique_documents": len({c.source for c in chunks}),
            "index": str(args.output),
        }
    else:
        retriever = Retriever(
            load_index(args.index),
            mode=args.mode,
            model_name=args.embedding_model,
        )
        if args.command == "search":
            result = {
                "query": args.question,
                "hits": [h.to_dict() for h in retriever.search(args.question, args.k)],
            }
        elif args.command == "ask":
            result = answer(
                args.question, retriever, args.k, args.generator,
                args.llm_model, args.min_score,
            ).to_dict()
        else:
            result = evaluate(retriever, args.dataset, args.k)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
