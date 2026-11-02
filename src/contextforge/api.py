"""Optional lightweight FastAPI application."""

from __future__ import annotations

from pathlib import Path


def make_app(index: Path = Path("artifacts/index.json")):
    try:
        from fastapi import FastAPI, HTTPException
        from pydantic import BaseModel
    except ImportError as exc:
        raise RuntimeError("Install API dependencies: pip install fastapi uvicorn") from exc

    from .answer import answer
    from .retrieval import Retriever
    from .store import load_index

    app = FastAPI(title="ContextForge", version="0.1.0")

    class AskRequest(BaseModel):
        question: str
        top_k: int = 5

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.post("/ask")
    def ask(request: AskRequest) -> dict:
        try:
            retriever = Retriever(load_index(index))
            return answer(request.question, retriever, request.top_k).to_dict()
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return app
