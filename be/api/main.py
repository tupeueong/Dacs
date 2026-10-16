"""FastAPI backend for the web UI: /api/health, /api/index-info, /api/chat."""

from __future__ import annotations

import logging
import os
import threading
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from retrieval.qdrant_store import QdrantStore  # also loads .env

logger = logging.getLogger("api")
COLLECTION = os.getenv("QDRANT_COLLECTION", "legal_chunks")
EMBEDDING_MODEL = "BAAI/bge-m3"
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"

app = FastAPI(title="Legal AI API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)

_lock = threading.RLock()  # models are not shown to be thread-safe: one chat at a time
_state: dict[str, Any] = {}


class ChatRequest(BaseModel):
    message: str
    history: list[dict[str, Any]] = []  # accepted for the FE contract; the pipeline is single-turn


def _error(status: int, message: str) -> HTTPException:
    return HTTPException(status_code=status, detail={"message": message})


def get_store() -> QdrantStore:
    with _lock:
        if "store" not in _state:
            _state["store"] = QdrantStore(collection_name=COLLECTION)
        return _state["store"]


def get_pipeline() -> Any:
    """Load the heavy models on first use so importing the app never needs Qdrant or Groq."""
    with _lock:
        if "pipeline" not in _state:
            from embeddings.bge_m3 import BgeM3Embedder
            from embeddings.bge_reranker import CrossEncoderReranker
            from llm.groq import GroqClient
            from rag.pipeline import RagPipeline
            from retrieval.retriever import RerankingRetriever, Retriever
            from router.semantic_router import SemanticRouter

            llm = GroqClient()  # fail on a missing key before loading models
            embedder = BgeM3Embedder()
            retriever = RerankingRetriever(Retriever(embedder, get_store()), CrossEncoderReranker())
            _state["pipeline"] = RagPipeline(SemanticRouter(embedder), retriever, llm)
        return _state["pipeline"]


def to_sources(citations: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Map pipeline citations (validated verbatim quotes) to the FE source shape."""
    return [
        {
            "doc_title": c["doc_name"],
            "article": c["article"],
            "clause": " ".join(str(c[k]) for k in ("clause", "point") if c.get(k)),
            "text": c["quote"],
        }
        for c in citations
    ]


@app.get("/api/health")
def health() -> dict[str, Any]:
    checks = {"qdrant": False, "collection": False}
    try:
        checks["qdrant"] = True
        checks["collection"] = get_store().client.collection_exists(COLLECTION)
    except Exception as exc:  # health must report, not raise
        logger.warning("health check failed: %s", exc)
        checks["qdrant"] = False
    ok = all(checks.values())
    body = {"status": "ok" if ok else "error", "service": "legal-rag", "checks": checks,
            "message": "Hệ thống sẵn sàng" if ok else "Qdrant hoặc collection chưa sẵn sàng"}
    if not ok:
        raise HTTPException(status_code=503, detail=body)
    return body


@app.get("/api/index-info")
def index_info() -> dict[str, Any]:
    try:
        info = get_store().client.get_collection(COLLECTION)
    except Exception as exc:
        logger.warning("index-info failed: %s", exc)
        raise _error(503, "Không đọc được thông tin collection Qdrant.") from None
    vectors = info.config.params.vectors
    return {
        "status": "ready",
        "corpus_name": f"Qdrant collection {COLLECTION}",
        "chunk_count": info.points_count,
        "article_count": "—",  # not stored in Qdrant; shown as a dash instead of a made-up number
        "embedding_dimension": vectors.size,
        "distance_metric": str(getattr(vectors.distance, "value", vectors.distance)).lower(),
        "embedding_model": EMBEDDING_MODEL,
        "rerank_model": RERANK_MODEL,
        "generator_model": os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"),
        "retrieval_accuracy": "Chưa đánh giá",
    }


@app.post("/api/chat")
def chat(req: ChatRequest) -> dict[str, Any]:
    try:
        with _lock:
            result = get_pipeline().answer(req.message)
    except ValueError as exc:  # empty question, missing .env / GROQ_API_KEY
        raise _error(400, str(exc)) from None
    except RuntimeError as exc:  # Groq failure (details already logged by GroqClient)
        raise _error(502, str(exc)) from None
    except Exception:
        logger.exception("chat failed")
        raise _error(503, "Dịch vụ truy hồi chưa sẵn sàng (kiểm tra Qdrant và model).") from None
    return {"answer": result["answer"], "sources": to_sources(result.get("citations", [])),
            "status": result.get("status")}
