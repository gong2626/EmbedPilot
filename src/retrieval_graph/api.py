"""Small read-only API and local demo for EmbedPilot."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from retrieval_graph.agent_graph import graph
from retrieval_graph.local_index import LocalIndex, index_dir
from retrieval_graph.utils import get_message_text

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")
app = FastAPI(
    title="EmbedPilot",
    version="0.1.0",
    description="Evidence-backed Zephyr knowledge and build-log analysis",
)


class AskRequest(BaseModel):
    """Question or pasted compiler output; no code execution is performed."""

    question: str = Field(min_length=1, max_length=12000)


@app.get("/")
def demo() -> FileResponse:
    """Serve the local demonstration page."""
    return FileResponse(ROOT / "static" / "embedpilot.html")


@app.get("/health")
def health() -> dict:
    """Report whether the local index and hosted model are configured."""
    path = index_dir() / "index_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    return {
        "index_ready": manifest is not None,
        "documents": manifest["documents"] if manifest else 0,
        "embedding_model": manifest["embedding_model"] if manifest else None,
        "generation_configured": bool(os.environ.get("DEEPSEEK_API_KEY")),
    }


@app.get("/search")
async def search(
    q: str = Query(min_length=1, max_length=1000),
    mode: Literal["keyword", "vector", "hybrid"] = "hybrid",
    k: int = Query(default=5, ge=1, le=10),
) -> dict:
    """Inspect retrieved evidence and its original source metadata."""
    try:
        hits = await asyncio.to_thread(LocalIndex().search, q, k=k, mode=mode)
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    return {
        "query": q,
        "mode": mode,
        "hits": [
            {
                "section": hit.document.metadata.get("section"),
                "path": hit.document.metadata.get("path"),
                "source_url": hit.document.metadata.get("source_url"),
                "snippet": hit.document.page_content[:500],
                "keyword_score": hit.keyword_score,
                "vector_score": hit.vector_score,
                "fusion_score": hit.fusion_score,
            }
            for hit in hits
        ],
    }


@app.post("/ask")
async def ask(request: AskRequest) -> dict:
    """Run the bounded LangGraph workflow and return grounded output."""
    try:
        result = await graph.ainvoke(
            {"messages": [HumanMessage(content=request.question)]}
        )
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    return {
        "answer": get_message_text(result["messages"][-1]),
        "route": result["route"],
        "attempts": result["attempts"],
        "evidence_ok": result["evidence_ok"],
        "diagnostics": result["diagnostics"],
        "citations": result["citations"],
        "mode": (
            "hosted"
            if result["generation_used"]
            else "evidence_preview"
            if result["citations"]
            else "abstained"
        ),
        "citation_ids_valid": result["citation_ids_valid"],
        "cited_ids": result["cited_ids"],
        "model_usage": result.get("model_usage", {}),
    }
