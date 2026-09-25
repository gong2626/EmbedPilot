"""A small, inspectable local hybrid index for the EmbedPilot corpus."""

from __future__ import annotations

import asyncio
import json
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import numpy as np
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi  # type: ignore[import-untyped]

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_INDEX_DIR = Path(__file__).resolve().parents[2] / "data" / "index"
STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "can",
        "do",
        "does",
        "for",
        "from",
        "how",
        "i",
        "in",
        "is",
        "of",
        "on",
        "or",
        "the",
        "to",
        "what",
        "when",
        "where",
        "which",
        "why",
        "with",
        "you",
        "zephyr",
    }
)


def index_dir() -> Path:
    """Resolve the index location without depending on the shell's cwd."""
    return (
        Path(os.environ.get("EMBEDPILOT_INDEX_DIR", DEFAULT_INDEX_DIR))
        .expanduser()
        .resolve()
    )


def tokenize(text: str) -> list[str]:
    """Tokenize English identifiers and Chinese runs for BM25 search."""
    aliases = {
        "threads": "thread",
        "semaphores": "semaphore",
        "mutexes": "mutex",
        "queues": "queue",
        "timers": "timer",
    }
    return [
        aliases.get(token, token)
        for token in re.findall(
            r"[A-Za-z_][A-Za-z_0-9]*|[\u4e00-\u9fff]+", text.lower()
        )
        if token not in STOPWORDS
    ]


@lru_cache(maxsize=1)
def _encoder(cache_dir: str):
    """Load the CPU embedding model lazily, downloading it on first use."""
    from fastembed import TextEmbedding

    return TextEmbedding(model_name=EMBEDDING_MODEL, cache_dir=cache_dir)


def _embed(texts: Sequence[str], location: Path) -> np.ndarray:
    """Compute L2-normalized vectors for documents or queries."""
    model = _encoder(str(location / "model_cache"))
    vectors = np.asarray(list(model.embed(list(texts))), dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[0] != len(texts):
        raise ValueError("Embedding model returned an unexpected shape")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)


@dataclass
class SearchHit:
    """One retrieved document with its modality-specific scores."""

    document: Document
    keyword_score: float
    vector_score: float | None
    fusion_score: float


class LocalIndex:
    """Persist documents and optional embeddings without an external service."""

    def __init__(self, location: Path | None = None):
        """Initialize a local index at an explicit or default path."""
        self.location = (location or index_dir()).resolve()

    def build(
        self, documents: Sequence[Document], *, with_embeddings: bool = True
    ) -> dict:
        """Replace the index with the supplied, traceable document chunks."""
        if not documents:
            raise ValueError("Cannot build an empty index")
        self.location.mkdir(parents=True, exist_ok=True)
        vectors = (
            _embed([doc.page_content for doc in documents], self.location)
            if with_embeddings
            else None
        )
        records = [
            {"page_content": doc.page_content, "metadata": doc.metadata}
            for doc in documents
        ]
        temp_docs = self.location / "documents.jsonl.tmp"
        with temp_docs.open("w", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        temp_docs.replace(self.location / "documents.jsonl")
        if vectors is not None:
            temp_vectors = self.location / "vectors.npy.tmp"
            with temp_vectors.open("wb") as stream:
                np.save(stream, vectors, allow_pickle=False)
            temp_vectors.replace(self.location / "vectors.npy")
        manifest = {
            "documents": len(documents),
            "embedding_model": EMBEDDING_MODEL if with_embeddings else None,
            "vector_dimensions": int(vectors.shape[1]) if vectors is not None else None,
        }
        (self.location / "index_manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        return manifest

    def load(self) -> tuple[list[Document], np.ndarray | None]:
        """Load the persisted index and verify its shape."""
        path = self.location / "documents.jsonl"
        if not path.exists():
            raise FileNotFoundError(
                f"No local index at {path}. Run `python -m retrieval_graph.cli ingest` first."
            )
        with path.open(encoding="utf-8") as stream:
            documents = [
                Document(**json.loads(line)) for line in stream if line.strip()
            ]
        manifest = json.loads(
            (self.location / "index_manifest.json").read_text(encoding="utf-8")
        )
        if len(documents) != manifest["documents"]:
            raise ValueError("Index document count does not match its manifest")
        vectors = None
        if manifest["embedding_model"]:
            vectors = np.load(self.location / "vectors.npy", allow_pickle=False)
            if vectors.shape != (len(documents), manifest["vector_dimensions"]):
                raise ValueError("Index vector shape does not match its manifest")
        return documents, vectors

    def search(
        self, query: str, *, k: int = 5, mode: str = "hybrid", user_id: str = "public"
    ) -> list[SearchHit]:
        """Rank an authorized subset with BM25, vectors, or reciprocal-rank fusion."""
        if mode not in {"keyword", "vector", "hybrid"}:
            raise ValueError(f"Unknown search mode: {mode}")
        if k < 1:
            raise ValueError("k must be positive")
        documents, all_vectors = self.load()
        selected = [
            i
            for i, doc in enumerate(documents)
            if doc.metadata.get("user_id", "public") == user_id
        ]
        if not selected:
            return []
        docs = [documents[i] for i in selected]
        tokens = [
            tokenize(doc.page_content)
            + tokenize(str(doc.metadata.get("section", ""))) * 2
            + tokenize(Path(str(doc.metadata.get("path", ""))).stem) * 3
            for doc in docs
        ]
        bm25 = BM25Okapi(tokens)
        keyword_scores = np.asarray(bm25.get_scores(tokenize(query)), dtype=float)
        vector_scores: np.ndarray | None = None
        if mode != "keyword" and all_vectors is not None:
            query_vector = _embed([query], self.location)[0]
            vector_scores = all_vectors[selected] @ query_vector
        if mode == "vector" and vector_scores is None:
            raise ValueError(
                "This index has no embeddings; re-ingest without --keyword-only"
            )
        keyword_order = list(np.argsort(-keyword_scores, kind="stable"))
        vector_order = (
            list(np.argsort(-vector_scores, kind="stable"))
            if vector_scores is not None
            else []
        )
        fusion = np.zeros(len(docs), dtype=float)
        if mode in {"keyword", "hybrid"}:
            for rank, index in enumerate(keyword_order):
                if keyword_scores[index] > 0:
                    fusion[index] += 1 / (60 + rank + 1)
        if mode in {"vector", "hybrid"}:
            for rank, index in enumerate(vector_order):
                fusion[index] += 1 / (60 + rank + 1)
        # Prevent many chunks from one long file crowding out other sources.
        order = []
        per_source: dict[str, int] = {}
        for index in np.argsort(-fusion, kind="stable"):
            source = str(docs[index].metadata.get("path", index))
            if fusion[index] <= 0 or per_source.get(source, 0) >= 2:
                continue
            order.append(index)
            per_source[source] = per_source.get(source, 0) + 1
            if len(order) == k:
                break
        return [
            SearchHit(
                document=docs[i],
                keyword_score=float(keyword_scores[i]),
                vector_score=float(vector_scores[i])
                if vector_scores is not None
                else None,
                fusion_score=float(fusion[i]),
            )
            for i in order
        ]


class LocalRetriever:
    """Adapt LocalIndex to the template's asynchronous retriever interface."""

    def __init__(
        self,
        *,
        location: Path | None = None,
        user_id: str = "public",
        k: int = 5,
        mode: str = "hybrid",
    ):
        """Bind a local index to one namespace and search mode."""
        self.index = LocalIndex(location)
        self.user_id = user_id
        self.k = k
        self.mode = mode

    async def ainvoke(self, query: str, config=None) -> list[Document]:
        """Retrieve documents without blocking the event loop."""
        hits = await asyncio.to_thread(
            self.index.search, query, k=self.k, mode=self.mode, user_id=self.user_id
        )
        return [
            Document(
                page_content=hit.document.page_content,
                metadata={
                    **hit.document.metadata,
                    "keyword_score": hit.keyword_score,
                    "vector_score": hit.vector_score,
                    "fusion_score": hit.fusion_score,
                },
            )
            for hit in hits
        ]

    async def aadd_documents(self, documents: Sequence[Document]) -> list[str]:
        """Append new documents by rebuilding the small local index."""
        try:
            existing, vectors = self.index.load()
            with_embeddings = vectors is not None
        except FileNotFoundError:
            existing = []
            with_embeddings = True
        await asyncio.to_thread(
            self.index.build, [*existing, *documents], with_embeddings=with_embeddings
        )
        return [str(doc.metadata.get("id", "")) for doc in documents]
