import asyncio

import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document

from retrieval_graph.agent_graph import graph, validate_citation_ids
from retrieval_graph.api import app
from retrieval_graph.build_log import parse_build_log
from retrieval_graph.corpus import (
    ZEPHYR_COMMIT,
    ZEPHYR_PATHS,
    chunk_source,
    load_zephyr_corpus,
)
from retrieval_graph.local_index import LocalIndex, tokenize


def test_corpus_chunks_preserve_source() -> None:
    source = (
        "Threads\n=======\n\n" + "A thread has a stack and scheduling priority. " * 50
    )
    docs = chunk_source(
        "doc/kernel/services/threads/index.rst", source, chunk_size=220, overlap=20
    )
    assert len(docs) > 1
    assert all(doc.metadata["section"] == "Threads" for doc in docs)
    assert all(ZEPHYR_COMMIT in doc.metadata["source_url"] for doc in docs)
    assert len({doc.metadata["id"] for doc in docs}) == len(docs)


def test_corpus_rejects_corrupt_cached_source(tmp_path) -> None:
    target = tmp_path / ZEPHYR_PATHS[0]
    target.parent.mkdir(parents=True)
    target.write_text("corrupt", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_zephyr_corpus(tmp_path)


def test_build_log_parser_handles_gcc_and_msvc() -> None:
    diagnostics = parse_build_log(
        "src/main.c:42:7: error: implicit declaration of function k_sem_take\n"
        "main.cpp(12,3): warning C4996: unsafe function"
    )
    assert len(diagnostics) == 2
    assert diagnostics[0].path == "src/main.c"
    assert diagnostics[0].line == 42
    assert diagnostics[1].code == "C4996"


def test_keyword_index_is_local_and_namespaced(tmp_path) -> None:
    docs = [
        Document(
            page_content="Semaphore permits synchronize producers and consumers.",
            metadata={"user_id": "public", "section": "Semaphore"},
        ),
        Document(
            page_content="Mutex priority inheritance protects shared data.",
            metadata={"user_id": "public", "section": "Mutex"},
        ),
        Document(
            page_content="Private secret mutex note.",
            metadata={"user_id": "private", "section": "Private"},
        ),
        Document(
            page_content="Device tree describes hardware nodes.",
            metadata={"user_id": "public", "section": "Device tree"},
        ),
    ]
    index = LocalIndex(tmp_path)
    index.build(docs, with_embeddings=False)
    hits = index.search("mutex priority inheritance", mode="keyword")
    assert hits[0].document.metadata["section"] == "Mutex"
    assert all(hit.document.metadata["user_id"] == "public" for hit in hits)
    assert "what" not in tokenize("what is mutex")


def test_graph_routes_build_log_without_api_key(tmp_path, monkeypatch) -> None:
    docs = [
        Document(
            page_content=f"The function k_sem_take takes a semaphore. {i}",
            metadata={
                "user_id": "public",
                "section": "Taking a Semaphore",
                "path": "semaphores.rst",
                "offset": i,
                "source_url": "https://example.com/semaphores.rst",
            },
        )
        for i in range(4)
    ]
    LocalIndex(tmp_path).build(docs, with_embeddings=False)
    monkeypatch.setenv("EMBEDPILOT_INDEX_DIR", str(tmp_path))
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    result = asyncio.run(
        graph.ainvoke(
            {"messages": [("user", "main.c:7:2: error: k_sem_take undeclared")]}
        )
    )
    assert result["route"] == "build_log"
    assert result["diagnostics"][0]["line"] == 7
    assert result["attempts"] <= 2
    assert result["messages"][-1].content


def test_api_serves_search_and_keyless_analysis(tmp_path, monkeypatch) -> None:
    documents = [
        Document(
            page_content=f"Semaphore k_sem_take synchronizes worker thread {i}.",
            metadata={
                "user_id": "public",
                "section": "Semaphore",
                "path": "semaphores.rst",
                "offset": i,
                "source_url": "https://example.com/semaphores.rst",
            },
        )
        for i in range(4)
    ]
    LocalIndex(tmp_path).build(documents, with_embeddings=False)
    monkeypatch.setenv("EMBEDPILOT_INDEX_DIR", str(tmp_path))
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    client = TestClient(app)
    assert client.get("/health").json()["index_ready"]
    assert (
        client.get("/search", params={"q": "k_sem_take", "mode": "keyword"}).status_code
        == 200
    )
    answer = client.post(
        "/ask", json={"question": "main.c:7:2: error: k_sem_take undeclared"}
    )
    assert answer.status_code == 200
    assert answer.json()["route"] == "build_log"
    assert answer.json()["mode"] == "abstained"
    assert answer.json()["citations"] == []


def test_chinese_question_abstains_without_translation_key(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("EMBEDPILOT_INDEX_DIR", str(tmp_path))
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    result = asyncio.run(
        graph.ainvoke({"messages": [("user", "Zephyr 互斥锁优先级继承是什么？")]})
    )
    assert result["unsupported_language"]
    assert result["evidence_ok"] is False
    assert result["citations"] == []
    assert result["attempts"] == 0


def test_citation_id_validation_is_syntactic_only() -> None:
    sources = [{"id": 1}, {"id": 2}]
    assert validate_citation_ids("Supported by [1][2].", sources) == (True, [1, 2])
    assert validate_citation_ids("Unsupported [3].", sources) == (False, [3])
    assert validate_citation_ids("No references.", sources) == (False, [])


def test_out_of_scope_question_abstains_before_model_call(
    tmp_path, monkeypatch
) -> None:
    documents = [
        Document(
            page_content="Mutex priority inheritance protects shared data.",
            metadata={"user_id": "public", "section": "Mutex"},
        ),
        Document(
            page_content="Semaphore synchronizes worker threads.",
            metadata={"user_id": "public", "section": "Semaphore"},
        ),
        Document(
            page_content="Timer expiry wakes a thread.",
            metadata={"user_id": "public", "section": "Timer"},
        ),
    ]
    LocalIndex(tmp_path).build(documents, with_embeddings=False)
    monkeypatch.setenv("EMBEDPILOT_INDEX_DIR", str(tmp_path))
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fake-key-should-never-be-used")
    result = asyncio.run(
        graph.ainvoke({"messages": [("user", "How do I bake sourdough bread?")]})
    )
    assert result["evidence_ok"] is False
    assert result["generation_used"] is False
    assert result["citations"] == []
