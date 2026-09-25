"""Command-line entry points for reproducible ingestion and retrieval checks."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from retrieval_graph.corpus import load_zephyr_corpus, write_corpus_manifest
from retrieval_graph.local_index import LocalIndex, index_dir


def main() -> None:
    """Run a local corpus operation."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="EmbedPilot local knowledge base")
    commands = parser.add_subparsers(dest="command", required=True)
    ingest = commands.add_parser(
        "ingest", help="Fetch pinned Zephyr docs and build the index"
    )
    ingest.add_argument(
        "--keyword-only",
        action="store_true",
        help="Skip the one-time embedding model download",
    )
    search = commands.add_parser("search", help="Inspect retrieved source chunks")
    search.add_argument("query")
    search.add_argument(
        "--mode", choices=("keyword", "vector", "hybrid"), default="hybrid"
    )
    search.add_argument("--k", type=int, default=5)
    ask = commands.add_parser("ask", help="Run the bounded Agent workflow")
    ask.add_argument(
        "question", nargs="?", help="Question or pasted compiler diagnostic"
    )
    ask.add_argument("--log-file", type=Path, help="Read a build log from a local file")
    ask.add_argument(
        "--code-file", type=Path, help="Analyze a C/C++ source file via Cppcheck"
    )
    args = parser.parse_args()

    if args.command == "ingest":
        location = index_dir()
        documents, sources = load_zephyr_corpus(location.parent / "raw")
        manifest = LocalIndex(location).build(
            documents, with_embeddings=not args.keyword_only
        )
        write_corpus_manifest(location / "corpus_manifest.json", sources)
        print(
            json.dumps(
                {"sources": len(sources), **manifest, "index_dir": str(location)},
                indent=2,
            )
        )
    elif args.command == "search":
        hits = LocalIndex().search(args.query, k=args.k, mode=args.mode)
        for rank, hit in enumerate(hits, 1):
            meta = hit.document.metadata
            print(f"{rank}. {meta['section']} — {meta['path']}")
            print(f"   {meta['source_url']}")
            print(
                f"   keyword={hit.keyword_score:.3f} vector={hit.vector_score} fusion={hit.fusion_score:.4f}"
            )
            print(f"   {hit.document.page_content[:200].replace(chr(10), ' ')}…")
    elif args.command == "ask":
        from retrieval_graph.agent_graph import graph

        load_dotenv(Path(__file__).resolve().parents[2] / ".env")
        if args.log_file and args.code_file:
            parser.error("Choose only one of --log-file and --code-file")
        if args.code_file:
            suffix = args.code_file.suffix.lower()
            if suffix not in {".c", ".cc", ".cpp", ".cxx"}:
                parser.error("--code-file must be a C or C++ source file")
            language = "c" if suffix == ".c" else "cpp"
            code = args.code_file.read_text(encoding="utf-8", errors="replace")
            question = f"Please inspect this code:\n```{language}\n{code}\n```"
        elif args.log_file:
            question = args.log_file.read_text(encoding="utf-8", errors="replace")
        else:
            question = args.question
        if not question:
            parser.error("ask requires a question, --log-file, or --code-file")
        result = asyncio.run(graph.ainvoke({"messages": [HumanMessage(content=question)]}))
        print(result["messages"][-1].content)
        print(
            f"\nroute={result['route']} attempts={result['attempts']} evidence_ok={result['evidence_ok']}"
        )
        if result["diagnostics"]:
            print(json.dumps(result["diagnostics"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
