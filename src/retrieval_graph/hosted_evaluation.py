"""Run a small opt-in hosted-model smoke evaluation with auditable outputs."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from retrieval_graph.agent_graph import graph
from retrieval_graph.corpus import ZEPHYR_COMMIT
from retrieval_graph.utils import get_message_text

ROOT = Path(__file__).resolve().parents[2]


async def evaluate(cases_path: Path) -> dict:
    """Call DeepSeek sequentially for fixed public cases and record checkable facts."""
    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    rows = []
    for case in cases:
        start = time.perf_counter()
        result = await graph.ainvoke(
            {"messages": [HumanMessage(content=case["question"])]}
        )
        cited_paths = [item["path"] for item in result["citations"]]
        rows.append(
            {
                "id": case["id"],
                "question": case["question"],
                "answer": get_message_text(result["messages"][-1]),
                "route": result["route"],
                "route_correct": result["route"] == case["expected_route"],
                "evidence_ok": result["evidence_ok"],
                "generation_used": result["generation_used"],
                "generation_expected": case["expect_generation"],
                "generation_match": result["generation_used"]
                == case["expect_generation"],
                "citation_ids_valid": result["citation_ids_valid"],
                "cited_ids": result["cited_ids"],
                "cited_paths": cited_paths,
                "expected_source_path": case.get("expected_source_path"),
                "expected_source_retrieved": (
                    case["expected_source_path"] in cited_paths
                    if case.get("expected_source_path")
                    else None
                ),
                "expected_diagnostic_code": case.get("expected_diagnostic_code"),
                "expected_diagnostic_found": (
                    any(
                        item.get("code") == case["expected_diagnostic_code"]
                        for item in result["diagnostics"]
                    )
                    if case.get("expected_diagnostic_code")
                    else None
                ),
                "diagnostics": result["diagnostics"],
                "model_usage": result.get("model_usage", {}),
                "latency_ms": round((time.perf_counter() - start) * 1000, 2),
            }
        )
    generated = [row for row in rows if row["generation_used"]]
    citation_applicable = [row for row in generated if row["cited_paths"]]
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": "deepseek-flash",
        "agent_search_mode": os.environ.get("EMBEDPILOT_AGENT_SEARCH_MODE", "keyword"),
        "corpus_revision": ZEPHYR_COMMIT,
        "case_count": len(rows),
        "route_correct_count": sum(row["route_correct"] for row in rows),
        "generation_match_count": sum(row["generation_match"] for row in rows),
        "citation_id_valid_count": sum(
            row["citation_ids_valid"] is True for row in citation_applicable
        ),
        "citation_check_applicable_count": len(citation_applicable),
        "generated_count": len(generated),
        "scope": "Automated route, generation, source-presence and citation-number checks only. Claim-level factual correctness requires manual review.",
        "rows": rows,
    }


def main() -> None:
    """Require an explicit billing opt-in before running live cases."""
    parser = argparse.ArgumentParser(
        description="EmbedPilot live DeepSeek smoke evaluation"
    )
    parser.add_argument(
        "--allow-api",
        action="store_true",
        help="Confirm these cases may incur API charges",
    )
    parser.add_argument(
        "--cases", type=Path, default=ROOT / "eval" / "hosted_cases.json"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "reports" / "hosted_smoke.json"
    )
    args = parser.parse_args()
    if not args.allow_api:
        parser.error("Pass --allow-api to run hosted-model calls")
    load_dotenv(ROOT / ".env")
    if not os.environ.get("DEEPSEEK_API_KEY"):
        parser.error("DEEPSEEK_API_KEY is not configured in the ignored local .env")
    report = asyncio.run(evaluate(args.cases))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"Cases: {report['case_count']}; route checks: {report['route_correct_count']}; generation checks: {report['generation_match_count']}; citation IDs: {report['citation_id_valid_count']}/{report['citation_check_applicable_count']}"
    )
    print(f"Report: {args.output.resolve()}")


if __name__ == "__main__":
    main()
