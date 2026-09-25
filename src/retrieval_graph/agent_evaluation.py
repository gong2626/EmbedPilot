"""Evaluate offline Agent routing and abstention without spending API credits."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from unittest.mock import patch

from langchain_core.messages import HumanMessage

from retrieval_graph.agent_graph import graph

ROOT = Path(__file__).resolve().parents[2]


async def evaluate(cases_path: Path) -> dict:
    """Run expected route, tool parse, and evidence outcomes on fixed cases."""
    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    rows = []
    with patch.dict(os.environ, {"DEEPSEEK_API_KEY": ""}):
        for case in cases:
            result = await graph.ainvoke(
                {"messages": [HumanMessage(content=case["input"])]}
            )
            actual = {
                "route": result["route"],
                "diagnostics": len(result["diagnostics"]),
                "evidence": bool(result["evidence_ok"]),
                "unsupported_language": bool(result.get("unsupported_language")),
            }
            checks = {
                key: actual[key.removeprefix("expected_")] == expected
                for key, expected in case.items()
                if key.startswith("expected_")
            }
            rows.append(
                {
                    "id": case["id"],
                    "actual": actual,
                    "checks": checks,
                    "passed": all(checks.values()),
                }
            )
    return {
        "cases": len(rows),
        "passed": sum(row["passed"] for row in rows),
        "scope": "offline heuristic routing, log parsing, and retrieval evidence gate only; not generated answer quality",
        "rows": rows,
    }


def main() -> None:
    """Run the offline cases and save an auditable report."""
    parser = argparse.ArgumentParser(description="EmbedPilot offline Agent evaluation")
    parser.add_argument(
        "--cases", type=Path, default=ROOT / "eval" / "agent_cases.json"
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = asyncio.run(evaluate(args.cases))
    print(f"Offline Agent checks: {report['passed']}/{report['cases']}")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"Full report: {args.output.resolve()}")


if __name__ == "__main__":
    main()
