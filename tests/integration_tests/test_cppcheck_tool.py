from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from retrieval_graph.api import app
from retrieval_graph.cppcheck_tool import analyze_code, resolve_cppcheck


def test_cppcheck_detects_synthetic_bounds_bug() -> None:
    try:
        resolve_cppcheck()
    except FileNotFoundError:
        pytest.skip("Cppcheck is not installed; run tools/setup_cppcheck.ps1")
    root = Path(__file__).resolve().parents[2]
    sample = (root / "samples" / "bounds_bug.c").read_text(encoding="utf-8")
    findings = analyze_code(sample)
    assert any(
        item["code"] == "arrayIndexOutOfBounds" and item["line"] == 5
        for item in findings
    )
    assert all(item["path"] == "<snippet>" for item in findings)


def test_api_analyzes_synthetic_code_without_model_call(monkeypatch) -> None:
    try:
        resolve_cppcheck()
    except FileNotFoundError:
        pytest.skip("Cppcheck is not installed; run tools/setup_cppcheck.ps1")
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    code = "int read_sensor(void) { int samples[2]={1,2}; return samples[2]; }"
    response = TestClient(app).post(
        "/analyze-code", json={"code": code, "language": "c"}
    )
    assert response.status_code == 200
    assert any(
        item["code"] == "arrayIndexOutOfBounds" for item in response.json()["findings"]
    )
