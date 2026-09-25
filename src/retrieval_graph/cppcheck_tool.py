"""Run pinned Cppcheck on a temporary C/C++ snippet without executing it."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from xml.etree import ElementTree

LOCAL_EXE = (
    Path(__file__).resolve().parents[2]
    / "tools"
    / "cppcheck"
    / "extracted"
    / "PFiles"
    / "Cppcheck"
    / "cppcheck.exe"
)


def resolve_cppcheck() -> Path:
    """Prefer the verified local binary, then a user-supplied or PATH binary."""
    configured = os.environ.get("EMBEDPILOT_CPPCHECK")
    if configured:
        executable = Path(configured).expanduser().resolve()
        if not executable.is_file():
            raise FileNotFoundError(
                f"Cppcheck not found at configured path: {executable}"
            )
        return executable
    if LOCAL_EXE.is_file():
        return LOCAL_EXE
    on_path = shutil.which("cppcheck")
    if on_path:
        return Path(on_path).resolve()
    raise FileNotFoundError(
        "Cppcheck is unavailable. Run `powershell -File tools/setup_cppcheck.ps1` in the project root."
    )


def parse_cppcheck_xml(xml_text: str) -> list[dict]:
    """Convert Cppcheck XML findings to stable, path-safe records."""
    root = ElementTree.fromstring(xml_text)
    findings = []
    for error in root.findall("./errors/error")[:30]:
        location = error.find("location")
        findings.append(
            {
                "tool": "cppcheck",
                "code": error.get("id"),
                "severity": error.get("severity"),
                "message": error.get("msg"),
                "cwe": error.get("cwe"),
                "path": "<snippet>",
                "line": int(location.get("line", "0"))
                if location is not None
                else None,
                "column": int(location.get("column", "0"))
                if location is not None
                else None,
            }
        )
    return findings


def analyze_code(code: str, *, language: str = "c") -> list[dict]:
    """Analyze a bounded snippet with no shell, code execution, or repo edits."""
    if language not in {"c", "cpp"}:
        raise ValueError("language must be 'c' or 'cpp'")
    if not code.strip() or len(code) > 12000:
        raise ValueError("code must contain 1 to 12000 characters")
    executable = resolve_cppcheck()
    with tempfile.TemporaryDirectory(prefix="embedpilot-cppcheck-") as directory:
        source = Path(directory) / ("snippet.c" if language == "c" else "snippet.cpp")
        source.write_text(code, encoding="utf-8")
        try:
            completed = subprocess.run(
                [
                    str(executable),
                    "--xml",
                    "--xml-version=2",
                    "--enable=warning,style",
                    "--error-exitcode=0",
                    "--quiet",
                    str(source),
                ],
                cwd=directory,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=15,
                shell=False,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            raise TimeoutError("Cppcheck timed out after 15 seconds") from error
    if completed.returncode != 0:
        raise RuntimeError(f"Cppcheck failed with exit code {completed.returncode}")
    return parse_cppcheck_xml(completed.stderr)
