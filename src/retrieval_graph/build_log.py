"""Read-only parsing of compiler and build diagnostics."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

GCC_PATTERN = re.compile(
    r"^(?P<path>.+?):(?P<line>\d+)(?::(?P<column>\d+))?:\s*"
    r"(?P<severity>fatal error|error|warning|note):\s*(?P<message>.+)$",
    re.IGNORECASE,
)
MSVC_PATTERN = re.compile(
    r"^(?P<path>.+?)\((?P<line>\d+)(?:,(?P<column>\d+))?\)\s*:\s*"
    r"(?P<severity>fatal error|error|warning|note)\s*(?P<code>[A-Z]+\d+)?:?\s*(?P<message>.+)$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Diagnostic:
    """A normalized compiler finding, without executing or editing code."""

    path: str
    line: int
    column: int | None
    severity: str
    message: str
    code: str | None = None

    def as_dict(self) -> dict:
        """Return a JSON-friendly record."""
        return asdict(self)


def parse_build_log(text: str, *, limit: int = 30) -> list[Diagnostic]:
    """Extract GCC/Clang/MSVC diagnostics from untrusted pasted text."""
    findings: list[Diagnostic] = []
    for line in text.splitlines()[:2000]:
        if len(line) > 4000:
            continue
        match = GCC_PATTERN.match(line.strip()) or MSVC_PATTERN.match(line.strip())
        if not match:
            continue
        fields = match.groupdict()
        findings.append(
            Diagnostic(
                path=fields["path"],
                line=int(fields["line"]),
                column=int(fields["column"]) if fields.get("column") else None,
                severity=fields["severity"].lower(),
                message=fields["message"],
                code=fields.get("code"),
            )
        )
        if len(findings) >= limit:
            break
    return findings
