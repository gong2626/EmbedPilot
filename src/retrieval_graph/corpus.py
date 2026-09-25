"""Fetch and chunk a version-pinned public Zephyr documentation corpus."""

from __future__ import annotations

import hashlib
import json
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from langchain_core.documents import Document

ZEPHYR_COMMIT = "684c9e8f32e4373a21098559f748f06915f950c9"  # v4.4.0
ZEPHYR_PATHS = (
    "doc/kernel/services/threads/index.rst",
    "doc/kernel/services/synchronization/semaphores.rst",
    "doc/kernel/services/synchronization/mutexes.rst",
    "doc/kernel/services/data_passing/queues.rst",
    "doc/kernel/services/data_passing/fifos.rst",
    "doc/kernel/services/data_passing/lifos.rst",
    "doc/kernel/services/timing/timers.rst",
    "doc/kernel/services/scheduling/index.rst",
    "doc/build/dts/index.rst",
    "doc/build/dts/intro.rst",
    "doc/build/dts/api-usage.rst",
    "doc/build/kconfig/index.rst",
    "doc/build/kconfig/setting.rst",
    "doc/build/kconfig/menuconfig.rst",
    "doc/develop/application/index.rst",
    "doc/develop/getting_started/index.rst",
)
SOURCE_HASHES = {
    "doc/kernel/services/threads/index.rst": "1193a2dcdf92da6a1752ea21eea0bc205183c3722e63f7ed76de5aee433df412",
    "doc/kernel/services/synchronization/semaphores.rst": "89055bbf916c29e062b8193e70ddbdfb7c3ee5474178293224b839a73f48da05",
    "doc/kernel/services/synchronization/mutexes.rst": "10cfd93af198eb8e78a758365f19aea93abeb1351217d35f4f487e031a912c33",
    "doc/kernel/services/data_passing/queues.rst": "4223a462fb55a01f03cf6496fecb1ccd429858dfa0f786557a5c575917f63a6b",
    "doc/kernel/services/data_passing/fifos.rst": "292b0ecb77f24f77944259be43e60ad2d5994f7b307ad833401e4acdf5c6a3a1",
    "doc/kernel/services/data_passing/lifos.rst": "949728295f67c63524df9ab68b748e9e3b0b771ba523ce47f495f5b303ea2dc9",
    "doc/kernel/services/timing/timers.rst": "ac26d2d8b7dfa3f0b8e881ee73d335ad0485f46651e7a33164e375a53af72413",
    "doc/kernel/services/scheduling/index.rst": "038989510e5dcdac8ac64a44c36db105e076ced353b7cb601580737f2794b4da",
    "doc/build/dts/index.rst": "d0274e46597aa81b119af9463e7a132c37e9f6ee1d9ad7b9efe83a0d4b865d55",
    "doc/build/dts/intro.rst": "2697c2fc29aa5707944ee6498726d1a6a6ca6eeaae226f5de8fc01dc1d7d73ee",
    "doc/build/dts/api-usage.rst": "a531cf722c35b85abaa61765a700783ae8c5afeab790051d859d103d8ebbdb46",
    "doc/build/kconfig/index.rst": "f713b672d204c93d2dbc5ea558e1173a1c177d723687ff2cec14dca38f1b612a",
    "doc/build/kconfig/setting.rst": "03c65eb809ad09a4be33c41ec1b3afebca7cce65839304e82edfa70a6eb74546",
    "doc/build/kconfig/menuconfig.rst": "e2422534e3e2e26f31d1cbe839386d8c8c3454cff78b6ba73d08470df6c5b60b",
    "doc/develop/application/index.rst": "270139c08ec8d4b497338abc89b514a928963f9f0084c9fa4a21a3556cb4c10a",
    "doc/develop/getting_started/index.rst": "126cce7bdb98b7ef93bf1084cd3efcc4e74eebaa13786cceaeebff04ec722bc2",
}
BASE_RAW_URL = (
    f"https://raw.githubusercontent.com/zephyrproject-rtos/zephyr/{ZEPHYR_COMMIT}"
)
BASE_SOURCE_URL = f"https://github.com/zephyrproject-rtos/zephyr/blob/{ZEPHYR_COMMIT}"


@dataclass(frozen=True)
class SourceRecord:
    """Describe a source file used to construct the index."""

    path: str
    sha256: str
    bytes: int


def _read_or_fetch(path: str, cache_dir: Path) -> bytes:
    """Read a local snapshot or download a pinned upstream source file."""
    target = cache_dir / path
    if target.exists():
        return target.read_bytes()
    url = f"{BASE_RAW_URL}/{path}"
    request = urllib.request.Request(url, headers={"User-Agent": "EmbedPilot/0.1"})
    with urllib.request.urlopen(request, timeout=30) as response:
        data = response.read()
    if not data or len(data) > 2_000_000:
        raise ValueError(f"Unexpected source size for {path}: {len(data)}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    return data


def _sections(text: str) -> list[tuple[str, str]]:
    """Split reStructuredText into heading-labeled sections."""
    lines = text.splitlines()
    sections: list[tuple[str, str]] = []
    title = "Introduction"
    content: list[str] = []
    index = 0
    while index < len(lines):
        if index + 1 < len(lines):
            underline = lines[index + 1].strip()
            heading = lines[index].strip()
            if (
                heading
                and len(underline) >= len(heading)
                and len(underline) >= 3
                and len(set(underline)) == 1
                and underline[0] in "=-~^\"'`:+*#"
            ):
                if content:
                    sections.append((title, "\n".join(content).strip()))
                title = heading
                content = []
                index += 2
                continue
        content.append(lines[index])
        index += 1
    if content:
        sections.append((title, "\n".join(content).strip()))
    return [(heading, body) for heading, body in sections if len(body) >= 40]


def chunk_source(
    path: str, raw_text: str, chunk_size: int = 1100, overlap: int = 150
) -> list[Document]:
    """Create traceable chunks while keeping upstream section labels."""
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    documents: list[Document] = []
    for section, body in _sections(raw_text):
        step = chunk_size - overlap
        for offset in range(0, len(body), step):
            chunk = body[offset : offset + chunk_size].strip()
            if len(chunk) < 40:
                continue
            source_id = hashlib.sha256(
                f"{path}:{section}:{offset}".encode()
            ).hexdigest()[:16]
            documents.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "id": source_id,
                        "path": path,
                        "section": section,
                        "offset": offset,
                        "source_url": f"{BASE_SOURCE_URL}/{path}",
                        "revision": ZEPHYR_COMMIT,
                        "license": "Apache-2.0",
                        "user_id": "public",
                    },
                )
            )
    return documents


def load_zephyr_corpus(cache_dir: Path) -> tuple[list[Document], list[SourceRecord]]:
    """Load all selected public files and return documents plus provenance."""
    documents: list[Document] = []
    sources: list[SourceRecord] = []
    for path in ZEPHYR_PATHS:
        raw = _read_or_fetch(path, cache_dir)
        actual_hash = hashlib.sha256(raw).hexdigest()
        if actual_hash != SOURCE_HASHES[path]:
            raise ValueError(
                f"Source checksum mismatch for {path}; remove the bad cache file and retry"
            )
        decoded = raw.decode("utf-8")
        documents.extend(chunk_source(path, decoded))
        sources.append(SourceRecord(path=path, sha256=actual_hash, bytes=len(raw)))
    if not documents:
        raise ValueError("No documents were produced from the corpus")
    return documents, sources


def write_corpus_manifest(path: Path, sources: list[SourceRecord]) -> None:
    """Write the precise revision and checksums of fetched files."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "repository": "https://github.com/zephyrproject-rtos/zephyr",
                "revision": ZEPHYR_COMMIT,
                "license": "Apache-2.0",
                "files": [record.__dict__ for record in sources],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
