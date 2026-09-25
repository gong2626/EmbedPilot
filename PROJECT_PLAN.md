# EmbedPilot development plan

This repository starts from [langchain-ai/retrieval-agent-template](https://github.com/langchain-ai/retrieval-agent-template) (MIT license). The upstream Git history and `LICENSE` are retained. Work on `embedpilot-development` records the independent additions.

## Target

Build an embedded software development assistant that can answer cited questions from public technical documentation, inspect C/C++ build or static analysis logs, and produce actionable defect reports. It must be runnable and evaluated on this computer before it is described on a resume.

## Implementation status (2026-09-25)

- Completed: pinned sixteen-file Zephyr v4.4.0 documentation corpus, 254 local chunks, source checksums, 384-dimensional CPU vectors, BM25/vector/hybrid search, provenance metadata, and keyword-only fallback.
- Completed: bounded LangGraph workflow, read-only GCC/Clang/MSVC diagnostic parsing, keyless evidence preview, FastAPI endpoints, local demo page, and seven passing unit/API tests.
- Completed as a **small exploratory evaluation only**: 15 labeled development questions, 10 additional questions, and 7 offline Agent checks. On the expanded corpus, hybrid Recall@5/MRR@5 is 0.8667/0.8133 on the development set and 0.7500/0.6333 on the additional set. Keyword-only is stronger on both. Do not claim a hybrid performance gain.
- Not completed: live DeepSeek API validation, held-out/citation/abstention evaluation, real compiler/static-analysis integration, optional Docker verification, and resume-ready evidence.

## Local environment checked on 2026-09-25

- Windows laptop, Ryzen 7 4800H, 16 GB RAM, NVIDIA GTX 1650 Ti (4 GB VRAM).
- Python 3.12.7, uv, Git, and Docker client available. Docker Desktop engine is not currently running.
- CMake, GCC/Clang, and cppcheck are not yet available on `PATH`.
- `uv sync --group dev` succeeded; upstream unit tests: 1 passed.
- No model API key is configured in this project. Upstream integration tests require external services and have not been run.

## Milestones and acceptance checks

1. **Reproducible local baseline.** Replace the template's mandatory cloud vector store and embedding dependencies with a local document index and local embeddings. Add a pinned public corpus, ingestion command, and a keyless retrieval smoke test. Record the upstream baseline and the changed modules.
2. **Embedded knowledge base.** Index a curated, version-pinned subset of Zephyr documentation and example code. Preserve source URL, file path, section, and license metadata. Add keyword/vector hybrid retrieval and source citations. No private lab, employer, or candidate files are included.
3. **Agent workflow and tools.** Add query routing, bounded retrieval retry, evidence check, and abstention. Provide read-only tools for source/document search and parsing compiler or static analysis output. Add a human-review gate before any proposed code change or externally posted comment. Expose an API and a local demo interface.
4. **Evaluation.** Create a reviewed question set covering factual lookup, multi-document questions, build-log diagnosis, tool choice, and out-of-corpus cases. Compare vector-only and hybrid retrieval using Recall@k/MRR, citation correctness, abstention accuracy, tool-call success, latency, and API cost where applicable. Publish raw cases, evaluation script, and a failure analysis. Never carry upstream sample metrics into this project's report.
5. **Delivery.** Package local startup instructions and an optional Docker setup, run tests and an end-to-end demo, document known limits, and prepare resume wording based only on measured results and independently implemented work.

## Runtime choices

- Local embeddings and index run on CPU; the 4 GB GPU is not needed for these stages.
- The user selected DeepSeek-V4.1-Flash for the final demo. DeepSeek's documented API model ID is `deepseek-flash` and its OpenAI-compatible base URL is `https://api.deepseek.com` ([API guide](https://api-docs.deepseek.com/guides/harness), [release notes](https://api-docs.deepseek.com/updates/)). Keep endpoint and model configurable; use deterministic local stubs for keyless development tests. Secrets stay outside Git.
- AutoDL is unnecessary for this scope. Revisit only if the goal changes to fine-tuning, large local models, or large-scale benchmark runs.

## User inputs eventually needed

- The API provider and model have been confirmed. Never send a key in chat; place it in the ignored local `.env` file when requested. Confirm that the available account can call `deepseek-flash` before the hosted-model end-to-end demo.
- Review the final demo and confirm which claims accurately reflect personal work before putting them on a resume.
