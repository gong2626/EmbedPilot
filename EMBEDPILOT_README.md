# EmbedPilot：嵌入式研发知识与编译日志分析助手

这是基于 [LangChain retrieval-agent-template](https://github.com/langchain-ai/retrieval-agent-template) 的二次开发项目，保留原仓库历史与 MIT 许可。它不是原模板的直接演示：新增了固定版本的 Zephyr 文档语料、本地 BM25 + 向量检索、有限重试的 LangGraph 工作流、编译日志解析、只读 API/网页和可复跑的评测。

## 当前能力

- 从 Zephyr v4.4.0 的固定提交 `684c9e8f32e4373a21098559f748f06915f950c9` 下载 16 个公开文档文件，切成 254 个片段；每个片段记录文件、章节、源码链接、版本和许可。下载后按项目中固定的 SHA-256 校验。原文来自 [Zephyr 官方仓库](https://github.com/zephyrproject-rtos/zephyr)。首次构建需要联网；原文快照和索引在 `data/` 下，不纳入 Git。
- 在 CPU 上使用 `BAAI/bge-small-en-v1.5` 生成 384 维向量，结合 BM25 和倒数排名融合检索，无需 Elasticsearch、Pinecone 或 GPU。
- 识别 GCC、Clang、MSVC 风格的编译诊断，并提取文件、行号、严重程度、消息；工具只解析文本，不执行编译命令、不修改代码。
- LangGraph 按“识别输入 → 检索 → 判断证据 → 最多重试一次 → 回答/拒答”运行。离线中文提问会明确提示无法可靠检索；配置 API 后先把中文检索问题译为英文，再用 DeepSeek `deepseek-flash` 生成有来源的回答。
- FastAPI 提供 `/health`、`/search`、`/ask` 和本地网页。

## 在这台 Windows 电脑上运行

在 `E:\job\EmbedPilot` 打开 PowerShell：

```powershell
uv sync --group dev --extra dev
uv run python -m retrieval_graph.cli ingest
uv run python -m retrieval_graph.cli search "What is priority inheritance for a mutex?" --mode hybrid
uv run python -m retrieval_graph.cli ask "What is priority inheritance for a mutex?"
uv run uvicorn retrieval_graph.api:app --host 127.0.0.1 --port 8000
```

浏览器打开 `http://127.0.0.1:8000`。无 API 密钥时最后两条命令仍可用于离线证据预览。`uv sync` 创建项目自己的 `.venv`，不修改 `gxy`。向量模型首次下载需要联网，后续使用本地缓存。如果目录再次移动，请在新目录重新运行 `uv sync --reinstall --group dev --extra dev` 修复 Windows 虚拟环境中的旧路径。

需要模型回答时，在项目根目录的 `.env` 中填写 `DEEPSEEK_API_KEY`。使用官方接口时保持 `DEEPSEEK_BASE_URL=https://api.deepseek.com`；`DeepSeek-V4.1-Flash` 的 API 模型标识是 `deepseek-flash`，以 [DeepSeek 官方接口说明](https://api-docs.deepseek.com/guides/harness) 为准。`.env` 已被 Git 忽略，**不要把密钥发到聊天或提交到仓库**。目前尚未用用户密钥做真实 API 端到端验证。

## 验证与评测

```powershell
uv run pytest tests/unit_tests -q
uv run --extra dev ruff check src/retrieval_graph tests/unit_tests
uv run python -m retrieval_graph.evaluation --cases eval/retrieval_cases_v2.json --output reports/retrieval_v2.json
uv run python -m retrieval_graph.agent_evaluation --output reports/agent_offline_v2.json
```

`eval/retrieval_cases_v2.json` 是 15 条人工编写的英文开发题，按源文件标注相关性。`eval/holdout_cases_v2.json` 是 10 条额外题目，但语料扩充和标签修订发生在评测过程中，因此也不能视作严格盲测 benchmark。这些题只观察检索，不评估生成回答的正确性。当前 16 文件语料上的结果：

| 模式 | 开发题 Recall@5 / MRR@5 | 额外题 Recall@5 / MRR@5 |
| --- | ---: | ---: |
| BM25 关键词 | 0.8667 / 0.8333 | 0.8500 / 0.6900 |
| 向量 | 0.7667 / 0.7333 | 0.7500 / 0.5833 |
| 混合 | 0.8667 / 0.8133 | 0.7500 / 0.6333 |

混合检索没有超过关键词基线；不能将其描述为性能提升。当前逐题结果见 `reports/retrieval_v2.json` 和 `reports/holdout_v2.json`。十文件语料的旧报告保留作开发记录，但因语料和标签均已改变，不能直接和上表比较。离线 Agent 的 7 条固定检查全部通过，见 `reports/agent_offline_v2.json`；这不是回答质量分数。后续需要补充独立的新题集、引用正确性、真实模型回答和费用评测。

## 边界与后续

- 当前语料只有 16 个 Zephyr 文档文件，不覆盖完整 SDK、板级驱动或真实企业缺陷库；不要把问答结果当作已验证修复。
- 英文向量模型对中文问题可能检索较差；尚未完成跨语言查询处理。
- 编译日志工具只解析粘贴的文本，尚未安装或验证真实编译器/静态分析器。
- 工作流用可解释的规则选择工具，不应表述为“模型自主多工具规划”。
- 证据阈值是启发式，拒答准确率尚未测定；调用 DeepSeek 的成本和质量也尚未测定。
- 源文档是 Zephyr 项目公开资料；遵守其许可和出处，不在索引中放入个人简历、研究数据或雇主私有代码。

因此项目目前是可复现的 **开发版**，不是可以直接写进简历的最终版。剩余事项与验收标准见 [PROJECT_PLAN.md](PROJECT_PLAN.md)。
