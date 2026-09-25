# EmbedPilot：嵌入式研发知识与编译日志分析助手

这是基于 [LangChain retrieval-agent-template](https://github.com/langchain-ai/retrieval-agent-template) 的二次开发项目，保留原仓库历史与 MIT 许可。它不是原模板的直接演示：新增了固定版本的 Zephyr 文档语料、本地 BM25 + 向量检索、有限重试的 LangGraph 工作流、编译日志解析、只读 API/网页和可复跑的评测。

## 当前能力

- 从 Zephyr v4.4.0 的固定提交 `684c9e8f32e4373a21098559f748f06915f950c9` 下载 16 个公开文档文件，切成 254 个片段；每个片段记录文件、章节、源码链接、版本和许可。下载后按项目中固定的 SHA-256 校验。原文来自 [Zephyr 官方仓库](https://github.com/zephyrproject-rtos/zephyr)。首次构建需要联网；原文快照和索引在 `data/` 下，不纳入 Git。
- 在 CPU 上使用 `BAAI/bge-small-en-v1.5` 生成 384 维向量，并实现 BM25 与倒数排名融合检索，无需 Elasticsearch、Pinecone 或 GPU。因当前评测中关键词检索更稳，Agent 默认使用 BM25；`/search` 仍可选择向量/混合检索做对照。
- 识别 GCC、Clang、MSVC 风格的编译诊断，并提取文件、行号、严重程度、消息。明确提交的 C/C++ 代码块由 Cppcheck 2.22.0 在临时文件上静态分析，返回结构化发现；不编译、执行原程序或修改用户原文件。
- LangGraph 按“识别输入 → 检索 → 判断证据 → 最多重试一次 → 回答/拒答”运行。离线中文提问会明确提示无法可靠检索；配置 API 后先把中文检索问题译为英文，再用 DeepSeek `deepseek-flash` 生成有来源的回答。
- FastAPI 提供 `/health`、`/search`、`/analyze-code`、`/ask` 和本地网页。

## 在这台 Windows 电脑上运行

在 `E:\job\EmbedPilot` 打开 PowerShell：

```powershell
uv sync --group dev --extra dev
uv run python -m retrieval_graph.cli ingest
uv run python -m retrieval_graph.cli search "What is priority inheritance for a mutex?" --mode hybrid
uv run python -m retrieval_graph.cli ask "What is priority inheritance for a mutex?"
powershell -File tools/setup_cppcheck.ps1
uv run python -m retrieval_graph.cli ask --code-file samples/bounds_bug.c
uv run uvicorn retrieval_graph.api:app --host 127.0.0.1 --port 8000
```

浏览器打开 `http://127.0.0.1:8000`。无 API 密钥时仍可进行离线证据预览和 Cppcheck 分析。`uv sync` 创建项目自己的 `.venv`，不修改 `gxy`。向量模型首次下载需要联网，后续使用本地缓存。如果目录再次移动，请在新目录重新运行 `uv sync --reinstall --group dev --extra dev` 修复 Windows 虚拟环境中的旧路径。Cppcheck 设置脚本下载 [官方 2.22.0 Windows 安装包](https://github.com/cppcheck-opensource/cppcheck/releases/tag/2.22.0)、校验 SHA-256 并提取到被 Git 忽略的项目内工具目录；不向仓库分发其 GPL-3.0 二进制文件，也不做全局系统安装。

**数据边界**：`/analyze-code` 只在本机运行 Cppcheck；配置 API 后，`/ask` 会把用户问题、粘贴的日志/代码片段和检索片段发送给 DeepSeek 生成回答。请勿输入单位私有代码、个人敏感数据或未获授权的资料。服务按上例只绑定 `127.0.0.1`；不要将这个未加认证的演示接口直接暴露到公网。

需要模型回答时，在项目根目录的 `.env` 中填写 `DEEPSEEK_API_KEY`。使用官方接口时保持 `DEEPSEEK_BASE_URL=https://api.deepseek.com`；`DeepSeek-V4.1-Flash` 的 API 模型标识是 `deepseek-flash`，以 [DeepSeek 官方接口说明](https://api-docs.deepseek.com/guides/harness) 为准。`.env` 已被 Git 忽略，**不要把密钥发到聊天或提交到仓库**。目前尚未用用户密钥做真实 API 端到端验证。

## 验证与评测

```powershell
uv run pytest tests/unit_tests -q
uv run pytest tests/integration_tests/test_cppcheck_tool.py -q
uv run --extra dev ruff check src/retrieval_graph tests/unit_tests
uv run python -m retrieval_graph.evaluation --cases eval/retrieval_cases_v2.json --output reports/retrieval_v2.json
uv run python -m retrieval_graph.agent_evaluation --output reports/agent_offline_keyword.json
# Optional: uses your configured DeepSeek API and may incur charges
uv run python -m retrieval_graph.hosted_evaluation --allow-api --output reports/hosted_keyword_smoke.json
```

`eval/retrieval_cases_v2.json` 是 15 条人工编写的英文开发题，按源文件标注相关性。`eval/holdout_cases_v2.json` 是 10 条额外题目，但语料扩充和标签修订发生在评测过程中，因此也不能视作严格盲测 benchmark。这些题只观察检索，不评估生成回答的正确性。当前 16 文件语料上的结果：

| 模式 | 开发题 Recall@5 / MRR@5 | 额外题 Recall@5 / MRR@5 |
| --- | ---: | ---: |
| BM25 关键词 | 0.8667 / 0.8333 | 0.8500 / 0.6900 |
| 向量 | 0.7667 / 0.7333 | 0.7500 / 0.5833 |
| 混合 | 0.8667 / 0.8133 | 0.7500 / 0.6333 |

混合检索没有超过关键词基线；不能将其描述为性能提升。当前逐题结果见 `reports/retrieval_v2.json` 和 `reports/holdout_v2.json`。十文件语料的旧报告保留作开发记录，但因语料和标签均已改变，不能直接和上表比较。离线 Agent 的 7 条固定检查全部通过，见 `reports/agent_offline_v2.json`；这不是回答质量分数。

用户在本机配置密钥后，默认 BM25 模式下 5 条预设公开题的 DeepSeek 真实调用已完成：路由和是否调用模型的预期均为 5/5；4 条生成回答的引用**编号**均合法，预期来源文件均出现在引用中。另有 1 条合成 C 数组越界案例走通 Cppcheck→DeepSeek 链路。人工逐条核对的范围与保留意见见 `reports/HOSTED_MANUAL_REVIEW.md`。样本太小，不能据此宣称总体回答准确率；令牌用量和每题耗时保存在 `reports/hosted_keyword_smoke.json`、`reports/hosted_code_smoke.json`，未换算费用。旧混合模式报告 `reports/hosted_smoke.json` 保留作对照。

## 边界与后续

- 当前语料只有 16 个 Zephyr 文档文件，不覆盖完整 SDK、板级驱动或真实企业缺陷库；不要把问答结果当作已验证修复。
- 英文向量模型对中文问题可能检索较差；有 API 时先翻译查询，离线时拒绝不可靠的中文检索。目前只验证过一条中文真实回答，不能宣称广泛的中文能力。
- 编译日志工具只解析粘贴的文本；Cppcheck 已在合成 C 案例上验证，但尚未在真实 Zephyr 工程、板卡或编译数据库上验证，也未安装编译器。
- 工作流用可解释的规则选择工具，不应表述为“模型自主多工具规划”。
- 证据阈值是启发式，系统性的拒答准确率尚未测定；DeepSeek 的回答只经过小样本人工核查和引用编号检查，未做完整事实性或费用评测。
- 源文档是 Zephyr 项目公开资料；遵守其许可和出处，不在索引中放入个人简历、研究数据或雇主私有代码。

因此项目目前是可复现的 **开发版**，不是可以直接写进简历的最终版。剩余事项与验收标准见 [PROJECT_PLAN.md](PROJECT_PLAN.md)。
