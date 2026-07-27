# MathRAG

[![CI](https://github.com/xiaopangF/MathRAG/actions/workflows/ci.yml/badge.svg)](https://github.com/xiaopangF/MathRAG/actions/workflows/ci.yml)

MathRAG 是一个面向高等数学教材的本地优先 RAG 问答系统。项目覆盖 PDF 教材解析、结构化切分、混合检索、二阶段重排、低置信度拒答、DeepSeek 生成、来源引用、多知识库管理和自动化评测，并提供带教材检索与安全符号计算工具的受控 Math Agent，适合作为 RAG/Agent 工程实践、课程项目和面试展示项目。

当前主线是 `FastAPI + React + FAISS + BM25 + BGE Reranker + DeepSeek`。

## 项目亮点

- 教材解析不是简单抽文本：支持 PyMuPDF 块级坐标、双栏阅读顺序、重复页眉页脚清理、扫描页 OCR 回退、原生 PDF 表格转 Markdown。
- 切分面向数学教材结构：按章、节、定义、定理、例题等结构切块，并保留页码、章节、标题和来源文件元数据。
- 检索链路完整：原问题走 BGE 向量召回，规则 Query Rewrite 扩展 BM25 词法召回，再经 RRF 排名融合和 BGE Reranker 精排。
- 面向数学文本做了检索标准化：公式、定理名、中文数学符号进入 `search_text`，回答仍使用教材原文。
- 有低置信度拒答机制：检索不足时减少无依据生成。
- 提供 RAG / Agent 双模式：RAG 走一次固定检索生成流水线；Agent 在有限步数内自行选择教材检索或受限 SymPy 计算，并在前端展示可折叠的工具执行摘要。
- Agent 不是无限自治循环：只开放 `search_textbook` 和 `calculate_math` 两个白名单工具，默认最多调用 4 次，跳过重复调用，并在工具额度耗尽后强制生成最终答案。
- 数学计算不直接执行模型生成的代码：表达式先经过 AST 白名单解析，再在可超时终止的独立子进程中运行 SymPy；纯计算答案还会校验操作、表达式、变量和边界参数确实对应用户问题，并直接展示 SymPy 的确定性结果。
- 教材型问题必须有达到阈值的检索片段；最终答案会再经过一次仅依据达标片段的 grounding 重写，并逐次校验 `[1]`、`[2]` 的编号、片段分数、引用句内容和逐句引用覆盖，失败则拒答。
- 支持多知识库：上传 PDF 后异步构建索引，可查看任务历史、失败重试、删除知识库。
- 工程可靠性比较完整：SQLite WAL、任务幂等、重启恢复、RAG 并发保护、请求体限制、结构化日志、`X-Request-ID`。
- CI 覆盖后端测试、前端构建、Docker Compose 健康检查和评测集结构校验。
- 评测不是只看关键词：维护 grounded 评测集，标注页码和章节，用于暴露切分与元数据问题。

## 当前基线

自动化测试覆盖检索、生成、API、并发可靠性、Agent 工具编排、安全表达式解析和前端构建；实际数量以当前 `pytest` 与 CI 输出为准。

默认索引：

```text
2223 vectors
Embedding: BAAI/bge-small-zh-v1.5
Reranker: BAAI/bge-reranker-base
Index: FAISS IndexFlatIP + normalized embeddings
```

100 题 grounded-dev 评测集：

| 指标 | Recall@1 | Recall@3 | Recall@5 | MRR |
|---|---:|---:|---:|---:|
| 关键词命中 | 88.42% | 98.95% | 100.00% | 0.9377 |
| 页码命中 | 55.79% | 82.11% | 93.68% | 0.6972 |
| 章节命中 | 64.21% | 81.05% | 82.11% | 0.7184 |

Query Rewrite 单变量消融（其余参数相同）：

| Variant | Keyword R@1/3/5 | Keyword MRR | Page R@1/3/5 | Page 未命中 |
|---|---:|---:|---:|---:|
| `full` | 88.42% / 98.95% / 100.00% | 0.9377 | 55.79% / 82.11% / 93.68% | 6 |
| `no_query_rewrite` | 89.47% / 98.95% / 100.00% | 0.9412 | 55.79% / 82.11% / 91.58% | 8 |

Query Rewrite 只作用于 BM25，不改写向量查询。它让 Page Recall@5 提升 2.10 个百分点，并把页码未命中从 8 个降到 6 个；代价是 Keyword Recall@1 下降 1.05 个百分点，但 Keyword Recall@5 仍为 100%。当前默认开启，优先提高引用依据的覆盖率。

上一轮检索组件消融（关闭 Query Rewrite，作为历史对照）：

| Variant | 说明 | Keyword R@1/3/5 | Keyword MRR | Page R@1/3/5 | Section R@1/3/5 |
|---|---|---:|---:|---:|---:|
| `full` | Embedding + BM25 + RRF + Reranker | 89.47% / 98.95% / 100.00% | 0.9412 | 55.79% / 82.11% / 91.58% | 65.26% / 81.05% / 82.11% |
| `no_rrf` | 去掉 RRF prior | 86.32% / 97.89% / 98.95% | 0.9219 | 58.95% / 82.11% / 89.47% | 65.26% / 80.00% / 82.11% |
| `no_bm25` | 去掉 BM25 候选 | 86.32% / 95.79% / 97.89% | 0.9153 | 57.89% / 81.05% / 90.53% | 63.16% / 78.95% / 80.00% |
| `no_reranker` | 去掉二阶段 Reranker | 82.11% / 92.63% / 95.79% | 0.8711 | 44.21% / 73.68% / 85.26% | 50.53% / 70.53% / 74.74% |
| `narrow_recall` | 一阶段候选深度降为 5 + 5 | 93.68% / 97.89% / 98.95% | 0.9588 | 55.79% / 81.05% / 88.42% | 65.26% / 76.84% / 77.89% |

组件消融结论：Reranker 对页码和章节定位贡献最大；BM25 和 RRF 对关键词 Top5 与结构化命中有正贡献；`narrow_recall` 虽然 Keyword R@1 更高，但 Top5、页码和章节均下降，不适合作为默认配置。

说明：

- 总题数 100，其中 95 题可回答、5 题越界拒答。
- 关键词 Recall@5 已达到 100%，说明候选片段基本能召回到答案依据。
- 页码和章节 Recall 仍有优化空间，当前报告保留 6 个页码未命中和 17 个章节未命中案例，后续继续用于修切分器和 metadata 继承。

历史 100 题严格关键词基线：

| 方法 | Recall@1 | Recall@3 | Recall@5 | MRR |
|---|---:|---:|---:|---:|
| BM25 + BGE + Reranker | 86.00% | 97.00% | 97.00% | 0.9117 |

## 快速启动

### 环境要求

- Docker 启动：Docker Desktop
- 本地开发：Python 3.12（最低 3.11）和 Node.js 22+
- 问答生成：DeepSeek API Key

### Docker Compose

复制配置：

```powershell
Copy-Item .env.example .env
```

编辑 `.env`：

```env
DEEPSEEK_API_KEY=your_deepseek_api_key_here
MATHRAG_LLM_MODEL=deepseek-v4-flash
MATHRAG_AGENT_ENABLED=true
```

首次运行建议预热模型：

```powershell
docker compose run --rm model-cache
```

启动：

```powershell
docker compose up -d --build
```

访问：

| 服务 | 地址 |
|---|---|
| React 前端 | http://127.0.0.1:5173 |
| FastAPI 文档 | http://127.0.0.1:8000/docs |
| Health | http://127.0.0.1:8000/health |
| Readiness | http://127.0.0.1:8000/api/readiness |

查看日志：

```powershell
docker compose ps
docker compose logs -f backend
```

停止：

```powershell
docker compose down
```

### Windows 本地开发

安装依赖：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup.ps1
```

填写 `.env` 后启动：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start-dev.ps1
```

端口冲突时：

```powershell
.\scripts\start-dev.ps1 -BackendPort 8001 -FrontendPort 5174
```

### 手动启动

后端：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
.venv\Scripts\python.exe -m uvicorn backend.main:app --reload --port 8000
```

前端：

```powershell
cd frontend
npm ci
npm run dev
```

旧版 Streamlit 调试入口仍保留：

```powershell
streamlit run app.py
```

项目主线以 FastAPI + React 为准。

## 使用流程

1. 打开前端，确认 readiness 状态。
2. 上传 PDF 教材。
3. 创建索引任务。
4. 等待任务状态变为 `success`。
5. 选择知识库，并在顶部选择 `RAG` 或 `Agent` 模式。
6. 用“回答策略”滑块在“更快”和“参考更充分”之间调整返回片段数量。
7. 提问并查看回答、引用片段、页码和相关性；Agent 模式还可以展开查看工具名称、状态和执行摘要。
8. 对答案点赞、点踩或提交文字反馈。

两种回答模式的定位：

- `RAG`：固定执行一次检索、置信度判断和生成，调用链更短，适合普通教材问答。
- `Agent`：模型在受控循环内选择教材检索或数学计算，适合“先查定义再计算”、符号求导、积分、极限等需要工具的任务。
- 两种模式都复用当前知识库、Reranker 阈值和后端并发槽位；Agent 模式不会获得文件系统、网络、Shell 或数据库工具权限。

可尝试的问题：

```text
什么是导数？
洛必达法则的适用条件是什么？
定积分的几何意义是什么？
泰勒公式有什么用？
```

知识库约定：

- `default` 是内置默认知识库，索引位于 `data/faiss_index/`。
- 用户上传的 PDF、索引和任务状态位于 `storage/`。
- 旧索引可以继续查询，但新增的表格 Markdown、公式标准化和解析诊断只会写入新索引；升级后建议从原 PDF 重建重要知识库。

## 系统架构

```text
PDF 上传
  -> 文件校验与持久化
  -> 后台索引任务
  -> PDF 块级版面提取 / OCR / 表格 Markdown
  -> StructuralSplitter 结构化切分
  -> BGE Embedding + FAISS
  -> BM25 关键词索引

用户问题 + mode
  ├─ RAG
  │    -> 数学检索标准化 -> 向量召回
  │    -> Query Rewrite 术语别名和检索意图扩展 -> BM25 召回
  │    -> 候选合并去重 -> RRF 排名融合 -> BGE Reranker 精排
  │    -> 低置信度判断 -> DeepSeek 基于教材上下文生成
  └─ Agent
       -> DeepSeek 选择白名单工具
       -> search_textbook：复用混合检索与 Reranker
       -> calculate_math：安全解析 -> 独立子进程 -> SymPy
       -> 工具结果回填，有限次数循环
       -> grounding 重写与逐句证据校验 -> 回答或拒答

React 展示答案、教材来源和反馈；Agent 回答额外提供可折叠的结构化执行摘要
```

### Agent 请求流程与安全边界

1. `POST /api/chat` 收到 `mode: agent` 后，在现有 RAG 并发槽位内创建本次 `MathAgent`，不会递归调用问答接口。
2. 第一次模型请求要求选择工具，后续请求允许继续调用工具或直接回答；总调用次数由 `MATHRAG_AGENT_MAX_TOOL_CALLS` 限制，默认 4，合法范围为 1 到 8。
3. `search_textbook` 的参数经过 Pydantic 校验，单次 `top_k` 最大为 5；片段按 `vector_id` 去重，一轮最多保留 8 个上下文。
4. `calculate_math` 只接受声明过的操作、变量、常量、运算符和数学函数。解析器拒绝属性访问、导入、文件调用、未知名称和额外参数，不使用 `eval`、`sympify` 或模型生成代码。
5. 符号计算在独立进程中运行，默认 4 秒超时；表达式最长 240 字符、AST 最多 80 个节点，常量、指数和输出长度也有限制。纯计算路径只有在操作、原表达式、变量、上下限或极限点与用户问题匹配时才接受工具结果，最终答案显示被计算的表达式。
6. 相同工具与相同参数的重复调用会跳过。达到调用上限后，系统只允许基于已有工具结果生成最终答案。
7. 定义、定理、条件、证明和教材解释类问题必须有达到 `MATHRAG_MIN_RERANK_SCORE` 的教材片段；纯计算问题可以只依赖成功的计算结果。
8. 有达标教材片段时，草稿会经过一次独立 grounding 重写；随后逐次检查每个 `[n]` 的范围、对应片段分数、引用句与片段的内容绑定，并拒绝夹带无引用事实句的回答。任一检查失败都返回证据不足，不把模型记忆当教材结论。

前端执行摘要只展示工具标签、工具名、状态和简短结果，不展示模型内部思维过程。Agent 仍是教材问答功能，不是通用自动化代理；它没有访问 Shell、任意文件、网络或数据库的能力。

后端可靠性层：

```text
类型化配置
  + SQLite WAL / busy timeout
  + 任务原子认领和重启恢复
  + RAG 并发槽位
  + LLM 超时与重试
  + 请求体限制和输入校验
  + JSON 日志和 request_id
```

## 配置

算法配置位于 `config/config.yaml`，包括模型、索引路径、召回数量、Reranker 批大小和 LLM 生成参数。

运行配置放在 `.env`：

| 变量 | 默认值 | 作用 |
|---|---:|---|
| `DEEPSEEK_API_KEY` | 无 | DeepSeek API Key |
| `DEEPSEEK_BASE_URL` | `https://api.deepseek.com/v1` | OpenAI 兼容 API 地址 |
| `MATHRAG_LLM_MODEL` | `deepseek-v4-flash` | DeepSeek 聊天与工具调用模型 |
| `MATHRAG_LLM_THINKING_ENABLED` | `false` | 是否启用 DeepSeek thinking；默认关闭以缩短普通问答与工具循环延迟 |
| `HF_ENDPOINT` | `https://huggingface.co` | HuggingFace 下载端点 |
| `MATHRAG_EMBEDDING_MODEL` | `BAAI/bge-small-zh-v1.5` | Embedding 模型 ID 或本地目录 |
| `MATHRAG_RERANKER_MODEL` | `BAAI/bge-reranker-base` | Reranker 模型 ID 或本地目录 |
| `MATHRAG_MIN_RERANK_SCORE` | `0.2` | 低置信度拒答阈值 |
| `MATHRAG_ENVIRONMENT` | `development` | 运行环境标识 |
| `MATHRAG_ALLOW_RUNTIME_API_KEY` | 开发环境为 `true` | 是否允许通过 API 临时修改 DeepSeek Key |
| `MATHRAG_LOG_LEVEL` | `INFO` | 后端日志级别 |
| `MATHRAG_LOG_JSON` | `true` | 是否输出 JSON 结构化日志 |
| `MATHRAG_CORS_ORIGINS` | 本地 5173 地址 | 允许的前端来源，逗号分隔 |
| `MATHRAG_MAX_UPLOAD_MB` | `50` | 单个 PDF 上传上限 |
| `MATHRAG_MAX_JSON_BODY_MB` | `1` | 普通 API 请求体上限 |
| `MATHRAG_PDF_OCR_ENABLED` | 本机 `false`，Docker `true` | 是否对扫描页执行 OCR 回退 |
| `MATHRAG_PDF_OCR_LANGUAGES` | `chi_sim+eng` | Tesseract OCR 语言组合 |
| `MATHRAG_PDF_TABLE_DETECTION_ENABLED` | 本机 `false`，Docker `true` | 是否识别原生 PDF 表格并转换为 Markdown |
| `MATHRAG_SQLITE_TIMEOUT_SECONDS` | `10` | SQLite 锁等待时间 |
| `MATHRAG_JOB_MAX_ATTEMPTS` | `3` | 索引任务最大尝试次数 |
| `MATHRAG_RAG_MAX_CONCURRENCY` | `2` | 同时进入模型流水线的请求数 |
| `MATHRAG_QUERY_REWRITE_ENABLED` | `true` | 是否对 BM25 查询启用规则改写 |
| `MATHRAG_AGENT_ENABLED` | `true` | 是否允许 `/api/chat` 使用 Agent 模式 |
| `MATHRAG_AGENT_MAX_TOOL_CALLS` | `4` | 单次 Agent 问答的工具调用上限，范围 1 到 8 |
| `MATHRAG_LLM_TIMEOUT_SECONDS` | `30` | 单次 LLM 请求超时 |
| `MATHRAG_LLM_MAX_RETRIES` | `2` | LLM SDK 最大重试次数 |

`POST /api/settings/deepseek-key` 只修改当前后端进程，服务重启后失效；生产环境默认禁用该接口。

## 后端 API

完整交互文档见 `/docs`。

| 方法 | 路径 | 作用 |
|---|---|---|
| `GET` | `/health` | 容器健康检查 |
| `GET` | `/api/readiness` | 索引、模型和 API Key 就绪检查 |
| `POST` | `/api/chat` | 基于指定知识库问答 |
| `POST` | `/api/documents/upload` | 上传 PDF |
| `POST` | `/api/index/build` | 创建或复用索引任务 |
| `GET` | `/api/jobs` | 最近任务历史 |
| `GET` | `/api/jobs/{job_id}` | 查询任务状态 |
| `POST` | `/api/jobs/{job_id}/retry` | 重试失败任务 |
| `GET` | `/api/knowledge-bases` | 知识库列表 |
| `DELETE` | `/api/knowledge-bases/{knowledge_base_id}` | 删除知识库和索引 |
| `GET` | `/api/eval/latest` | 获取最近评测结果 |
| `GET` | `/api/feedback/summary` | 查看用户反馈统计 |
| `GET` | `/api/feedback` | 分页查看用户反馈 |
| `POST` | `/api/feedback` | 保存回答反馈 |
| `GET` | `/api/settings` | 查询运行设置状态 |
| `POST` | `/api/settings/deepseek-key` | 开发环境临时更新 DeepSeek Key |

问答请求示例：

```json
{
  "question": "洛必达法则的适用条件是什么？",
  "top_k": 3,
  "knowledge_base_id": "default"
}
```

请求可以增加 `mode` 字段，取值为 `rag` 或 `agent`；省略时保持向后兼容并使用 `rag`。例如，教材解释配合符号求导的任务可以把 `mode` 设置为 `agent`。

问答响应通过 `mode` 标记实际模式；Agent 响应额外包含 `agent_steps`，每一步提供 `tool`、`label`、`status`、`input` 和 `summary`。这些字段是结构化执行记录，不包含模型的内部思维过程。普通 RAG 响应的 `agent_steps` 为空数组。

所有响应都包含 `X-Request-ID`。错误响应保留前端兼容的 `detail`，并提供稳定错误编码：

```json
{
  "detail": "问答服务当前繁忙，请稍后重试",
  "error": {
    "code": "service_unavailable",
    "request_id": "a1b2c3d4"
  }
}
```

## 检索评测

运行默认评测：

```powershell
python evaluate_retrieval.py
```

运行 100 题 grounded-dev：

```powershell
python evaluate_retrieval.py `
  --eval-path data/eval/questions.grounded.dev.jsonl `
  --index-dir data/faiss_index `
  --top-k 5 `
  --top-k-embedding 20 `
  --top-k-bm25 20 `
  --rerank-batch-size 64 `
  --rrf-k 60 `
  --rrf-weight 1.0 `
  --output-json reports/retrieval_metrics_grounded_dev.json
```

运行检索消融：

```powershell
python evaluate_retrieval_ablation.py `
  --eval-path data/eval/questions.grounded.dev.jsonl `
  --index-dir data/faiss_index `
  --top-k 5 `
  --top-k-embedding 20 `
  --top-k-bm25 20 `
  --rerank-batch-size 64 `
  --rrf-k 60 `
  --rrf-weight 1.0 `
  --output-json reports/retrieval_ablation_grounded_dev.json
```

只对比 Query Rewrite 开关：

```powershell
python evaluate_retrieval_ablation.py `
  --eval-path data/eval/questions.grounded.dev.jsonl `
  --index-dir data/faiss_index `
  --variants full,no_query_rewrite `
  --output-json reports/retrieval_ablation_query_rewrite.json
```

也可以在单次评测命令中追加 `--no-query-rewrite`，关闭规则改写。

提交或扩展评测集前先跑结构校验：

```powershell
python validate_eval_dataset.py --eval-path eval/questions.sample.jsonl --profile grounded-smoke
python validate_eval_dataset.py --eval-path data/eval/questions.jsonl --profile keyword-100
python validate_eval_dataset.py --eval-path data/eval/questions.grounded.dev.jsonl --profile grounded-locked
```

评测集分层：

- `grounded-smoke`：5 题快速验收。
- `grounded-dev`：日常调参使用，当前已扩展到 100 题。
- `grounded-locked`：至少 100 题，用于版本发布对比。

注意：grounded 题目的页码和章节必须人工核验，不能直接把检索结果当真值。

## 测试与验收

后端测试：

```powershell
.venv\Scripts\python.exe -m pytest -q
```

只运行 Agent 专项测试：

```powershell
.venv\Scripts\python.exe -m pytest -q tests/test_math_agent.py tests/test_llm_generator.py tests/test_backend_reliability.py
```

Agent 测试覆盖恶意表达式拒绝、基础微积分与独立计算进程、教材检索、`top_k` 限制、重复调用跳过、确定性纯计算回答、错误操作或算式拒绝、概念题计算绕过、低分或无关引用、弱内容重合、重复引用夹带、无引用事实句、运行异常脱敏，以及 DeepSeek 工具消息中的 `reasoning_content` 回放。模型调用使用测试替身，不要求在 CI 中配置真实 API Key。

前端生产构建：

```powershell
cd frontend
npm run build
```

Compose 配置检查：

```powershell
docker compose config --quiet
```

容器验收：

```powershell
docker compose up -d --build
docker compose ps
curl.exe http://127.0.0.1:8000/health
curl.exe http://127.0.0.1:8000/api/readiness
```

GitHub Actions 会自动完成：

- Ubuntu Python 3.11 / 3.12 后端测试
- Windows Python 3.12 后端测试
- 运行时生成 PDF 的真实文件 I/O 集成测试
- Python 依赖一致性检查
- 前端生产构建
- Docker Compose 启动和健康检查
- 评测集结构校验
- Agent 工具编排、安全计算和 API 模式回归测试

## 项目结构

```text
MathRAG/
├── backend/                 # FastAPI 路由、Schema、服务和运行时核心
├── frontend/                # React + Vite 管理台
├── src/
│   ├── agent/               # 受控 Agent 状态机与安全 SymPy 工具
│   ├── loader/              # PDF 文本提取
│   ├── splitter/            # 结构化切分
│   ├── retriever/           # FAISS、BM25 和 Reranker
│   ├── generation/          # DeepSeek 生成器
│   └── pipeline/            # 完整问答流水线
├── config/config.yaml       # 算法与检索配置
├── scripts/                 # 安装、启动和模型预热脚本
├── eval/                    # 可提交的评测样例
├── data/eval/               # 本地 grounded 评测集
├── reports/                 # 检索评测报告
├── tests/                   # 单元、API、并发和可靠性测试
├── Dockerfile
├── compose.yaml
├── evaluate_retrieval.py
├── evaluate_retrieval_ablation.py
└── validate_eval_dataset.py
```

## 数据与持久化

| 路径 | 内容 |
|---|---|
| `data/faiss_index/` | 默认知识库 FAISS 索引和元数据 |
| `data/chunks/` | 默认教材切分结果 |
| `data/feedback/mathrag.db` | 用户反馈 SQLite 数据库 |
| `storage/documents/` | 上传的 PDF |
| `storage/indexes/` | 多知识库索引 |
| `storage/mathrag_backend.db` | 文档、知识库和任务状态 |

每个新知识库的 `processed/` 目录会保存 `pages.jsonl`、`full_text.txt` 和 `extraction_summary.json`。运行数据默认不提交到 Git。

## 已知限制

- OCR 效果依赖扫描清晰度和 Tesseract 语言数据。
- 无边框表格、跨页表格和复杂合并单元格仍可能需要专用版面模型。
- 复杂公式目前进行符号标准化和候选识别，尚未恢复完整 LaTeX 结构。
- 页码和章节 metadata 在复杂排版下仍会误继承或丢失；当前 100 题报告已经暴露了这类问题。
- 当前是本地单用户模型，没有账号、权限隔离和租户数据边界。
- 后台任务使用 FastAPI `BackgroundTasks`，适合单实例、本地使用和项目演示；多实例生产部署需要迁移到 Celery/RQ + Redis 等外部队列。
- 答案忠实度和引用正确性仍需单独评测。
- Agent 会增加一次或多次 LLM 请求，因此通常比固定 RAG 模式更慢、消耗更多 API 配额；工具上限控制了单次请求的最坏循环次数，但不代表结果一定正确。
- 安全 SymPy 工具只支持已声明的变量、函数与六类操作，不是完整计算机代数输入接口；超出白名单或超过复杂度、时间和输出限制的表达式会被拒绝。

## 路线图

- [x] FastAPI + React 前后端分离
- [x] 多知识库、任务历史、失败恢复和重试
- [x] BM25 + 向量混合检索、RRF 融合和 Reranker
- [x] 规则 Query Rewrite，扩展数学术语别名和常见问法
- [x] PDF 块级解析、页眉页脚清理和扫描页 OCR 回退
- [x] 数学公式检索标准化和 PDF 表格 Markdown 保留
- [x] Docker Compose、Windows 脚本和 GitHub Actions
- [x] 类型化配置、结构化日志和 request ID
- [x] SQLite WAL、任务幂等、并发保护和输入限制
- [x] 100 题 grounded 评测集和 Recall@K / MRR 报告
- [x] RAG / Agent 双模式、受控工具循环和前端结构化执行摘要
- [x] AST 白名单解析、独立进程超时的 SymPy 数学工具
- [x] Agent 教材证据约束、引用校验和失败拒答
- [ ] 修复页眉导致的章节 metadata 丢失和误继承
- [ ] 父子块上下文扩展
- [ ] 引用一致性、答案忠实度和低置信度评测
- [ ] Agent 答案正确性、工具选择和端到端成本评测
- [ ] Redis + Celery/RQ 生产任务队列
- [ ] 用户认证、权限隔离和在线演示部署
