# MathRAG

[![CI](https://github.com/xiaopangF/MathRAG/actions/workflows/ci.yml/badge.svg)](https://github.com/xiaopangF/MathRAG/actions/workflows/ci.yml)

MathRAG 是一个面向高等数学教材的本地优先 RAG 问答系统。它可以上传 PDF 教材，完成文本提取、结构化切分、混合检索、重排和低置信度判断，再调用 DeepSeek 根据教材片段生成带来源的回答。

当前版本已经具备 FastAPI 后端、React 管理台、多知识库、持久任务、结构化日志、请求追踪、Docker Compose 和自动化测试，适合作为 RAG 工程实践、课程项目和面试演示项目。

## 当前能力

- PDF 教材上传、格式校验和 50 MB 默认上传限制
- PDF 块级坐标解析、双栏阅读顺序和重复页眉页脚清理
- 页面解析质量诊断和扫描页按需 OCR 回退
- 数学公式标准化检索文本，保留教材原文用于回答和引用
- 原生 PDF 表格提取为 Markdown，避免单元格文本错序和重复
- 按章、节、定义、定理、例题等结构切分教材
- BGE Embedding + FAISS 向量召回
- BM25 关键词召回，补强公式名和定理名匹配
- BGE Reranker 二阶段重排
- 低置信度拒答，减少无依据生成
- DeepSeek 上下文问答和 `[1]`、`[2]` 来源编号
- 来源文件、页码范围、章节、标题和相关性展示
- 多知识库、构建任务历史、失败重试和知识库删除
- 点赞、点踩和文字反馈持久化
- JSON 结构化日志和 `X-Request-ID` 请求追踪
- SQLite WAL、任务恢复、幂等构建和 RAG 并发保护
- Docker Compose、Windows 开发脚本和 GitHub Actions

当前 v0.4.1 默认索引的 100 题严格关键词匹配基线：

| 方法 | Recall@1 | Recall@3 | Recall@5 | MRR |
|---|---:|---:|---:|---:|
| BM25 + BGE + Reranker | 86.00% | 97.00% | 97.00% | 0.9117 |

新版索引包含 3084 个向量、完整页码、表格 Markdown 和数学标准化检索文本。3 个严格关键词未命中案例均返回了语义相关片段，已保留在评测报告中供下一轮切块和评测器优化。

5 题人工结构标注样例的历史 Recall@5：关键词 `100%`、页码 `100%`、章节 `80%`。切分器现已向细分块传递章/节 metadata，但仓库不包含生成默认索引所需的原 PDF；需要用原教材重建索引并重跑 grounded 评测后，才能更新该历史指标。

当前自动化测试基线：`127 passed`。

## 快速启动

### 环境要求

- Docker 启动：Docker Desktop
- 本地开发：Python 3.12（最低 3.11）和 Node.js 22+
- 使用问答功能：DeepSeek API Key

### 方式一：Docker Compose

复制环境变量模板：

```powershell
Copy-Item .env.example .env
```

编辑 `.env`，至少填写：

```env
DEEPSEEK_API_KEY=your_deepseek_api_key_here
```

首次运行建议预热 Embedding 和 Reranker 模型：

```powershell
docker compose run --rm model-cache
```

构建并启动：

```powershell
docker compose up -d --build
```

启动后访问：

| 服务 | 地址 |
|---|---|
| React 前端 | http://127.0.0.1:5173 |
| FastAPI 文档 | http://127.0.0.1:8000/docs |
| Health | http://127.0.0.1:8000/health |
| Readiness | http://127.0.0.1:8000/api/readiness |

查看状态和日志：

```powershell
docker compose ps
docker compose logs -f backend
```

停止服务：

```powershell
docker compose down
```

HuggingFace 模型缓存在具名卷 `huggingface-cache` 中，重新创建容器不会重复下载。Windows 本地模型路径不能直接在 Linux 容器中使用；Docker 模式建议使用模型 ID，或显式挂载模型目录。

### 方式二：Windows 开发脚本

首次安装后端和前端依赖：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup.ps1
```

编辑生成的 `.env`，填写 DeepSeek API Key，然后启动开发环境：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start-dev.ps1
```

脚本会分别打开后端和前端日志窗口。端口被占用时可以指定其他端口：

```powershell
.\scripts\start-dev.ps1 -BackendPort 8001 -FrontendPort 5174
```

### 手动开发启动

安装 Python 依赖：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
```

安装前端依赖：

```powershell
cd frontend
npm ci
cd ..
```

分别启动两个终端：

```powershell
.venv\Scripts\python.exe -m uvicorn backend.main:app --reload --port 8000
```

```powershell
cd frontend
npm run dev
```

旧版 Streamlit 调试入口仍保留：

```powershell
streamlit run app.py
```

项目主线以 FastAPI + React 为准。

## 使用流程

1. 打开前端并检查顶部 readiness 状态。
2. 上传 PDF 教材，后端返回 `document_id`。
3. 创建索引任务，后端返回 `job_id` 和 `knowledge_base_id`。
4. 等待任务状态变为 `success`。
5. 选择知识库并提问。
6. 展开来源片段，查看教材文件、页码和相关性。
7. 对答案点赞、点踩或提交文字反馈。

可以尝试：

```text
什么是导数？
洛必达法则的适用条件是什么？
定积分的几何意义是什么？
泰勒公式有什么用？
```

同一文档存在 `pending` 或 `running` 任务时，重复构建会复用原任务，并返回 `reused: true`。失败任务可以通过前端或重试接口再次执行。

### 知识库约定

- `default` 是内置默认知识库，索引位于 `data/faiss_index/`，不出现在上传知识库的持久记录中，也不能通过删除接口移除。
- 用户上传的 PDF、索引和任务状态位于 `storage/`，可以在前端或 API 中单独删除，不会影响默认知识库。
- 旧索引可以继续查询，但本轮新增的表格 Markdown、公式 `search_text` 和完整解析诊断只会写入新索引。升级后应从原 PDF 重新构建需要更新的知识库。

## 系统架构

```text
PDF 上传
  -> 文件校验与持久化
  -> 后台索引任务
  -> PyMuPDF 块级版面提取与解析质量诊断
  -> StructuralSplitter 结构化切分
  -> BGE Embedding + FAISS
  -> BM25 关键词索引

用户问题
  -> 向量召回 + BM25 召回
  -> 候选合并去重
  -> BGE Reranker 精排
  -> 低置信度判断
  -> DeepSeek 基于教材上下文生成
  -> React 展示答案、引用和反馈
```

后端可靠性层负责：

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

算法和检索配置位于 `config/config.yaml`，包括模型、索引路径、召回数量、Reranker 批大小和 LLM 生成参数。

运行配置放在 `.env`：

| 变量 | 默认值 | 作用 |
|---|---:|---|
| `DEEPSEEK_API_KEY` | 无 | DeepSeek API Key |
| `DEEPSEEK_BASE_URL` | `https://api.deepseek.com/v1` | OpenAI 兼容 API 地址 |
| `HF_ENDPOINT` | `https://huggingface.co` | HuggingFace 下载端点 |
| `MATHRAG_EMBEDDING_MODEL` | `BAAI/bge-small-zh-v1.5` | Embedding 模型 ID 或本地目录 |
| `MATHRAG_RERANKER_MODEL` | `BAAI/bge-reranker-base` | Reranker 模型 ID 或本地目录 |
| `MATHRAG_MIN_RERANK_SCORE` | `0.2` | 低置信度拒答阈值 |
| `MATHRAG_ENVIRONMENT` | `development` | 运行环境标识 |
| `MATHRAG_ALLOW_RUNTIME_API_KEY` | 开发环境为 `true` | 是否允许通过 API 临时修改 DeepSeek Key |
| `MATHRAG_LOG_LEVEL` | `INFO` | 后端日志级别 |
| `MATHRAG_LOG_JSON` | `true` | 是否输出 JSON 结构化日志 |
| `MATHRAG_CORS_ORIGINS` | 本地 5173 地址 | 允许的前端来源，逗号分隔 |
| `MATHRAG_CORS_ORIGIN_REGEX` | 本地地址正则 | 可选的动态本地端口来源规则 |
| `MATHRAG_MAX_UPLOAD_MB` | `50` | 单个 PDF 上传上限 |
| `MATHRAG_MAX_JSON_BODY_MB` | `1` | 普通 API 请求体上限 |
| `MATHRAG_PDF_OCR_ENABLED` | 本机 `false`，Docker `true` | 是否对扫描页执行 OCR 回退 |
| `MATHRAG_PDF_OCR_LANGUAGES` | `chi_sim+eng` | Tesseract OCR 语言组合 |
| `MATHRAG_PDF_OCR_DPI` | `200` | OCR 渲染 DPI |
| `MATHRAG_PDF_OCR_MAX_PAGES` | `100` | 单个文档最多 OCR 页数 |
| `MATHRAG_PDF_TABLE_DETECTION_ENABLED` | 本机 `false`，Docker `true` | 是否识别原生 PDF 表格并转换为 Markdown |
| `MATHRAG_SQLITE_TIMEOUT_SECONDS` | `10` | SQLite 锁等待时间 |
| `MATHRAG_JOB_MAX_ATTEMPTS` | `3` | 索引任务最大尝试次数 |
| `MATHRAG_RAG_MAX_CONCURRENCY` | `2` | 同时进入模型流水线的请求数 |
| `MATHRAG_RAG_ACQUIRE_TIMEOUT_SECONDS` | `2` | 等待推理槽位的时间 |
| `MATHRAG_LLM_TIMEOUT_SECONDS` | `30` | 单次 LLM 请求超时 |
| `MATHRAG_LLM_MAX_RETRIES` | `2` | LLM SDK 最大重试次数 |

无效数值、布尔值或日志级别会在启动时直接报错。`POST /api/settings/deepseek-key` 只修改当前后端进程，服务重启后失效；持久配置应写入 `.env`。生产环境默认禁用该临时设置接口。

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
| `POST` | `/api/feedback` | 保存回答反馈 |
| `GET` | `/api/settings` | 查询运行设置状态 |
| `POST` | `/api/settings/deepseek-key` | 开发环境临时更新 DeepSeek Key |

问答请求：

```json
{
  "question": "洛必达法则的适用条件是什么？",
  "top_k": 3,
  "knowledge_base_id": "default"
}
```

### Readiness

`GET /api/readiness` 不加载模型、不访问网络，只检查本地状态：

- `ready`：依赖已经可用
- `download_required`：远程模型尚未缓存
- `missing`：索引、Key 或本地模型目录缺失

响应同时包含 `can_answer_default`、`can_build_index` 和具体阻塞原因。

### 错误响应与请求追踪

所有响应都包含 `X-Request-ID`。调用方传入合法 request ID 时后端会沿用，否则自动生成。

错误响应保留前端兼容的 `detail`，并提供稳定错误编码：

```json
{
  "detail": "问答服务当前繁忙，请稍后重试",
  "error": {
    "code": "service_unavailable",
    "request_id": "a1b2c3d4"
  }
}
```

常见错误编码：

| HTTP | code | 含义 |
|---:|---|---|
| 400 | `bad_request` | 请求参数或业务输入错误 |
| 404 | `not_found` | 任务或知识库不存在 |
| 409 | `conflict` | 状态冲突或达到重试上限 |
| 413 | `payload_too_large` | 请求体超过限制 |
| 422 | `validation_error` | Schema 校验失败 |
| 429 | `rate_limited` | 上游模型限流 |
| 500 | `internal_error` | 未捕获服务端错误 |
| 502 | `upstream_error` | LLM 上游错误 |
| 503 | `service_unavailable` | 网络错误或推理容量耗尽 |

500 响应不会返回内部路径、密钥或堆栈。完整异常只写入服务端日志，并通过 `request_id` 关联。

## 后端可靠性

- SQLite 使用 WAL 和 busy timeout，降低并发读写冲突
- SQLite 使用显式 Schema 版本迁移，并拒绝打开更高版本数据库
- 索引任务持久化 `status`、`progress`、`attempt_count`、起止时间和错误原因
- 任务通过条件更新原子认领，重复后台执行不会重复构建
- 服务重启会把未完成任务标记为失败，用户可以显式重试
- 删除知识库使用暂存目录；数据库失败会回滚，进程中断后启动恢复
- 上传入库失败会删除已写入文件，避免普通数据库故障产生孤儿文件
- 文档读取使用受控存储目录，并自动修复 Windows/Docker 之间的旧绝对路径
- 删除知识库后会失效内存中的 Retriever 缓存
- RAG 使用并发槽位保护，超时后快速返回 503
- DeepSeek 调用使用显式超时和重试，不覆盖容器环境变量
- 普通请求体、PDF、问题、反馈和上下文都有大小限制
- PDF 解析保留文本块坐标、阅读顺序和质量标记，并对扫描页按需 OCR
- 向量索引和 BM25 使用“标题 + 正文”的数学标准化 `search_text`，回答仍使用原始 `text`
- Reranker 使用标题补充结构语义，批量查询与单条查询执行相同的数学标准化
- Embedding 和 Reranker 优先读取本地缓存，缓存缺失时才访问 HuggingFace
- 表格文本以 Markdown 进入切块，并排除重复的散乱单元格文本

当前任务执行器仍是 FastAPI `BackgroundTasks`，适合单实例、本地使用和项目演示。多实例生产部署需要迁移到 Celery/RQ + Redis 等外部队列。

## 检索评测

默认运行：

```powershell
python evaluate_retrieval.py
```

指定评测集和输出：

```powershell
python evaluate_retrieval.py `
  --eval-path data/eval/questions.jsonl `
  --index-dir data/faiss_index `
  --top-k 5 `
  --top-k-embedding 20 `
  --top-k-bm25 20 `
  --rerank-batch-size 64 `
  --output-json reports/retrieval_metrics.json
```

创建本地评测集：

```powershell
New-Item -ItemType Directory -Force data/eval
Copy-Item eval/questions.sample.jsonl data/eval/questions.jsonl
```

每行一条 JSON：

```json
{"id":"q001","question":"什么是导数？","expected_chunk_keywords":["导数","变化率","切线斜率"],"type":"definition"}
```

当前指标包括 Recall@1、Recall@3、Recall@5 和 MRR。详细说明见 [docs/evaluation.md](docs/evaluation.md)，基线报告位于：

```text
reports/retrieval_metrics_100_vector_only.json
reports/retrieval_metrics_100_hybrid.json
reports/retrieval_metrics_grounded_sample.json
```

提交或扩展评测集前先运行结构校验：

```powershell
python validate_eval_dataset.py --eval-path eval/questions.sample.jsonl --profile grounded-smoke
python validate_eval_dataset.py --eval-path data/eval/questions.jsonl --profile keyword-100
```

评测集分三层维护：`grounded-smoke` 保留 5 题快速验收，`grounded-dev` 至少 30 题用于日常调参，`grounded-locked` 至少 100 题用于版本发布对比。`grounded-dev` 和 `grounded-locked` 的每题都必须有人核验页码和章节，不能直接把检索结果当作真值。

## 测试与验收

后端测试：

```powershell
.venv\Scripts\python.exe -m pytest -q
```

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

GitHub Actions 会自动完成以下检查：

- Ubuntu 上的 Python 3.11 和 3.12 后端测试
- Windows 上的 Python 3.12 后端测试
- 使用运行时生成 PDF 的真实文件 I/O 集成测试
- Python 依赖一致性检查和前端生产构建
- Pull Request 与 `main` 分支上的 Docker Compose 启动及健康检查

测试报告和前端 `dist` 构建结果会作为短期 Actions artifacts 保存，便于定位失败和检查构建产物。

## 项目结构

```text
MathRAG/
├── backend/                 # FastAPI 路由、Schema、服务和运行时核心
├── frontend/                # React + Vite 管理台
├── src/
│   ├── loader/              # PDF 文本提取
│   ├── splitter/            # 结构化切分
│   ├── retriever/           # FAISS、BM25 和 Reranker
│   ├── generation/          # DeepSeek 生成器
│   └── pipeline/            # 完整问答流水线
├── config/config.yaml       # 算法与检索配置
├── scripts/                 # 安装、启动和模型预热脚本
├── eval/                    # 可提交的评测样例
├── reports/                 # 检索评测报告
├── tests/                   # 单元、API、并发和可靠性测试
├── Dockerfile               # 后端镜像
├── compose.yaml             # 前后端与模型缓存编排
├── evaluate_retrieval.py    # 检索评测入口
└── app.py                   # Streamlit 旧版调试入口
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

每个新知识库的 `processed/` 目录会保存 `pages.jsonl`、`full_text.txt` 和 `extraction_summary.json`。其中解析摘要包含文本页数、空白页、版面类型、公式块和表格数量、表格识别错误页、被移除的页眉页脚数量，以及 OCR 建议、成功、失败和跳过的页码。

这些运行数据默认不提交到 Git。旧索引可能没有页码、表格或 `search_text` 字段；重新上传 PDF 并构建索引后才能获得完整的解析与检索元数据。

## 已知限制

- OCR 效果依赖扫描清晰度和 Tesseract 语言数据；本机开发需要单独安装对应语言包
- 无边框表格、跨页表格和复杂合并单元格仍可能需要专用版面模型
- 复杂公式目前进行符号标准化和候选识别，尚未恢复完整 LaTeX 结构
- 检索效果依赖教材结构、切分质量和评测集覆盖度
- 页码定位在复杂排版下可能只能精确到页码范围
- 当前是本地单用户模型，没有账号、权限隔离和租户数据边界
- DeepSeek Key 临时设置接口仅面向本地开发，生产环境默认关闭
- 后台任务尚未使用独立队列，不支持多实例任务协调
- 答案忠实度和引用正确性仍需单独评测

## 路线图

- [x] 100 题检索评测集和 Recall@K / MRR 基线
- [x] BM25 + 向量混合检索和 Reranker
- [x] FastAPI + React 前后端分离
- [x] 多知识库、任务历史、失败恢复和重试
- [x] Docker Compose、Windows 脚本和 GitHub Actions
- [x] 类型化配置、结构化日志和 request ID
- [x] SQLite WAL、任务幂等、并发保护和输入限制
- [x] PDF 块级解析、页眉页脚清理和扫描页 OCR 回退
- [x] 数学公式检索标准化和 PDF 表格 Markdown 保留
- [ ] Query Rewrite、RRF 融合和父子块上下文扩展
- [ ] 引用一致性、答案忠实度和低置信度评测
- [ ] SymPy 数学工具调用和答案验证
- [ ] Redis + Celery/RQ 生产任务队列
- [ ] 用户认证、权限隔离和在线演示部署
