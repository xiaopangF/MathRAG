# MathRAG

[![CI](https://github.com/xiaopangF/MathRAG/actions/workflows/ci.yml/badge.svg)](https://github.com/xiaopangF/MathRAG/actions/workflows/ci.yml)

MathRAG 是一个面向高等数学教材的本地优先 RAG 问答系统。项目覆盖 PDF 教材解析、结构化切分、混合检索、二阶段重排、低置信度拒答、DeepSeek 生成、来源引用、多知识库管理和自动化评测，适合作为 RAG 工程实践、课程项目和面试展示项目。

当前主线是 `FastAPI + React + FAISS + BM25 + BGE Reranker + DeepSeek`。

## 项目亮点

- 教材解析不是简单抽文本：支持 PyMuPDF 块级坐标、双栏阅读顺序、重复页眉页脚清理、扫描页 OCR 回退、原生 PDF 表格转 Markdown。
- 切分面向数学教材结构：按章、节、定义、定理、例题等结构切块，并保留页码、章节、标题和来源文件元数据。
- 检索链路完整：BGE 向量召回 + BM25 词法召回 + RRF 排名融合 + BGE Reranker 精排。
- 面向数学文本做了检索标准化：公式、定理名、中文数学符号进入 `search_text`，回答仍使用教材原文。
- 有低置信度拒答机制：检索不足时减少无依据生成。
- 支持多知识库：上传 PDF 后异步构建索引，可查看任务历史、失败重试、删除知识库。
- 工程可靠性比较完整：SQLite WAL、任务幂等、重启恢复、RAG 并发保护、请求体限制、结构化日志、`X-Request-ID`。
- CI 覆盖后端测试、前端构建、Docker Compose 健康检查和评测集结构校验。
- 评测不是只看关键词：维护 grounded 评测集，标注页码和章节，用于暴露切分与元数据问题。

## 当前基线

自动化测试：

```text
128 passed
```

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
| 关键词命中 | 89.47% | 98.95% | 100.00% | 0.9412 |
| 页码命中 | 55.79% | 82.11% | 91.58% | 0.6907 |
| 章节命中 | 65.26% | 81.05% | 82.11% | 0.7214 |

说明：

- 总题数 100，其中 95 题可回答、5 题越界拒答。
- 关键词 Recall@5 已达到 100%，说明候选片段基本能召回到答案依据。
- 页码和章节 Recall 仍有优化空间，当前报告保留 8 个页码未命中和 17 个章节未命中案例，后续继续用于修切分器和 metadata 继承。

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
5. 选择知识库并提问。
6. 查看回答、引用片段、页码和相关性。
7. 对答案点赞、点踩或提交文字反馈。

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

用户问题
  -> 向量召回 + BM25 召回
  -> 候选合并去重
  -> RRF 排名融合
  -> BGE Reranker 精排
  -> 低置信度判断
  -> DeepSeek 基于教材上下文生成
  -> React 展示答案、引用和反馈
```

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
├── data/eval/               # 本地 grounded 评测集
├── reports/                 # 检索评测报告
├── tests/                   # 单元、API、并发和可靠性测试
├── Dockerfile
├── compose.yaml
├── evaluate_retrieval.py
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

## 路线图

- [x] FastAPI + React 前后端分离
- [x] 多知识库、任务历史、失败恢复和重试
- [x] BM25 + 向量混合检索、RRF 融合和 Reranker
- [x] PDF 块级解析、页眉页脚清理和扫描页 OCR 回退
- [x] 数学公式检索标准化和 PDF 表格 Markdown 保留
- [x] Docker Compose、Windows 脚本和 GitHub Actions
- [x] 类型化配置、结构化日志和 request ID
- [x] SQLite WAL、任务幂等、并发保护和输入限制
- [x] 100 题 grounded 评测集和 Recall@K / MRR 报告
- [ ] 修复页眉导致的章节 metadata 丢失和误继承
- [ ] Query Rewrite 和父子块上下文扩展
- [ ] 引用一致性、答案忠实度和低置信度评测
- [ ] SymPy 数学工具调用和答案验证
- [ ] Redis + Celery/RQ 生产任务队列
- [ ] 用户认证、权限隔离和在线演示部署
