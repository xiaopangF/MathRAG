# MathRAG

MathRAG 是一个面向高等数学教材的 RAG 问答系统。项目支持上传 PDF 教材，自动进行文本提取、结构化切分、向量索引构建，并基于检索到的教材片段生成答案。

## Features

- PDF 教材文本提取与基础清洗
- 基于章、节、定义、定理、例题等结构的知识块切分
- 使用 BGE Embedding 进行向量召回
- 使用 BM25 进行关键词召回，补强公式名、定理名等精确匹配问题
- 使用 FAISS 构建本地向量索引
- 合并 BM25 与向量召回结果，并使用 BGE Reranker 进行二阶段重排
- 调用 DeepSeek API 基于教材上下文生成答案
- 提供 FastAPI 后端接口，支持前后端分离
- 提供 React 前端管理台，支持多知识库选择、任务历史和删除知识库
- 支持 Markdown/LaTeX 公式渲染
- 支持用户对回答进行点赞/点踩和文字反馈
- 保留 Streamlit 旧版入口，便于快速本地调试
- 提供检索评测脚本，支持 Recall@K 和 MRR 指标

## Architecture

```text
PDF 教材
  ↓
PDFLoader 文本提取
  ↓
StructuralSplitter 结构化切分
  ↓
BGE Embedding 向量化 + BM25 关键词召回
  ↓
FAISS 向量索引 + 候选合并去重
  ↓
Top-K 混合召回
  ↓
BGE Reranker 重排
  ↓
DeepSeek 生成答案
  ↓
FastAPI API 层
  ↓
React 前端展示答案、参考片段和用户反馈
```

## Project Structure

```text
MathRAG/
├── app.py                         # Streamlit 旧版本地调试入口
├── evaluate_retrieval.py          # 检索评测脚本
├── requirements.txt               # Python 依赖
├── backend/                       # FastAPI 后端
│   ├── main.py                    # API 入口
│   ├── api/                       # 路由
│   ├── schemas/                   # 请求/响应模型
│   └── services/                  # RAG、反馈、评测、知识库服务
├── frontend/                      # React + Vite 前端
│   ├── package.json
│   └── src/
├── config/
│   └── config.yaml                # 默认配置
├── eval/
│   └── questions.sample.jsonl     # 可提交的评测样例
├── docs/
│   └── evaluation.md              # 评测说明
├── src/
│   ├── loader/
│   │   └── pdf_loader.py          # PDF 文本提取
│   ├── splitter/
│   │   └── structural_splitter.py # 文档结构切分
│   ├── retriever/
│   │   ├── vector_indexer.py      # 向量索引构建
│   │   └── retriever.py           # 双阶段检索器
│   ├── generation/
│   │   └── llm_generator.py       # LLM 答案生成
│   └── pipeline/
│       └── qa_pipeline.py         # 问答流水线
└── tests/
    └── test_*.py                  # 单元测试和后端接口测试
```

## Quick Start

### 1. 创建虚拟环境

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### 2. 安装依赖

```powershell
pip install -r requirements.txt
```

### 3. 配置 API Key

复制 `.env.example` 为 `.env`，并填写 DeepSeek API Key：

```env
DEEPSEEK_API_KEY=your_deepseek_api_key_here
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
HF_ENDPOINT=https://hf-mirror.com
MATHRAG_EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5
MATHRAG_RERANKER_MODEL=BAAI/bge-reranker-base
MATHRAG_MIN_RERANK_SCORE=0.2
```

如果当前网络无法访问 HuggingFace 或镜像站，可以先把模型下载到本地，然后把模型变量改成本地路径：

```env
MATHRAG_EMBEDDING_MODEL=C:/models/bge-small-zh-v1.5
MATHRAG_RERANKER_MODEL=C:/models/bge-reranker-base
```

### 4. 启动前后端分离版本

启动后端 API：

```powershell
uvicorn backend.main:app --reload --port 8000
```

启动前端：

```powershell
cd frontend
npm install
npm run dev
```

前端地址：

```text
http://127.0.0.1:5173
```

这些地址是本地开发地址，不是线上部署地址。当前项目尚未部署到公网服务器；如果要给别人远程访问，需要使用云服务器、内网穿透或其他部署方式。

### 5. 可选：启动 Streamlit 旧版入口

Streamlit 版本仍保留在 `app.py`，适合快速本地调试旧流程。当前主线功能以 FastAPI + React 为准。

```powershell
streamlit run app.py
```

## Usage Examples

可以尝试提问：

```text
什么是导数？
洛必达法则的适用条件是什么？
定积分的几何意义是什么？
泰勒公式有什么用？
```

系统会基于教材中检索到的相关片段生成回答。

回答正文会尽量使用 `[1]`、`[2]` 这样的编号标注依据来源。页面中的“查看检索到的相关片段”会展示对应片段的来源文件、页码范围、章节、标题、类型和相关性分数。

## Data And Generated Files

项目运行后会在 `data/` 下生成文本切分结果和 FAISS 索引，例如：

```text
data/
├── chunks/
│   ├── parents/
│   ├── children/
│   └── metadata.jsonl
└── faiss_index/
    ├── index.faiss
    ├── chunks_meta.jsonl
    └── index_config.json
```

这些文件由本地教材生成，默认不会提交到 Git。

重建知识库后，`metadata.jsonl` 和 `chunks_meta.jsonl` 会尽量保留引用元数据：

```json
{
  "source_file": "高等数学.pdf",
  "page_start": 47,
  "page_end": 48,
  "title": "导数的定义"
}
```

如果使用旧索引，历史 chunk 可能没有页码字段；需要重新上传 PDF 并重建知识库后，新的问答结果才会显示页码引用。

## Evaluation

项目提供检索评测脚本，用于衡量检索模块是否能把正确知识片段召回到 Top-K 结果中。

默认评测命令：

```powershell
python evaluate_retrieval.py
```

带参数的评测命令：

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

评测数据建议放在 `data/eval/questions.jsonl`。仓库中提供了可提交的样例文件 `eval/questions.sample.jsonl`，可以复制后扩展：

```powershell
New-Item -ItemType Directory -Force data/eval
Copy-Item eval/questions.sample.jsonl data/eval/questions.jsonl
```

每行是一条 JSON：

```json
{"id": "q001", "question": "什么是导数？", "expected_chunk_keywords": ["导数", "变化率", "切线斜率"], "type": "definition"}
```

当前评测指标包括：

- Recall@1
- Recall@3
- Recall@5
- MRR

更详细的评测集构建方法见 `docs/evaluation.md`。

### Current Results

当前本地评测集包含 100 条高等数学问题，覆盖函数、极限与连续、导数与微分、中值定理、泰勒公式、不定积分、定积分、反常积分和常微分方程等内容。

| Method | Questions | Recall@1 | Recall@3 | Recall@5 | MRR |
|---|---:|---:|---:|---:|---:|
| BGE Embedding + Reranker | 100 | 85.00% | 95.00% | 96.00% | 0.8987 |
| BM25 + BGE + Reranker | 100 | 88.00% | 97.00% | 99.00% | 0.9262 |

评测结果已保存到：

```text
reports/retrieval_metrics_100_vector_only.json
reports/retrieval_metrics_100_hybrid.json
```

当前默认检索流程是混合召回：

```text
BGE 向量召回 Top-K + BM25 关键词召回 Top-K
  → 按 vector_id 合并去重
  → CrossEncoder reranker 精排
  → 返回最终 Top-K
```

可以在 `config/config.yaml` 中调整：

```yaml
retrieval:
  use_hybrid_search: true
  top_k_embedding: 20
  top_k_bm25: 20
  top_k_rerank: 3
  rerank_batch_size: 64
```

问答阶段还会做低置信度保护。默认情况下，如果最高 reranker 分数低于 `MATHRAG_MIN_RERANK_SCORE=0.2`，系统会直接返回“未找到足够可靠的依据”，而不会调用大模型硬生成答案。这个阈值可以在 `.env` 中调整。

## Configuration

默认配置位于：

```text
config/config.yaml
```

当前包含模型、索引路径、检索参数和 LLM 参数。后续可以将代码中的硬编码参数逐步迁移到该配置文件。

## Backend API

项目已提供 FastAPI 后端，便于后续前后端分离。默认知识库仍使用 `data/faiss_index`，同时已经支持上传新 PDF、后台构建独立知识库，并在问答时通过 `knowledge_base_id` 指定知识库。

安装依赖后启动：

```powershell
pip install -r requirements.txt
uvicorn backend.main:app --reload --port 8000
```

当前接口：

```text
GET  /health
POST /api/chat
POST /api/documents/upload
POST /api/index/build
GET  /api/jobs
GET  /api/jobs/{job_id}
GET  /api/knowledge-bases
DELETE /api/knowledge-bases/{knowledge_base_id}
GET  /api/eval/latest?method=hybrid
GET  /api/eval/latest?method=vector_only
POST /api/feedback
GET  /api/settings
POST /api/settings/deepseek-key
```

多知识库构建流程：

```text
1. POST /api/documents/upload 上传 PDF，得到 document_id
2. POST /api/index/build 传入 document_id，得到 job_id 和 knowledge_base_id
3. GET /api/jobs/{job_id} 查询构建进度，status=success 表示完成
4. POST /api/chat 时传入 knowledge_base_id，基于该知识库问答
```

知识库管理：

```text
GET /api/jobs 查看最近构建任务历史
DELETE /api/knowledge-bases/{knowledge_base_id} 删除知识库记录和对应索引目录
```

问答请求示例：

```json
{
  "question": "洛必达法则的适用条件是什么？",
  "top_k": 3,
  "knowledge_base_id": "default"
}
```

用户点赞/点踩和文字反馈会保存到本地 SQLite：

```text
data/feedback/mathrag.db
```

上传文档、多知识库索引和后端任务数据库会保存在：

```text
storage/
```

这些文件属于本地运行数据，不提交到 Git。当前后台索引构建使用 FastAPI `BackgroundTasks`，适合本地演示；如果要部署给多人并发使用，建议升级为 Celery/RQ + Redis 队列。

## Frontend

项目已加入独立 React 前端，默认连接本地后端 `http://127.0.0.1:8000`。这是本地开发地址，当前 README 不提供线上访问地址。

启动后端：

```powershell
uvicorn backend.main:app --reload --port 8000
```

启动前端：

```powershell
cd frontend
npm install
npm run dev
```

前端地址：

```text
http://127.0.0.1:5173
```

如果部署到服务器或使用内网穿透，需要通过 `VITE_API_BASE` 或反向代理配置前端 API 地址。

当前前端支持知识库选择、PDF 上传、索引构建进度、任务历史、删除知识库、教材问答、Markdown/LaTeX 公式渲染、检索片段查看、点赞/点踩和文字反馈。

如果页面提示缺少 `DEEPSEEK_API_KEY`，推荐复制 `.env.example` 为 `.env`，并写入自己的 Key：

```env
DEEPSEEK_API_KEY=your_deepseek_api_key_here
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
```

后端也保留了 `POST /api/settings/deepseek-key`，可用于本地临时设置当前进程的 Key；重启后端后会失效。

## Roadmap

- [x] 增加 100 题高数检索评测集
- [x] 答案和检索片段保留文档名与页码范围引用
- [x] 支持 BM25 + 向量混合检索
- [x] 增加 FastAPI 后端 API 骨架
- [x] 支持多知识库隔离与后台索引构建
- [x] 增加 React 前端管理台
- [x] 增加任务历史与删除知识库
- [x] 增加用户文字反馈
- [x] 优化 Markdown/LaTeX 公式渲染
- [ ] 支持生产级异步队列、并发限流与多用户权限
- [ ] 增加 Docker 部署
- [ ] 增加 GitHub Actions 自动测试

## Limitations

当前版本仍处于原型阶段，主要限制包括：

- PDF 公式和复杂排版的解析效果依赖 PyMuPDF 的提取质量
- 检索效果依赖教材切分质量和 embedding 模型
- 页码引用依赖 PDF 文本抽取和 `[PAGE]` 标记，复杂排版下可能只能定位到页码范围
- 当前多知识库管理仍是本地单用户版本，尚未加入用户账号、权限隔离和并发队列
- 答案生成依赖外部大模型 API
