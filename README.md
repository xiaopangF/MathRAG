  # MathRAG

  MathRAG 是一个面向高等数学教材的 RAG 问答系统。项目支持上传 PDF 教材，自动进行文本提取、结构化切分、向量索引构建，并基
  于检索到的教材片段生成答案。

  ## Features

  - PDF 教材文本提取与基础清洗
  - 基于章、节、定义、定理、例题等结构的知识块切分
  - 使用 BGE Embedding 进行向量召回
  - 使用 FAISS 构建本地向量索引
  - 使用 BGE Reranker 进行二阶段重排
  - 调用 DeepSeek API 基于教材上下文生成答案
  - 提供 Streamlit Web 界面，支持上传教材、构建知识库和对话问答
  - 提供检索评测脚本，支持 Recall@K 和 MRR 指标

  ## Architecture

  `	ext
  PDF 教材
    ↓
  PDFLoader 文本提取
    ↓
  StructuralSplitter 结构化切分
    ↓
  BGE Embedding 向量化
    ↓
  FAISS 向量索引
    ↓
  Top-K 召回
    ↓
  BGE Reranker 重排
    ↓
  DeepSeek 生成答案
    ↓
  Streamlit 展示答案和参考片段

  ## Project Structure

  MathRAG/
  ├── app.py                         # Streamlit Web 应用
  ├── evaluate_retrieval.py          # 检索评测脚本
  ├── requirements.txt               # Python 依赖
  ├── config/
  │   └── config.yaml                # 默认配置
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
      └── test_pdf_loader.py

  ## Quick Start

  ### 1. 创建虚拟环境

  python -m venv .venv
  .venv\Scripts\activate

  ### 2. 安装依赖

  pip install -r requirements.txt

  ### 3. 配置 API Key

  复制 .env.example 为 .env，并填写 DeepSeek API Key：

  DEEPSEEK_API_KEY=your_deepseek_api_key_here
  DEEPSEEK_BASE_URL=https://api.deepseek.com/v1

  ### 4. 启动应用

  streamlit run app.py

  启动后，在页面左侧上传高等数学 PDF 教材，并点击“构建知识库”。

  ## Usage Examples

  可以尝试提问：

  什么是导数？
  洛必达法则的适用条件是什么？
  定积分的几何意义是什么？
  泰勒公式有什么用？

  系统会基于教材中检索到的相关片段生成回答。

  ## Data And Generated Files

  项目运行后会在 data/ 下生成文本切分结果和 FAISS 索引，例如：

  data/
  ├── chunks/
  │   ├── parents/
  │   ├── children/
  │   └── metadata.jsonl
  └── faiss_index/
      ├── index.faiss
      ├── chunks_meta.jsonl
      └── index_config.json

  这些文件由本地教材生成，默认不会提交到 Git。

  ## Evaluation

  项目提供检索评测脚本：

  python evaluate_retrieval.py

  评测数据建议放在：

  data/eval/questions.jsonl

  示例格式：

  {"id": "q001", "question": "什么是导数？", "expected_chunk_keywords": ["变化率", "切线斜率"]}
  {"id": "q002", "question": "洛必达法则的适用条件是什么？", "expected_chunk_keywords": ["0/0", "无穷/无穷", "可导"]}

  当前评测指标包括：

  - Recall@1
  - Recall@3
  - Recall@5
  - MRR

  ## Configuration

  config/config.yaml

  当前包含模型、索引路径、检索参数和 LLM 参数。后续可以将代码中的硬编码参数逐步迁移到该配置文件。

  ## Roadmap

  - [ ] 增加标准高数问答评测集
  - [ ] 答案增加文档名、章节和页码引用
  - [ ] 支持 BM25 + 向量混合检索
  - [ ] 优化数学公式提取与 LaTeX 渲染
  - [ ] 支持多教材知识库管理
  - [ ] 增加 Docker 部署
  - [ ] 增加 GitHub Actions 自动测试

  ## Limitations

  当前版本仍处于原型阶段，主要限制包括：

  - PDF 公式和复杂排版的解析效果依赖 PyMuPDF 的提取质量
  - 检索效果依赖教材切分质量和 embedding 模型
  - 暂未实现页码级精确引用
  - 暂未支持多教材管理
  - 答案生成依赖外部大模型 API
