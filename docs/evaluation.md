# Retrieval Evaluation

本项目的第二阶段目标是让检索效果可以被量化，而不是只依赖主观体验。

## 评测目标

评测脚本衡量的是：用户问题进入检索器后，正确教材片段是否出现在 Top-K 结果中。

当前指标：

- `Recall@1`：正确片段是否出现在第 1 个结果中
- `Recall@3`：正确片段是否出现在前 3 个结果中
- `Recall@5`：正确片段是否出现在前 5 个结果中
- `MRR`：第一个命中结果的倒数排名均值

## 数据格式

评测文件使用 JSONL，每行一个问题：

```json
{"id": "q001", "question": "什么是导数？", "expected_keywords": ["导数", "极限", "变化率"], "expected_chunk_keywords": ["导数", "变化率", "切线斜率"], "type": "definition", "difficulty": "easy"}
```

字段说明：

- `id`：问题编号，建议使用 `q001`、`q002` 这类稳定编号
- `question`：用户问题
- `expected_keywords`：答案中期望出现的关键词，用于后续生成质量评估
- `expected_chunk_keywords`：正确检索片段中应出现的关键词，用于当前检索评测
- `type`：问题类型，例如 `definition`、`theorem`、`formula`、`method`
- `difficulty`：难度，例如 `easy`、`medium`、`hard`

当前脚本优先使用 `expected_chunk_keywords` 判断检索命中；如果缺少该字段，会退回使用 `expected_keywords`。

## 构建评测集

仓库提供样例：

```powershell
eval/questions.sample.jsonl
```

本地真实评测集建议放在：

```powershell
data/eval/questions.jsonl
```

初始化命令：

```powershell
New-Item -ItemType Directory -Force data/eval
Copy-Item eval/questions.sample.jsonl data/eval/questions.jsonl
```

`data/` 默认不提交到 Git，适合放本地教材、索引和完整评测集。如果要公开一部分评测题，可以补充到 `eval/questions.sample.jsonl`。

## 推荐题型比例

第一版建议整理 100 条问题：

- 概念定义题：25 条，例如“什么是导数？”
- 定理条件题：20 条，例如“罗尔定理的条件是什么？”
- 公式题：20 条，例如“牛顿-莱布尼茨公式是什么？”
- 方法题：20 条，例如“如何判断函数单调性？”
- 易错题：15 条，例如“什么时候不能使用洛必达法则？”

## 运行评测

默认运行：

```powershell
python evaluate_retrieval.py
```

指定参数：

```powershell
python evaluate_retrieval.py `
  --eval-path data/eval/questions.jsonl `
  --index-dir data/faiss_index `
  --top-k 5 `
  --top-k-embedding 20 `
  --output-json reports/retrieval_metrics.json
```

如果缺少索引，需要先在 Web 页面上传 PDF 并构建知识库，或直接调用索引构建脚本。

## 记录实验结果

建议在 README 或实验记录中维护下面的表格：

```markdown
| Method | Questions | Recall@1 | Recall@3 | Recall@5 | MRR |
|---|---:|---:|---:|---:|---:|
| BGE Embedding + Reranker | 100 | - | - | - | - |
```

后续可以加入对比实验：

- 只使用 embedding，不使用 reranker
- 调整 `top_k_embedding`
- 调整 chunk 切分策略
- 加入 BM25 + 向量混合检索
