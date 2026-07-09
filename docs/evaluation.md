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

## 关键词标注规范

`expected_chunk_keywords` 应该标注“正确教材片段中稳定出现的词”，不要只标注自己脑海里的标准说法。

建议：

- 优先选择教材原文中的短语，例如 `牛顿-莱布尼茨`、`切线斜率`
- 同一个知识点可以标多个候选关键词，例如 `["凹凸", "凸性", "下凸", "上凸"]`
- 对 OCR 容易识别错的公式、符号和标点，尽量补充中文文字关键词
- 不要使用过于宽泛的词，例如只写 `函数`、`公式`、`定理`
- 每个问题建议至少 2 个关键词，减少误判

评测脚本会忽略空白和常见中英文标点，因此 `牛顿-莱布尼茨` 可以匹配 OCR 中的 `牛顿" 莱布尼茨`。但它不会自动理解同义词，例如 `凹凸` 和 `凸性`，这类情况需要在评测集中都标出来。

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
  --top-k-bm25 20 `
  --output-json reports/retrieval_metrics.json
```

如果要做消融实验，可以关闭混合检索：

```powershell
python evaluate_retrieval.py --no-hybrid-search
```

如果缺少索引，需要先在 Web 页面上传 PDF 并构建知识库，或直接调用索引构建脚本。

## 记录实验结果

建议在 README 或实验记录中维护下面的表格：

```markdown
| Method | Questions | Recall@1 | Recall@3 | Recall@5 | MRR |
|---|---:|---:|---:|---:|---:|
| BGE Embedding + Reranker | 100 | - | - | - | - |
| BM25 + BGE + Reranker | 100 | - | - | - | - |
```

当前本地 100 题评测结果：

| Method | Questions | Recall@1 | Recall@3 | Recall@5 | MRR |
|---|---:|---:|---:|---:|---:|
| BGE Embedding + Reranker | 100 | 85.00% | 95.00% | 96.00% | 0.8987 |
| BM25 + BGE + Reranker | 100 | 88.00% | 97.00% | 99.00% | 0.9262 |

对应 JSON 结果保存在：

```text
reports/retrieval_metrics_100_vector_only.json
reports/retrieval_metrics_100_hybrid.json
```

后续可以加入对比实验：

- 只使用 embedding，不使用 reranker
- 调整 `top_k_embedding`
- 调整 chunk 切分策略
- 对比 BM25 + 向量混合检索开启/关闭后的指标差异
