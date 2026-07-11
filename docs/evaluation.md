# Retrieval Evaluation

本项目的第二阶段目标是让检索效果可以被量化，而不是只依赖主观体验。

## 评测目标

评测脚本衡量的是：用户问题进入检索器后，正确教材片段是否出现在 Top-K 结果中。

当前指标：

- `Recall@1`：正确片段是否出现在第 1 个结果中
- `Recall@3`：正确片段是否出现在前 3 个结果中
- `Recall@5`：正确片段是否出现在前 5 个结果中
- `MRR`：第一个命中结果的倒数排名均值

评测器分别计算三类命中信号：

- 关键词：输出在顶层 `recall_at_*`，兼容已有报告和前端
- 页码：输出在 `page_metrics`，判断检索片段页码与标注范围是否重叠
- 章节：输出在 `section_metrics`，匹配结果的章、节或标题元数据

## 数据格式

评测文件使用 JSONL，每行一个问题：

```json
{"id": "q001", "question": "什么是导数？", "expected_keywords": ["导数", "极限", "变化率"], "expected_chunk_keywords": ["导数", "变化率", "切线斜率"], "expected_page_ranges": [109], "expected_sections": ["二、导数的定义"], "type": "definition", "difficulty": "easy"}
```

字段说明：

- `id`：问题编号，建议使用 `q001`、`q002` 这类稳定编号
- `question`：用户问题
- `expected_keywords`：答案中期望出现的关键词，用于后续生成质量评估
- `expected_chunk_keywords`：正确检索片段中应出现的关键词，用于当前检索评测
- `expected_page_ranges`：可选，正确 PDF 物理页码；支持单页 `109` 或范围 `[164, 165]`
- `expected_sections`：可选，正确的章、节或标题；可以提供多个教材原文别名
- `type`：问题类型，例如 `definition`、`theorem`、`formula`、`method`
- `difficulty`：难度，例如 `easy`、`medium`、`hard`

当前脚本优先使用 `expected_chunk_keywords` 判断关键词命中；如果缺少该字段，会退回使用 `expected_keywords`。页码和章节标注是可选字段，没有标注时不会计入对应指标的分母。

## 页码与章节标注规范

- 页码使用解析产物中的 1-based PDF 物理页码，不使用教材页脚印刷页码
- 连续内容使用范围，例如 `[[164, 165]]`；多个离散位置可以写成 `[109, [164, 165]]`
- 页码命中采用范围重叠，只要检索片段覆盖任一标注页即可
- 章节优先填写教材原文标题，例如 `二、导数的定义`
- OCR 存在异体字或标点差异时，可以提供多个可接受标题
- 页码和章节应从 `data/processed/pages.jsonl` 或原 PDF 人工核验，不能直接复制检索结果作为真值

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

当前 v0.4.1 默认索引的 100 题严格关键词评测结果：

| Method | Questions | Recall@1 | Recall@3 | Recall@5 | MRR |
|---|---:|---:|---:|---:|---:|
| BM25 + BGE + Reranker | 100 | 86.00% | 97.00% | 97.00% | 0.9117 |

对应 JSON 结果保存在：

```text
reports/retrieval_metrics_100_vector_only.json
reports/retrieval_metrics_100_hybrid.json
reports/retrieval_metrics_grounded_sample.json
```

其中 `retrieval_metrics_grounded_sample.json` 使用 5 条人工核验的页码和章节标注。历史 Recall@5 为：关键词 100%、页码 100%、章节 80%。当前切分器已让定义、定理和例题块继承章/节 metadata，并将结构上下文用于索引和重排；该修复只对新索引生效。仓库未包含原 PDF，因此历史报告保持不变，取得原教材后需要重建默认索引并重新运行 grounded 评测。

后端运行后可以通过 `GET /api/eval/latest?method=grounded_sample` 读取该结构化基线；原有 `hybrid` 和 `vector_only` 方法保持兼容。

后续可以加入对比实验：

- 只使用 embedding，不使用 reranker
- 调整 `top_k_embedding`
- 调整 chunk 切分策略
- 对比 BM25 + 向量混合检索开启/关闭后的指标差异
