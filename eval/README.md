# 评测集扩充模板

目标：当前 `eval/questions.jsonl` 有 22 条，继续扩到 **30–50 条**，让切分参数、混合检索、rerank 这些改动有可比较的基线。

> 注意：评测脚本只读取 `eval/questions.jsonl`。
> `eval/questions.template.jsonl` 只是示例，不会被自动加载，复制需要的行到 `questions.jsonl` 即可。

## 1. 每行一个 JSON 对象

```json
{"question": "问题", "type": "answerable", "category": "fact", "gold_sources": ["文件名.md"], "gold_answer": "参考答案", "gold_keywords": ["关键词1", "关键词2"], "notes": "为什么值得测"}
```

| 字段 | 必填 | 说明 |
|---|---|---|
| `question` | 是 | 用户会真实问的问题，避免“这道题考什么”这种元问题 |
| `type` | 是 | 只能是 `answerable` 或 `unanswerable`；脚本靠它区分可回答/知识库外 |
| `category` | 建议 | 题型：`fact` / `list` / `paraphrase` / `keyword` / `multi_hop` / `numeric` / `out_of_scope` / `false_premise` |
| `gold_sources` | 是 | 标准答案所在文件，必须和 `data/raw/` 里的文件名完全一致；知识库外问题填 `[]` |
| `gold_answer` | 建议 | 一句话参考答案；知识库外问题可以留空 |
| `gold_keywords` | 可回答题必填 | 2–5 个“正确答案里应该出现”的关键词，必须是原文中真实出现的词 |
| `notes` | 可选 | 出题理由、对应原文句子、容易错在哪里 |

## 2. 题型与写法

| category | 目的 | 示例 | 出题要点 |
|---|---|---|---|
| `fact` | 单点事实 | “LangGraph 的 Checkpointer 有什么作用？” | 答案能在一段原文里找到 |
| `list` | 列举/归纳 | “RAG 的检索优化常见方法有哪些？” | `gold_keywords` 覆盖所有要点 |
| `paraphrase` | 语义检索 | “为什么不能把整本书直接塞给大模型？” | 不要照抄原文措辞 |
| `keyword` | 精确词/参数 | “Chroma 查询时用哪个参数控制返回条数？” | 放一个必须命中的英文词或参数名，用来测试 BM25 |
| `multi_hop` | 跨段/跨文档综合 | “RAG 和 Agent 的工作流程有什么不同？” | `gold_sources` 列出所有涉及的文件 |
| `numeric` | 数字/日期 | “中文文档切分建议从多少字开始实验？” | 关键词里带数字和单位 |
| `out_of_scope` | 知识库外 | “今天杭州的天气怎么样？” | 问题自然、合理，但知识库确实没有 |
| `false_premise` | 错误前提 | “LangGraph 的 Checkpointer 默认存在哪个数据库？” | 文档提到相关概念，但没给这个细节；专门测试幻觉 |

## 3. 建议的覆盖矩阵（32 条）

| 类别 | 数量 |
|---|---:|
| `fact` | 10 |
| `list` | 4 |
| `paraphrase` | 5 |
| `keyword` | 4 |
| `multi_hop` | 3 |
| `numeric` | 2 |
| `out_of_scope` | 2 |
| `false_premise` | 2 |

另外保证：
- 每篇文档至少有 5 道题；
- 知识库外问题占 20%–25%；
- 不要出现两道问法几乎一样、答案也一样的题。

## 4. 出题质量检查清单

- [ ] 答案能在原文里找到，并在 `notes` 里记下大致出处（哪一段/哪句话）。
- [ ] `gold_sources` 的文件名和 `data/raw/` 完全一致。
- [ ] `gold_keywords` 是原文真实出现的词，不是同义词或你的概括。
- [ ] 关键词有区分度：不要用“模型、系统、方法”这种每篇都有的词。
- [ ] `paraphrase` 题不要直接抄原文句子，否则测不出语义检索。
- [ ] `keyword` 题要包含一个精确词（如 `n_results`、`RRF`、`Checkpointer`），用来测 BM25。
- [ ] `unanswerable` 问题要“看起来合理”，不要问“火星上有没有税务局”这种明显无关题。
- [ ] 没有重复/近重复问题。
- [ ] 可回答题的 `gold_keywords` 不为空，`unanswerable` 的 `gold_keywords` 为 `[]`。

## 5. 写完怎么验证

### 5.1 格式与数量

```bat
python -c "import json,collections; items=[json.loads(l) for l in open('eval/questions.jsonl',encoding='utf-8') if l.strip()]; print('总数:',len(items)); print('type:',collections.Counter(i['type'] for i in items)); print('category:',collections.Counter(i.get('category','未标注') for i in items)); print('重复:',len(items)-len({i['question'] for i in items}))"
```

### 5.2 跑检索评测

```bat
python scripts/evaluate.py --retrieval-only
```

重点检查：
- `keyword_questions` 是否等于可回答题数量（说明每道可回答题都标了关键词）；
- `report.json` 的 `details` 里有没有 `rank: null` 或 `keyword_top1_coverage: 0` 的题；
- 新增的知识库外问题是否被正确拒答（`refusal_accuracy`）。

### 5.3 看分布

```bat
python -c "import json,collections; items=[json.loads(l) for l in open('eval/questions.jsonl',encoding='utf-8') if l.strip()]; print('来源分布:',collections.Counter(s for i in items for s in i.get('gold_sources',[])))"
```

每篇文档至少 5 道题；如果某篇为 0，说明覆盖不均。

## 6. 空白模板

```json
{"question": "【问题】", "type": "answerable", "category": "fact", "gold_sources": ["【文件名.md】"], "gold_answer": "【参考答案】", "gold_keywords": ["【关键词1】", "【关键词2】"], "notes": "【对应原文/出题理由】"}
```

知识库外问题：

```json
{"question": "【知识库外的问题】", "type": "unanswerable", "category": "out_of_scope", "gold_sources": [], "gold_answer": "", "gold_keywords": [], "notes": "【为什么知识库回答不了】"}
```

## 7. 示例文件

`eval/questions.template.jsonl` 里有 12 条基于当前示例文档的题目，覆盖了所有 category，可以直接复制、修改后合并进 `questions.jsonl`。

建议一次加 5–10 条，加完就跑一次 `--retrieval-only`，确认新题没有格式错误、指标没有异常下降，再继续加。

## 8. 工具调用评测集

`eval/tool_calls.jsonl` 用于评测 Agent 是否选对工具、是否传对 `source` 参数。每行一个 JSON：

```json
{"question": "知识库里有哪些文档？", "history": [], "expected_tools": ["list_knowledge_base_documents"], "category": "list", "notes": "文档列表工具"}
```

| 字段 | 必填 | 说明 |
|---|---|---|
| `question` | 是 | 用户问题 |
| `history` | 否 | 多轮历史，格式同 Agent API |
| `expected_tools` | 是 | 必须调用的工具集合；不调用工具时填 `[]` |
| `allowed_extra_tools` | 否 | 允许出现的额外工具，例如先列文档再检索 |
| `expected_source` | 否 | 期望传给 `source` 参数的文件名 |
| `category` | 否 | `no_tool` / `list` / `stats` / `doc_info` / `search` 等 |
| `notes` | 否 | 出题理由 |

运行：

```bat
python scripts/evaluate_tool_calling.py
```

输出 `eval/results/tool_calls_report.json`，包含逐题的期望工具、实际工具、source 参数、答案和延迟。该脚本会真实调用 DeepSeek，产生费用；建议每次改动工具描述或 `AGENT_TOOL_SOURCE` 后运行一次。
