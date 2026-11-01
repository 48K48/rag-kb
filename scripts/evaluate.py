"""
跑评测集：python scripts/evaluate.py [--retrieval-only] [--judge]

这个脚本是「RAG 项目从 demo 变成工程」的关键：
    不靠感觉判断效果，而是用一组固定问题 + 指标，对比每次改动的收益。

指标说明：
    - Hit@k（hit@k）：
        前 k 条检索结果里，只要有一条来自 gold_sources，就算命中。
        衡量「检索有没有把正确文档找出来」。
    - MRR（Mean Reciprocal Rank）：
        第一个命中结果的排名倒数，再对所有可回答问题求平均。
        排第 1 得 1 分，排第 2 得 0.5 分，排第 3 得 0.33 分。
        比 Hit@k 更严格：不仅要求命中，还要求排得靠前。
    - keyword_top1_coverage / keyword_top3_coverage：
        答案关键词出现在 top-1 / top-3 块中的比例（块级指标）。
        Hit@k 是文件级指标，语料小的时候容易饱和（全是 1.0），
        关键词覆盖率能区分不同切分粒度的实际效果。
    - keyword_top1_all_rate：
        top-1 块包含全部答案关键词的题目比例。
        这个指标最严格：要求答案完整地落在一个块里。
    - refusal_accuracy：
        知识库外问题（type=unanswerable）是否正确拒答。
        完整模式下看 response.refused；
        --retrieval-only 模式下用「没有召回任何块」近似。
    - avg_latency_ms：
        每个问题的平均处理耗时（毫秒）。
    - judge_avg：
        用 DeepSeek 给回答打 1-5 分的平均值（需要 --judge 和 API Key）。
        这是 LLM-as-Judge 的简化版，存在偏差，建议人工抽检。

运行模式：
    1. python scripts/evaluate.py --retrieval-only
       只评检索，不调用大模型。适合没配 Key、或只想调 chunk 参数时使用。
    2. python scripts/evaluate.py
       检索 + 生成，但不打分。
    3. python scripts/evaluate.py --judge
       完整评测：检索 + 生成 + LLM 打分。

输出：
    - 终端打印汇总指标；
    - eval/results/report.json 保存汇总 + 每条问题的明细，
      方便定位「哪道题没检索到」「哪道题答错了」。
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

# Windows 终端默认 GBK，中文输出会乱码；统一改成 UTF-8。
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# 允许直接运行 scripts/evaluate.py；必须在 import app.* 之前执行。
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openai import OpenAI  # noqa: E402

from app.config import settings  # noqa: E402
from app.deps import get_chain, get_retriever  # noqa: E402


def load_questions(path: Path) -> list[dict]:
    """读取 JSONL 评测集。

    JSONL 的格式是「一行一个 JSON 对象」，
    相比一个大 JSON 数组，更方便追加、diff 和逐行处理。
    """
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def first_hit_rank(sources: list[str], gold_sources: list[str]) -> int | None:
    """返回第一个命中 gold_sources 的排名（从 1 开始）。

    参数：
        sources: 检索结果里的来源文件名列表，按相关性从高到低；
        gold_sources: 标准答案对应的来源文件名列表。

    返回：
        命中排名；没有命中返回 None。
    """
    for rank, source in enumerate(sources, start=1):
        if source in gold_sources:
            return rank
    return None


def keyword_metrics(results: list, gold_keywords: list[str]) -> dict | None:
    """计算答案关键词在 top-1 / top-3 块中的覆盖情况。

    参数：
        results: 检索结果（RetrievedChunk 列表，已按相关性排序、已过阈值）；
        gold_keywords: 标准答案里的关键词列表。

    返回：
        None（题目没有标注关键词时），或包含三个指标的 dict：
            top1_coverage: top-1 块里命中的关键词比例
            top3_coverage: top-3 块合并后命中的关键词比例
            top1_all:      top-1 块是否包含全部关键词（1/0）
    """
    if not gold_keywords:
        return None

    top1_text = results[0].text if results else ""
    top3_text = "\n".join(item.text for item in results[:3])

    top1_hits = sum(1 for keyword in gold_keywords if keyword in top1_text)
    top3_hits = sum(1 for keyword in gold_keywords if keyword in top3_text)

    return {
        "top1_coverage": top1_hits / len(gold_keywords),
        "top3_coverage": top3_hits / len(gold_keywords),
        "top1_all": 1 if top1_hits == len(gold_keywords) else 0,
    }


def judge_answer(question: str, gold_answer: str, answer: str) -> int | None:
    """让 DeepSeek 给回答打 1-5 分；没有配置 Key 时跳过。

    这是一个简化的 LLM-as-Judge：
        - 把问题、参考答案、模型回答一起交给模型；
        - 要求只输出一个数字；
        - 用正则提取 1-5 分。

    注意：
        - 模型打分有偏差，不能完全替代人工；
        - 建议随机抽 10 条人工复核，并在 README 里说明局限。
    """
    if not settings.deepseek_api_key:
        return None

    client = OpenAI(
        api_key=settings.deepseek_api_key, base_url=settings.deepseek_base_url
    )
    prompt = f"""请根据参考答案，给模型回答打分（1-5 分）。
5 分：事实正确、完整、没有编造
3 分：大意正确但有遗漏
1 分：错误或编造
只输出一个数字，不要解释。

问题：{question}
参考答案：{gold_answer}
模型回答：{answer}"""
    response = client.chat.completions.create(
        model=settings.deepseek_model,
        messages=[{"role": "user", "content": prompt}],
        # temperature=0：打分要稳定，不要发挥。
        temperature=0,
    )
    text = response.choices[0].message.content or ""
    match = re.search(r"[1-5]", text)
    return int(match.group()) if match else None


def main() -> None:
    """解析参数、逐题评测、计算指标、写出报告。"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--questions",
        default=str(settings.base_dir / "eval" / "questions.jsonl"),
        help="评测集路径（JSONL）",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=settings.top_k,
        help="检索条数，默认取配置里的 TOP_K",
    )
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="只评测检索，不调用大模型（适合没配置 Key 时）",
    )
    parser.add_argument(
        "--judge",
        action="store_true",
        help="调用 DeepSeek 给答案打 1-5 分",
    )
    args = parser.parse_args()

    questions = load_questions(Path(args.questions))
    retriever = get_retriever()
    chain = get_chain()

    # 计数器：只用 int / float，最后统一算比率，避免每道题都做除法。
    answerable = hits = refused_correct = unanswerable = 0
    mrr_total = 0.0
    keyword_questions = 0
    keyword_top1_total = 0.0
    keyword_top3_total = 0.0
    keyword_top1_all_total = 0
    latencies: list[int] = []
    judge_scores: list[int] = []
    details: list[dict] = []

    for item in questions:
        question = item["question"]
        gold_sources = item.get("gold_sources", [])
        question_type = item.get("type", "answerable")

        # 计时从「检索开始」到「回答结束」，包含 embedding + 向量检索 + 大模型生成。
        start = time.perf_counter()

        # 检索：注意这里直接调用 retriever.retrieve，
        # 它内部已经做了 MAX_DISTANCE 阈值过滤。
        results = retriever.retrieve(question, top_k=args.top_k)
        retrieved_sources = [result.metadata.get("source", "") for result in results]
        rank = first_hit_rank(retrieved_sources, gold_sources) if gold_sources else None

        # 块级指标：答案关键词有没有落在 top-1 / top-3 块里。
        # 只有标注了 gold_keywords 的题目才参与统计。
        gold_keywords = item.get("gold_keywords", [])
        keyword_stats = keyword_metrics(results, gold_keywords)
        if keyword_stats:
            keyword_questions += 1
            keyword_top1_total += keyword_stats["top1_coverage"]
            keyword_top3_total += keyword_stats["top3_coverage"]
            keyword_top1_all_total += keyword_stats["top1_all"]

        # 可回答问题：累计 Hit@k 和 MRR。
        if question_type == "answerable":
            answerable += 1
            if rank:
                hits += 1
                mrr_total += 1.0 / rank
        else:
            unanswerable += 1

        # 生成：--retrieval-only 时跳过，节省时间和费用。
        response = None
        if not args.retrieval_only:
            response = chain.answer(question, top_k=args.top_k)

        latency = int((time.perf_counter() - start) * 1000)
        latencies.append(latency)

        # 拒答判定：
        #   有模型时 -> 看 ChatResponse.refused（检索为空或模型输出拒答话术）；
        #   只评检索时 -> 用「没有召回任何块」近似。
        if response is not None:
            refused = response.refused
        elif question_type == "unanswerable":
            refused = len(results) == 0
        else:
            refused = None

        if question_type == "unanswerable" and refused:
            refused_correct += 1

        # LLM-as-Judge：只给「可回答且没有拒答」的答案打分，
        # 避免把拒答话术拿去和参考答案比较。
        judge_score = None
        if (
            args.judge
            and response
            and not response.refused
            and item.get("gold_answer")
        ):
            judge_score = judge_answer(
                question, item["gold_answer"], response.answer
            )
            if judge_score is not None:
                judge_scores.append(judge_score)

        # 保存每题明细，便于定位问题。
        details.append(
            {
                "question": question,
                "type": question_type,
                "gold_sources": gold_sources,
                "gold_keywords": gold_keywords,
                "retrieved_sources": retrieved_sources,
                "retrieved_distances": [round(result.distance, 4) for result in results],
                "rank": rank,
                "keyword_top1_coverage": (
                    round(keyword_stats["top1_coverage"], 4)
                    if keyword_stats
                    else None
                ),
                "keyword_top3_coverage": (
                    round(keyword_stats["top3_coverage"], 4)
                    if keyword_stats
                    else None
                ),
                "keyword_top1_all": (
                    keyword_stats["top1_all"] if keyword_stats else None
                ),
                "answer": response.answer if response else None,
                "refused": refused,
                "latency_ms": latency,
                "judge_score": judge_score,
            }
        )

    # 汇总指标。
    # 分母为 0 时返回 0 或 None，避免 ZeroDivisionError。
    report = {
        "total": len(questions),
        "answerable": answerable,
        "hit@k": round(hits / answerable, 4) if answerable else 0.0,
        "mrr": round(mrr_total / answerable, 4) if answerable else 0.0,
        "keyword_questions": keyword_questions,
        "keyword_top1_coverage": (
            round(keyword_top1_total / keyword_questions, 4)
            if keyword_questions
            else None
        ),
        "keyword_top3_coverage": (
            round(keyword_top3_total / keyword_questions, 4)
            if keyword_questions
            else None
        ),
        "keyword_top1_all_rate": (
            round(keyword_top1_all_total / keyword_questions, 4)
            if keyword_questions
            else None
        ),
        "refusal_accuracy": (
            round(refused_correct / unanswerable, 4) if unanswerable else None
        ),
        "avg_latency_ms": (
            round(sum(latencies) / len(latencies), 1) if latencies else 0
        ),
        "judge_avg": (
            round(sum(judge_scores) / len(judge_scores), 2) if judge_scores else None
        ),
        "details": details,
    }

    # 写出报告：ensure_ascii=False 保留中文，indent=2 方便人工阅读。
    output = settings.base_dir / "eval" / "results" / "report.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # 终端只打印汇总，明细在 report.json 里。
    summary = {key: value for key, value in report.items() if key != "details"}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"详细结果：{output}")


if __name__ == "__main__":
    main()
