"""工具调用评测：检查 Agent 是否选对工具、是否传对 source 参数。

用法：
    python scripts/evaluate_tool_calling.py
    python scripts/evaluate_tool_calling.py --url http://127.0.0.1:8000
    python scripts/evaluate_tool_calling.py --questions eval/tool_calls.jsonl

前置条件：
    - FastAPI 已启动；
    - 已配置 AGENT_API_KEY / DEEPSEEK_API_KEY；
    - 如果 AGENT_TOOL_SOURCE=auto，建议先启动 MCP 服务，这样测的是 MCP 工具。

指标：
    - tool_set_accuracy：期望工具是否全部调用，且没有出现未授权的额外工具；
    - no_tool_accuracy：期望“不调用工具”的题目中，正确直接回答的比例；
    - source_arg_accuracy：需要指定文件名的题目中，source 参数正确的比例；
    - used_rag_accuracy：used_rag 标记是否符合期望；
    - avg_latency_ms：平均端到端耗时；
    - tool_source_counts：本轮工具来自 MCP 还是本地。

注意：
    - 这是端到端评测，会真实调用 DeepSeek，产生费用；
    - 模型有随机性，temperature=0.1，建议跑 2-3 次取平均；
    - 题目集很小，指标只适合做方向性对比，不适合统计显著性。
"""

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

# Windows 终端默认 GBK，中文输出会乱码；统一改成 UTF-8。
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# 允许直接运行 scripts/evaluate_tool_calling.py；必须在 import app.* 之前执行。
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx  # noqa: E402

from app.config import settings  # noqa: E402


def load_cases(path: Path) -> list[dict]:
    """读取 JSONL 评测集。"""
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Agent 工具调用评测")
    parser.add_argument(
        "--questions",
        default=str(settings.base_dir / "eval" / "tool_calls.jsonl"),
        help="评测集路径（JSONL）",
    )
    parser.add_argument(
        "--url",
        default="http://127.0.0.1:8000",
        help="FastAPI 地址",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=180.0,
        help="单题超时（秒）",
    )
    args = parser.parse_args()

    cases = load_cases(Path(args.questions))
    details: list[dict] = []
    tool_source_counter: Counter[str] = Counter()

    for case in cases:
        question = case["question"]
        expected_tools = set(case.get("expected_tools", []))
        # 允许的额外工具：例如先列文档再检索是合理行为，不应判错。
        allowed_tools = expected_tools | set(
            case.get("allowed_extra_tools", [])
        )
        expected_source = case.get("expected_source")

        start = time.perf_counter()
        error = None
        data: dict = {}
        try:
            response = httpx.post(
                f"{args.url.rstrip('/')}/api/agent/chat",
                json={
                    "question": question,
                    "history": case.get("history", []),
                },
                timeout=args.timeout,
            )
            response.raise_for_status()
            data = response.json()
        except Exception as exc:  # noqa: BLE001
            error = str(exc)

        latency = int((time.perf_counter() - start) * 1000)
        tool_calls = data.get("tool_calls", []) or []
        actual_tools = {call.get("name", "") for call in tool_calls}
        # 期望工具必须全部调用，且不能出现未授权的额外工具。
        tool_ok = (
            error is None
            and expected_tools <= actual_tools
            and actual_tools <= allowed_tools
        )

        # search_knowledge_base 和 get_document_info 都可能带 source 参数。
        actual_sources = [
            (call.get("args") or {}).get("source")
            for call in tool_calls
            if "source" in (call.get("args") or {})
        ]
        source_ok = None
        if expected_source:
            source_ok = error is None and expected_source in actual_sources

        used_rag_expected = "search_knowledge_base" in expected_tools
        used_rag_ok = (
            error is None and bool(data.get("used_rag")) == used_rag_expected
        )

        tool_source = data.get("tool_source")
        if tool_source:
            tool_source_counter[tool_source] += 1

        details.append(
            {
                "question": question,
                "category": case.get("category"),
                "expected_tools": sorted(expected_tools),
                "allowed_tools": sorted(allowed_tools),
                "actual_tools": sorted(actual_tools),
                "tool_set_ok": tool_ok,
                "expected_source": expected_source,
                "actual_sources": actual_sources,
                "source_ok": source_ok,
                "used_rag_expected": used_rag_expected,
                "used_rag_ok": used_rag_ok,
                "tool_source": tool_source,
                "answer": data.get("answer"),
                "latency_ms": latency,
                "error": error,
            }
        )

    total = len(details)
    no_tool_cases = [item for item in details if not item["expected_tools"]]
    source_cases = [item for item in details if item["expected_source"]]
    errors = [item for item in details if item["error"]]

    report = {
        "total": total,
        "tool_set_accuracy": round(
            sum(item["tool_set_ok"] for item in details) / total, 4
        )
        if total
        else 0.0,
        "no_tool_accuracy": round(
            sum(item["tool_set_ok"] for item in no_tool_cases)
            / len(no_tool_cases),
            4,
        )
        if no_tool_cases
        else None,
        "source_arg_accuracy": round(
            sum(bool(item["source_ok"]) for item in source_cases)
            / len(source_cases),
            4,
        )
        if source_cases
        else None,
        "used_rag_accuracy": round(
            sum(item["used_rag_ok"] for item in details) / total, 4
        )
        if total
        else 0.0,
        "avg_latency_ms": round(
            sum(item["latency_ms"] for item in details) / total, 1
        )
        if total
        else 0.0,
        "tool_source_counts": dict(tool_source_counter),
        "errors": len(errors),
        "details": details,
    }

    output = settings.base_dir / "eval" / "results" / "tool_calls_report.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    summary = {key: value for key, value in report.items() if key != "details"}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\n逐题明细：{output}")

    # 终端额外打印失败题目，方便直接定位。
    failures = [item for item in details if not item["tool_set_ok"]]
    if failures:
        print("\n未通过题目：")
        for item in failures:
            print(
                f"- {item['question']} | 期望 {item['expected_tools']} "
                f"实际 {item['actual_tools']} | {item['error'] or ''}"
            )


if __name__ == "__main__":
    main()
