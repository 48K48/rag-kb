"""
切分参数对比实验：自动跑多组 chunk_size / chunk_overlap，输出 Markdown 表格。

用法：
    python scripts/run_chunk_experiment.py
    python scripts/run_chunk_experiment.py --configs 300:50,500:100,800:150
    python scripts/run_chunk_experiment.py --modes dense,hybrid
    python scripts/run_chunk_experiment.py --judge    # 同时跑 LLM 打分（慢、会产生费用）

--modes 说明：
    dense  = 纯向量检索（HYBRID_ENABLED=false）
    hybrid = 向量 + BM25 + RRF（HYBRID_ENABLED=true）
    默认两种都跑，方便直接对比混合检索的收益。

为什么要用独立目录：
    不同切分参数会产生不同的 chunk id。如果都写进同一个 Chroma 目录，
    上一组配置的旧块会残留在库里，污染下一组的评测结果。
    所以每组配置使用 data/chroma_{size}_{overlap}/，跑之前先清空。

为什么用子进程 + 环境变量：
    app/config.py 是在 import 时读取环境变量的。
    如果在同一个进程里修改 os.environ 再 import，代码已经加载完，改不动；
    用 subprocess 启动新的 Python 进程，环境变量才能在 import 前生效。

指标说明：
    - Hit@k / MRR 是文件级指标，语料小的时候容易饱和（全是 1.0）；
    - Top1覆盖 / Top3覆盖 / Top1全覆盖是块级指标，
      检查答案关键词是否落在检索到的具体文本块里，能区分切分粒度。

输出：
    - 终端打印 Markdown 对比表；
    - eval/results/chunk_experiment.md 保存表格；
    - eval/results/report_{size}_{overlap}.json 保存每一组的明细。
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

# Windows 终端默认 GBK，中文表头会乱码；统一改成 UTF-8。
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# 项目根目录（scripts/ 的上一级）。
ROOT = Path(__file__).resolve().parent.parent

# 用当前解释器启动子进程，保证和运行本脚本的是同一个虚拟环境。
PYTHON = sys.executable


def parse_configs(raw: str) -> list[tuple[int, int]]:
    """把 "300:50,500:100" 解析成 [(300, 50), (500, 100)]。"""
    configs: list[tuple[int, int]] = []
    for item in raw.split(","):
        size, overlap = item.split(":")
        configs.append((int(size), int(overlap)))
    return configs


def parse_modes(raw: str) -> list[str]:
    """解析 --modes，只允许 dense / hybrid。"""
    modes = [item.strip().lower() for item in raw.split(",") if item.strip()]
    invalid = set(modes) - {"dense", "hybrid"}
    if invalid:
        raise SystemExit(f"不支持的 mode：{invalid}，只能是 dense 或 hybrid")
    return modes


def run_one(size: int, overlap: int, hybrid: bool, use_judge: bool) -> dict:
    """跑一组配置：清空目录 -> 入库 -> 评测 -> 读取报告。

    hybrid=False 时设置 HYBRID_ENABLED=false，走纯向量 baseline；
    hybrid=True 时设置 HYBRID_ENABLED=true，走向量 + BM25 + RRF。
    """
    chroma_dir = ROOT / "data" / f"chroma_{size}_{overlap}"
    # 清空上一轮同配置的数据，保证这次是从零开始。
    shutil.rmtree(chroma_dir, ignore_errors=True)

    # 环境变量优先级高于 .env，所以子进程会使用这里的参数。
    env = os.environ.copy()
    env["CHUNK_SIZE"] = str(size)
    env["CHUNK_OVERLAP"] = str(overlap)
    env["CHROMA_DIR"] = str(chroma_dir)
    env["HYBRID_ENABLED"] = "true" if hybrid else "false"

    # 1) 入库
    ingest = subprocess.run(
        [PYTHON, "scripts/ingest.py"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="ignore",
    )
    if ingest.returncode != 0:
        print(ingest.stdout)
        print(ingest.stderr)
        raise SystemExit(f"入库失败：CHUNK_SIZE={size}, OVERLAP={overlap}")

    # 从「文件数：4，切块数：8，库内总数：8」里提取切块数。
    match = re.search(r"切块数：(\d+)", ingest.stdout)
    chunks = int(match.group(1)) if match else None

    # 2) 评测
    # 默认只评检索，速度快、不花钱；--judge 时再加 LLM 打分。
    evaluate_cmd = [PYTHON, "scripts/evaluate.py", "--retrieval-only"]
    if use_judge:
        evaluate_cmd = [PYTHON, "scripts/evaluate.py", "--judge"]

    evaluate = subprocess.run(
        evaluate_cmd,
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="ignore",
    )
    if evaluate.returncode != 0:
        print(evaluate.stdout)
        print(evaluate.stderr)
        raise SystemExit(f"评测失败：CHUNK_SIZE={size}, OVERLAP={overlap}")

    # 3) 读取报告，并另存一份带参数名的副本，避免被下一组覆盖。
    report_path = ROOT / "eval" / "results" / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    target = ROOT / "eval" / "results" / f"report_{size}_{overlap}.json"
    target.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    return {
        "mode": "hybrid" if hybrid else "dense",
        "chunk_size": size,
        "chunk_overlap": overlap,
        "chunks": chunks,
        "hit@k": report["hit@k"],
        "mrr": report["mrr"],
        # 块级指标：Hit@k 饱和时，用关键词覆盖率区分切分粒度。
        "keyword_top1_coverage": report.get("keyword_top1_coverage"),
        "keyword_top3_coverage": report.get("keyword_top3_coverage"),
        "keyword_top1_all_rate": report.get("keyword_top1_all_rate"),
        "refusal_accuracy": report["refusal_accuracy"],
        "avg_latency_ms": report["avg_latency_ms"],
        "judge_avg": report["judge_avg"],
        "report": str(target),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--configs",
        default="300:50,500:100,800:150",
        help="格式：size:overlap,size:overlap",
    )
    parser.add_argument(
        "--modes",
        default="dense,hybrid",
        help="要对比的检索模式：dense（纯向量）或 hybrid（向量+BM25+RRF）",
    )
    parser.add_argument(
        "--judge",
        action="store_true",
        help="同时调用 DeepSeek 给答案打分（每组 8 次调用，较慢且有费用）",
    )
    args = parser.parse_args()

    configs = parse_configs(args.configs)
    modes = parse_modes(args.modes)
    results = [
        run_one(size, overlap, mode == "hybrid", args.judge)
        for size, overlap in configs
        for mode in modes
    ]

    # 生成 Markdown 表格
    lines = [
        "| 模式 | chunk_size | overlap | chunks | Hit@k | MRR | Top1覆盖 | Top3覆盖 | Top1全覆盖 | 拒答准确率 | 平均延迟(ms) | Judge |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        judge = "-" if result["judge_avg"] is None else f'{result["judge_avg"]:.2f}'
        refusal = (
            "-"
            if result["refusal_accuracy"] is None
            else f'{result["refusal_accuracy"]:.4f}'
        )
        top1 = (
            "-"
            if result["keyword_top1_coverage"] is None
            else f'{result["keyword_top1_coverage"]:.4f}'
        )
        top3 = (
            "-"
            if result["keyword_top3_coverage"] is None
            else f'{result["keyword_top3_coverage"]:.4f}'
        )
        top1_all = (
            "-"
            if result["keyword_top1_all_rate"] is None
            else f'{result["keyword_top1_all_rate"]:.4f}'
        )
        lines.append(
            f'| {result["mode"]} | {result["chunk_size"]} | {result["chunk_overlap"]} | '
            f'{result["chunks"]} | {result["hit@k"]:.4f} | {result["mrr"]:.4f} | '
            f'{top1} | {top3} | {top1_all} | '
            f'{refusal} | {result["avg_latency_ms"]:.1f} | {judge} |'
        )

    table = "\n".join(lines)
    print(table)

    output = ROOT / "eval" / "results" / "chunk_experiment.md"
    output.write_text(table + "\n", encoding="utf-8")
    print(f"\n结果已保存：{output}")


if __name__ == "__main__":
    main()
