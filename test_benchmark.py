"""
检索策略评测脚本。

用法:
  python test_benchmark.py run              # 跑评测集并计算 hit rate
  python test_benchmark.py lookup 空调 制冷  # 按文本查 chunk_id（标注 ground truth）

config.yaml 开关:
  query_rewrite.enabled          — 是否改写问句
  retrieval.keyword_search.enabled — 是否启用关键词检索路径
  verification.enabled           — 召回后是否 qwen 核对
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config_loader import load_config
from retrieve.chunk_lookup import (
    find_chunk_ids_by_text,
    format_matches,
    unique_chunk_ids,
)


@dataclass
class BenchmarkCase:
    question: str
    expected_texts: list[str] = field(default_factory=list)
    expected_chunk_ids: list[str] = field(default_factory=list)


# 填写 expected_texts 后先跑 lookup 确认 chunk_id，再填入 expected_chunk_ids
BENCHMARK_CASES: list[BenchmarkCase] = [
    BenchmarkCase(
        question="夏天车里被晒得像烤箱，怎么能最快把冷风开到最大？",
        expected_texts=["极速制冷/制热"],
        expected_chunk_ids=["3d442394-ea4c-4781-9233-d44bc8bfd27a"]
    ),
    BenchmarkCase(
        question="仪表盘右下角那个 0% PWR 进度条是干嘛的？",
        expected_texts=["瞬时功率百分比"],
        expected_chunk_ids=["70a31c61-5b55-4b42-ac4f-c0fedfc091bd","17965218-fc92-44e7-813c-2d03dc010be7"]
    ),
    BenchmarkCase(
        question="我是五座版 M9，后排中间的那个座位能不能用 ISOFIX 固定接口装儿童座椅？",
        expected_texts=["ISOFIX", "不同的 ISOFIX 位置对 ISOFIX 儿童约束系统的适应性信息"],
        expected_chunk_ids=["64bafccd-60b5-4405-a626-9f87634ede41"]
    ),
    BenchmarkCase(
        question="车门外把手没有自己弹出来，我现在在车外，怎么把门打开？",
        expected_texts=["按压车门外把手前端使车门外把手翘起，再拉动车门外把手"],
        expected_chunk_ids=["081fb53d-0420-4336-ab7d-6cf4d0a0ce3c"]
    ),
    BenchmarkCase(
        question="刚启动车，发现屏幕上弹出一个红色的小人前面有个球的灯亮了，这是咋回事？",
        expected_chunk_ids=["70a31c61-5b55-4b42-ac4f-c0fedfc091bd","4a936e25-655b-41fe-8a43-38b595634693","b0e05530-a996-4525-a369-c8fa17ba7c70"]
    ),
    BenchmarkCase(
        question="我家小孩 5 岁，差不多 18 公斤，我的五座版 M9 后排应该怎么弄座椅？",
        expected_chunk_ids=['64bafccd-60b5-4405-a626-9f87634ede41']
    ),
    BenchmarkCase(
        question="前面路段有积水，我可以直接开过去吗？要注意啥？",
        expected_chunk_ids=['0784d768-c91c-458d-a529-1e99cdac0754']
    ),
    BenchmarkCase(
        question="方向盘位置太低了，怎么调高点？",
        expected_chunk_ids=['0f15c4b3-b03c-4ea6-9bbc-d0baaec8584c']
    ),
    BenchmarkCase(
        question="这周末准备带全家出去玩，车上坐满 6 个人，后备箱也塞满了。这种满载情况下，我的前轮胎压和后轮胎压应该分别打到多少 bar？",
        expected_chunk_ids=['8e75321a-3592-49e2-86f7-278fb7582f43']
    ),
    BenchmarkCase(
        question="仪表盘上亮了个红色的灯，像一个小帆船在水上飘（或者像个水壶下面有波浪线），这车还能继续开吗？",
        expected_chunk_ids= ['70a31c61-5b55-4b42-ac4f-c0fedfc091bd', '4a936e25-655b-41fe-8a43-38b595634693']
    ),
    # BenchmarkCase(
    #     question="我车后面挂了房车，用了那个电动拖挂辅助，怎么现在 NCA（领航辅助）和自适应巡航不能用了？",
    #     expected_texts=["电动拖挂辅助开启后会禁用 ADS 相关功能"],
    # ),
]


def resolve_expected_ids(case: BenchmarkCase, index_path: Path) -> list[str]:
    if case.expected_chunk_ids:
        return case.expected_chunk_ids
    ids: list[str] = []
    for text in case.expected_texts:
        matches = find_chunk_ids_by_text(text, index_path)
        ids.extend(unique_chunk_ids(matches))
    return ids


def compute_metrics(retrieved_ids: list[str], expected_ids: list[str]) -> dict:
    if not expected_ids:
        return {"hit": None, "recall": None, "matched_ids": []}
    retrieved_set = set(retrieved_ids)
    expected_set = set(expected_ids)
    matched = retrieved_set & expected_set
    return {
        "hit": len(matched) > 0,
        "recall": len(matched) / len(expected_set),
        "matched_ids": sorted(matched),
    }


def print_config_flags(config) -> None:
    rw = config.get("query_rewrite", "enabled")
    kw = config.get("retrieval", "keyword_search", "enabled")
    ver = config.get("verification", "enabled")
    print("=== 当前检索策略开关 ===")
    print(f"  rewrite:          {rw}")
    print(f"  keyword_search:   {kw}")
    print(f"  verification:     {ver}")
    print(
        f"  top_k:            keyword={config.get('retrieval', 'keyword_top_k', default=2)}"
        f" + rewritten={config.get('retrieval', 'rewritten_top_k', default=5)}"
    )
    print()


def cmd_lookup(args) -> None:
    config = load_config(args.config, strategy_override=args.strategy)
    index_path = config.index_path()
    needles = args.texts
    if not needles:
        print("请提供至少一段文本，例如: python test_benchmark.py lookup 空调 ISOFIX")
        sys.exit(1)

    print(f"索引: {index_path}\n")
    for needle in needles:
        matches = find_chunk_ids_by_text(needle, index_path)
        ids = unique_chunk_ids(matches)
        print(f"查询文本: 「{needle}」 → {len(ids)} 个 chunk")
        print(format_matches(matches))
        if ids:
            print(f"  chunk_ids: {ids}")
        print("-" * 50)


def cmd_run(args) -> None:
    from retrieve.pipeline import Retriever

    config = load_config(args.config, strategy_override=args.strategy)
    index_path = config.index_path()

    if not (index_path / "faiss").exists():
        print(f"报错：找不到向量索引 {index_path}，请先执行 python main.py build")
        sys.exit(1)

    print_config_flags(config)
    retriever = Retriever(config, index_path)
    retriever.load()

    total = 0
    hits = 0
    recalls: list[float] = []

    for i, case in enumerate(BENCHMARK_CASES, start=1):
        expected_ids = resolve_expected_ids(case, index_path)
        print(f"\n========== [{i}/{len(BENCHMARK_CASES)}] {case.question} ==========")

        if not expected_ids:
            print("  ⚠ 未配置 expected_chunk_ids / expected_texts 无匹配，跳过 hit 计算")
            print(f"  提示: python test_benchmark.py lookup {' '.join(case.expected_texts)}")

        result = retriever.retrieve_stateless(case.question)

        if result.keyword != case.question:
            print(f"  关键词: {result.keyword}")
        if result.rewritten_query != case.question:
            print(f"  改写:   {result.rewritten_query}")
        if result.pre_verify_count:
            print(f"  核对:   {len(result.docs)}/{result.pre_verify_count} 条通过")

        retrieved_ids = [d.metadata["chunk_id"] for d in result.docs]
        metrics = compute_metrics(retrieved_ids, expected_ids)

        print(f"  召回 {len(retrieved_ids)} 条:")
        for rank, doc in enumerate(result.docs, start=1):
            cid = doc.metadata["chunk_id"]
            score = result.scores.get(cid, 0.0)
            mark = "✓" if cid in set(expected_ids) else " "
            preview = doc.page_content.strip().replace("\n", " ")[:80]
            print(f"    [{rank}]{mark} {cid} score={score:.4f} | {preview}…")

        if metrics["hit"] is not None:
            total += 1
            if metrics["hit"]:
                hits += 1
            recalls.append(metrics["recall"])
            print(
                f"  Hit: {metrics['hit']} | Recall: {metrics['recall']:.2%}"
                f" | 命中: {metrics['matched_ids']}"
            )
            print(f"  期望: {expected_ids}")

    print("\n========== 汇总 ==========")
    if total:
        print(f"  Hit Rate:  {hits}/{total} = {hits / total:.2%}")
        print(f"  Avg Recall:{sum(recalls) / len(recalls):.2%}")
    else:
        print("  无有效标注用例，请先配置 expected_texts 或 expected_chunk_ids")


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG 检索策略评测")
    parser.add_argument("--config", default=str(ROOT / "config.yaml"))
    parser.add_argument("--strategy", choices=["hierarchy", "semantic", "fixed_size"])
    sub = parser.add_subparsers(dest="command")

    p_run = sub.add_parser("run", help="运行评测集")
    p_run.set_defaults(func=cmd_run)

    p_lookup = sub.add_parser("lookup", help="按文本查 chunk_id")
    p_lookup.add_argument("texts", nargs="+", help="手册内文本片段")
    p_lookup.set_defaults(func=cmd_lookup)

    args = parser.parse_args()
    if args.command is None:
        args.command = "run"
        args.func = cmd_run
    args.func(args)


if __name__ == "__main__":
    main()
