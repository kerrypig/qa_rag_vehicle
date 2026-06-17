"""
检索策略评测脚本。

用法:
  python test_benchmark.py run              # 完整检索过程 + hit rate，自动保存到 logs/
  python test_benchmark.py run --compact    # 仅摘要
  python test_benchmark.py run -o report.txt  # 指定输出路径
  python test_benchmark.py run --no-save    # 不写入文件
  python test_benchmark.py lookup 空调 制冷  # 按文本查 chunk_id（标注 ground truth）

config.yaml 开关:
  query_rewrite.enabled              — 是否改写问句
  retrieval.keyword_search.enabled   — 是否启用关键词检索路径
  retrieval.bookmark_match.enabled   — 是否 PDF 书签匹配
  retrieval.bookmark_match.max_matches — 书签最大匹配数
  verification.enabled               — 召回后是否 qwen 核对（不含书签 chunk）
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
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
    vehicle_model: str = ""  # 强制车型过滤（留空则按问句自动识别/不过滤）


# 填写 expected_texts 后先跑 lookup 确认 chunk_id，再填入 expected_chunk_ids
BENCHMARK_CASES: list[BenchmarkCase] = [
    # BenchmarkCase(
    #     question="夏天车里被晒得像烤箱，怎么能最快把冷风开到最大？",
    #     expected_texts=["极速制冷/制热"],
    #     expected_chunk_ids=["b8e278c0-4350-4709-93d0-270c74656e74"]
    # ),
    BenchmarkCase(
        question="仪表盘右下角那个 0% PWR 进度条是干嘛的？",
        expected_texts=["瞬时功率百分比"],
        expected_chunk_ids=['81179c9e-3d75-4cc5-a22b-bf5a56d1f070', '91ffdbce-f57d-4f18-99cc-8d171e8e02ec'],
        vehicle_model="M9-EVR-2025",
    ),
    # BenchmarkCase(
    #     question="我是五座版 M9，后排中间的那个座位能不能用 ISOFIX 固定接口装儿童座椅？",
    #     expected_texts=["ISOFIX", "不同的 ISOFIX 位置对 ISOFIX 儿童约束系统的适应性信息"],
    #     expected_chunk_ids=['aed3fc1d-c510-47e2-a115-47c4271b88a5']
    # ),
    # BenchmarkCase(
    #     question="车门外把手没有自己弹出来，我现在在车外，怎么把门打开？",
    #     expected_texts=["按压车门外把手前端使车门外把手翘起，再拉动车门外把手"],
    #     expected_chunk_ids=["812892d0-908e-44df-9b75-4b62bdd396e8"]
    # ),
    # BenchmarkCase(
    #     question="刚启动车，发现屏幕上弹出一个红色的小人前面有个球的灯亮了，这是咋回事？",
    #     expected_chunk_ids=["70a31c61-5b55-4b42-ac4f-c0fedfc091bd","4a936e25-655b-41fe-8a43-38b595634693","b0e05530-a996-4525-a369-c8fa17ba7c70"]
    # ),
    # BenchmarkCase(
    #     question="我家小孩 5 岁，差不多 18 公斤，我的五座版 M9 后排应该怎么弄座椅？",
    #     expected_chunk_ids=['64bafccd-60b5-4405-a626-9f87634ede41']
    # ),
    # BenchmarkCase(
    #     question="前面路段有积水，我可以直接开过去吗？要注意啥？",
    #     expected_chunk_ids=['0784d768-c91c-458d-a529-1e99cdac0754']
    # ),
    # BenchmarkCase(
    #     question="方向盘位置太低了，怎么调高点？",
    #     expected_chunk_ids=['0f15c4b3-b03c-4ea6-9bbc-d0baaec8584c']
    # ),
    # BenchmarkCase(
    #     question="这周末准备带全家出去玩，车上坐满 6 个人，后备箱也塞满了。这种满载情况下，我的前轮胎压和后轮胎压应该分别打到多少 bar？",
    #     expected_chunk_ids=['8e75321a-3592-49e2-86f7-278fb7582f43']
    # ),
    # BenchmarkCase(
    #     question="仪表盘上亮了个红色的灯，像一个小帆船在水上飘（或者像个水壶下面有波浪线），这车还能继续开吗？",
    #     expected_chunk_ids= ['70a31c61-5b55-4b42-ac4f-c0fedfc091bd', '4a936e25-655b-41fe-8a43-38b595634693']
    # ),
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


def format_config_flags(config) -> str:
    lines = [
        "=== 当前检索策略开关 ===",
        f"  rewrite:          {config.rewrite_enabled}",
        f"  keyword_search:   {config.keyword_search_enabled}",
        f"  bookmark_match:   {config.bookmark_match_enabled}",
        f"  bookmark_max:     {config.bookmark_max_matches}",
        f"  verification:     {config.verification_enabled}",
        (
            f"  top_k:            keyword={config.get('retrieval', 'keyword_top_k', default=2)}"
            f" + rewritten={config.get('retrieval', 'rewritten_top_k', default=5)}"
        ),
        "",
    ]
    return "\n".join(lines)


def print_config_flags(config) -> None:
    print(format_config_flags(config))


def _format_chunk_block(
    rank: int,
    doc,
    scores: dict[str, float],
    *,
    source: str = "",
    expected_ids: set[str] | None = None,
) -> str:
    meta = doc.metadata
    cid = meta.get("chunk_id", "")
    score = scores.get(cid, 0.0)
    page = meta.get("page", "?")
    section = meta.get("section_path", "")
    mark = " [GT✓]" if expected_ids and cid in expected_ids else ""
    src = source or meta.get("retrieval_source", "hybrid")
    extra = ""
    if meta.get("bookmark_title"):
        extra = f" | 书签: {meta['bookmark_title']}"
    header = (
        f"  [{rank}] chunk_id={cid}{mark}\n"
        f"      来源={src} | P.{page} | {section}{extra} | score={score:.4f}"
    )
    body = doc.page_content.strip().replace("\n", " ").replace("\r", " ")
    return f"{header}\n      ---\n      {body}\n      ---"


def print_retrieval_trace(result, expected_ids: list[str]) -> str:
    """生成完整检索过程文本（各阶段 + chunk 全文）。"""
    t = result.trace
    if t is None:
        return "  ⚠ 无 trace 数据\n"

    gt = set(expected_ids)
    scores = result.scores
    parts: list[str] = []

    def add(text: str = "") -> None:
        parts.append(text)

    add(f"\n{'=' * 60}")
    add("【阶段 1】Query Rewrite / Keyword 提取")
    add(f"{'=' * 60}")
    add(f"  原始问题:     {result.query}")
    add(f"  rewrite 开关: {t.rewrite_enabled}")
    add(f"  keyword 开关: {t.keyword_search_enabled}")
    add(f"  → 关键词:     {t.keyword}")
    add(f"  → 改写问句:   {t.rewritten_query}")

    def stage(title: str, lines: list[str]) -> None:
        add(f"\n{'─' * 60}")
        add(f"▶ {title}")
        add(f"{'─' * 60}")
        if lines:
            add("\n".join(lines))
        else:
            add("  （无）")

    rw_lines = [
        _format_chunk_block(i, d, t.rewritten_scores, source="rewritten_hybrid", expected_ids=gt)
        for i, d in enumerate(t.rewritten_docs, 1)
    ]
    stage(f"【阶段 2a】rewritten_q Hybrid 检索 (top {len(t.rewritten_docs)})", rw_lines)

    if t.keyword_search_enabled:
        kw_lines = [
            _format_chunk_block(i, d, t.keyword_scores, source="keyword_hybrid", expected_ids=gt)
            for i, d in enumerate(t.keyword_docs, 1)
        ]
        stage(f"【阶段 2b】keyword Hybrid 检索 (top {len(t.keyword_docs)})", kw_lines)

    merged_lines = [
        _format_chunk_block(i, d, scores, source="hybrid_merged", expected_ids=gt)
        for i, d in enumerate(t.hybrid_merged, 1)
    ]
    stage(f"【阶段 3】双路合并 (共 {len(t.hybrid_merged)} 条)", merged_lines)

    if t.threshold_removed:
        rm_lines = [
            f"  - {r['chunk_id']} | vector_sim={r['vector_sim']:.4f} < threshold={r['threshold']}"
            for r in t.threshold_removed
        ]
        stage(f"【阶段 4】score_threshold 过滤 (移除 {len(t.threshold_removed)} 条)", rm_lines)
    else:
        stage("【阶段 4】score_threshold 过滤", ["  （无移除）"])

    after_th_lines = [
        _format_chunk_block(i, d, scores, source="hybrid", expected_ids=gt)
        for i, d in enumerate(t.hybrid_after_threshold, 1)
    ]
    stage(f"【阶段 4 结果】阈值过滤后 hybrid (共 {len(t.hybrid_after_threshold)} 条)", after_th_lines)

    if t.bookmark_enabled:
        add(f"\n{'─' * 60}")
        add("▶ 【阶段 5】PDF 书签匹配")
        add(f"{'─' * 60}")
        add(f"  LLM 原始输出: {t.bookmark_llm_raw or '（无）'}")
        if t.bookmark_fallback:
            add(f"  回退方式:     {t.bookmark_fallback}")
        add(f"  选中书签:     {', '.join(t.bookmark_selected) if t.bookmark_selected else '（无）'}")
        add(f"  （目录为叶子小节 + 完整路径，如 车辆控制>空调）")
        bm_lines = [
            _format_chunk_block(i, d, scores, source="bookmark", expected_ids=gt)
            for i, d in enumerate(t.bookmark_docs, 1)
        ]
        if bm_lines:
            add("\n".join(bm_lines))
        else:
            add("  （无书签 chunk）")
    else:
        stage("【阶段 5】PDF 书签匹配", ["  （已关闭）"])

    if t.verification_enabled and t.hybrid_before_verify:
        add(f"\n{'─' * 60}")
        add("▶ 【阶段 6】qwen2.5 Validation（仅 hybrid，书签跳过）")
        add(f"{'─' * 60}")
        for v in t.verify_verdicts:
            tag = "通过" if v["verdict"] else "拒绝"
            add(
                f"  [{tag}] {v['chunk_id']} | P.{v['page']} | {v['section']}\n"
                f"         LLM 回复: {v['raw']}"
            )
        add(f"\n  核对前: {len(t.hybrid_before_verify)} 条 → 核对后: {len(t.hybrid_after_verify)} 条")
        after_v_lines = [
            _format_chunk_block(i, d, scores, source="hybrid_verified", expected_ids=gt)
            for i, d in enumerate(t.hybrid_after_verify, 1)
        ]
        if after_v_lines:
            add("\n".join(after_v_lines))
    elif t.verification_enabled:
        stage("【阶段 6】qwen2.5 Validation", ["  （hybrid 为空，跳过）"])
    else:
        stage("【阶段 6】qwen2.5 Validation", ["  （已关闭）"])

    final_lines = [
        _format_chunk_block(i, d, scores, expected_ids=gt)
        for i, d in enumerate(t.final_docs, 1)
    ]
    stage(f"【阶段 7】最终合并输出 (书签置顶 + hybrid，共 {len(t.final_docs)} 条)", final_lines)

    return "\n".join(parts) + "\n"

def default_report_path(config) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return config.log_dir / f"test_{ts}.txt"


def save_report(report_lines: list[str], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(report_lines), encoding="utf-8")
    return path.resolve()


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

    report_lines: list[str] = []

    def emit(text: str = "") -> None:
        print(text)
        report_lines.append(text)

    emit(format_config_flags(config).rstrip())

    retriever = Retriever(config, index_path)
    retriever.load()

    total = 0
    hits = 0
    recalls: list[float] = []

    for i, case in enumerate(BENCHMARK_CASES, start=1):
        expected_ids = resolve_expected_ids(case, index_path)
        emit(f"\n{'#' * 70}")
        emit(f"# [{i}/{len(BENCHMARK_CASES)}] {case.question}")
        emit(f"{'#' * 70}")

        if not expected_ids:
            emit("  ⚠ 未配置 expected_chunk_ids / expected_texts 无匹配，跳过 hit 计算")
            emit(f"  提示: python test_benchmark.py lookup {' '.join(case.expected_texts)}")

        force = [case.vehicle_model] if case.vehicle_model else None
        result = retriever.retrieve_stateless(
            case.question, trace=not args.compact, force_models=force
        )

        if args.compact:
            if result.keyword != case.question:
                emit(f"  关键词: {result.keyword}")
            if result.rewritten_query != case.question:
                emit(f"  改写:   {result.rewritten_query}")
            if result.bookmark_titles:
                emit(f"  书签:   {', '.join(result.bookmark_titles)} ({result.bookmark_count} chunks)")
            if config.verification_enabled and result.pre_verify_count:
                hybrid_kept = len(result.docs) - result.bookmark_count
                emit(f"  核对:   hybrid {hybrid_kept}/{result.pre_verify_count} 条保留")
            emit(f"  召回 {len(result.docs)} 条:")
            for rank, doc in enumerate(result.docs, start=1):
                cid = doc.metadata["chunk_id"]
                score = result.scores.get(cid, 0.0)
                mark = "✓" if cid in set(expected_ids) else " "
                src = "书签" if doc.metadata.get("retrieval_source") == "bookmark" else "hybrid"
                preview = doc.page_content.strip().replace("\n", " ")[:80]
                emit(f"    [{rank}]{mark} [{src}] {cid} score={score:.4f} | {preview}…")
        else:
            trace_text = print_retrieval_trace(result, expected_ids)
            emit(trace_text.rstrip())

        retrieved_ids = [d.metadata["chunk_id"] for d in result.docs]
        metrics = compute_metrics(retrieved_ids, expected_ids)

        emit(f"\n{'─' * 60}")
        emit("【评测结果】")
        if metrics["hit"] is not None:
            total += 1
            if metrics["hit"]:
                hits += 1
            recalls.append(metrics["recall"])
            emit(
                f"  Hit: {metrics['hit']} | Recall: {metrics['recall']:.2%}"
                f" | 命中: {metrics['matched_ids']}"
            )
            emit(f"  期望 GT: {expected_ids}")
        else:
            emit("  （无 GT 标注，跳过指标）")

    emit(f"\n{'=' * 70}")
    emit("【汇总】")
    if total:
        emit(f"  Hit Rate:   {hits}/{total} = {hits / total:.2%}")
        emit(f"  Avg Recall: {sum(recalls) / len(recalls):.2%}")
    else:
        emit("  无有效标注用例，请先配置 expected_texts 或 expected_chunk_ids")

    if not args.no_save:
        out_path = Path(args.output) if args.output else default_report_path(config)
        saved = save_report(report_lines, out_path)
        print(f"\n完整报告已写入: {saved}")


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG 检索策略评测")
    parser.add_argument("--config", default=str(ROOT / "config.yaml"))
    parser.add_argument("--strategy", choices=["hierarchy", "semantic", "fixed_size"])
    sub = parser.add_subparsers(dest="command")

    p_run = sub.add_parser("run", help="运行评测集（默认输出完整检索过程）")
    p_run.add_argument("--compact", action="store_true", help="仅输出摘要，不展示各阶段全文")
    p_run.add_argument("-o", "--output", help="报告输出路径（默认 logs/test_YYYYMMDD_HHMMSS.txt）")
    p_run.add_argument("--no-save", action="store_true", help="不保存报告文件")
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
