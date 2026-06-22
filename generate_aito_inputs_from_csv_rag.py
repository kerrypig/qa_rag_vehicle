#!/usr/bin/env python3
"""从真实汽车问答 CSV 制作问界/AITO「input candidates」。

流程：读取真实问题 → 清洗/去重 → 规则分类与车型/动力推断 → 轻度规则改写 →
调用项目现有 RAG 检索做可回答性筛选 → 输出 accepted / rejected 两个 JSON。
只产出 input 候选，不生成最终 answer；不修改任何既有训练数据文件。

用法示例：
  # 自动识别/车型不明确
  python generate_aito_inputs_from_csv_rag.py \
      --source_csv doc/AutoMaster_TrainSet.csv --count 200 \
      --output data/aito_input_candidates_200.json --seed 42

  # 锁定到指定车型：改写前缀与 RAG 检索都限定到该车型 chunk
  python generate_aito_inputs_from_csv_rag.py --models M7 --count 200 \
      --output data/aito_m7_inputs.json --seed 42
  # 多车型 / 指定动力形式：--models "M9纯电,M8增程"

依赖说明：仅用本地嵌入 + FAISS 检索（不调用云端 LLM，不受 DashScope 配额影响）。
加 --verify 时会用本地 Ollama(qwen2.5:7b) 做检索相关性核对（更准但更慢）。
"""
from __future__ import annotations

import os

os.environ["HF_HUB_OFFLINE"] = "1"

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tqdm import tqdm

from aito_inputs.answerability import judge
from aito_inputs.candidate import build_candidate, build_rejected
from aito_inputs.classify import classify_task_type, risk_tags
from aito_inputs.dedup import jaccard, normalize
from aito_inputs.filters import (
    describes_specific_other_vehicle,
    is_ice_specific,
    is_off_topic_intent,
    needs_field_diagnosis,
)
from aito_inputs.models import available_models, resolve_models
from aito_inputs.powertrain import infer_scope, is_fuel_car_only, mentions_other_brand
from aito_inputs.rewrite import rule_rewrite
from config_loader import load_config
from dataset_gen.cleaning import extract_input_text, load_rows
from dataset_gen.config_overrides import apply_overrides
from retrieve.model_router import detect_models
from retrieve.pipeline import Retriever

# 关闭检索链路里所有 Ollama 步骤 → 轻量 hybrid（向量+BM25），快且离线
LIGHT_OVERRIDES = {
    "query_rewrite.enabled": False,
    "retrieval.keyword_search.enabled": False,
    "retrieval.bookmark_match.enabled": False,
    "verification.enabled": False,
}

# 触发「安全兜底」的主题标签
_SAFETY_TAGS = {"安全停车"}
# 太短/无意义问题的最小长度
_MIN_Q_LEN = 5
# 近重复阈值（与已 accepted 问题的 bigram Jaccard ≥ 此值则跳过）
_NEAR_DUP_TH = 0.85

# 新能源通用场景关键词：命中任一才保留（把 8 万条 CSV 预筛到相关子集，
# 同时排除大量传统燃油车闲聊，显著加速并提升相关性）。
KEEP_KEYWORDS = (
    "故障灯", "指示灯", "报警", "充电", "充电桩", "充电枪", "续航", "电池", "亏电",
    "保养", "质保", "三包", "保修", "保险", "理赔", "辅助驾驶", "智驾", "领航",
    "雷达", "自动泊车", "钥匙", "解锁", "空调", "制冷", "暖风", "除雾", "轮胎",
    "胎压", "异响", "异味", "事故", "碰撞", "气囊", "高压", "车机", "中控", "蓝牙",
    "车门", "车窗", "后备箱", "天窗", "后视镜", "座椅", "仪表", "刹车", "制动",
    "拖车", "增程", "加油", "机油", "冷却液", "灯",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="从真实 CSV 制作问界/AITO input candidates（RAG 筛选）")
    p.add_argument("--source_csv", default="doc/AutoMaster_TrainSet.csv", help="输入 CSV 路径")
    p.add_argument("--count", type=int, default=200, help="目标 accepted 数量")
    p.add_argument("--output", default="data/aito_input_candidates.json", help="accepted 输出 JSON")
    p.add_argument("--rejected_output", default="data/aito_input_rejected.json", help="rejected 输出 JSON")
    p.add_argument("--seed", type=int, default=42, help="随机种子（可复现）")
    p.add_argument("--models", default="",
                   help="目标车型（逗号分隔，如 'M7' 或 'M9纯电,M8增程'）。"
                        "指定后改写前缀与 RAG 检索都锁定到这些车型；留空=自动识别/车型不明确")
    p.add_argument("--rewrite", choices=["rule", "llm"], default="rule", help="改写方式（当前仅 rule）")
    p.add_argument("--verify", action="store_true", help="启用 Ollama 检索相关性核对（更准更慢）")
    p.add_argument("--min_score", type=float, default=0.45, help="弱检索向量相似度下限")
    p.add_argument("--min_chunks", type=int, default=2, help="有效 chunk 下限")
    p.add_argument("--max_attempts", type=int, default=0, help="最多处理候选数；0=count*8")
    p.add_argument("--csv_encoding", default="gb18030")
    p.add_argument("--config", default=str(ROOT / "config.yaml"))
    return p.parse_args()


def load_clean_questions(args) -> tuple[int, list[str]]:
    """读 CSV → 提取真实问题 → 去过短 → 关键词预筛 → 精确去重 → 按种子打散。

    近重复（near-dup）放到主循环里对已 accepted 增量判定，避免在 8 万条上做
    O(n²) 相似度比较导致卡死。
    """
    rows = load_rows(str(ROOT / args.source_csv) if not os.path.isabs(args.source_csv)
                     else args.source_csv, encoding=args.csv_encoding)
    total = len(rows)
    raw = [extract_input_text(r) for r in rows]
    raw = [q for q in raw if q and len(q) >= _MIN_Q_LEN]
    raw = [q for q in raw if any(k in q for k in KEEP_KEYWORDS)]
    seen: set[str] = set()
    pool: list[str] = []
    for q in raw:
        n = normalize(q)
        if n and n not in seen:
            seen.add(n)
            pool.append(q)
    rng = random.Random(args.seed)
    rng.shuffle(pool)
    return total, pool


def _pre_reject_reason(source_q: str) -> str:
    """检索前的高精度剔除；返回剔除原因，空串表示放行。"""
    if mentions_other_brand(source_q):
        return "明确提及其它汽车品牌/车型，非问界场景"
    if describes_specific_other_vehicle(source_q):
        return "描述具体年款/车辆，多为他品牌真实车况，无法干净迁移"
    if needs_field_diagnosis(source_q):
        return "依赖读码仪/拆解/反复维修等现场信息，手册无法支撑"
    if is_ice_specific(source_q):
        return "内燃机/柴油/传统传动专属特征，无法迁移到问界新能源车"
    if is_off_topic_intent(source_q):
        return "购车/选车/价格等元问题，非车主手册可答范围"
    return ""


def build_verify_fn(args):
    """按需返回一个 (question, docs, scores) -> relevant_count 的核对函数。"""
    if not args.verify:
        return None
    from retrieve.verifier import verify_chunks

    def _verify(question, docs, scores):
        kept = verify_chunks(question, docs, scores=scores, min_keep=0)
        return len(kept)

    return _verify


def main() -> None:
    args = parse_args()
    max_attempts = args.max_attempts or args.count * 8

    config = load_config(args.config)
    apply_overrides(config.raw, LIGHT_OVERRIDES)

    index_path = config.index_path()
    if not (index_path / "faiss").exists():
        print(f"索引不存在: {index_path}\n请先运行: python main.py build")
        sys.exit(1)

    retriever = Retriever(config, index_path)
    retriever.load()
    verify_fn = build_verify_fn(args)

    # 目标车型解析（指定后锁定改写前缀 + RAG 检索范围）
    force_ids: list[str] = []
    force_label = ""
    if args.models.strip():
        tokens = [t.strip() for t in args.models.split(",") if t.strip()]
        force_ids, labels, unresolved = resolve_models(tokens, config)
        if unresolved:
            print(f"无法识别车型: {unresolved}\n可选车型：")
            for name in available_models(config):
                print(f"  - {name}")
            sys.exit(1)
        force_label = "、".join(labels)
        print(f"锁定车型: {force_label} → {len(force_ids)} 个变体: {force_ids}")

    total_csv, questions = load_clean_questions(args)
    print(f"CSV 总问题数: {total_csv} | 清洗+去重后候选数: {len(questions)}")

    accepted: list[dict] = []
    rejected: list[dict] = []
    attempts = 0

    bar = tqdm(questions, desc="筛选", unit="条")
    for source_q in bar:
        if len(accepted) >= args.count or attempts >= max_attempts:
            break
        attempts += 1

        # 结构化高精度剔除（非问界 / 具体他车 / 现场诊断 / 内燃机专属），不进检索
        pre_reason = _pre_reject_reason(source_q)
        if pre_reason:
            rejected.append({
                "source_question": source_q, "input": source_q,
                "vehicle_scope": "非问界", "powertrain": "不明确",
                "task_type": classify_task_type(source_q),
                "answerability": "not_answerable",
                "reason": pre_reason,
                "retrieved": False, "rag_evidence_summary": "未检索",
            })
            bar.set_postfix(ok=len(accepted), rej=len(rejected))
            continue

        if force_ids:
            # 指定车型：powertrain/wrong_premise 仍由 infer_scope 依据车型名推断，
            # 但车型范围用简洁标签、且不再追问（车型已明确）。
            scope = infer_scope(source_q, source_q, force_ids, config)
            scope.vehicle_scope = force_label
            scope.needs_clarification = False
            input_text = rule_rewrite(source_q, scope, force=True)
        else:
            detected = detect_models(source_q, "", config)
            scope = infer_scope(source_q, source_q, detected, config)
            input_text = rule_rewrite(source_q, scope)
        if not input_text:
            continue

        # 近重复门控：与已入选问题高度雷同则跳过，保证 accepted 多样性
        if any(jaccard(source_q, c["source_question"]) >= _NEAR_DUP_TH for c in accepted):
            continue

        task_type = classify_task_type(source_q)
        tags = risk_tags(f"{source_q} {input_text}")
        fuel_only = is_fuel_car_only(f"{source_q} {input_text}")
        is_safety = task_type == "安全提醒" or bool(_SAFETY_TAGS & set(tags))

        try:
            result = retriever.retrieve_stateless(
                input_text, force_models=force_ids or None)
        except Exception as e:  # noqa: BLE001 — 单条失败不应中断整批
            rejected.append({
                "source_question": source_q, "input": input_text,
                "reason": f"检索异常: {e}", "retrieved": False,
            })
            bar.set_postfix(ok=len(accepted), rej=len(rejected))
            continue

        relevant_count = None
        if verify_fn and result.docs:
            try:
                relevant_count = verify_fn(input_text, result.docs, result.scores)
            except Exception:  # noqa: BLE001 — verify 失败则退回纯检索分判定
                relevant_count = None

        decision = judge(
            result.docs, result.scores,
            is_safety=is_safety,
            needs_clarification=scope.needs_clarification,
            wrong_premise=scope.wrong_premise,
            is_fuel_car_only=fuel_only,
            min_chunks=args.min_chunks, min_score=args.min_score,
            relevant_count=relevant_count,
        )

        if decision.accepted:
            accepted.append(build_candidate(
                len(accepted) + 1, source_q, input_text, scope, task_type, decision, tags))
        else:
            rejected.append(build_rejected(
                source_q, input_text, scope, task_type, decision, retrieved=len(result.docs) > 0))
        bar.set_postfix(ok=len(accepted), rej=len(rejected))
    bar.close()

    _write_json(ROOT / args.output, accepted)
    _write_json(ROOT / args.rejected_output, rejected)
    _print_stats(total_csv, len(questions), accepted, rejected, attempts, args)


def _write_json(path: Path, data: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _print_stats(total_csv, cleaned, accepted, rejected, attempts, args) -> None:
    print("\n" + "=" * 48)
    print(f"1. CSV 总问题数:        {total_csv}")
    print(f"2. 清洗后候选数:        {cleaned}")
    print(f"   实际处理候选数:      {attempts}")
    rag_ok = sum(1 for c in accepted if c["answerability"] != "not_answerable")
    print(f"3. RAG 可回答数量:      {rag_ok}")
    print(f"4. accepted 数量:       {len(accepted)}")
    print(f"5. rejected 数量:       {len(rejected)}")

    if len(accepted) < args.count:
        print(f"   ⚠ 未达目标 {args.count} 条：高质量可回答问题不足，"
              f"已处理 {attempts}/{cleaned} 候选。可调低 --min_score 或增大 --max_attempts。")

    print("6. task_type 分布:      " + _fmt(Counter(c["task_type"] for c in accepted)))
    print("7. vehicle_scope 分布:  " + _fmt(Counter(c["vehicle_scope"] for c in accepted)))
    print("   powertrain 分布:     " + _fmt(Counter(c["powertrain"] for c in accepted)))
    nc = sum(1 for c in accepted if c["answerability"] == "needs_clarification")
    print(f"8. needs_clarification: {nc}")
    print("   answerability 分布:  " + _fmt(Counter(c["answerability"] for c in accepted)))
    print("9. rejected 主要原因:")
    for reason, n in Counter(r.get("reason", "未知") for r in rejected).most_common(8):
        print(f"     {n:>4}  {reason}")
    print("=" * 48)
    print(f"accepted → {args.output}")
    print(f"rejected → {args.rejected_output}")


def _fmt(counter: Counter) -> str:
    return "  ".join(f"{k}:{v}" for k, v in counter.most_common())


if __name__ == "__main__":
    main()
