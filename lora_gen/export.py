"""导出 dataset / meta / rejected / report / train-dev split。"""
from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

from lora_gen.schema import Rejected, Sample, SampleMeta


def write_dataset(samples: list[Sample], path: Path) -> None:
    path.write_text(
        json.dumps([s.to_record() for s in samples], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_meta(metas: list[SampleMeta], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for m in metas:
            f.write(json.dumps(m.to_record(), ensure_ascii=False) + "\n")


def write_rejected(rejected: list[Rejected], path: Path) -> None:
    path.write_text(
        json.dumps([r.to_record() for r in rejected], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def stratified_split(
    pairs: list[tuple[Sample, str]], *, train_ratio: float, rng: random.Random
) -> tuple[list[Sample], list[Sample]]:
    by_tt: dict[str, list[Sample]] = {}
    for s, tt in pairs:
        by_tt.setdefault(tt, []).append(s)
    train: list[Sample] = []
    dev: list[Sample] = []
    for tt in sorted(by_tt):
        items = list(by_tt[tt])
        rng.shuffle(items)
        n_train = round(len(items) * train_ratio)
        train.extend(items[:n_train])
        dev.extend(items[n_train:])
    return train, dev


def build_report(
    *,
    accepted_tiers: list[str],
    task_types: list[str],
    vehicles: list[str],
    reject_reasons: list[str],
    corpus_fingerprint: str,
    backend: str,
    config_hash: str,
    manual_check_ratio: float,
) -> str:
    def fmt(counter: Counter) -> str:
        return "\n".join(f"- {k}: {v}" for k, v in sorted(counter.items()))

    accepted = len(accepted_tiers)
    rejected = len(reject_reasons)
    total = accepted + rejected
    acc_rate = f"{accepted / total:.1%}" if total else "n/a"
    lines = [
        "# LoRA 数据生成报告",
        "",
        f"- corpus 指纹: `{corpus_fingerprint}`",
        f"- backend: {backend}",
        f"- dataset_gen.yaml hash: `{config_hash}`",
        f"- accepted: {accepted} / rejected: {rejected} / 通过率: {acc_rate}",
        f"- 人工抽检比例: {manual_check_ratio:.0%}",
        "",
        "## accept_tier 分布",
        fmt(Counter(accepted_tiers)),
        "",
        "## task_type 分布（accepted）",
        fmt(Counter(task_types)),
        "",
        "## 车型分布（accepted）",
        fmt(Counter(vehicles)),
        "",
        "## 拒绝原因分布",
        fmt(Counter(reject_reasons)),
        "",
    ]
    return "\n".join(lines)
