"""车型 round-robin 采样计划：model_id → chunk → task_type。"""
from __future__ import annotations

import random
from dataclasses import dataclass

from lora_gen.chunks import Chunk, section_to_task_type


@dataclass
class PlanItem:
    model_id: str
    chunk_id: str
    section_path: str
    task_type: str


def build_plan(
    chunks_by_model: dict[str, list[Chunk]],
    *,
    target: int,
    per_vehicle_min: int,
    per_vehicle_max: int,
    vehicle_subset: list[str],
    rng: random.Random,
) -> list[PlanItem]:
    models = list(vehicle_subset) if vehicle_subset else sorted(chunks_by_model)
    pools: dict[str, list[Chunk]] = {m: list(chunks_by_model.get(m, [])) for m in models}
    for m in models:
        rng.shuffle(pools[m])

    counts = {m: 0 for m in models}
    cursor = {m: 0 for m in models}
    plan: list[PlanItem] = []

    progressed = True
    while len(plan) < target and progressed:
        progressed = False
        for m in models:
            if len(plan) >= target:
                break
            if counts[m] >= per_vehicle_max or cursor[m] >= len(pools[m]):
                continue
            ch = pools[m][cursor[m]]
            cursor[m] += 1
            counts[m] += 1
            plan.append(
                PlanItem(
                    model_id=m,
                    chunk_id=ch.chunk_id,
                    section_path=ch.section_path,
                    task_type=section_to_task_type(ch.section_path),
                )
            )
            progressed = True
    return plan


def under_min_models(plan: list[PlanItem], models: list[str], per_vehicle_min: int) -> list[str]:
    """返回未达 per_vehicle_min 的车型，供 report 警示。"""
    from collections import Counter

    counts = Counter(p.model_id for p in plan)
    return [m for m in models if counts.get(m, 0) < per_vehicle_min]
