import random
from lora_gen.chunks import Chunk
from lora_gen.registry import PlanItem, build_plan

def _mk(model, n, section="车辆控制>空调"):
    return [Chunk(f"{model}-c{i}", "x" * 300, model, "owner_manual", section, 1) for i in range(n)]

def test_round_robin_balances_and_respects_max():
    by_model = {"A": _mk("A", 20), "B": _mk("B", 20), "C": _mk("C", 20)}
    plan = build_plan(by_model, target=12, per_vehicle_min=1, per_vehicle_max=10,
                      vehicle_subset=[], rng=random.Random(1))
    assert len(plan) == 12
    from collections import Counter
    counts = Counter(p.model_id for p in plan)
    # 12 / 3 模型 → 每个 4 条，均衡
    assert counts == {"A": 4, "B": 4, "C": 4}

def test_per_vehicle_max_caps():
    by_model = {"A": _mk("A", 20), "B": _mk("B", 2)}
    plan = build_plan(by_model, target=100, per_vehicle_min=1, per_vehicle_max=5,
                      vehicle_subset=[], rng=random.Random(1))
    from collections import Counter
    counts = Counter(p.model_id for p in plan)
    assert counts["A"] == 5  # 受 max 限制
    assert counts["B"] == 2  # chunk 不足

def test_subset_filters_models():
    by_model = {"A": _mk("A", 10), "B": _mk("B", 10)}
    plan = build_plan(by_model, target=6, per_vehicle_min=1, per_vehicle_max=10,
                      vehicle_subset=["A"], rng=random.Random(1))
    assert {p.model_id for p in plan} == {"A"}

def test_task_type_assigned_from_section():
    by_model = {"A": _mk("A", 4, section="维护保养>更换雨刮")}
    plan = build_plan(by_model, target=2, per_vehicle_min=1, per_vehicle_max=10,
                      vehicle_subset=[], rng=random.Random(1))
    assert all(p.task_type == "步骤指导" for p in plan)
