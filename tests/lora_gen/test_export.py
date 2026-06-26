import json
import random
from lora_gen.schema import Sample
from lora_gen.export import stratified_split, build_report

def test_stratified_split_ratio():
    pairs = [(Sample("i", f"q{i}", "a"), "直接问答") for i in range(8)] + \
            [(Sample("i", f"s{i}", "a"), "步骤指导") for i in range(2)]
    train, dev = stratified_split(pairs, train_ratio=0.5, rng=random.Random(1))
    # 每类按比例：直接问答 8→4/4，步骤指导 2→1/1
    assert len(train) == 5 and len(dev) == 5

def test_build_report_counts():
    accepted_tiers = ["strong", "strong", "ok", "partial_ok"]
    task_types = ["直接问答", "步骤指导", "故障分析", "直接问答"]
    vehicles = ["A", "A", "B", "B"]
    reject_reasons = ["seed_not_returned", "low_score", "seed_not_returned"]
    md = build_report(
        accepted_tiers=accepted_tiers, task_types=task_types, vehicles=vehicles,
        reject_reasons=reject_reasons, corpus_fingerprint="fp", backend="cloud",
        config_hash="abc123", manual_check_ratio=0.1,
    )
    assert "seed_not_returned" in md and "strong" in md
    assert "cloud" in md and "abc123" in md
