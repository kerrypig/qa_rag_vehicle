from config_loader import load_config
from lora_gen.schema import Sample
from lora_gen.quality import (
    strip_leakage, ungrounded_numbers, has_over_promise,
    insurance_warranty_mix, unsafe_without_guard, foreign_vehicles, run_quality,
)

CONFIG = load_config()

def test_strip_leakage():
    out = strip_leakage("根据手册，空调请按AUTO键。参考第12页。[1]")
    assert "根据手册" not in out and "第12页" not in out and "[1]" not in out
    assert "空调请按AUTO键" in out

def test_ungrounded_numbers():
    ev = "胎压建议为 2.5 bar。"
    assert ungrounded_numbers("请保持 2.5 bar 胎压", ev) == []
    assert "3.0bar" in [x.replace(" ", "") for x in ungrounded_numbers("请保持 3.0 bar 胎压", ev)]

def test_over_promise():
    assert set(has_over_promise("该服务永久免费", "服务说明")) == {"永久", "免费"}
    assert has_over_promise("正常保养即可", "保养说明") == []

def test_insurance_warranty_mix():
    assert insurance_warranty_mix("此故障保险全赔", "电池故障说明") is True
    assert insurance_warranty_mix("请检查电池", "电池故障说明") is False

def test_unsafe_without_guard():
    assert unsafe_without_guard("动力电池起火时请自行拆解处理") is True
    assert unsafe_without_guard("动力电池起火时请立即联系授权服务中心") is False

def test_foreign_vehicles_detected():
    # 绑定 M9-2026 增程版，但 output 提到 M7 → 冲突
    foreign = foreign_vehicles("问界M7 2026增程版也支持", "问界M9-2026款增程版", CONFIG)
    assert foreign  # 非空

def test_foreign_vehicles_bound_alias_not_flagged():
    # 用别名提到绑定车型本身 → detect_models 解析回同一 id → 不算 foreign
    foreign = foreign_vehicles("M92026增程版也支持该功能", "问界M9-2026款增程版", CONFIG)
    assert foreign == []

def test_normalize_question_dedup_equivalence():
    from lora_gen.quality import normalize_question
    assert normalize_question(" 问界M9 空调 怎么 开？") == normalize_question("问界M9空调怎么开?")

def test_run_quality_vehicle_conflict_checks_input_and_output():
    s = Sample(instruction="i", input="问界M7 2026增程版怎么样", output="正常使用即可")
    v = run_quality(s, evidence_text="正常使用即可", model_id="问界M9-2026款增程版",
                    config=CONFIG, max_chars=400)
    assert not v.ok and v.reason == "vehicle_conflict"

def test_run_quality_pass_returns_cleaned():
    s = Sample(instruction="i", input="问界M9 2026增程版空调怎么开",
               output="根据手册，按下AUTO键即可。")
    v = run_quality(s, evidence_text="按下AUTO键即可开启空调", model_id="问界M9-2026款增程版",
                    config=CONFIG, max_chars=400)
    assert v.ok
    assert "根据手册" not in v.cleaned_output
