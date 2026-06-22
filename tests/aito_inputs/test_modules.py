"""aito_inputs 纯逻辑模块单测。"""
from __future__ import annotations

from aito_inputs.answerability import Decision, judge, max_score, summarize_evidence
from aito_inputs.candidate import build_candidate, build_rejected
from aito_inputs.classify import classify_task_type, risk_tags
from aito_inputs.dedup import dedup_questions, jaccard, normalize
from aito_inputs.models import resolve_models
from aito_inputs.filters import (
    describes_specific_other_vehicle,
    is_ice_specific,
    is_off_topic_intent,
    needs_field_diagnosis,
)
from aito_inputs.powertrain import (
    ScopeResult,
    infer_scope,
    is_fuel_car_only,
    mentions_other_brand,
)
from aito_inputs.rewrite import rule_rewrite


# ---- 测试替身 ----
_MODELS = [
    {"id": "问界M7-2026款增程版", "name": "问界M7 2026款增程版", "aliases": ["M72026款增程版", "M7 2026增程版"]},
    {"id": "问界M7-2026款纯电版", "name": "问界M7 2026款纯电版", "aliases": ["M72026款纯电版"]},
    {"id": "问界M7-Max智驾版", "name": "问界M7 Max智驾版", "aliases": ["M7Max"]},
    {"id": "问界M9-2025款纯电版", "name": "问界M9 2025款纯电版", "aliases": ["M92025款纯电版"]},
    {"id": "问界M9纯电版", "name": "问界M9纯电版", "aliases": ["M9纯电"]},
    {"id": "问界M8增程版", "name": "问界M8增程版", "aliases": ["M8增程"]},
]


class FakeConfig:
    """最小 config：infer_scope 用 model_display，resolve_models 用 models。"""

    @property
    def models(self) -> list[dict]:
        return _MODELS

    def model_display(self, mid: str) -> str:
        for m in _MODELS:
            if m["id"] == mid:
                return m.get("name", mid)
        return mid


class Doc:
    def __init__(self, section_path: str):
        self.metadata = {"section_path": section_path}


# ---- classify ----
def test_task_type_safety_first():
    assert classify_task_type("安全气囊灯亮了") == "安全提醒"


def test_task_type_fault():
    assert classify_task_type("发动机故障灯亮了还能开吗") == "故障分析"


def test_task_type_steps():
    assert classify_task_type("怎么连接车机蓝牙") == "步骤指导"


def test_task_type_term():
    assert classify_task_type("ADS是什么意思") == "术语解释"


def test_task_type_default():
    assert classify_task_type("今天心情不错") == "直接问答"


def test_risk_tags():
    tags = risk_tags("充电枪拔不出来")
    assert "充电" in tags


# ---- powertrain ----
def test_infer_fuel_unknown_model_needs_clarify():
    r = infer_scope("加多少号油", "", [], FakeConfig())
    assert r.powertrain == "增程" and r.needs_clarification is True


def test_infer_variant_sensitive_needs_clarify():
    r = infer_scope("充电上限设多少合适", "", [], FakeConfig())
    assert r.needs_clarification is True


def test_infer_generic_topic():
    r = infer_scope("安全气囊怎么回事", "", [], FakeConfig())
    assert r.vehicle_scope == "AITO通用" and r.needs_clarification is False


def test_infer_pure_ev_wrong_premise():
    r = infer_scope("是不是也要加95号油", "", ["问界M9纯电版"], FakeConfig())
    assert r.powertrain == "纯电" and r.wrong_premise is True


def test_infer_range_extender():
    r = infer_scope("增程器异响", "", ["问界M9增程版"], FakeConfig())
    assert r.powertrain == "增程" and r.needs_clarification is False


def test_fuel_car_only():
    assert is_fuel_car_only("变速箱顿挫") is True
    assert is_fuel_car_only("胎压低") is False


def test_other_brand():
    assert mentions_other_brand("我的别克凯越") is True
    assert mentions_other_brand("我的问界M9") is False


# ---- models ----
def test_resolve_family_m7():
    ids, labels, unresolved = resolve_models(["M7"], FakeConfig())
    assert set(ids) == {"问界M7-2026款增程版", "问界M7-2026款纯电版", "问界M7-Max智驾版"}
    assert labels == ["问界M7"] and unresolved == []


def test_resolve_m9_pure_electric():
    ids, labels, unresolved = resolve_models(["M9纯电"], FakeConfig())
    assert set(ids) == {"问界M9-2025款纯电版", "问界M9纯电版"}
    assert unresolved == []


def test_resolve_multiple_tokens():
    ids, labels, _ = resolve_models(["M8增程", "M9纯电"], FakeConfig())
    assert "问界M8增程版" in ids and "问界M9纯电版" in ids
    assert labels == ["问界M8增程", "问界M9纯电"]


def test_resolve_unresolved():
    ids, labels, unresolved = resolve_models(["M3"], FakeConfig())
    assert ids == [] and unresolved == ["M3"]


def test_resolve_keeps_aito_prefix_label():
    _, labels, _ = resolve_models(["问界M9纯电版"], FakeConfig())
    assert labels == ["问界M9纯电版"]


# ---- rewrite ----
def test_rewrite_specific_model_prefix():
    scope = ScopeResult("问界M9增程版", "增程", False)
    out = rule_rewrite("发动机故障灯亮了还能开吗", scope)
    assert out.startswith("我的问界M9增程版，")
    assert "故障灯" in out


def test_rewrite_unknown_model_prefix():
    scope = ScopeResult("车型不明确", "不明确", True)
    out = rule_rewrite("充电上限设多少", scope)
    assert out.startswith("我的问界车，")


def test_rewrite_keeps_already_aito():
    scope = ScopeResult("AITO通用", "通用", False)
    out = rule_rewrite("问界M9充电怎么看是否在充", scope)
    assert out.startswith("问界M9")  # 已自带品牌，不再加前缀


def test_rewrite_strips_other_brand_prefix():
    scope = ScopeResult("车型不明确", "不明确", False)
    out = rule_rewrite("10年凯越发动机故障灯亮了", scope)
    assert "凯越" not in out
    assert "故障灯" in out


def test_rewrite_empty():
    assert rule_rewrite("   ", ScopeResult("车型不明确", "不明确", False)) == ""


def test_rewrite_force_overrides_already_aito():
    scope = ScopeResult("问界M7", "增程", False)
    out = rule_rewrite("M9充电怎么看是否在充", scope, force=True)
    assert out.startswith("我的问界M7，")


# ---- dedup ----
def test_dedup_exact():
    out = dedup_questions(["胎压灯亮了怎么办", "胎压灯亮了怎么办", "蓝牙怎么连"])
    assert len(out) == 2


def test_dedup_near():
    out = dedup_questions(["胎压灯亮了怎么办", "胎压灯亮了怎么办呀"], threshold=0.85)
    assert len(out) == 1


def test_jaccard_identical():
    assert jaccard("充电", "充电") == 1.0


def test_normalize_strips_punct():
    assert normalize("胎压，灯亮了？") == "胎压灯亮了"


# ---- filters ----
def test_describes_specific_vehicle():
    assert describes_specific_other_vehicle("11年A6l空调出热风") is True
    assert describes_specific_other_vehicle("18年款的新远景后视镜灯") is True
    assert describes_specific_other_vehicle("2013年12月生产的车转向异响") is True
    assert describes_specific_other_vehicle("胎压灯亮了还能开吗") is False


def test_is_ice_specific():
    assert is_ice_specific("途观2.0T更换水泵") is True
    assert is_ice_specific("夏利三缸车速一过60嗡嗡响") is True
    assert is_ice_specific("断开预热塞供电故障灯还亮") is True
    assert is_ice_specific("空调不制冷怎么办") is False


def test_is_ice_specific_drivetrain_terms():
    assert is_ice_specific("差速器和内球笼异响") is True
    assert is_ice_specific("节温器和防冻液问题") is True
    assert is_ice_specific("胎压灯亮了") is False


def test_needs_field_diagnosis():
    assert needs_field_diagnosis("431读取52副驾驶侧故障码") is True
    assert needs_field_diagnosis("拆解后发现电阻值异常") is True
    assert needs_field_diagnosis("助力泵都换了还是没解决") is True
    assert needs_field_diagnosis("故障灯亮了还能继续开吗") is False


def test_is_off_topic_intent():
    assert is_off_topic_intent("6万左右买什么车好呢") is True
    assert is_off_topic_intent("这是哪款车的灯") is True
    assert is_off_topic_intent("胎压灯亮了还能开吗") is False


# ---- answerability ----
def test_judge_fuel_car_only_rejected():
    d = judge([Doc("x")], {"a": 0.9}, is_safety=False, needs_clarification=False,
              wrong_premise=False, is_fuel_car_only=True, min_chunks=2, min_score=0.45)
    assert d.accepted is False and d.answerability == "not_answerable"


def test_judge_no_evidence_rejected():
    d = judge([], {}, is_safety=False, needs_clarification=False, wrong_premise=False,
              is_fuel_car_only=False, min_chunks=2, min_score=0.45)
    assert d.accepted is False and "检索不到" in d.reason


def test_judge_strong_answerable():
    docs = [Doc("充电>充电口"), Doc("充电>故障")]
    d = judge(docs, {"a": 0.6, "b": 0.55}, is_safety=False, needs_clarification=False,
              wrong_premise=False, is_fuel_car_only=False, min_chunks=2, min_score=0.45)
    assert d.accepted and d.answerability == "answerable_by_rag" and d.expected_behavior == "直接回答"


def test_judge_safety_fallback():
    docs = [Doc("故障救援>安全气囊")]
    d = judge(docs, {"a": 0.40}, is_safety=True, needs_clarification=False,
              wrong_premise=False, is_fuel_car_only=False, min_chunks=2, min_score=0.45)
    assert d.accepted and d.answerability == "safety_fallback_supported"


def test_judge_needs_clarification():
    docs = [Doc("充电>设置")]
    d = judge(docs, {"a": 0.40}, is_safety=False, needs_clarification=True,
              wrong_premise=False, is_fuel_car_only=False, min_chunks=2, min_score=0.45)
    assert d.accepted and d.answerability == "needs_clarification"


def test_judge_wrong_premise():
    docs = [Doc("增程>加油"), Doc("纯电>充电")]
    d = judge(docs, {"a": 0.6, "b": 0.5}, is_safety=False, needs_clarification=False,
              wrong_premise=True, is_fuel_car_only=False, min_chunks=2, min_score=0.45)
    assert d.accepted and d.expected_behavior == "纠正错误前提"


def test_judge_low_score_rejected():
    d = judge([Doc("x")], {"a": 0.2}, is_safety=False, needs_clarification=False,
              wrong_premise=False, is_fuel_car_only=False, min_chunks=2, min_score=0.45)
    assert d.accepted is False


def test_judge_relevant_count_zero_rejects_strong():
    docs = [Doc("a"), Doc("b")]
    d = judge(docs, {"a": 0.6, "b": 0.5}, is_safety=False, needs_clarification=False,
              wrong_premise=False, is_fuel_car_only=False, min_chunks=2, min_score=0.45,
              relevant_count=0)
    assert d.accepted is False


def test_summarize_evidence():
    assert "充电" in summarize_evidence([Doc("充电>充电口")])
    assert summarize_evidence([]) == "未检索到相关章节"


def test_max_score_empty():
    assert max_score({}) == 0.0


# ---- candidate ----
def test_build_candidate_schema():
    scope = ScopeResult("问界M9增程版", "增程", False)
    dec = Decision(True, "answerable_by_rag", "直接回答", "", "检索到手册章节：充电>充电口")
    c = build_candidate(1, "充电口红灯", "我的问界M9增程版，充电口红灯？", scope,
                        "故障分析", dec, ["充电", "故障灯"])
    assert c["id"] == "input_001"
    assert set(c) == {"id", "source_question", "input", "vehicle_scope", "powertrain",
                      "task_type", "answerability", "expected_behavior",
                      "rag_evidence_summary", "risk_tags"}
    assert "output" not in c and "answer" not in c


def test_build_rejected_schema():
    scope = ScopeResult("车型不明确", "不明确", False)
    dec = Decision(False, "not_answerable", "", "RAG 检索不到足够相关依据", "未检索到相关章节")
    r = build_rejected("变速箱异响", "我的问界车，变速箱异响？", scope, "故障分析", dec, False)
    assert r["reason"] == "RAG 检索不到足够相关依据"
    assert r["retrieved"] is False
