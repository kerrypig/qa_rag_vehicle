"""质检拒绝规则（纯函数）+ run_quality 编排。"""
from __future__ import annotations

import re
from dataclasses import dataclass

from retrieve.model_router import detect_models
from lora_gen.schema import Sample

_LEAK_PATTERNS = [
    r"根据手册[，,]?", r"根据(上述|以上|资料|内容)[，,]?",
    r"作为(一个)?\s*AI[^，。,.\n]*[，,]?", r"作为(智能)?助手[，,]?",
    r"参考第?\s*\d+\s*页", r"见第?\s*\d+\s*(章|节|页)", r"\[\d+\]",
]
_LEAK_RE = [re.compile(p) for p in _LEAK_PATTERNS]

_NUM_UNIT = re.compile(
    r"\d+(?:\.\d+)?\s*(?:kWh|kW|km/h|km|kPa|MPa|bar|V|A|Nm|N·m|℃|°C|%|升|L|公里|千米|巴|伏|安|牛·米|分钟|小时|秒)",
    re.I,
)
_OVER_PROMISE = ["免费", "一定", "永久", "保证", "保险全赔", "绝对", "100%"]
_INSURANCE_TERMS = ["保险", "质保", "三包"]
_DANGER = ["高压", "动力电池", "电池起火", "救援", "拖车", "起火", "触电"]
_GUARD = ["授权", "服务中心", "专业人员", "售后", "客服", "维修站"]


@dataclass
class QualityVerdict:
    ok: bool
    reason: str
    detail: str
    cleaned_output: str


def strip_leakage(text: str) -> str:
    out = text
    for rx in _LEAK_RE:
        out = rx.sub("", out)
    return out.strip()


def ungrounded_numbers(output: str, evidence: str) -> list[str]:
    ev = evidence.replace(" ", "")
    bad: list[str] = []
    for m in _NUM_UNIT.finditer(output):
        token = m.group(0).replace(" ", "")
        if token in ev:
            continue
        num = re.match(r"\d+(?:\.\d+)?", token).group(0)
        if num not in ev:
            bad.append(token)
    return bad


def has_over_promise(output: str, evidence: str) -> list[str]:
    return [w for w in _OVER_PROMISE if w in output and w not in evidence]


def insurance_warranty_mix(output: str, evidence: str) -> bool:
    return any(t in output and t not in evidence for t in _INSURANCE_TERMS)


def unsafe_without_guard(output: str) -> bool:
    if any(d in output for d in _DANGER):
        return not any(g in output for g in _GUARD)
    return False


def foreign_vehicles(text: str, model_id: str, config) -> list[str]:
    # detect_models 已把别名/展示名归一化为 model_id；故与 bound id 比对即可，
    # 绑定车型的别名会解析回同一 id，不会误判为 foreign。
    detected = detect_models(text, "", config)
    return [m for m in detected if m != model_id]


def normalize_question(q: str) -> str:
    """去空白 + 全角标点折叠 + 小写，用于 normalized exact 去重。"""
    s = re.sub(r"\s+", "", q).lower()
    trans = str.maketrans("？！，。：；（）", "?!,.:;()")
    return s.translate(trans)


def run_quality(
    sample: Sample, *, evidence_text: str, model_id: str, config, max_chars: int
) -> QualityVerdict:
    cleaned = strip_leakage(sample.output)

    foreign = foreign_vehicles(sample.input, model_id, config) + foreign_vehicles(
        cleaned, model_id, config
    )
    if foreign:
        return QualityVerdict(False, "vehicle_conflict", f"foreign={foreign}", cleaned)

    bad_nums = ungrounded_numbers(cleaned, evidence_text)
    if bad_nums:
        return QualityVerdict(False, "ungrounded_number", f"nums={bad_nums}", cleaned)

    if insurance_warranty_mix(cleaned, evidence_text):
        return QualityVerdict(False, "insurance_warranty_mix", "", cleaned)

    if unsafe_without_guard(cleaned):
        return QualityVerdict(False, "unsafe_danger_advice", "", cleaned)

    promises = has_over_promise(cleaned, evidence_text)
    if promises:
        return QualityVerdict(False, "over_promise", f"words={promises}", cleaned)

    if not cleaned or not sample.input.strip() or not sample.instruction.strip():
        return QualityVerdict(False, "field_incomplete", "", cleaned)

    if len(cleaned) > max_chars:
        return QualityVerdict(False, "length_out_of_bounds", f"len={len(cleaned)}", cleaned)

    return QualityVerdict(True, "", "", cleaned)
