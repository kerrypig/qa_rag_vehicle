"""规则模板「轻度」改写（纯逻辑，无 I/O / 无 LLM）。

原则：保留原问题的措辞与核心意图，只做最小处理——
- 清洗空白、去掉对话噪声、补全问号；
- 若原句未点明问界/车型，自然地加上车型或「问界车」前缀（真实车主口吻）；
- 绝不扩写成不同的复杂问题（这是任务明确禁止的）。

llm 改写留作后续扩展（本版本仅 rule）。
"""
from __future__ import annotations

import re

from aito_inputs.powertrain import OTHER_BRAND_KEYWORDS, ScopeResult

# 原句已自带问界/车型标识时不再加前缀
_ALREADY_AITO = ("问界", "aito", "m5", "m7", "m8", "m9", "赛力斯")
# 原句已是第一人称「我的车/我的…」开头，避免叠加「我的」
_FIRST_PERSON_PREFIX = ("我的", "我家", "我车", "我想问", "我", "咱")


def _clean(question: str) -> str:
    """归一空白、去掉常见对话残留、补全句末问号。"""
    q = re.sub(r"\s+", "", str(question or "")).strip()
    q = q.strip("。.!！,，、 ")
    if not q:
        return ""
    if not q.endswith(("？", "?", "。", "！")):
        q += "？"
    return q


def _strip_other_brand_prefix(q: str) -> str:
    """去掉句首的「10年凯越」「我的别克」之类其它品牌限定，保留核心问题。"""
    for brand in OTHER_BRAND_KEYWORDS:
        idx = q.find(brand)
        if idx != -1 and idx <= 6:  # 仅处理靠句首的品牌限定
            tail = q[idx + len(brand):]
            tail = tail.lstrip("，,。 的款年")
            if len(tail) >= 4:
                return tail
    return q


def rule_rewrite(question: str, scope: ScopeResult, *, force: bool = False) -> str:
    """把真实问题轻度改写为问界/AITO 场景下的自然问句。

    force=True 时（指定了目标车型），无论原句是否已含品牌，都强制加上
    scope.vehicle_scope 作为车型前缀（如「我的问界M7，…」）。
    """
    q = _clean(question)
    if not q:
        return ""

    low = q.lower()
    # 句首其它品牌限定 → 去掉，换成问界语境
    q = _strip_other_brand_prefix(q)
    low = q.lower()

    if not force and any(tok in low for tok in _ALREADY_AITO):
        return q  # 已自带品牌/车型，保持原貌

    # 构造车型前缀
    if scope.vehicle_scope not in ("车型不明确", "AITO通用", ""):
        prefix = f"我的{scope.vehicle_scope}"
    else:
        prefix = "我的问界车"

    # 已是第一人称开头则用「，」轻接，否则直接前缀
    if any(q.startswith(p) for p in _FIRST_PERSON_PREFIX):
        # 去掉原句开头的「我/我的」，避免「我的问界车我的…」
        body = re.sub(r"^(我的|我家|我车|我想问|我|咱)+", "", q).lstrip("，,的 ")
        return f"{prefix}，{body}" if body else f"{prefix}，{q}"
    return f"{prefix}，{q}"
