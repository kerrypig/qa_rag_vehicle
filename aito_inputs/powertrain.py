"""车型范围 / 动力形式推断与门控规则（纯逻辑，无 I/O）。

依据任务要求：
- 加油/汽油/增程器/机油/冷却液/燃油补电/OBD 等燃油话题 → 只能落到「增程」，
  或车型不明确时标记需追问。
- 纯电话题不混入燃油内容，除非用于「纠正错误前提」（前提错误样本单独识别）。
- 高压/儿童安全/事故/辅助驾驶等 → AITO 通用。
- 只说「我的问界车/我的M9/这车」且答案因车型而异 → 标记需追问，不强行指定车型。
"""
from __future__ import annotations

from dataclasses import dataclass, field

# 燃油/增程专属话题（纯电车不应涉及）
FUEL_TOPIC_KEYWORDS: tuple[str, ...] = (
    "加油", "汽油", "95号", "92号", "98号", "号油", "机油", "增程器", "增程",
    "燃油", "补电", "油箱", "加注", "obd", "排放", "尾气", "积碳",
)
# 纯电明确标识
PURE_EV_KEYWORDS: tuple[str, ...] = ("纯电", "电动版", "ev")
# 增程明确标识
RANGE_EXT_KEYWORDS: tuple[str, ...] = ("增程",)
# AITO 通用安全/功能话题（与动力形式无关）
GENERIC_TOPIC_KEYWORDS: tuple[str, ...] = (
    "气囊", "高压", "事故", "碰撞", "起火", "儿童", "安全座椅", "辅助驾驶",
    "ads", "智驾", "领航", "雷达", "钥匙", "车门", "车窗", "空调", "座椅",
)
# 答案可能因车型/年款不同而不同的「易歧义」话题 → 车型不明时应追问
VARIANT_SENSITIVE_KEYWORDS: tuple[str, ...] = (
    "充电上限", "充满", "充电设置", "续航", "保养周期", "首保", "里程",
    "质保", "三包", "胎压", "几个气囊", "选装", "配置",
)

# 传统燃油车专属零件/系统 + 天然气燃料：增程器是发电机、无变速箱/离合器/正时/火花塞，
# 纯电更无；问界全系无 CNG。命中这些 → 无法合理迁移到问界新能源车，应剔除。
FUEL_CAR_ONLY_KEYWORDS: tuple[str, ...] = (
    "变速箱", "变速器", "正时皮带", "正时链条", "火花塞", "离合器", "三元催化",
    "排气管", "喷油嘴", "气缸", "缸线", "双离合", "at变速", "cvt", "档位顿挫",
    "烧气", "天然气", "cng", "油改气", "双燃料", "化油器", "节气门", "kdss",
)

# 其它汽车品牌/车型（用于识别明显非问界的真实问题）。
# 仅收录多字、不会成为技术名词子串的安全 token（避免如 "cc" 命中 "ACC"）。
OTHER_BRAND_KEYWORDS: tuple[str, ...] = (
    # 品牌
    "别克", "大众", "丰田", "本田", "奥迪", "宝马", "奔驰", "福特", "日产", "现代",
    "起亚", "雪佛兰", "标致", "马自达", "比亚迪", "特斯拉", "蔚来", "理想", "小鹏",
    "吉利", "长安", "哈弗", "五菱", "铃木", "斯柯达", "雪铁龙", "荣威", "名爵",
    "奇瑞", "长城", "宝骏", "领克", "沃尔沃", "捷豹", "路虎", "凯迪拉克",
    # 常见燃油车型名
    "凯越", "英朗", "君威", "君越", "速腾", "朗逸", "帕萨特", "桑塔纳", "普桑", "途观",
    "卡罗拉", "凯美瑞", "雅阁", "思域", "轩逸", "天籁", "普拉多", "汉兰达", "威驰",
    "速锐", "天语", "宝来", "捷达", "迈腾", "高尔夫", "蒙迪欧", "福克斯", "科鲁兹",
    "迈锐宝", "雅特", "途胜", "奥拓", "远景", "赛拉图", "帕杰罗", "威虎", "夏利",
    "切诺基", "乐风", "polo", "宝马x", "奥迪a", "奥迪q", "3系", "5系", "7系",
)


def is_fuel_car_only(text: str) -> bool:
    """命中传统燃油车专属零件/系统（增程/纯电均无）→ 不可迁移。"""
    return _has(text, FUEL_CAR_ONLY_KEYWORDS)


def mentions_other_brand(text: str) -> bool:
    """问题明显提到其它汽车品牌/车型。"""
    return _has(text, OTHER_BRAND_KEYWORDS)


@dataclass
class ScopeResult:
    vehicle_scope: str          # 具体车型名 / "车型不明确" / "AITO通用"
    powertrain: str             # "增程" / "纯电" / "通用" / "不明确"
    needs_clarification: bool   # 是否应先追问车型/动力形式
    wrong_premise: bool = False  # 是否疑似错误前提（如纯电问加油）
    notes: list[str] = field(default_factory=list)


def _has(text: str, kws: tuple[str, ...]) -> bool:
    low = text.lower()
    return any(k in text or k in low for k in kws)


def infer_scope(
    question: str,
    rewritten: str,
    detected_models: list[str],
    config,
) -> ScopeResult:
    """综合原问题、改写问句、车型识别结果，推断范围/动力形式与是否需追问。"""
    text = f"{question} {rewritten}"
    is_fuel = _has(text, FUEL_TOPIC_KEYWORDS)
    is_generic = _has(text, GENERIC_TOPIC_KEYWORDS)
    is_variant_sensitive = _has(text, VARIANT_SENSITIVE_KEYWORDS)

    # 1) 已识别到具体车型
    if detected_models:
        names = [config.model_display(m) for m in detected_models]
        scope = "、".join(names)
        low = text.lower()
        explicit_pure = any("纯电" in n for n in names) or _has(text, PURE_EV_KEYWORDS)
        explicit_range = any("增程" in n for n in names) or "增程" in text
        if explicit_pure and not explicit_range:
            powertrain = "纯电"
            # 纯电 + 燃油话题 → 错误前提（可用于训练纠正）
            wrong = is_fuel
            notes = ["纯电车型出现燃油话题，疑似错误前提"] if wrong else []
            return ScopeResult(scope, powertrain, needs_clarification=False,
                               wrong_premise=wrong, notes=notes)
        if explicit_range:
            return ScopeResult(scope, "增程", needs_clarification=False)
        # 车型已知但未点明动力形式
        pt = "增程" if is_fuel else "通用"
        return ScopeResult(scope, pt, needs_clarification=False)

    # 2) 车型不明确
    if is_fuel:
        # 燃油话题但车型不明 → 倾向增程，但需追问确认（增程/纯电差异大）
        return ScopeResult("车型不明确", "增程", needs_clarification=True,
                           notes=["燃油/增程话题但未指明车型，需确认是否增程版"])
    if is_variant_sensitive:
        return ScopeResult("车型不明确", "不明确", needs_clarification=True,
                           notes=["答案因车型/年款而异，需先确认车型"])
    if is_generic:
        return ScopeResult("AITO通用", "通用", needs_clarification=False)
    # 其余：车型不明，话题通用度未知 → 标记不明确，由 RAG 决定可回答性
    return ScopeResult("车型不明确", "不明确", needs_clarification=False)
