"""证据覆盖判定（纯逻辑）：检索内容是否真的命中问题的核心主题。

解决「检索到东西就判 answerable」的误判：把问题归入若干主题家族（轮胎/空调/
充电…），只有当检索 chunk 的章节路径或正文里出现该家族的词，且至少一条来自
非「前言/随车工具/SOS」等泛章节时，才认为证据真正覆盖了问题核心。
"""
from __future__ import annotations

import re

# 主题家族 → 同义/相关词集合（命中其一即视为该主题被覆盖）
FAMILIES: dict[str, tuple[str, ...]] = {
    "轮胎": ("轮胎", "胎压", "动平衡", "四轮定位", "换胎", "备胎", "轮毂", "胎噪", "车轮", "花纹", "胎面"),
    "空调": ("空调", "制冷", "制热", "压缩机", "暖风", "冷风", "出风", "滤芯", "除雾", "鼓风机", "雪种", "冷媒", "空气净化"),
    "充电": ("充电", "充电枪", "充电口", "充电桩", "快充", "慢充", "放电", "补能", "充电盖"),
    "电池电瓶": ("蓄电池", "电瓶", "电量", "续航", "亏电", "低压", "充满", "充电上限"),
    "钥匙门锁": ("钥匙", "解锁", "落锁", "门锁", "无钥匙", "遥控", "钥匙卡"),
    "车门车窗": ("车门", "车窗", "后备箱", "尾门", "天窗", "后视镜", "玻璃", "电动尾门"),
    "灯光": ("大灯", "车灯", "车外灯", "转向灯", "示宽灯", "雾灯", "灯罩", "灯泡", "远光", "近光", "氛围灯"),
    "仪表故障灯": ("仪表", "故障灯", "指示灯", "警告灯", "报警灯", "胎压灯"),
    "刹车制动": ("刹车", "制动", "刹车盘", "刹车片", "驻车", "手刹", "abs"),
    "辅助驾驶": ("辅助驾驶", "智驾", "领航", "ads", "lcc", "nca", "acc", "自动泊车", "雷达", "摄像头", "巡航"),
    "增程": ("增程", "增程器", "机油", "冷却液", "油箱", "加油", "燃油", "补电"),
    "座椅安全": ("座椅", "安全带", "气囊", "儿童", "安全座椅", "isofix"),
    "保养质保": ("保养", "质保", "三包", "保修", "首保", "保养周期"),
    "保险": ("保险", "理赔", "出险", "赔付", "保单", "交强险"),
    "拖车": ("拖车", "牵引", "拖挂", "脱困"),
    "车机娱乐": ("车机", "中控", "导航", "蓝牙", "音响", "音效", "屏幕", "ota", "升级"),
}

# 泛章节标识（归一后子串匹配）：这些只算「泛泛相关」，不足以单独支撑具体回答
_GENERIC_MARKERS: tuple[str, ...] = (
    "前言", "随车工具", "驾驶设置", "主动安全", "sos", "应急救援", "重要提示",
    "整车参数", "储物空间", "音效", "情景智能", "目录", "保修登记",
)


def _norm(s: str) -> str:
    return re.sub(r"\s+", "", str(s or "")).lower()


def core_families(question: str) -> list[str]:
    """问题命中的主题家族列表。"""
    q = _norm(question)
    return [fam for fam, terms in FAMILIES.items() if any(t in q for t in terms)]


def _is_generic_section(section_path: str) -> bool:
    n = _norm(section_path)
    return any(g in n for g in _GENERIC_MARKERS)


def _doc_text(doc) -> tuple[str, str]:
    meta = getattr(doc, "metadata", {}) or {}
    section = meta.get("section_path") or meta.get("section") or ""
    content = getattr(doc, "page_content", "") or ""
    return section, _norm(section + content[:400])


def evidence_covers(families: list[str], docs) -> tuple[bool, bool]:
    """返回 (是否覆盖核心主题, 是否有非泛章节『章节路径』命中)。

    - covered：主题词出现在任一 chunk 的章节路径或正文（泛泛相关）。
    - non_generic_hit：主题词出现在某条非「前言/随车工具/SOS」类章节的『章节路径』里
      ——这才说明检索到了真正对口的小节，而不是某段正文里偶然提到。
      判 answerable 必须满足 non_generic_hit。

    families 为空（问题无可识别主题）时返回 (False, False)，交由上层从严处理。
    """
    if not families:
        return False, False
    covered = False
    non_generic_hit = False
    for fam in families:
        terms = FAMILIES[fam]
        for doc in docs:
            section, blob = _doc_text(doc)
            nsec = _norm(section)
            if any(t in blob for t in terms):
                covered = True
            if any(t in nsec for t in terms) and not _is_generic_section(section):
                non_generic_hit = True
    return covered, non_generic_hit
