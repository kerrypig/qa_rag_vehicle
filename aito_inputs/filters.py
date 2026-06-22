"""高精度剔除规则（纯逻辑）。

AutoMaster 数据以传统燃油车、其它品牌的真实问题为主，且常带具体「年款+车型」
描述与维修工位诊断细节。这些无法干净地迁移成问界新能源车主问题，应结构化剔除：

- describes_specific_other_vehicle：出现「X年/X款/20XX年X月生产」等具体他车描述；
- needs_field_diagnosis：依赖读码仪/故障码/拆解/柴油件等现场维修信息；
- is_ice_specific：明确的内燃机配置/柴油特征（三缸、2.0T、柴油等）。
"""
from __future__ import annotations

import re

# 「11年A6l」「18年款」「10款」「2013年12月生产」等具体年款车辆描述。
# 注意：避免误伤「开了3年」「用了10年了」这类车龄表述（年后接 了/的/多 等）。
_YEAR_MODEL_RE = re.compile(
    r"\d{2,4}\s*年款"                  # 18年款 / 2014年款
    r"|\d{1,2}\s*款"                   # 10款 / 14款
    r"|(?:19|20)\d{2}\s*年"            # 2013年（四位完整年份）
    r"|\d{2}\s*年(?![了的多半，。、\s])"  # 11年A6（两位年份后紧跟车型描述）
)
# 排量/缸数等内燃机配置（CJK 是 \w，故用否定前瞻而非 \b）
_ICE_CONFIG_RE = re.compile(r"\d\.\d\s*[tTlL](?![a-zA-Z])|v[68](?![a-zA-Z0-9])|[一二三四五六两]缸", re.I)

# 需维修工位/读码仪/拆解才能判断的细节（手册无法支撑）
_DIAG_KEYWORDS: tuple[str, ...] = (
    "故障码", "读码", "解码", "431", "电脑检测", "诊断仪", "数据流", "示波器",
    "拆解", "拆开", "拆下", "万用表", "波形", "电阻值", "对地电压",
    "修理厂", "修理工", "修了", "都换了", "换了还", "更换过", "换过",
)

# 购车/选车/价格咨询或「这是哪款车」等元问题 → 非手册可答场景
_OFF_TOPIC_KEYWORDS: tuple[str, ...] = (
    "买什么车", "买什么", "哪款车", "选车", "值得买", "落地价", "裸车", "提车",
    "二手车值", "落地多少", "哪个车好", "推荐买",
)


def is_off_topic_intent(text: str) -> bool:
    """购车咨询/价格/「这是哪款车」等元问题，手册无法回答。"""
    t = text or ""
    return any(k in t for k in _OFF_TOPIC_KEYWORDS)
# 柴油/内燃机/传统传动专属部件、现象与操作（增程器是发电机、纯电更无；
# 问界为无钥匙启动、单速变速、无怠速/转速表/拧钥匙点火等传统行为）
_ICE_PART_KEYWORDS: tuple[str, ...] = (
    # 部件
    "预热塞", "柴油", "喷油泵", "高压油泵", "高压油管", "共轨", "涡轮增压器", "积碳",
    "缸压", "氧传感器", "爆震", "缸盖", "活塞环", "化油器", "分电器", "怠速马达",
    "节温器", "恒温器", "防冻液", "水箱", "助力泵", "助力油", "差速器", "内球笼",
    "下支臂", "正时", "离合", "黑烟", "白烟", "烧机油", "机油灯",
    "点火线圈", "点火高压", "曲轴", "凸轮轴", "真空助力",
    # 传统启动/变速行为（问界无）
    "怠速", "转速", "拧钥匙", "打火", "打不着", "着火", "灭火", "熄火",
    "自动挡", "手动挡", "换挡", "挂挡", "挂档", "空挡", "挂d档", "挂n档", "挂r档",
)


def describes_specific_other_vehicle(text: str) -> bool:
    """出现具体年款/生产日期描述 → 多为在描述某一辆具体的（他品牌）车。"""
    return bool(_YEAR_MODEL_RE.search(text or ""))


def is_ice_specific(text: str) -> bool:
    """明确的内燃机配置/柴油特征。"""
    t = text or ""
    if _ICE_CONFIG_RE.search(t):
        return True
    return any(k in t.lower() for k in _ICE_PART_KEYWORDS)


def needs_field_diagnosis(text: str) -> bool:
    """依赖读码仪/拆解等现场维修信息，手册无法支撑。"""
    t = (text or "").lower()
    return any(k in t for k in _DIAG_KEYWORDS)
