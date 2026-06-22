"""数据集生成 Prompt：5 类指令 + 1~2 个 Few-Shot 示例（本地 7B 用）。"""
from __future__ import annotations

TASK_GUIDE = {
    "直接问答": "用一问一答方式，直接、准确地回答用户的汽车使用问题。",
    "步骤指导": "把操作拆成清晰的编号步骤，便于用户照做。",
    "故障分析": "根据用户描述的现象，推断可能原因并给出处理建议。",
    "术语解释": "用通俗语言解释一个汽车专业名词或功能。",
    "安全提醒": "针对该场景给出安全或注意事项提醒。",
}

# 1~2 个完整 Few-Shot：展示 context→{instruction,input,output} 的映射与 AITO 语气
FEW_SHOT = (
    "示例1（直接问答）：\n"
    "【可用资料】\n"
    "胎压监测：当某个轮胎气压低于设定阈值时，仪表会显示对应车轮的低压报警，"
    "建议尽快检查并充气至标准值。\n"
    "【车主原始问题】胎压灯亮了是咋回事啊\n"
    "【输出】\n"
    '{"instruction": "回答用户关于问界车辆的使用问题。", '
    '"input": "仪表上的胎压报警灯亮了是什么原因？", '
    '"output": "这通常表示某个轮胎的气压低于设定阈值。仪表会指出对应车轮的位置，'
    '建议尽快检查该轮胎并将气压充至标准值，报警随后会自动解除。"}\n\n'
    "示例2（步骤指导）：\n"
    "【可用资料】\n"
    "蓝牙连接：在中控屏进入蓝牙设置并开启蓝牙，在手机端搜索车辆名称完成配对，"
    "配对成功后可使用蓝牙电话与音乐。\n"
    "【车主原始问题】手机咋连车上蓝牙\n"
    "【输出】\n"
    '{"instruction": "给出在问界车辆上完成指定操作的步骤。", '
    '"input": "怎么把手机连接到车机蓝牙？", '
    '"output": "1. 在中控屏进入蓝牙设置并开启蓝牙；\\n'
    "2. 在手机端打开蓝牙，搜索并选择车辆名称；\\n"
    "3. 确认配对码完成配对；\\n"
    '4. 配对成功后即可使用蓝牙拨打电话和播放音乐。"}'
)

DATASET_PROMPT_TEMPLATE = (
    "你是问界（AITO）汽车技术文档训练数据构造专家。\n"
    "任务：把车主的口语化问题，结合可用资料，改写成一条高质量的指令微调样本。\n\n"
    "【本条样本类型】{task_type}：{task_hint}\n\n"
    "【硬性规则】\n"
    "1. output 只能依据【可用资料】，不得编造资料里没有的数字、配置或功能。\n"
    "2. instruction 是给汽车助手的任务说明；input 是清晰规范的用户问句；"
    "output 是准确专业的回答。\n"
    "3. 严禁出现：「根据手册」「根据资料」「资料显示」「作为AI」「作为助手」等字样，"
    "也不要出现 [资料1] 之类引用标记。\n"
    "4. 语气贴合问界车主助手，面向「问界车辆」通用表述，不要编造具体车型年款。\n"
    "5. 只输出一个 JSON 对象，且仅含 instruction、input、output 三个键，"
    "不要任何额外文字。\n\n"
    "{few_shot}\n\n"
    "现在请处理这一条：\n"
    "【可用资料】\n{context}\n\n"
    "【车主原始问题】{question}\n"
    "【输出】\n"
)


def build_dataset_prompt(context: str, question: str, task_type: str) -> str:
    return DATASET_PROMPT_TEMPLATE.format(
        task_type=task_type,
        task_hint=TASK_GUIDE[task_type],
        few_shot=FEW_SHOT,
        context=context,
        question=question,
    )
