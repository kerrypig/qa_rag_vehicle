"""集中管理 instruction 模板池与各阶段 prompt。"""
from __future__ import annotations

import random

TASK_TYPES = ["直接问答", "步骤指导", "故障分析", "术语解释", "安全提醒"]

INSTRUCTION_POOLS: dict[str, list[str]] = {
    "直接问答": [
        "作为汽车技术助手，准确回答车主关于本车型的问题。",
        "请基于车主手册，回答以下汽车使用问题。",
        "你是车型技术顾问，请专业、简洁地回答车主提问。",
        "根据该车型官方资料，解答车主的疑问。",
    ],
    "步骤指导": [
        "请给出完成以下操作的分步骤说明。",
        "作为汽车助手，分步指导车主完成该操作。",
        "列出该项检查/操作的具体流程步骤。",
        "请按顺序说明车主应如何完成此操作。",
    ],
    "故障分析": [
        "根据车主描述的现象，分析可能原因并给出建议。",
        "作为汽车诊断助手，推断该故障现象的成因与处理方式。",
        "请根据以下故障现象，判断原因并提供应对建议。",
        "分析该报警/异常现象，说明含义与车主应采取的措施。",
    ],
    "术语解释": [
        "请解释以下汽车专业术语的含义。",
        "作为汽车助手，向车主通俗解释该名词。",
        "说明该术语在本车型中的定义与作用。",
        "请用车主能理解的语言解释这个专业概念。",
    ],
    "安全提醒": [
        "请给出与该操作相关的安全注意事项。",
        "作为汽车安全助手，提醒车主相关风险与正确做法。",
        "说明涉及的安全警告及车主须遵守的事项。",
        "请强调该场景下的安全要点与禁止行为。",
    ],
}


def pick_instruction(task_type: str, rng: random.Random) -> str:
    pool = INSTRUCTION_POOLS.get(task_type) or INSTRUCTION_POOLS["直接问答"]
    return rng.choice(pool)


def question_gen_prompt(*, model_display: str, section_path: str, chunk_text: str, task_type: str) -> str:
    return (
        f"你在为「{model_display}」构建问答训练数据。下面是其车主手册中"
        f"「{section_path}」章节的一段内容：\n---\n{chunk_text}\n---\n"
        f"请基于且仅基于这段内容，站在真实车主角度提出一个【{task_type}】类型的问题。\n"
        f"要求：\n"
        f"1. 问题必须在开头明确点出车型「{model_display}」。\n"
        f"2. 问题要具体、口语化，能被这段内容回答，不要泛泛而问。\n"
        f"3. 不要在问题里包含答案或引用「手册/章节/页码」。\n"
        f'只输出 JSON：{{"question": "..."}}'
    )


def answer_gen_prompt(*, model_display: str, instruction: str, question: str, evidence_text: str, task_type: str) -> str:
    return (
        f"你是「{model_display}」的汽车技术助手。以下是从该车型手册检索到的资料：\n"
        f"---\n{evidence_text}\n---\n"
        f"任务类型：{task_type}\n车主问题：{question}\n"
        f"请严格依据上述资料作答，资料未提及的内容不要编造或承诺。\n"
        f"回答要专业、通顺、贴合汽车助手口吻；不要出现「根据手册/根据资料/作为AI」"
        f"等字样，不要引用页码或编号。\n"
        f'只输出 JSON：{{"output": "..."}}'
    )


def judge_prompt(*, question: str, evidence_text: str) -> str:
    return (
        f"判断下面的「资料」能否完整回答「问题」。\n"
        f"问题：{question}\n资料：\n---\n{evidence_text}\n---\n"
        f"label 取值：full=可完整回答；partial=只能部分回答；no=无法回答。\n"
        f"conflict：资料内部是否存在相互矛盾的信息（true/false）。\n"
        f'只输出 JSON：{{"label": "full|partial|no", "conflict": true|false}}'
    )
