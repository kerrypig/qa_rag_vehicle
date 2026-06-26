import random
from lora_gen.prompts import (
    INSTRUCTION_POOLS, pick_instruction, question_gen_prompt,
    answer_gen_prompt, judge_prompt, TASK_TYPES,
)

def test_every_task_type_has_pool_min4():
    for tt in TASK_TYPES:
        assert len(INSTRUCTION_POOLS[tt]) >= 4

def test_pick_instruction_deterministic_in_pool():
    rng = random.Random(7)
    val = pick_instruction("步骤指导", rng)
    assert val in INSTRUCTION_POOLS["步骤指导"]

def test_question_prompt_includes_model_and_chunk():
    p = question_gen_prompt(model_display="问界M9 2026款增程版",
                            section_path="车辆控制>空调", chunk_text="空调使用说明……",
                            task_type="直接问答")
    assert "问界M9 2026款增程版" in p
    assert "空调使用说明" in p
    assert "JSON" in p

def test_answer_prompt_uses_evidence():
    p = answer_gen_prompt(model_display="问界M9", instruction="请解释",
                          question="空调怎么开？", evidence_text="按下AUTO键……",
                          task_type="直接问答")
    assert "按下AUTO键" in p

def test_judge_prompt_asks_conflict():
    p = judge_prompt(question="x", evidence_text="y")
    assert "conflict" in p
