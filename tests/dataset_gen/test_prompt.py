import pytest

from dataset_gen.prompt import TASK_GUIDE, build_dataset_prompt


def test_prompt_contains_inputs():
    p = build_dataset_prompt(context="问界胎压资料", question="胎压灯亮了", task_type="直接问答")
    assert "问界胎压资料" in p
    assert "胎压灯亮了" in p
    assert "直接问答" in p
    assert TASK_GUIDE["直接问答"] in p


def test_prompt_has_few_shot_examples():
    p = build_dataset_prompt(context="c", question="q", task_type="步骤指导")
    assert "示例1" in p and "示例2" in p


def test_prompt_bans_leaky_phrases():
    p = build_dataset_prompt(context="c", question="q", task_type="术语解释")
    assert "根据手册" in p  # 出现在「严禁」清单里
    assert "严禁" in p


def test_all_five_task_types_supported():
    for t in ["直接问答", "步骤指导", "故障分析", "术语解释", "安全提醒"]:
        assert t in TASK_GUIDE
        assert t in build_dataset_prompt(context="c", question="q", task_type=t)


def test_unknown_task_type_raises():
    with pytest.raises(KeyError):
        build_dataset_prompt(context="c", question="q", task_type="不存在")
