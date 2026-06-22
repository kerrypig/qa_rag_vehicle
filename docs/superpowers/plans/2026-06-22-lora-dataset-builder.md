# LoRA 微调数据集构建器 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 `doc/AutoMaster_TrainSet.csv` 通用汽车问答，经问界手册 RAG 检索后改写为合法 JSON 数组 LoRA 微调数据集 `ito_lora_dataset.json`，支持本地/云端后端、弱检索跳过、断点续传。

**Architecture:** 新增 `dataset_gen/` 纯逻辑包（清洗/质量/Prompt/后端/断点）+ 编排脚本 `scripts/build_lora_dataset.py` + 独立配置 `dataset_gen.yaml`。复用既有 `Retriever.retrieve_stateless`、`format_context`、`QwenClient`、`ollama`，对既有 pipeline 零改动。轻量检索通过把 `query_rewrite/keyword_search/bookmark_match/verification` 全部覆盖为关闭实现，检索阶段零 Ollama 调用。

**Tech Stack:** Python 3.12、pandas、ollama（本地 qwen2.5:7b，`format='json'`）、OpenAI SDK（云端 DashScope）、LangChain/FAISS/BM25（既有检索）、tqdm、pytest。

**关键约定：**
- 解释器一律用项目 venv：`../.venv/Scripts/python.exe`（仅此 venv 有依赖；conda 不可用）。
- 所有命令在项目根 `qa_rag_vehicle/` 下执行。
- spec：`docs/superpowers/specs/2026-06-22-lora-dataset-builder-design.md`。

---

## 文件结构

| 文件 | 职责 |
|---|---|
| `dataset_gen/__init__.py` | 包标记（空） |
| `dataset_gen/config_overrides.py` | 点路径覆盖项写入 `config.raw` |
| `dataset_gen/cleaning.py` | CSV 载入、input 提取、关键词过滤、去重、采样 |
| `dataset_gen/quality.py` | 弱检索判定、禁语、样本校验 |
| `dataset_gen/prompt.py` | 数据集 Prompt 模板 + Few-Shot + 构造器 |
| `dataset_gen/backends.py` | `extract_json` + Local/Cloud 后端 + `make_backend` |
| `dataset_gen/checkpoint.py` | 原子写 JSON、追加 meta、读已完成 QID |
| `scripts/build_lora_dataset.py` | CLI 编排（整合上述） |
| `dataset_gen.yaml` | 数据集生成配置 |
| `conftest.py` | 让 pytest 把项目根加入 sys.path |
| `tests/dataset_gen/test_*.py` | 各纯逻辑模块单测 |

---

## Task 0: 项目准备（依赖、包骨架、pytest 接通）

**Files:**
- Modify: `requirements.txt`
- Create: `dataset_gen/__init__.py`
- Create: `tests/dataset_gen/__init__.py`
- Create: `conftest.py`

- [ ] **Step 1: 在 requirements.txt 追加缺失依赖**

在文件末尾追加三行（pandas/tqdm 已在 venv 但未登记，pytest 缺失）：

```
pandas>=2.0.0
tqdm>=4.66.0
pytest>=8.0.0
```

- [ ] **Step 2: 安装 pytest 到项目 venv**

Run: `../.venv/Scripts/python.exe -m pip install "pytest>=8.0.0"`
Expected: 末尾出现 `Successfully installed pytest-...`

- [ ] **Step 3: 创建空包标记与 conftest**

Create `dataset_gen/__init__.py`：

```python
"""LoRA 微调数据集构建：纯逻辑模块集合。"""
```

Create `tests/dataset_gen/__init__.py`（空文件，内容为单个换行）：

```python
```

Create `conftest.py`（位于项目根，使 pytest 把根目录加入 sys.path，从而能 `import dataset_gen`）：

```python
"""pytest 根配置：确保项目根在 sys.path，便于导入 dataset_gen / 既有模块。"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
```

- [ ] **Step 4: 验证 pytest 可运行（暂无测试）**

Run: `../.venv/Scripts/python.exe -m pytest tests/ -q`
Expected: `no tests ran` 或 `collected 0 items`，无导入错误。

- [ ] **Step 5: 提交**

```bash
git add requirements.txt dataset_gen/__init__.py tests/dataset_gen/__init__.py conftest.py
git commit -m "chore(dataset): 数据集构建依赖与包骨架"
```

---

## Task 1: config_overrides — 点路径覆盖

**Files:**
- Create: `dataset_gen/config_overrides.py`
- Test: `tests/dataset_gen/test_config_overrides.py`

- [ ] **Step 1: 写失败测试**

Create `tests/dataset_gen/test_config_overrides.py`：

```python
from dataset_gen.config_overrides import apply_overrides


def test_sets_existing_nested_key():
    raw = {"query_rewrite": {"enabled": True}, "retrieval": {"keyword_search": {"enabled": True}}}
    apply_overrides(raw, {
        "query_rewrite.enabled": False,
        "retrieval.keyword_search.enabled": False,
    })
    assert raw["query_rewrite"]["enabled"] is False
    assert raw["retrieval"]["keyword_search"]["enabled"] is False


def test_creates_missing_path():
    raw = {}
    apply_overrides(raw, {"a.b.c": 1})
    assert raw["a"]["b"]["c"] == 1


def test_overwrites_non_dict_intermediate():
    raw = {"a": 5}
    apply_overrides(raw, {"a.b": 2})
    assert raw["a"] == {"b": 2}
```

- [ ] **Step 2: 运行确认失败**

Run: `../.venv/Scripts/python.exe -m pytest tests/dataset_gen/test_config_overrides.py -v`
Expected: FAIL，`ModuleNotFoundError: No module named 'dataset_gen.config_overrides'`

- [ ] **Step 3: 实现**

Create `dataset_gen/config_overrides.py`：

```python
"""把点路径覆盖项就地写入 config.raw（用于关闭检索中的 Ollama 步骤）。"""
from __future__ import annotations

from typing import Any


def apply_overrides(raw: dict[str, Any], overrides: dict[str, Any]) -> None:
    """overrides 形如 {"query_rewrite.enabled": False}；按 '.' 分段写入 raw。

    中间节点不存在或不是 dict 时，新建为 dict 后继续。
    """
    for dotted, value in overrides.items():
        keys = dotted.split(".")
        node = raw
        for key in keys[:-1]:
            nxt = node.get(key)
            if not isinstance(nxt, dict):
                nxt = {}
                node[key] = nxt
            node = nxt
        node[keys[-1]] = value
```

- [ ] **Step 4: 运行确认通过**

Run: `../.venv/Scripts/python.exe -m pytest tests/dataset_gen/test_config_overrides.py -v`
Expected: 3 passed

- [ ] **Step 5: 提交**

```bash
git add dataset_gen/config_overrides.py tests/dataset_gen/test_config_overrides.py
git commit -m "feat(dataset): 点路径配置覆盖"
```

---

## Task 2: cleaning — CSV 载入/过滤/采样

**Files:**
- Create: `dataset_gen/cleaning.py`
- Test: `tests/dataset_gen/test_cleaning.py`

- [ ] **Step 1: 写失败测试**

Create `tests/dataset_gen/test_cleaning.py`：

```python
import pandas as pd

from dataset_gen.cleaning import (
    dedup_by_question,
    extract_input_text,
    filter_by_keywords,
    load_rows,
    sample_rows,
)


def test_extract_input_prefers_question():
    assert extract_input_text({"Question": "胎压灯亮了", "Dialogue": "x|y"}) == "胎压灯亮了"


def test_extract_input_falls_back_to_dialogue_first_segment():
    row = {"Question": "", "Dialogue": "技师说蓝牙怎么连|车主说好的"}
    assert extract_input_text(row) == "技师说蓝牙怎么连"


def test_filter_by_keywords_keeps_only_matches():
    rows = [
        {"Question": "胎压灯亮了怎么办"},
        {"Question": "发动机正时皮带多久换"},
        {"Question": "空调不制冷"},
    ]
    kept = filter_by_keywords(rows, ["胎压", "空调"])
    assert [r["Question"] for r in kept] == ["胎压灯亮了怎么办", "空调不制冷"]


def test_dedup_by_question():
    rows = [{"Question": "蓝牙怎么连"}, {"Question": "蓝牙怎么连"}, {"Question": "雷达报警"}]
    assert len(dedup_by_question(rows)) == 2


def test_sample_rows_deterministic_with_seed():
    rows = [{"Question": str(i)} for i in range(20)]
    a = sample_rows(rows, seed=42)
    b = sample_rows(rows, seed=42)
    assert a == b
    assert sorted(r["Question"] for r in a) == sorted(r["Question"] for r in rows)


def test_load_rows_reads_gb18030(tmp_path):
    csv = tmp_path / "t.csv"
    df = pd.DataFrame({"QID": ["Q1"], "Question": ["空调不制冷"], "Dialogue": [""]})
    df.to_csv(csv, index=False, encoding="gb18030")
    rows = load_rows(str(csv), encoding="gb18030")
    assert rows[0]["Question"] == "空调不制冷"
    assert rows[0]["QID"] == "Q1"
```

- [ ] **Step 2: 运行确认失败**

Run: `../.venv/Scripts/python.exe -m pytest tests/dataset_gen/test_cleaning.py -v`
Expected: FAIL，`ModuleNotFoundError: No module named 'dataset_gen.cleaning'`

- [ ] **Step 3: 实现**

Create `dataset_gen/cleaning.py`：

```python
"""CSV 载入、input 文本提取、关键词过滤、去重、随机采样。"""
from __future__ import annotations

import random

import pandas as pd


def load_rows(csv_path: str, encoding: str = "gb18030") -> list[dict]:
    """按指定编码读 CSV，失败则依次回退 gbk / utf-8。空值填空串。"""
    last_err: Exception | None = None
    for enc in [encoding, "gbk", "utf-8"]:
        try:
            df = pd.read_csv(csv_path, encoding=enc, dtype=str)
            df = df.fillna("")
            return df.to_dict(orient="records")
        except UnicodeDecodeError as e:
            last_err = e
    raise last_err  # type: ignore[misc]


def extract_input_text(row: dict) -> str:
    """优先用 Question；为空时回退取 Dialogue 第一段（以 '|' 分隔）。"""
    q = str(row.get("Question", "")).strip()
    if q:
        return q
    dlg = str(row.get("Dialogue", "")).strip()
    return dlg.split("|")[0].strip() if dlg else ""


def filter_by_keywords(rows: list[dict], keywords: list[str]) -> list[dict]:
    """保留 input 文本命中任一关键词的行。"""
    return [r for r in rows if any(kw in extract_input_text(r) for kw in keywords)]


def dedup_by_question(rows: list[dict]) -> list[dict]:
    """按 input 文本去重，保留首次出现；丢弃 input 为空的行。"""
    seen: set[str] = set()
    out: list[dict] = []
    for r in rows:
        q = extract_input_text(r)
        if q and q not in seen:
            seen.add(q)
            out.append(r)
    return out


def sample_rows(rows: list[dict], seed: int) -> list[dict]:
    """用固定种子打散整池，返回全部（调用方按 sample_size/target_size 控制截断）。"""
    rng = random.Random(seed)
    pool = list(rows)
    rng.shuffle(pool)
    return pool
```

- [ ] **Step 4: 运行确认通过**

Run: `../.venv/Scripts/python.exe -m pytest tests/dataset_gen/test_cleaning.py -v`
Expected: 6 passed

- [ ] **Step 5: 提交**

```bash
git add dataset_gen/cleaning.py tests/dataset_gen/test_cleaning.py
git commit -m "feat(dataset): CSV 清洗与采样"
```

---

## Task 3: quality — 弱检索/禁语/样本校验

**Files:**
- Create: `dataset_gen/quality.py`
- Test: `tests/dataset_gen/test_quality.py`

- [ ] **Step 1: 写失败测试**

Create `tests/dataset_gen/test_quality.py`：

```python
from dataset_gen.quality import (
    has_banned_phrase,
    is_weak_retrieval,
    max_score,
    validate_sample,
)


def test_max_score_empty():
    assert max_score({}) == 0.0


def test_weak_when_too_few_chunks():
    assert is_weak_retrieval(["d"], {"a": 0.9}, min_chunks=2, min_score=0.3) is True


def test_weak_when_score_below_threshold():
    assert is_weak_retrieval(["d1", "d2"], {"a": 0.1, "b": 0.2}, min_chunks=2, min_score=0.3) is True


def test_not_weak_when_enough_and_strong():
    assert is_weak_retrieval(["d1", "d2"], {"a": 0.5}, min_chunks=2, min_score=0.3) is False


def test_banned_phrase_detected():
    assert has_banned_phrase("根据手册，应当检查胎压") is True
    assert has_banned_phrase("请检查胎压并充气") is False


def test_validate_ok():
    obj = {"instruction": "回答问题", "input": "胎压灯亮了？", "output": "请检查并充气。"}
    ok, msg = validate_sample(obj)
    assert ok is True and msg == ""


def test_validate_missing_field():
    ok, msg = validate_sample({"instruction": "x", "input": "y"})
    assert ok is False and "output" in msg


def test_validate_empty_field():
    ok, msg = validate_sample({"instruction": "x", "input": "  ", "output": "z"})
    assert ok is False and "input" in msg


def test_validate_extra_field():
    obj = {"instruction": "x", "input": "y", "output": "z", "note": "多余"}
    ok, msg = validate_sample(obj)
    assert ok is False and "多余字段" in msg


def test_validate_banned_in_output():
    obj = {"instruction": "x", "input": "y", "output": "根据手册请检查"}
    ok, msg = validate_sample(obj)
    assert ok is False and "禁语" in msg


def test_validate_non_dict():
    ok, msg = validate_sample(["not", "a", "dict"])
    assert ok is False
```

- [ ] **Step 2: 运行确认失败**

Run: `../.venv/Scripts/python.exe -m pytest tests/dataset_gen/test_quality.py -v`
Expected: FAIL，`ModuleNotFoundError: No module named 'dataset_gen.quality'`

- [ ] **Step 3: 实现**

Create `dataset_gen/quality.py`：

```python
"""检索强度判定、禁语过滤、样本结构校验。"""
from __future__ import annotations

BANNED_PHRASES = [
    "根据手册", "根据资料", "根据上述", "根据提供的", "根据以上",
    "资料显示", "手册显示", "手册中", "资料中", "如上所述",
    "作为AI", "作为人工智能", "作为助手", "作为一个",
    "[资料", "资料1", "资料2", "资料3",
]

_ALLOWED_KEYS = {"instruction", "input", "output"}


def max_score(scores: dict[str, float]) -> float:
    return max(scores.values()) if scores else 0.0


def is_weak_retrieval(docs, scores: dict[str, float], *, min_chunks: int, min_score: float) -> bool:
    """chunk 数不足，或最高向量相似度低于阈值，判为弱检索（应跳过）。"""
    if len(docs) < min_chunks:
        return True
    return max_score(scores) < min_score


def has_banned_phrase(text: str) -> bool:
    return any(p in text for p in BANNED_PHRASES)


def validate_sample(obj) -> tuple[bool, str]:
    """校验生成结果：必须是仅含三键的 dict，三字段非空，output 无禁语。"""
    if not isinstance(obj, dict):
        return False, "不是 JSON 对象"
    for key in ("instruction", "input", "output"):
        v = obj.get(key)
        if not isinstance(v, str) or not v.strip():
            return False, f"字段缺失或为空: {key}"
    extra = set(obj.keys()) - _ALLOWED_KEYS
    if extra:
        return False, f"含多余字段: {sorted(extra)}"
    if has_banned_phrase(obj["output"]):
        return False, "output 命中禁语"
    return True, ""
```

- [ ] **Step 4: 运行确认通过**

Run: `../.venv/Scripts/python.exe -m pytest tests/dataset_gen/test_quality.py -v`
Expected: 11 passed

- [ ] **Step 5: 提交**

```bash
git add dataset_gen/quality.py tests/dataset_gen/test_quality.py
git commit -m "feat(dataset): 弱检索与样本质量校验"
```

---

## Task 4: prompt — 数据集 Prompt + Few-Shot

**Files:**
- Create: `dataset_gen/prompt.py`
- Test: `tests/dataset_gen/test_prompt.py`

- [ ] **Step 1: 写失败测试**

Create `tests/dataset_gen/test_prompt.py`：

```python
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
```

- [ ] **Step 2: 运行确认失败**

Run: `../.venv/Scripts/python.exe -m pytest tests/dataset_gen/test_prompt.py -v`
Expected: FAIL，`ModuleNotFoundError: No module named 'dataset_gen.prompt'`

- [ ] **Step 3: 实现**

Create `dataset_gen/prompt.py`：

```python
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
```

- [ ] **Step 4: 运行确认通过**

Run: `../.venv/Scripts/python.exe -m pytest tests/dataset_gen/test_prompt.py -v`
Expected: 5 passed

- [ ] **Step 5: 提交**

```bash
git add dataset_gen/prompt.py tests/dataset_gen/test_prompt.py
git commit -m "feat(dataset): 数据集生成 Prompt 与 Few-Shot"
```

---

## Task 5: backends — JSON 提取与生成后端

**Files:**
- Create: `dataset_gen/backends.py`
- Test: `tests/dataset_gen/test_backends.py`

- [ ] **Step 1: 写失败测试**（仅测纯函数 `extract_json`，不触发 Ollama/网络）

Create `tests/dataset_gen/test_backends.py`：

```python
import pytest

from dataset_gen.backends import extract_json


def test_plain_json():
    assert extract_json('{"a": 1, "b": "x"}') == {"a": 1, "b": "x"}


def test_fenced_json():
    text = "```json\n{\"a\": 1}\n```"
    assert extract_json(text) == {"a": 1}


def test_json_with_leading_prose():
    text = '好的，结果如下：{"instruction": "i", "input": "q", "output": "o"}'
    assert extract_json(text) == {"instruction": "i", "input": "q", "output": "o"}


def test_invalid_json_raises():
    with pytest.raises(ValueError):
        extract_json("这里没有任何 JSON")
```

- [ ] **Step 2: 运行确认失败**

Run: `../.venv/Scripts/python.exe -m pytest tests/dataset_gen/test_backends.py -v`
Expected: FAIL，`ModuleNotFoundError: No module named 'dataset_gen.backends'`

- [ ] **Step 3: 实现**

Create `dataset_gen/backends.py`：

```python
"""生成后端：本地 Ollama（format=json）/ 云端 DashScope，统一返回 dict。"""
from __future__ import annotations

import json
import re
from typing import Protocol

import ollama


def extract_json(text: str) -> dict:
    """从模型输出中提取首个 JSON 对象。支持 ```json``` 围栏与前置说明文字。

    无法解析时抛 ValueError（json.JSONDecodeError 是其子类）。
    """
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if fence:
        candidate = fence.group(1)
    else:
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            raise ValueError(f"输出中未找到 JSON 对象: {text[:80]!r}")
        candidate = m.group(0)
    return json.loads(candidate)


class Backend(Protocol):
    def generate(self, prompt: str) -> dict: ...


class LocalOllamaBackend:
    """本地 Ollama，启用 format='json' 强约束。"""

    def __init__(self, model: str, temperature: float, num_predict: int, timeout_s: int):
        self.model = model
        self.temperature = temperature
        self.num_predict = num_predict
        self.client = ollama.Client(timeout=timeout_s)

    def generate(self, prompt: str) -> dict:
        resp = self.client.generate(
            model=self.model,
            prompt=prompt,
            format="json",
            options={"temperature": self.temperature, "num_predict": self.num_predict},
        )
        return extract_json(resp["response"])


class CloudBackend:
    """云端 DashScope，复用既有 QwenClient；用解析提取替代原生 JSON 模式。"""

    def __init__(self, qwen_client):
        self.client = qwen_client

    def generate(self, prompt: str) -> dict:
        text, _ = self.client.chat(prompt)
        return extract_json(text)


def make_backend(ds_cfg: dict, app_config) -> Backend:
    if ds_cfg.get("backend") == "cloud":
        from generate.qwen_client import QwenClient

        return CloudBackend(QwenClient(app_config))
    lc = ds_cfg["local"]
    return LocalOllamaBackend(
        model=lc["model"],
        temperature=lc["temperature"],
        num_predict=lc["num_predict"],
        timeout_s=lc["timeout_s"],
    )
```

- [ ] **Step 4: 运行确认通过**

Run: `../.venv/Scripts/python.exe -m pytest tests/dataset_gen/test_backends.py -v`
Expected: 4 passed

- [ ] **Step 5: 提交**

```bash
git add dataset_gen/backends.py tests/dataset_gen/test_backends.py
git commit -m "feat(dataset): JSON 提取与本地/云端生成后端"
```

---

## Task 6: checkpoint — 断点续传

**Files:**
- Create: `dataset_gen/checkpoint.py`
- Test: `tests/dataset_gen/test_checkpoint.py`

- [ ] **Step 1: 写失败测试**

Create `tests/dataset_gen/test_checkpoint.py`：

```python
from dataset_gen.checkpoint import (
    append_meta,
    load_done_qids,
    load_samples,
    write_samples,
)


def test_write_then_load_roundtrip(tmp_path):
    p = tmp_path / "out.json"
    samples = [{"instruction": "i", "input": "q", "output": "中文输出"}]
    write_samples(str(p), samples)
    assert load_samples(str(p)) == samples
    # 中文不转义
    assert "中文输出" in p.read_text(encoding="utf-8")


def test_load_samples_missing_returns_empty(tmp_path):
    assert load_samples(str(tmp_path / "nope.json")) == []


def test_load_samples_corrupt_returns_empty(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("{not json", encoding="utf-8")
    assert load_samples(str(p)) == []


def test_append_meta_and_load_done_qids(tmp_path):
    m = tmp_path / "meta.jsonl"
    append_meta(str(m), {"QID": "Q1", "task_type": "直接问答"})
    append_meta(str(m), {"QID": "Q2", "task_type": "步骤指导"})
    assert load_done_qids(str(m)) == {"Q1", "Q2"}


def test_load_done_qids_missing_returns_empty(tmp_path):
    assert load_done_qids(str(tmp_path / "nope.jsonl")) == set()


def test_load_done_qids_skips_bad_lines(tmp_path):
    m = tmp_path / "meta.jsonl"
    m.write_text('{"QID": "Q1"}\n{bad}\n\n{"no_qid": 1}\n', encoding="utf-8")
    assert load_done_qids(str(m)) == {"Q1"}
```

- [ ] **Step 2: 运行确认失败**

Run: `../.venv/Scripts/python.exe -m pytest tests/dataset_gen/test_checkpoint.py -v`
Expected: FAIL，`ModuleNotFoundError: No module named 'dataset_gen.checkpoint'`

- [ ] **Step 3: 实现**

Create `dataset_gen/checkpoint.py`：

```python
"""断点续传：原子写 JSON 数组、追加 meta.jsonl、读已完成 QID。"""
from __future__ import annotations

import json
import os
from pathlib import Path


def load_done_qids(meta_path: str) -> set[str]:
    """从 meta.jsonl 收集已完成 QID；文件缺失或坏行均安全跳过。"""
    p = Path(meta_path)
    if not p.exists():
        return set()
    done: set[str] = set()
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                qid = json.loads(line)["QID"]
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
            done.add(qid)
    return done


def load_samples(json_path: str) -> list[dict]:
    """读现有输出数组；缺失或损坏返回空列表，便于重头累积。"""
    p = Path(json_path)
    if not p.exists():
        return []
    try:
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError:
        return []
    return data if isinstance(data, list) else []


def write_samples(json_path: str, samples: list[dict]) -> None:
    """原子写：先写 .tmp 再 os.replace，避免中断损坏输出。"""
    p = Path(json_path)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)
    os.replace(tmp, p)


def append_meta(meta_path: str, entry: dict) -> None:
    """逐行追加溯源记录（JSONL）。"""
    with open(meta_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
```

- [ ] **Step 4: 运行确认通过**

Run: `../.venv/Scripts/python.exe -m pytest tests/dataset_gen/test_checkpoint.py -v`
Expected: 6 passed

- [ ] **Step 5: 提交**

```bash
git add dataset_gen/checkpoint.py tests/dataset_gen/test_checkpoint.py
git commit -m "feat(dataset): 断点续传读写"
```

---

## Task 7: dataset_gen.yaml 配置

**Files:**
- Create: `dataset_gen.yaml`

- [ ] **Step 1: 创建配置文件**

Create `dataset_gen.yaml`（项目根）：

```yaml
dataset_gen:
  input_csv: "./doc/AutoMaster_TrainSet.csv"
  csv_encoding: "gb18030"          # CSV 为 GBK 系编码，UTF-8 读会乱码
  base_config: "./config.yaml"     # 复用既有检索/嵌入配置
  output_json: "./ito_lora_dataset.json"
  output_meta: "./ito_lora_dataset.meta.jsonl"

  sample_size: 600                 # 尝试候选行的上限（含被跳过的）
  target_size: 300                 # 目标成功条数；达到即停（200~400 区间）
  random_seed: 42

  # 新能源通用场景关键词（命中任一即保留）
  keep_keywords: [胎压, 空调, 异响, 蓝牙, 车机, 雷达, 钥匙, 充电, 刹车,
                  座椅, 车门, 后视镜, 灯光, 仪表, 续航, 充电桩, 中控]

  # 弱检索判定（任一不满足即跳过该行）
  min_chunks: 2
  min_score: 0.30

  task_types: [直接问答, 步骤指导, 故障分析, 术语解释, 安全提醒]

  backend: "local"                 # local | cloud
  local:
    model: "qwen2.5:7b"
    temperature: 0.3
    num_predict: 1024
    timeout_s: 120
    retries: 2
  # cloud 后端复用 config.yaml 的 generation 段

  checkpoint_every: 10

  # 关闭检索中所有 Ollama 步骤 → 轻量 hybrid（向量+BM25）
  overrides:
    query_rewrite.enabled: false
    retrieval.keyword_search.enabled: false
    retrieval.bookmark_match.enabled: false
    verification.enabled: false
```

- [ ] **Step 2: 校验 YAML 可解析**

Run: `../.venv/Scripts/python.exe -c "import yaml; print(list(yaml.safe_load(open('dataset_gen.yaml',encoding='utf-8'))['dataset_gen'].keys()))"`
Expected: 打印出含 `input_csv`、`overrides`、`local` 等键的列表，无异常。

- [ ] **Step 3: 提交**

```bash
git add dataset_gen.yaml
git commit -m "feat(dataset): 数据集生成配置文件"
```

---

## Task 8: 编排脚本 build_lora_dataset.py

**Files:**
- Create: `scripts/build_lora_dataset.py`

- [ ] **Step 1: 实现编排脚本**

Create `scripts/build_lora_dataset.py`：

```python
#!/usr/bin/env python3
"""把通用汽车问答 CSV 经问界手册 RAG 检索改写为 LoRA 微调数据集。"""
from __future__ import annotations

import os

os.environ["HF_HUB_OFFLINE"] = "1"

import argparse
import logging
import random
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml
from tqdm import tqdm

from config_loader import load_config
from dataset_gen.backends import make_backend
from dataset_gen.checkpoint import (
    append_meta,
    load_done_qids,
    load_samples,
    write_samples,
)
from dataset_gen.cleaning import (
    dedup_by_question,
    extract_input_text,
    filter_by_keywords,
    load_rows,
    sample_rows,
)
from dataset_gen.config_overrides import apply_overrides
from dataset_gen.prompt import build_dataset_prompt
from dataset_gen.quality import is_weak_retrieval, validate_sample
from generate.prompt_builder import format_context
from retrieve.pipeline import Retriever

log = logging.getLogger("dataset_gen")


def setup_logging(log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.FileHandler(log_dir / f"dataset_gen_{ts}.log", encoding="utf-8")],
    )


def load_ds_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)["dataset_gen"]


def generate_one(backend, prompt: str, retries: int):
    """调用后端生成并校验，最多重试 retries 次。成功返回 dict，否则 None。"""
    last = "未知错误"
    for attempt in range(retries + 1):
        try:
            obj = backend.generate(prompt)
        except Exception as e:  # noqa: BLE001 — 单条失败不应中断整批
            last = f"生成异常: {e}"
            continue
        ok, msg = validate_sample(obj)
        if ok:
            return obj
        last = msg
    log.warning("生成校验失败（已重试%d次）: %s", retries, last)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="构建问界 LoRA 微调数据集")
    parser.add_argument("--config", default=str(ROOT / "dataset_gen.yaml"))
    parser.add_argument("--backend", choices=["local", "cloud"])
    parser.add_argument("--target", type=int, help="覆盖 target_size")
    parser.add_argument("--sample", type=int, help="覆盖 sample_size")
    args = parser.parse_args()

    ds = load_ds_config(args.config)
    if args.backend:
        ds["backend"] = args.backend
    if args.target:
        ds["target_size"] = args.target
    if args.sample:
        ds["sample_size"] = args.sample

    app_config = load_config(str(ROOT / ds["base_config"].lstrip("./")))
    apply_overrides(app_config.raw, ds["overrides"])
    setup_logging(app_config.log_dir)

    index_path = app_config.index_path()
    if not (index_path / "faiss").exists():
        print(f"索引不存在: {index_path}\n请先运行: python main.py build")
        sys.exit(1)

    retriever = Retriever(app_config, index_path)
    retriever.load()
    backend = make_backend(ds, app_config)

    rows = load_rows(str(ROOT / ds["input_csv"].lstrip("./")), encoding=ds["csv_encoding"])
    rows = filter_by_keywords(rows, ds["keep_keywords"])
    rows = dedup_by_question(rows)
    pool = sample_rows(rows, seed=ds["random_seed"])
    print(f"过滤+去重后池大小: {len(pool)}")

    out_json = str(ROOT / ds["output_json"].lstrip("./"))
    out_meta = str(ROOT / ds["output_meta"].lstrip("./"))
    samples = load_samples(out_json)
    done = load_done_qids(out_meta)
    ok = len(samples)
    skip = fail = attempts = 0

    rng = random.Random(ds["random_seed"])
    target = ds["target_size"]
    sample_cap = ds["sample_size"]
    every = ds["checkpoint_every"]
    retries = ds["local"].get("retries", 2)

    bar = tqdm(pool, desc="生成", unit="条")
    for row in bar:
        if ok >= target or attempts >= sample_cap:
            break
        qid = str(row.get("QID", "")).strip()
        if not qid or qid in done:
            continue
        question = extract_input_text(row)
        if not question:
            continue
        attempts += 1

        try:
            result = retriever.retrieve_stateless(question)
        except Exception as e:  # noqa: BLE001
            fail += 1
            log.warning("检索异常 QID=%s: %s", qid, e)
            bar.set_postfix(ok=ok, skip=skip, fail=fail)
            continue

        if is_weak_retrieval(
            result.docs, result.scores,
            min_chunks=ds["min_chunks"], min_score=ds["min_score"],
        ):
            skip += 1
            log.info("弱检索跳过 QID=%s: %s", qid, question)
            bar.set_postfix(ok=ok, skip=skip, fail=fail)
            continue

        context = format_context(result.docs)
        task_type = rng.choice(ds["task_types"])
        prompt = build_dataset_prompt(context, question, task_type)

        obj = generate_one(backend, prompt, retries)
        if obj is None:
            fail += 1
            bar.set_postfix(ok=ok, skip=skip, fail=fail)
            continue

        samples.append(obj)
        append_meta(out_meta, {
            "QID": qid,
            "task_type": task_type,
            "source_sections": [d.metadata.get("section_path", "") for d in result.docs],
            "max_score": max(result.scores.values()) if result.scores else 0.0,
            "backend": ds["backend"],
        })
        done.add(qid)
        ok += 1
        if ok % every == 0:
            write_samples(out_json, samples)
            log.info("checkpoint: 已写 %d 条", ok)
        bar.set_postfix(ok=ok, skip=skip, fail=fail)

    write_samples(out_json, samples)
    bar.close()
    print(f"\n完成: 成功 {ok} | 跳过 {skip} | 失败 {fail} | 尝试 {attempts}")
    print(f"数据集: {out_json}")
    print(f"溯源:   {out_meta}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 静态导入自检（不实际生成）**

Run: `../.venv/Scripts/python.exe -c "import ast; ast.parse(open('scripts/build_lora_dataset.py',encoding='utf-8').read()); print('syntax ok')"`
Expected: `syntax ok`

- [ ] **Step 3: CLI 帮助自检（验证 import 链与 argparse 正常）**

Run: `../.venv/Scripts/python.exe scripts/build_lora_dataset.py --help`
Expected: 打印 usage，列出 `--config/--backend/--target/--sample`，无 ImportError。

- [ ] **Step 4: 提交**

```bash
git add scripts/build_lora_dataset.py
git commit -m "feat(dataset): LoRA 数据集编排脚本"
```

---

## Task 9: 端到端冒烟运行 + 全量单测 + 交付说明

**Files:**
- Create: `docs/dataset_provenance.md`
- 运行产物：`ito_lora_dataset.json`、`ito_lora_dataset.meta.jsonl`

前置：本机已 `ollama serve` 且已 `ollama pull qwen2.5:7b`；`indexes/hierarchy/corpus/faiss` 已建库（已存在）。

- [ ] **Step 1: 全量单测通过**

Run: `../.venv/Scripts/python.exe -m pytest tests/dataset_gen/ -v`
Expected: 全部 passed（约 35 项），0 failed。

- [ ] **Step 2: 小批冒烟运行（target=3）**

Run: `../.venv/Scripts/python.exe scripts/build_lora_dataset.py --target 3 --sample 40`
Expected: 进度条结束后打印 `完成: 成功 3 | 跳过 .. | 失败 .. | 尝试 ..`，生成两个输出文件。

- [ ] **Step 3: 校验输出为合法 JSON 且字段齐全**

Run:
```bash
../.venv/Scripts/python.exe -c "import json; d=json.load(open('ito_lora_dataset.json',encoding='utf-8')); assert isinstance(d,list) and len(d)>=3; assert all(set(x)=={'instruction','input','output'} and all(x.values()) for x in d); print('OK', len(d), '条'); print(d[0])"
```
Expected: `OK 3 条` 并打印第一条样本；断言不触发。

- [ ] **Step 4: 校验禁语未泄漏到 output**

Run:
```bash
../.venv/Scripts/python.exe -c "import json; from dataset_gen.quality import has_banned_phrase; d=json.load(open('ito_lora_dataset.json',encoding='utf-8')); bad=[x for x in d if has_banned_phrase(x['output'])]; print('禁语条数:', len(bad)); assert not bad"
```
Expected: `禁语条数: 0`

- [ ] **Step 5: 验证断点续传（再跑一次 target=5，应跳过已完成 QID 并续写）**

Run: `../.venv/Scripts/python.exe scripts/build_lora_dataset.py --target 5 --sample 60`
Expected: 最终 `成功 5`（在已有 3 条基础上续写到 5），`ito_lora_dataset.json` 总条数为 5，无重复 QID（meta 中 QID 唯一）。

校验无重复 QID：
```bash
../.venv/Scripts/python.exe -c "import json; q=[json.loads(l)['QID'] for l in open('ito_lora_dataset.meta.jsonl',encoding='utf-8') if l.strip()]; print('meta 条数', len(q), '唯一', len(set(q))); assert len(q)==len(set(q))"
```
Expected: `meta 条数 5 唯一 5`

- [ ] **Step 6: 写交付溯源说明**

Create `docs/dataset_provenance.md`：

```markdown
# ito_lora_dataset 数据来源说明

## 生成方式
1. **种子**：真实车主问题脱敏取自 `doc/AutoMaster_TrainSet.csv`（AutoMaster 通用汽车问答），
   仅保留与新能源通用场景相关的问题（胎压/空调/蓝牙/车机/雷达/钥匙/充电/刹车等关键词过滤）。
2. **依据**：每个问题经 RAG 检索问界（AITO）官方车主手册（合并 `corpus` 索引，向量+BM25 hybrid），
   仅当检索到 ≥2 条相关 chunk 且最高相似度 ≥0.30 时才采用，否则跳过，避免无依据编造。
3. **改写**：本地 qwen2.5:7b（`format='json'` + Few-Shot 约束）按 5 类指令
   （直接问答/步骤指导/故障分析/术语解释/安全提醒）改写为 {instruction, input, output}。
4. **过滤**：自动剔除字段缺失、含「根据手册/作为AI」等穿帮语、引用标记的样本。

## 质量复核
- 每条样本的来源章节与检索分数记录于 `ito_lora_dataset.meta.jsonl`，可逐条溯源抽查。
- 交付前建议人工抽查 meta 中 max_score 偏低的样本，确认 output 与来源章节一致。

## 字段
- `instruction`：给汽车助手的任务说明
- `input`：规范化的用户问句
- `output`：基于问界手册的准确回答
```

- [ ] **Step 7: 提交（仅脚本与文档；产物按需另议是否入库）**

```bash
git add docs/dataset_provenance.md
git commit -m "docs(dataset): 数据来源与溯源说明"
```

注：`ito_lora_dataset.json` / `.meta.jsonl` 是否纳入版本库由用户决定（数据产物通常不入库或用 LFS）；本任务默认不 `git add` 产物。

---

## Self-Review 记录

- **Spec 覆盖**：CSV 清洗采样(Task2)、本地/云端后端+JSON 约束(Task5)、Few-Shot(Task4)、5 类指令(Task4/7)、弱检索跳过(Task3/8)、断点续传(Task6/8)、tqdm+异常隔离(Task8)、溯源 meta(Task8/9)、数据来源说明(Task9) — 均有对应任务。
- **轻量检索**：overrides 关闭 query_rewrite + keyword_search + bookmark + verification（Task7），与 spec 修正一致，检索零 Ollama 调用。
- **类型一致**：`extract_input_text/is_weak_retrieval/validate_sample/extract_json/make_backend/write_samples/append_meta/load_done_qids/load_samples/build_dataset_prompt` 在 Task8 的引用与各定义任务签名一致。
- **无占位符**：所有代码步骤均含完整代码与可执行命令。
```
