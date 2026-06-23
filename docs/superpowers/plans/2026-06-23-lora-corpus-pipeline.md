# LoRA 语料生成流水线 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建 RAG 证据驱动、单车型绑定、chunk 反推的 LoRA 语料生成流水线，输出严格 `{instruction,input,output}` 数据集 + 完整 meta/rejected/report。

**Architecture:** 新建独立包 `lora_gen/`，复用现成 `config_loader` / `retrieve.Retriever` / `retrieve.model_router` / `corpus` 索引。纯函数模块（schema/chunks/registry/prompts/quality/answerability/export）先行 TDD；`backends`/`pipeline` 编排 LLM 与检索，断点续传；CLI `scripts/build_lora_dataset.py` 串联。

**Tech Stack:** Python 3.12、pytest、PyYAML、DashScope(OpenAI SDK) `qwen3.6-plus`、本地 ollama `qwen2.5:7b`、现有 FAISS+BM25 检索栈。

**约定：** 所有命令用项目 venv 解释器：`../.venv/Scripts/python.exe`（项目根 = `week1/qa_rag_vehicle`）。每个 Task 末尾 commit。设计依据见 `docs/superpowers/specs/2026-06-23-lora-corpus-pipeline-design.md`。

## File Structure

| 文件 | 职责 |
|---|---|
| `lora_gen/__init__.py` | 包标识（空） |
| `lora_gen/dgconfig.py` | 加载 `config/dataset_gen.yaml`，含默认值与 config hash |
| `lora_gen/schema.py` | `Sample` / `SampleMeta` / `Rejected` dataclass + `to_record` |
| `lora_gen/chunks.py` | `Chunk`、`load_corpus`、可用性过滤、`section_blacklist`、`section_to_task_type` |
| `lora_gen/registry.py` | `PlanItem`、`build_plan`（round-robin + per-vehicle min/max） |
| `lora_gen/prompts.py` | instruction 模板池 + `pick_instruction` + 三个 prompt 构造函数 |
| `lora_gen/backends.py` | `Backend` 协议、Cloud/Local 实现、`make_backend`、`extract_json` |
| `lora_gen/answerability.py` | `RetrievedChunk`、`Verdict`、`evaluate`（分档门 + partial 配额） |
| `lora_gen/quality.py` | 各拒绝规则纯函数 + `strip_leakage` + `run_quality` |
| `lora_gen/export.py` | 写 dataset/meta/rejected、`stratified_split`、`build_report` |
| `lora_gen/pipeline.py` | 编排 + checkpoint/resume 助手 + `run` |
| `scripts/build_lora_dataset.py` | CLI 入口 |
| `config/dataset_gen.yaml` | 阈值/分布/黑名单/长度配置 |
| `tests/lora_gen/test_*.py` | 各模块单测 |

---

### Task 0: 包脚手架与配置

**Files:**
- Create: `lora_gen/__init__.py`, `tests/lora_gen/__init__.py`, `config/dataset_gen.yaml`, `lora_gen/dgconfig.py`
- Test: `tests/lora_gen/test_dgconfig.py`

- [ ] **Step 1: 创建空包文件与配置**

`lora_gen/__init__.py`：空文件。
`tests/lora_gen/__init__.py`：空文件。

`config/dataset_gen.yaml`：
```yaml
target_size: 100
oversample_factor: 2.5     # 候选计划量 = ceil(target_size * oversample_factor)，应对回检/质检拒绝
max_attempts: 400          # 处理候选数硬上限，避免无限循环
backend: cloud
judge_backend: local
local_model: "qwen2.5:7b"
vehicle_subset: []
per_vehicle_min: 3
per_vehicle_max: 10
generation_retries: 2
seed: 20260623
task_distribution: {直接问答: 0.30, 步骤指导: 0.25, 故障分析: 0.20, 术语解释: 0.15, 安全提醒: 0.10}
answerability:
  topk: 5
  min_retrieved: 2
  seed_score_min: 0.30
  seed_score_min_single: 0.35
  max_score_min: 0.35
  strong_seed_rank_max: 3
  strong_max_score_min: 0.45
  strong_same_section_min: 2
  partial_ok_quota: 0.08
chunks:
  min_chars: 200
  section_blacklist: [前言, 目录, 免责声明, 版权, 隐私, 单位, 术语, 缩略语, 修订记录, 联系方式]
quality:
  max_output_chars:
    直接问答: 400
    步骤指导: 800
    故障分析: 700
    术语解释: 400
    安全提醒: 500
  dup_threshold: 0.90
export:
  train_dev_split: 0.9
```

- [ ] **Step 2: 写失败测试**

`tests/lora_gen/test_dgconfig.py`：
```python
from lora_gen.dgconfig import load_dg_config

def test_loads_defaults_and_hash():
    cfg = load_dg_config()
    assert cfg.target_size == 100
    assert cfg.answerability["partial_ok_quota"] == 0.08
    assert cfg.quality["max_output_chars"]["步骤指导"] == 800
    assert isinstance(cfg.config_hash, str) and len(cfg.config_hash) == 12

def test_override_target(tmp_path):
    cfg = load_dg_config(target_override=50)
    assert cfg.target_size == 50
```

- [ ] **Step 3: 运行确认失败**

Run: `../.venv/Scripts/python.exe -m pytest tests/lora_gen/test_dgconfig.py -v`
Expected: FAIL（ModuleNotFoundError: lora_gen.dgconfig）

- [ ] **Step 4: 实现 dgconfig.py**

`lora_gen/dgconfig.py`：
```python
"""加载 config/dataset_gen.yaml。"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PATH = ROOT / "config" / "dataset_gen.yaml"


@dataclass
class DGConfig:
    raw: dict[str, Any]
    config_hash: str

    @property
    def target_size(self) -> int:
        return int(self.raw["target_size"])

    @property
    def backend(self) -> str:
        return self.raw["backend"]

    @property
    def judge_backend(self) -> str:
        return self.raw["judge_backend"]

    @property
    def answerability(self) -> dict:
        return self.raw["answerability"]

    @property
    def quality(self) -> dict:
        return self.raw["quality"]

    @property
    def chunks(self) -> dict:
        return self.raw["chunks"]

    def get(self, *keys: str, default: Any = None) -> Any:
        node: Any = self.raw
        for k in keys:
            if not isinstance(node, dict) or k not in node:
                return default
            node = node[k]
        return node


def load_dg_config(path: str | Path | None = None, *, target_override: int | None = None) -> DGConfig:
    p = Path(path) if path else DEFAULT_PATH
    text = p.read_text(encoding="utf-8")
    raw = yaml.safe_load(text)
    if target_override is not None:
        raw["target_size"] = target_override
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
    return DGConfig(raw=raw, config_hash=digest)
```

- [ ] **Step 5: 运行确认通过**

Run: `../.venv/Scripts/python.exe -m pytest tests/lora_gen/test_dgconfig.py -v`
Expected: PASS（2 passed）

- [ ] **Step 6: Commit**

```bash
git add lora_gen/__init__.py tests/lora_gen/__init__.py config/dataset_gen.yaml lora_gen/dgconfig.py tests/lora_gen/test_dgconfig.py
git commit -m "feat(lora_gen): 脚手架与 dataset_gen 配置加载"
```

---

### Task 1: schema.py — 数据结构

**Files:**
- Create: `lora_gen/schema.py`
- Test: `tests/lora_gen/test_schema.py`

- [ ] **Step 1: 写失败测试**

`tests/lora_gen/test_schema.py`：
```python
from lora_gen.schema import Sample, SampleMeta, Rejected

def test_sample_record_strict_keys():
    s = Sample(instruction="i", input="q", output="a")
    assert s.to_record() == {"instruction": "i", "input": "q", "output": "a"}

def test_meta_roundtrip_keys():
    m = SampleMeta(
        qid="Q1", model_id="M", model_display="Md", doc_type="owner_manual",
        section_path="A>B", task_type="直接问答", seed_chunk_id="c1",
        seed_preview="前50字", seed_score=0.4, evidence_chunk_ids=["c1", "c2"],
        evidence_previews=["p1", "p2"], retrieval_scores=[0.4, 0.3], max_score=0.4,
        seed_rank=1, same_section_count=2, evidence_sufficiency="full",
        accept_tier="strong", backend="cloud", gen_question_raw="{}",
    )
    rec = m.to_record()
    assert rec["accept_tier"] == "strong"
    assert rec["evidence_chunk_ids"] == ["c1", "c2"]

def test_rejected_defaults():
    r = Rejected(qid="Q1", model_id="M", section_path="A", task_type="直接问答",
                 seed_chunk_id="c1", seed_preview="p", reject_stage="quality",
                 reject_reason="vehicle_conflict", reject_detail="found M2")
    assert r.to_record()["question"] == ""
```

- [ ] **Step 2: 运行确认失败**

Run: `../.venv/Scripts/python.exe -m pytest tests/lora_gen/test_schema.py -v`
Expected: FAIL（ModuleNotFoundError）

- [ ] **Step 3: 实现 schema.py**

`lora_gen/schema.py`：
```python
"""LoRA 样本与中间字段数据结构。"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class Sample:
    instruction: str
    input: str
    output: str

    def to_record(self) -> dict:
        return {"instruction": self.instruction, "input": self.input, "output": self.output}


@dataclass
class SampleMeta:
    qid: str
    model_id: str
    model_display: str
    doc_type: str
    section_path: str
    task_type: str
    seed_chunk_id: str
    seed_preview: str
    seed_score: float
    evidence_chunk_ids: list[str]
    evidence_previews: list[str]
    retrieval_scores: list[float]
    max_score: float
    seed_rank: int
    same_section_count: int
    evidence_sufficiency: str
    accept_tier: str
    backend: str
    gen_question_raw: str

    def to_record(self) -> dict:
        return asdict(self)


@dataclass
class Rejected:
    qid: str
    model_id: str
    section_path: str
    task_type: str
    seed_chunk_id: str
    seed_preview: str
    reject_stage: str
    reject_reason: str
    reject_detail: str
    question: str = ""

    def to_record(self) -> dict:
        return asdict(self)
```

- [ ] **Step 4: 运行确认通过**

Run: `../.venv/Scripts/python.exe -m pytest tests/lora_gen/test_schema.py -v`
Expected: PASS（3 passed）

- [ ] **Step 5: Commit**

```bash
git add lora_gen/schema.py tests/lora_gen/test_schema.py
git commit -m "feat(lora_gen): Sample/SampleMeta/Rejected 数据结构"
```

---

### Task 2: chunks.py — 语料加载与可用性过滤

**Files:**
- Create: `lora_gen/chunks.py`
- Test: `tests/lora_gen/test_chunks.py`

- [ ] **Step 1: 写失败测试**

`tests/lora_gen/test_chunks.py`：
```python
import json
from lora_gen.chunks import (
    Chunk, load_corpus, is_usable_chunk, is_blacklisted, section_to_task_type,
)

def test_section_to_task_type():
    assert section_to_task_type("用户提示>故障警示灯说明") == "故障分析"
    assert section_to_task_type("附录>术语与缩略语") == "术语解释"
    assert section_to_task_type("维护保养>更换雨刮") == "步骤指导"
    assert section_to_task_type("行驶安全>儿童安全座椅") == "安全提醒"
    assert section_to_task_type("车辆控制>空调") == "直接问答"

def test_blacklist():
    bl = ["前言", "目录", "术语"]
    assert is_blacklisted("文档前言", bl) is True
    assert is_blacklisted("车辆控制>空调", bl) is False

def test_is_usable_chunk():
    bl = ["目录"]
    short = Chunk("c1", "太短", "M", "owner_manual", "车辆控制>空调", 1)
    good = Chunk("c2", "正常内容" * 60, "M", "owner_manual", "车辆控制>空调", 1)
    toc = Chunk("c3", "正常内容" * 60, "M", "owner_manual", "目录", 1)
    assert is_usable_chunk(short, min_chars=200, blacklist=bl) is False
    assert is_usable_chunk(good, min_chars=200, blacklist=bl) is True
    assert is_usable_chunk(toc, min_chars=200, blacklist=bl) is False

def test_load_corpus(tmp_path):
    d = tmp_path / "corpus"
    d.mkdir()
    payload = {
        "chunk_ids": ["c1"],
        "texts": ["内容"],
        "metadatas": [{"vehicle_model": "M9", "doc_type": "owner_manual",
                       "section_path": "A>B", "page": 3}],
    }
    (d / "bm25_corpus.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    chunks = load_corpus(d)
    assert len(chunks) == 1
    assert chunks[0].chunk_id == "c1" and chunks[0].vehicle_model == "M9"
    assert chunks[0].section_path == "A>B"
```

- [ ] **Step 2: 运行确认失败**

Run: `../.venv/Scripts/python.exe -m pytest tests/lora_gen/test_chunks.py -v`
Expected: FAIL（ModuleNotFoundError）

- [ ] **Step 3: 实现 chunks.py**

`lora_gen/chunks.py`：
```python
"""corpus 加载、可用性过滤、section→task_type 映射。"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

# 顺序敏感：先匹配到的规则胜出。
TASK_SECTION_RULES: list[tuple[tuple[str, ...], str]] = [
    (("故障", "警示", "报警", "异常", "警告灯", "提示灯"), "故障分析"),
    (("名词", "定义", "简介", "术语", "缩略", "释义"), "术语解释"),
    (("检查", "保养", "操作", "更换", "安装", "加注", "清洗", "调节"), "步骤指导"),
    (("安全", "儿童", "安全带", "气囊", "乘员", "警告"), "安全提醒"),
]


@dataclass
class Chunk:
    chunk_id: str
    text: str
    vehicle_model: str
    doc_type: str
    section_path: str
    page: object


def section_to_task_type(section_path: str) -> str:
    for keys, tt in TASK_SECTION_RULES:
        if any(k in section_path for k in keys):
            return tt
    return "直接问答"


def is_blacklisted(section_path: str, blacklist: list[str]) -> bool:
    return any(b in section_path for b in blacklist)


def is_usable_chunk(chunk: Chunk, *, min_chars: int, blacklist: list[str]) -> bool:
    text = chunk.text.strip()
    if len(text) < min_chars:
        return False
    if is_blacklisted(chunk.section_path, blacklist):
        return False
    return True


def load_corpus(index_path: Path) -> list[Chunk]:
    data = json.loads((index_path / "bm25_corpus.json").read_text(encoding="utf-8"))
    out: list[Chunk] = []
    for cid, text, meta in zip(
        data["chunk_ids"], data["texts"], data["metadatas"], strict=True
    ):
        out.append(
            Chunk(
                chunk_id=cid,
                text=text,
                vehicle_model=meta.get("vehicle_model", ""),
                doc_type=meta.get("doc_type", ""),
                section_path=meta.get("section_path", ""),
                page=meta.get("page", "?"),
            )
        )
    return out


def usable_chunks_by_model(
    chunks: list[Chunk], *, min_chars: int, blacklist: list[str]
) -> dict[str, list[Chunk]]:
    by_model: dict[str, list[Chunk]] = {}
    for c in chunks:
        if is_usable_chunk(c, min_chars=min_chars, blacklist=blacklist):
            by_model.setdefault(c.vehicle_model, []).append(c)
    return by_model
```

- [ ] **Step 4: 运行确认通过**

Run: `../.venv/Scripts/python.exe -m pytest tests/lora_gen/test_chunks.py -v`
Expected: PASS（4 passed）

- [ ] **Step 5: Commit**

```bash
git add lora_gen/chunks.py tests/lora_gen/test_chunks.py
git commit -m "feat(lora_gen): corpus 加载/可用性过滤/section→task_type"
```

---

### Task 3: registry.py — round-robin 采样计划

**Files:**
- Create: `lora_gen/registry.py`
- Test: `tests/lora_gen/test_registry.py`

- [ ] **Step 1: 写失败测试**

`tests/lora_gen/test_registry.py`：
```python
import random
from lora_gen.chunks import Chunk
from lora_gen.registry import PlanItem, build_plan

def _mk(model, n, section="车辆控制>空调"):
    return [Chunk(f"{model}-c{i}", "x" * 300, model, "owner_manual", section, 1) for i in range(n)]

def test_round_robin_balances_and_respects_max():
    by_model = {"A": _mk("A", 20), "B": _mk("B", 20), "C": _mk("C", 20)}
    plan = build_plan(by_model, target=12, per_vehicle_min=1, per_vehicle_max=10,
                      vehicle_subset=[], rng=random.Random(1))
    assert len(plan) == 12
    from collections import Counter
    counts = Counter(p.model_id for p in plan)
    # 12 / 3 模型 → 每个 4 条，均衡
    assert counts == {"A": 4, "B": 4, "C": 4}

def test_per_vehicle_max_caps():
    by_model = {"A": _mk("A", 20), "B": _mk("B", 2)}
    plan = build_plan(by_model, target=100, per_vehicle_min=1, per_vehicle_max=5,
                      vehicle_subset=[], rng=random.Random(1))
    from collections import Counter
    counts = Counter(p.model_id for p in plan)
    assert counts["A"] == 5  # 受 max 限制
    assert counts["B"] == 2  # chunk 不足

def test_subset_filters_models():
    by_model = {"A": _mk("A", 10), "B": _mk("B", 10)}
    plan = build_plan(by_model, target=6, per_vehicle_min=1, per_vehicle_max=10,
                      vehicle_subset=["A"], rng=random.Random(1))
    assert {p.model_id for p in plan} == {"A"}

def test_task_type_assigned_from_section():
    by_model = {"A": _mk("A", 4, section="维护保养>更换雨刮")}
    plan = build_plan(by_model, target=2, per_vehicle_min=1, per_vehicle_max=10,
                      vehicle_subset=[], rng=random.Random(1))
    assert all(p.task_type == "步骤指导" for p in plan)
```

- [ ] **Step 2: 运行确认失败**

Run: `../.venv/Scripts/python.exe -m pytest tests/lora_gen/test_registry.py -v`
Expected: FAIL（ModuleNotFoundError）

- [ ] **Step 3: 实现 registry.py**

`lora_gen/registry.py`：
```python
"""车型 round-robin 采样计划：model_id → chunk → task_type。"""
from __future__ import annotations

import random
from dataclasses import dataclass

from lora_gen.chunks import Chunk, section_to_task_type


@dataclass
class PlanItem:
    model_id: str
    chunk_id: str
    section_path: str
    task_type: str


def build_plan(
    chunks_by_model: dict[str, list[Chunk]],
    *,
    target: int,
    per_vehicle_min: int,
    per_vehicle_max: int,
    vehicle_subset: list[str],
    rng: random.Random,
) -> list[PlanItem]:
    models = list(vehicle_subset) if vehicle_subset else sorted(chunks_by_model)
    pools: dict[str, list[Chunk]] = {m: list(chunks_by_model.get(m, [])) for m in models}
    for m in models:
        rng.shuffle(pools[m])

    counts = {m: 0 for m in models}
    cursor = {m: 0 for m in models}
    plan: list[PlanItem] = []

    progressed = True
    while len(plan) < target and progressed:
        progressed = False
        for m in models:
            if len(plan) >= target:
                break
            if counts[m] >= per_vehicle_max or cursor[m] >= len(pools[m]):
                continue
            ch = pools[m][cursor[m]]
            cursor[m] += 1
            counts[m] += 1
            plan.append(
                PlanItem(
                    model_id=m,
                    chunk_id=ch.chunk_id,
                    section_path=ch.section_path,
                    task_type=section_to_task_type(ch.section_path),
                )
            )
            progressed = True
    return plan


def under_min_models(plan: list[PlanItem], models: list[str], per_vehicle_min: int) -> list[str]:
    """返回未达 per_vehicle_min 的车型，供 report 警示。"""
    from collections import Counter

    counts = Counter(p.model_id for p in plan)
    return [m for m in models if counts.get(m, 0) < per_vehicle_min]
```

- [ ] **Step 4: 运行确认通过**

Run: `../.venv/Scripts/python.exe -m pytest tests/lora_gen/test_registry.py -v`
Expected: PASS（4 passed）

- [ ] **Step 5: Commit**

```bash
git add lora_gen/registry.py tests/lora_gen/test_registry.py
git commit -m "feat(lora_gen): round-robin 采样计划 + per-vehicle 配额"
```

---

### Task 4: prompts.py — instruction 模板池与 prompt 构造

**Files:**
- Create: `lora_gen/prompts.py`
- Test: `tests/lora_gen/test_prompts.py`

- [ ] **Step 1: 写失败测试**

`tests/lora_gen/test_prompts.py`：
```python
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
```

- [ ] **Step 2: 运行确认失败**

Run: `../.venv/Scripts/python.exe -m pytest tests/lora_gen/test_prompts.py -v`
Expected: FAIL（ModuleNotFoundError）

- [ ] **Step 3: 实现 prompts.py**

`lora_gen/prompts.py`：
```python
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
```

- [ ] **Step 4: 运行确认通过**

Run: `../.venv/Scripts/python.exe -m pytest tests/lora_gen/test_prompts.py -v`
Expected: PASS（5 passed）

- [ ] **Step 5: Commit**

```bash
git add lora_gen/prompts.py tests/lora_gen/test_prompts.py
git commit -m "feat(lora_gen): instruction 模板池与各阶段 prompt"
```

---

### Task 5: backends.py — LLM 后端与 JSON 抽取

**Files:**
- Create: `lora_gen/backends.py`
- Test: `tests/lora_gen/test_backends.py`

- [ ] **Step 1: 写失败测试**

`tests/lora_gen/test_backends.py`：
```python
import pytest
from lora_gen.backends import extract_json, GenerationError

def test_extract_plain_json():
    assert extract_json('{"output": "ok"}') == {"output": "ok"}

def test_extract_fenced_json():
    raw = "```json\n{\"label\": \"full\", \"conflict\": false}\n```"
    assert extract_json(raw) == {"label": "full", "conflict": False}

def test_extract_with_surrounding_text():
    raw = "好的，结果是 {\"question\": \"空调怎么开\"} 以上。"
    assert extract_json(raw) == {"question": "空调怎么开"}

def test_extract_garbage_raises():
    with pytest.raises(GenerationError):
        extract_json("没有任何 JSON 内容")

def test_extract_broken_json_raises():
    with pytest.raises(GenerationError):
        extract_json('{"output": ')

def test_extract_nested_braces_not_greedy():
    # 贪婪正则会把后面的 } 一起吞掉；brace-balanced 应只取第一个完整对象
    raw = '前言 {"output": {"a": 1}} 结尾 {"b": 2}'
    assert extract_json(raw) == {"output": {"a": 1}}

def test_extract_fenced_priority_over_braces():
    raw = '忽略这个 {x} ```json\n{"label": "full"}\n``` 尾部'
    assert extract_json(raw) == {"label": "full"}
```

- [ ] **Step 2: 运行确认失败**

Run: `../.venv/Scripts/python.exe -m pytest tests/lora_gen/test_backends.py -v`
Expected: FAIL（ModuleNotFoundError）

- [ ] **Step 3: 实现 backends.py**

`lora_gen/backends.py`：
```python
"""LLM 后端抽象：cloud(qwen3.6-plus) / local(ollama)，及 JSON 抽取。"""
from __future__ import annotations

import json
import re
from typing import Protocol


class GenerationError(Exception):
    """JSON 解析失败 → reject_reason=json_parse_failed。"""


class Backend(Protocol):
    def complete(self, prompt: str) -> str: ...


class CloudBackend:
    def __init__(self, config):
        from generate.qwen_client import QwenClient

        self.client = QwenClient(config)

    def complete(self, prompt: str) -> str:
        text, _ = self.client.chat(prompt)
        return text


class LocalBackend:
    def __init__(self, model: str = "qwen2.5:7b"):
        import ollama

        self._ollama = ollama
        self.model = model

    def complete(self, prompt: str) -> str:
        resp = self._ollama.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            format="json",
            options={"temperature": 0.1},
        )
        return resp["message"]["content"]


def make_backend(name: str, config, dg_config) -> Backend:
    if name == "cloud":
        return CloudBackend(config)
    return LocalBackend(dg_config.get("local_model", default="qwen2.5:7b"))


_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.S)


def _balanced_object(text: str) -> str | None:
    """返回第一个括号平衡的 {...} 子串（考虑字符串内转义），无则 None。"""
    start = text.find("{")
    while start != -1:
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
        start = text.find("{", start + 1)
    return None


def extract_json(text: str) -> dict:
    # 1) fenced ```json {...}``` 优先
    m = _FENCE_RE.search(text)
    candidate = m.group(1) if m else _balanced_object(text)
    if candidate is None:
        raise GenerationError(f"json_parse_failed: 无 JSON 片段: {text[:80]!r}")
    try:
        return json.loads(candidate)
    except json.JSONDecodeError as e:
        raise GenerationError(f"json_parse_failed: {e}") from e
```

- [ ] **Step 4: 运行确认通过**

Run: `../.venv/Scripts/python.exe -m pytest tests/lora_gen/test_backends.py -v`
Expected: PASS（5 passed）

- [ ] **Step 5: Commit**

```bash
git add lora_gen/backends.py tests/lora_gen/test_backends.py
git commit -m "feat(lora_gen): LLM 后端抽象与 JSON 抽取"
```

---

### Task 6: answerability.py — 多信号分档门

**Files:**
- Create: `lora_gen/answerability.py`
- Test: `tests/lora_gen/test_answerability.py`

- [ ] **Step 1: 写失败测试**

`tests/lora_gen/test_answerability.py`：
```python
from lora_gen.answerability import RetrievedChunk, evaluate

CFG = {
    "topk": 5, "min_retrieved": 2,
    "seed_score_min": 0.30, "seed_score_min_single": 0.35, "max_score_min": 0.35,
    "strong_seed_rank_max": 3, "strong_max_score_min": 0.45, "strong_same_section_min": 2,
    "partial_ok_quota": 0.08,
}

def _r(cid, score, section="A"):
    return RetrievedChunk(chunk_id=cid, score=score, section_path=section)

def test_strong_accept():
    retr = [_r("seed", 0.5, "A"), _r("c2", 0.46, "A"), _r("c3", 0.2, "B")]
    v = evaluate(seed_chunk_id="seed", seed_section="A", retrieved=retr,
                 judge_label="full", judge_conflict=False, cfg=CFG,
                 partial_count=0, target=100)
    assert v.accept and v.tier == "strong"

def test_seed_not_returned_reject():
    retr = [_r("c2", 0.5, "A"), _r("c3", 0.4, "A")]
    v = evaluate(seed_chunk_id="seed", seed_section="A", retrieved=retr,
                 judge_label="full", judge_conflict=False, cfg=CFG,
                 partial_count=0, target=100)
    assert not v.accept and v.reason == "seed_not_returned"

def test_single_chunk_full_accept():
    retr = [_r("seed", 0.4, "A")]  # 只有 1 条，但 full 且 seed_score≥0.35
    v = evaluate(seed_chunk_id="seed", seed_section="A", retrieved=retr,
                 judge_label="full", judge_conflict=False, cfg=CFG,
                 partial_count=0, target=100)
    assert v.accept and v.tier == "single_chunk_full"

def test_too_few_chunks_reject_when_not_single_full():
    retr = [_r("seed", 0.4, "A")]  # 1 条但 judge=partial → 不符合 single_chunk_full
    v = evaluate(seed_chunk_id="seed", seed_section="A", retrieved=retr,
                 judge_label="partial", judge_conflict=False, cfg=CFG,
                 partial_count=0, target=100)
    assert not v.accept and v.reason == "too_few_chunks"

def test_partial_ok_within_quota():
    retr = [_r("seed", 0.32, "A"), _r("c2", 0.31, "A")]
    v = evaluate(seed_chunk_id="seed", seed_section="A", retrieved=retr,
                 judge_label="partial", judge_conflict=False, cfg=CFG,
                 partial_count=0, target=100)
    assert v.accept and v.tier == "partial_ok"

def test_partial_quota_full_reject():
    retr = [_r("seed", 0.32, "A"), _r("c2", 0.31, "A")]
    v = evaluate(seed_chunk_id="seed", seed_section="A", retrieved=retr,
                 judge_label="partial", judge_conflict=False, cfg=CFG,
                 partial_count=8, target=100)  # 8 >= floor(0.08*100)=8
    assert not v.accept and v.reason == "partial_quota_full"

def test_evidence_conflict_reject():
    retr = [_r("seed", 0.5, "A"), _r("c2", 0.46, "A")]
    v = evaluate(seed_chunk_id="seed", seed_section="A", retrieved=retr,
                 judge_label="full", judge_conflict=True, cfg=CFG,
                 partial_count=0, target=100)
    assert not v.accept and v.reason == "evidence_conflict"

def test_low_score_reject():
    retr = [_r("seed", 0.1, "A"), _r("c2", 0.1, "A")]
    v = evaluate(seed_chunk_id="seed", seed_section="A", retrieved=retr,
                 judge_label="full", judge_conflict=False, cfg=CFG,
                 partial_count=0, target=100)
    assert not v.accept and v.reason == "low_score"
```

- [ ] **Step 2: 运行确认失败**

Run: `../.venv/Scripts/python.exe -m pytest tests/lora_gen/test_answerability.py -v`
Expected: FAIL（ModuleNotFoundError）

- [ ] **Step 3: 实现 answerability.py**

`lora_gen/answerability.py`：
```python
"""RAG 回检门：多信号分档 + partial 配额。"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class RetrievedChunk:
    chunk_id: str
    score: float
    section_path: str


@dataclass
class Verdict:
    accept: bool
    tier: str
    reason: str
    signals: dict = field(default_factory=dict)


def evaluate(
    *,
    seed_chunk_id: str,
    seed_section: str,
    retrieved: list[RetrievedChunk],
    judge_label: str,
    judge_conflict: bool,
    cfg: dict,
    partial_count: int,
    target: int,
) -> Verdict:
    ordered = sorted(retrieved, key=lambda r: r.score, reverse=True)
    by_id = {r.chunk_id: i for i, r in enumerate(ordered)}
    seed_rank = by_id.get(seed_chunk_id, -1) + 1  # 1-based；0=未命中
    seed_score = next((r.score for r in ordered if r.chunk_id == seed_chunk_id), 0.0)
    max_score = ordered[0].score if ordered else 0.0
    same_section = sum(1 for r in ordered if r.section_path == seed_section)
    n = len(ordered)
    topk = cfg["topk"]
    seed_in_topk = 0 < seed_rank <= topk

    signals = {
        "seed_rank": seed_rank, "seed_score": seed_score, "max_score": max_score,
        "same_section_count": same_section, "retrieved": n,
        "judge_label": judge_label, "judge_conflict": judge_conflict,
    }

    def reject(reason: str) -> Verdict:
        return Verdict(accept=False, tier="", reason=reason, signals=signals)

    def accept(tier: str) -> Verdict:
        return Verdict(accept=True, tier=tier, reason="", signals=signals)

    if judge_conflict:
        return reject("evidence_conflict")
    if not seed_in_topk:
        return reject("seed_not_returned")
    if max_score < cfg["max_score_min"] or seed_score < cfg["seed_score_min"]:
        # single_chunk_full 用更高的 seed 阈值单独判
        if not (n == 1 and judge_label == "full" and seed_score >= cfg["seed_score_min_single"]):
            return reject("low_score")

    # strong
    if (
        judge_label == "full"
        and seed_rank <= cfg["strong_seed_rank_max"]
        and max_score >= cfg["strong_max_score_min"]
        and same_section >= cfg["strong_same_section_min"]
        and n >= cfg["min_retrieved"]
    ):
        return accept("strong")

    # single_chunk_full（min_retrieved 例外）
    if n == 1 and judge_label == "full" and seed_score >= cfg["seed_score_min_single"]:
        return accept("single_chunk_full")

    # ok
    if judge_label == "full" and n >= cfg["min_retrieved"] and seed_score >= cfg["seed_score_min"]:
        return accept("ok")

    # partial → 配额内 partial_ok
    if judge_label == "partial" and n >= cfg["min_retrieved"] and seed_score >= cfg["seed_score_min"]:
        quota = math.floor(cfg["partial_ok_quota"] * target)
        if partial_count < quota:
            return accept("partial_ok")
        return reject("partial_quota_full")

    if n < cfg["min_retrieved"]:
        return reject("too_few_chunks")
    return reject("insufficient_evidence")
```

- [ ] **Step 4: 运行确认通过**

Run: `../.venv/Scripts/python.exe -m pytest tests/lora_gen/test_answerability.py -v`
Expected: PASS（8 passed）

- [ ] **Step 5: Commit**

```bash
git add lora_gen/answerability.py tests/lora_gen/test_answerability.py
git commit -m "feat(lora_gen): answerability 多信号分档门 + partial 配额"
```

---

### Task 7: quality.py — 拒绝规则

**Files:**
- Create: `lora_gen/quality.py`
- Test: `tests/lora_gen/test_quality.py`

- [ ] **Step 1: 写失败测试**

`tests/lora_gen/test_quality.py`：
```python
from config_loader import load_config
from lora_gen.schema import Sample
from lora_gen.quality import (
    strip_leakage, ungrounded_numbers, has_over_promise,
    insurance_warranty_mix, unsafe_without_guard, foreign_vehicles, run_quality,
)

CONFIG = load_config()

def test_strip_leakage():
    out = strip_leakage("根据手册，空调请按AUTO键。参考第12页。[1]")
    assert "根据手册" not in out and "第12页" not in out and "[1]" not in out
    assert "空调请按AUTO键" in out

def test_ungrounded_numbers():
    ev = "胎压建议为 2.5 bar。"
    assert ungrounded_numbers("请保持 2.5 bar 胎压", ev) == []
    assert "3.0bar" in [x.replace(" ", "") for x in ungrounded_numbers("请保持 3.0 bar 胎压", ev)]

def test_over_promise():
    assert has_over_promise("该服务永久免费", "服务说明") == ["永久", "免费"][:2] or \
           set(has_over_promise("该服务永久免费", "服务说明")) == {"永久", "免费"}
    assert has_over_promise("正常保养即可", "保养说明") == []

def test_insurance_warranty_mix():
    assert insurance_warranty_mix("此故障保险全赔", "电池故障说明") is True
    assert insurance_warranty_mix("请检查电池", "电池故障说明") is False

def test_unsafe_without_guard():
    assert unsafe_without_guard("动力电池起火时请自行拆解处理") is True
    assert unsafe_without_guard("动力电池起火时请立即联系授权服务中心") is False

def test_foreign_vehicles_detected():
    # 绑定 M9-2026 增程版，但 output 提到 M7 → 冲突
    foreign = foreign_vehicles("问界M7 2026增程版也支持", "问界M9-2026款增程版", CONFIG)
    assert foreign  # 非空

def test_foreign_vehicles_bound_alias_not_flagged():
    # 用别名提到绑定车型本身 → detect_models 解析回同一 id → 不算 foreign
    foreign = foreign_vehicles("M92026增程版也支持该功能", "问界M9-2026款增程版", CONFIG)
    assert foreign == []

def test_normalize_question_dedup_equivalence():
    from lora_gen.quality import normalize_question
    assert normalize_question(" 问界M9 空调 怎么 开？") == normalize_question("问界M9空调怎么开?")

def test_run_quality_vehicle_conflict_checks_input_and_output():
    s = Sample(instruction="i", input="问界M7 2026增程版怎么样", output="正常使用即可",
               )
    v = run_quality(s, evidence_text="正常使用即可", model_id="问界M9-2026款增程版",
                    config=CONFIG, max_chars=400)
    assert not v.ok and v.reason == "vehicle_conflict"

def test_run_quality_pass_returns_cleaned():
    s = Sample(instruction="i", input="问界M9 2026增程版空调怎么开",
               output="根据手册，按下AUTO键即可。")
    v = run_quality(s, evidence_text="按下AUTO键即可开启空调", model_id="问界M9-2026款增程版",
                    config=CONFIG, max_chars=400)
    assert v.ok
    assert "根据手册" not in v.cleaned_output
```

- [ ] **Step 2: 运行确认失败**

Run: `../.venv/Scripts/python.exe -m pytest tests/lora_gen/test_quality.py -v`
Expected: FAIL（ModuleNotFoundError）

- [ ] **Step 3: 实现 quality.py**

`lora_gen/quality.py`：
```python
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
```

- [ ] **Step 4: 运行确认通过**

Run: `../.venv/Scripts/python.exe -m pytest tests/lora_gen/test_quality.py -v`
Expected: PASS（8 passed）。如 `test_over_promise` 断言形式不稳，改为对 `set(...)` 断言（实现返回顺序为 `_OVER_PROMISE` 列表顺序）。

- [ ] **Step 5: Commit**

```bash
git add lora_gen/quality.py tests/lora_gen/test_quality.py
git commit -m "feat(lora_gen): quality 拒绝规则（车型冲突/数字/危险/承诺等）"
```

---

### Task 8: export.py — 导出与报告

**Files:**
- Create: `lora_gen/export.py`
- Test: `tests/lora_gen/test_export.py`

- [ ] **Step 1: 写失败测试**

`tests/lora_gen/test_export.py`：
```python
import json
import random
from lora_gen.schema import Sample
from lora_gen.export import stratified_split, build_report

def test_stratified_split_ratio():
    pairs = [(Sample("i", f"q{i}", "a"), "直接问答") for i in range(8)] + \
            [(Sample("i", f"s{i}", "a"), "步骤指导") for i in range(2)]
    train, dev = stratified_split(pairs, train_ratio=0.5, rng=random.Random(1))
    # 每类按比例：直接问答 8→4/4，步骤指导 2→1/1
    assert len(train) == 5 and len(dev) == 5

def test_build_report_counts():
    accepted_tiers = ["strong", "strong", "ok", "partial_ok"]
    task_types = ["直接问答", "步骤指导", "故障分析", "直接问答"]
    vehicles = ["A", "A", "B", "B"]
    reject_reasons = ["seed_not_returned", "low_score", "seed_not_returned"]
    md = build_report(
        accepted_tiers=accepted_tiers, task_types=task_types, vehicles=vehicles,
        reject_reasons=reject_reasons, corpus_fingerprint="fp", backend="cloud",
        config_hash="abc123", manual_check_ratio=0.1,
    )
    assert "seed_not_returned" in md and "strong" in md
    assert "cloud" in md and "abc123" in md
```

- [ ] **Step 2: 运行确认失败**

Run: `../.venv/Scripts/python.exe -m pytest tests/lora_gen/test_export.py -v`
Expected: FAIL（ModuleNotFoundError）

- [ ] **Step 3: 实现 export.py**

`lora_gen/export.py`：
```python
"""导出 dataset / meta / rejected / report / train-dev split。"""
from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

from lora_gen.schema import Rejected, Sample, SampleMeta


def write_dataset(samples: list[Sample], path: Path) -> None:
    path.write_text(
        json.dumps([s.to_record() for s in samples], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_meta(metas: list[SampleMeta], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for m in metas:
            f.write(json.dumps(m.to_record(), ensure_ascii=False) + "\n")


def write_rejected(rejected: list[Rejected], path: Path) -> None:
    path.write_text(
        json.dumps([r.to_record() for r in rejected], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def stratified_split(
    pairs: list[tuple[Sample, str]], *, train_ratio: float, rng: random.Random
) -> tuple[list[Sample], list[Sample]]:
    by_tt: dict[str, list[Sample]] = {}
    for s, tt in pairs:
        by_tt.setdefault(tt, []).append(s)
    train: list[Sample] = []
    dev: list[Sample] = []
    for tt in sorted(by_tt):
        items = list(by_tt[tt])
        rng.shuffle(items)
        n_train = round(len(items) * train_ratio)
        train.extend(items[:n_train])
        dev.extend(items[n_train:])
    return train, dev


def build_report(
    *,
    accepted_tiers: list[str],
    task_types: list[str],
    vehicles: list[str],
    reject_reasons: list[str],
    corpus_fingerprint: str,
    backend: str,
    config_hash: str,
    manual_check_ratio: float,
) -> str:
    def fmt(counter: Counter) -> str:
        return "\n".join(f"- {k}: {v}" for k, v in sorted(counter.items()))

    accepted = len(accepted_tiers)
    rejected = len(reject_reasons)
    total = accepted + rejected
    acc_rate = f"{accepted / total:.1%}" if total else "n/a"
    lines = [
        "# LoRA 数据生成报告",
        "",
        f"- corpus 指纹: `{corpus_fingerprint}`",
        f"- backend: {backend}",
        f"- dataset_gen.yaml hash: `{config_hash}`",
        f"- accepted: {accepted} / rejected: {rejected} / 通过率: {acc_rate}",
        f"- 人工抽检比例: {manual_check_ratio:.0%}",
        "",
        "## accept_tier 分布",
        fmt(Counter(accepted_tiers)),
        "",
        "## task_type 分布（accepted）",
        fmt(Counter(task_types)),
        "",
        "## 车型分布（accepted）",
        fmt(Counter(vehicles)),
        "",
        "## 拒绝原因分布",
        fmt(Counter(reject_reasons)),
        "",
    ]
    return "\n".join(lines)
```

- [ ] **Step 4: 运行确认通过**

Run: `../.venv/Scripts/python.exe -m pytest tests/lora_gen/test_export.py -v`
Expected: PASS（2 passed）

- [ ] **Step 5: Commit**

```bash
git add lora_gen/export.py tests/lora_gen/test_export.py
git commit -m "feat(lora_gen): 导出 dataset/meta/rejected/report + 分层 split"
```

---

### Task 9: pipeline.py — 编排与断点续传

**Files:**
- Create: `lora_gen/pipeline.py`
- Test: `tests/lora_gen/test_pipeline_checkpoint.py`

- [ ] **Step 1: 写失败测试（仅测纯函数 checkpoint 助手）**

`tests/lora_gen/test_pipeline_checkpoint.py`：
```python
from lora_gen.pipeline import load_done_qids, append_checkpoint, make_qid

def test_make_qid_stable():
    a = make_qid("问界M9-2026款增程版", "chunk-7", "步骤指导")
    b = make_qid("问界M9-2026款增程版", "chunk-7", "步骤指导")
    assert a == b and a.startswith("Q")

def test_checkpoint_roundtrip(tmp_path):
    cp = tmp_path / "checkpoint.jsonl"
    append_checkpoint(cp, "Q1", "accepted")
    append_checkpoint(cp, "Q2", "rejected:low_score")
    done = load_done_qids(cp)
    assert done == {"Q1", "Q2"}

def test_load_done_missing_file(tmp_path):
    assert load_done_qids(tmp_path / "nope.jsonl") == set()
```

- [ ] **Step 2: 运行确认失败**

Run: `../.venv/Scripts/python.exe -m pytest tests/lora_gen/test_pipeline_checkpoint.py -v`
Expected: FAIL（ModuleNotFoundError）

- [ ] **Step 3: 实现 pipeline.py**

`lora_gen/pipeline.py`：
```python
"""编排：plan → 问题生成 → RAG 回检 → 答案生成 → 质检 → 落盘；断点续传。"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import random
from dataclasses import dataclass, field
from pathlib import Path

from lora_gen.answerability import RetrievedChunk, Verdict, evaluate
from lora_gen.backends import GenerationError, extract_json
from lora_gen.chunks import Chunk, load_corpus, usable_chunks_by_model
from lora_gen.prompts import (
    answer_gen_prompt, judge_prompt, pick_instruction, question_gen_prompt,
)
from lora_gen.quality import normalize_question, run_quality
from lora_gen.registry import build_plan
from lora_gen.schema import Rejected, Sample, SampleMeta

log = logging.getLogger(__name__)


def make_qid(model_id: str, chunk_id: str, task_type: str) -> str:
    raw = f"{model_id}|{chunk_id}|{task_type}"
    return "Q" + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:10]


def load_done_qids(checkpoint_path: Path) -> set[str]:
    if not checkpoint_path.exists():
        return set()
    done: set[str] = set()
    for line in checkpoint_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            done.add(json.loads(line)["qid"])
    return done


def append_checkpoint(checkpoint_path: Path, qid: str, status: str) -> None:
    with checkpoint_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"qid": qid, "status": status}, ensure_ascii=False) + "\n")


def preview(text: str, n: int = 50) -> str:
    return text.strip().replace("\n", " ")[:n]


@dataclass
class RunResult:
    accepted: list[Sample] = field(default_factory=list)
    metas: list[SampleMeta] = field(default_factory=list)
    rejected: list[Rejected] = field(default_factory=list)
    task_pairs: list[tuple[Sample, str]] = field(default_factory=list)


def _gen_json(backend, prompt: str, retries: int) -> dict:
    last: Exception | None = None
    for _ in range(retries + 1):
        try:
            return extract_json(backend.complete(prompt))
        except GenerationError as e:
            last = e
    raise last  # type: ignore[misc]


def run(
    *,
    config,
    dg,
    retriever,
    q_backend,
    answer_backend,
    judge_backend,
    out_dir: Path,
    rng: random.Random | None = None,
) -> RunResult:
    rng = rng or random.Random(dg.raw.get("seed", 0))
    index_path = config.index_path()
    chunks = load_corpus(index_path)
    chunk_by_id: dict[str, Chunk] = {c.chunk_id: c for c in chunks}
    by_model = usable_chunks_by_model(
        chunks, min_chars=dg.chunks["min_chars"], blacklist=dg.chunks["section_blacklist"]
    )
    # 过采样：候选量 = ceil(target * oversample_factor)，以 accepted 达标为目标
    plan_target = math.ceil(dg.target_size * dg.raw["oversample_factor"])
    plan = build_plan(
        by_model,
        target=plan_target,
        per_vehicle_min=dg.raw["per_vehicle_min"],
        per_vehicle_max=dg.raw["per_vehicle_max"],
        vehicle_subset=dg.raw["vehicle_subset"],
        rng=rng,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = out_dir / "checkpoint.jsonl"
    done = load_done_qids(checkpoint)
    result = RunResult()
    partial_count = 0
    attempts = 0
    seen_norm: set[str] = set()
    retries = dg.raw["generation_retries"]
    max_attempts = dg.raw["max_attempts"]
    a_cfg = dg.answerability

    for item in plan:
        # 目标：accepted 达到 target_size 即停；attempts 达上限兜底
        if len(result.accepted) >= dg.target_size or attempts >= max_attempts:
            break
        qid = make_qid(item.model_id, item.chunk_id, item.task_type)
        if qid in done:
            continue
        attempts += 1
        seed = chunk_by_id[item.chunk_id]
        model_display = config.model_display(item.model_id)
        base = Rejected(
            qid=qid, model_id=item.model_id, section_path=item.section_path,
            task_type=item.task_type, seed_chunk_id=seed.chunk_id,
            seed_preview=preview(seed.text), reject_stage="", reject_reason="", reject_detail="",
        )

        # 1) 问题生成
        try:
            qj = _gen_json(
                q_backend,
                question_gen_prompt(
                    model_display=model_display, section_path=item.section_path,
                    chunk_text=seed.text, task_type=item.task_type,
                ),
                retries,
            )
            question = (qj.get("question") or "").strip()
        except GenerationError as e:
            base.reject_stage, base.reject_reason, base.reject_detail = "generation", "json_parse_failed", str(e)
            result.rejected.append(base)
            append_checkpoint(checkpoint, qid, "rejected:json_parse_failed")
            continue
        if not question:
            base.reject_stage, base.reject_reason = "generation", "missing_required_field"
            result.rejected.append(base)
            append_checkpoint(checkpoint, qid, "rejected:missing_required_field")
            continue
        base.question = question

        # 1b) normalized exact 去重（在检索前拦截，省 LLM/检索开销）
        nq = normalize_question(question)
        if nq in seen_norm:
            base.reject_stage, base.reject_reason = "generation", "duplicate"
            result.rejected.append(base)
            append_checkpoint(checkpoint, qid, "rejected:duplicate")
            continue
        seen_norm.add(nq)

        # 2) RAG 回检（单车型）
        rr = retriever.retrieve_stateless(question, trace=True, force_models=[item.model_id])
        retrieved = [
            RetrievedChunk(
                chunk_id=d.metadata["chunk_id"],
                score=rr.scores.get(d.metadata["chunk_id"], 0.0),
                section_path=d.metadata.get("section_path", ""),
            )
            for d in rr.docs
        ]
        evidence_docs = rr.docs
        evidence_text = "\n".join(d.page_content.strip() for d in evidence_docs)

        # 3) 证据充分性 judge
        try:
            jj = _gen_json(judge_backend, judge_prompt(question=question, evidence_text=evidence_text), retries)
            judge_label = jj.get("label", "no")
            judge_conflict = bool(jj.get("conflict", False))
        except GenerationError:
            judge_label, judge_conflict = "no", False

        verdict: Verdict = evaluate(
            seed_chunk_id=seed.chunk_id, seed_section=seed.section_path, retrieved=retrieved,
            judge_label=judge_label, judge_conflict=judge_conflict, cfg=a_cfg,
            partial_count=partial_count, target=dg.target_size,
        )
        if not verdict.accept:
            base.reject_stage, base.reject_reason = "answerability", verdict.reason
            base.reject_detail = json.dumps(verdict.signals, ensure_ascii=False)
            result.rejected.append(base)
            append_checkpoint(checkpoint, qid, f"rejected:{verdict.reason}")
            continue

        # 4) 答案生成（以回检 evidence 为准）
        instruction = pick_instruction(item.task_type, rng)
        try:
            aj = _gen_json(
                answer_backend,
                answer_gen_prompt(
                    model_display=model_display, instruction=instruction, question=question,
                    evidence_text=evidence_text, task_type=item.task_type,
                ),
                retries,
            )
            output = (aj.get("output") or "").strip()
        except GenerationError as e:
            base.reject_stage, base.reject_reason, base.reject_detail = "generation", "json_parse_failed", str(e)
            result.rejected.append(base)
            append_checkpoint(checkpoint, qid, "rejected:json_parse_failed")
            continue

        sample = Sample(instruction=instruction, input=question, output=output)

        # 5) 质检
        max_chars = dg.quality["max_output_chars"].get(item.task_type, 800)
        qv = run_quality(sample, evidence_text=evidence_text, model_id=item.model_id, config=config, max_chars=max_chars)
        if not qv.ok:
            base.reject_stage, base.reject_reason, base.reject_detail = "quality", qv.reason, qv.detail
            result.rejected.append(base)
            append_checkpoint(checkpoint, qid, f"rejected:{qv.reason}")
            continue

        sample.output = qv.cleaned_output
        if verdict.tier == "partial_ok":
            partial_count += 1

        meta = SampleMeta(
            qid=qid, model_id=item.model_id, model_display=model_display, doc_type=seed.doc_type,
            section_path=item.section_path, task_type=item.task_type, seed_chunk_id=seed.chunk_id,
            seed_preview=preview(seed.text), seed_score=verdict.signals["seed_score"],
            evidence_chunk_ids=[r.chunk_id for r in retrieved],
            evidence_previews=[preview(d.page_content) for d in evidence_docs],
            retrieval_scores=[r.score for r in retrieved], max_score=verdict.signals["max_score"],
            seed_rank=verdict.signals["seed_rank"], same_section_count=verdict.signals["same_section_count"],
            evidence_sufficiency=judge_label, accept_tier=verdict.tier, backend=dg.backend,
            gen_question_raw=json.dumps(qj, ensure_ascii=False),
        )
        result.accepted.append(sample)
        result.metas.append(meta)
        result.task_pairs.append((sample, item.task_type))
        append_checkpoint(checkpoint, qid, "accepted")
        log.info("[accept:%s] %s | %s", verdict.tier, item.model_id, question[:30])

    return result
```

- [ ] **Step 4: 运行确认通过**

Run: `../.venv/Scripts/python.exe -m pytest tests/lora_gen/test_pipeline_checkpoint.py -v`
Expected: PASS（3 passed）

- [ ] **Step 5: 全量单测回归**

Run: `../.venv/Scripts/python.exe -m pytest tests/lora_gen -v`
Expected: PASS（全部）

- [ ] **Step 6: Commit**

```bash
git add lora_gen/pipeline.py tests/lora_gen/test_pipeline_checkpoint.py
git commit -m "feat(lora_gen): pipeline 编排与断点续传"
```

---

### Task 10: config 接口探测 + 接口自检脚本

**Files:**
- Create: `lora_gen/compat.py`, `scripts/check_interfaces.py`
- Test: `tests/lora_gen/test_compat.py`

- [ ] **Step 1: 写失败测试**

`tests/lora_gen/test_compat.py`：
```python
from pathlib import Path
from config_loader import load_config
from lora_gen.compat import probe_config, ConfigAdapter

class _Bare:
    """缺 index_path / model_display 的最小 config。"""
    def __init__(self):
        self.models = [{"id": "M1", "name": "显示名"}]
        self.doc_types = ["owner_manual"]
        self.raw = {"chunking": {"strategy": "hierarchy"}}
        self.index_dir = Path("indexes")
    def model_aliases(self):
        return [("M1", "M1")]

def test_probe_passthrough_when_complete():
    cfg = load_config()
    assert probe_config(cfg) is cfg  # 真实 config 已完整 → 原样返回

def test_probe_wraps_when_missing():
    wrapped = probe_config(_Bare())
    assert isinstance(wrapped, ConfigAdapter)
    assert wrapped.model_display("M1") == "显示名"
    assert wrapped.index_path().as_posix().endswith("hierarchy/corpus")

def test_probe_hard_missing_raises():
    import pytest
    class _NoModels:
        pass
    with pytest.raises(AttributeError):
        probe_config(_NoModels())
```

- [ ] **Step 2: 运行确认失败**

Run: `../.venv/Scripts/python.exe -m pytest tests/lora_gen/test_compat.py -v`
Expected: FAIL（ModuleNotFoundError）

- [ ] **Step 3: 实现 compat.py**

`lora_gen/compat.py`：
```python
"""config 接口探测：确认所需接口存在，缺失则用 fallback adapter 补齐。"""
from __future__ import annotations

from pathlib import Path

REQUIRED = ["index_path", "model_display", "model_aliases", "models", "doc_types"]
# 可被 adapter 补齐的软缺失；其余缺失视为硬错误。
SOFT = {"index_path", "model_display"}


class ConfigAdapter:
    """在原 config 上补齐缺失接口的轻量包装；其余属性透传。"""

    def __init__(self, config):
        self._c = config

    def __getattr__(self, name):
        return getattr(self._c, name)

    def model_display(self, model_id: str) -> str:
        for m in self._c.models:
            if m.get("id") == model_id:
                return m.get("name", model_id)
        return model_id

    def index_path(self, strategy: str | None = None) -> Path:
        s = strategy or self._c.raw["chunking"]["strategy"]
        return Path(self._c.index_dir) / s / "corpus"


def probe_config(config):
    missing = [name for name in REQUIRED if not hasattr(config, name)]
    if not missing:
        return config
    hard = [m for m in missing if m not in SOFT]
    if hard:
        raise AttributeError(f"config 缺少必需接口且无 fallback: {hard}")
    return ConfigAdapter(config)
```

- [ ] **Step 4: 运行确认通过**

Run: `../.venv/Scripts/python.exe -m pytest tests/lora_gen/test_compat.py -v`
Expected: PASS（3 passed）

- [ ] **Step 5: 实现接口自检脚本**

`scripts/check_interfaces.py`：
```python
"""smoke 前接口自检：corpus path / detect_models / Retriever / QwenClient / Ollama。"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config_loader import load_config
from lora_gen.compat import probe_config
from lora_gen.dgconfig import load_dg_config


def check() -> int:
    ok = True
    config = probe_config(load_config())
    dg = load_dg_config()

    idx = config.index_path()
    corpus = idx / "bm25_corpus.json"
    print(f"[corpus] {corpus} exists={corpus.exists()}")
    ok &= corpus.exists()

    try:
        from retrieve.model_router import detect_models

        d = detect_models("问界M9 2026增程版空调", "", config)
        print(f"[detect_models] -> {d}")
        ok &= isinstance(d, list)
    except Exception as e:  # noqa: BLE001
        print(f"[detect_models] FAIL {e}")
        ok = False

    try:
        from retrieve.pipeline import Retriever

        r = Retriever(config, idx)
        r.load()
        print("[Retriever] load ok")
    except Exception as e:  # noqa: BLE001
        print(f"[Retriever] FAIL {e}")
        ok = False

    if "cloud" in (dg.backend, dg.judge_backend):
        try:
            from generate.qwen_client import QwenClient

            QwenClient(config)
            print("[QwenClient] init ok")
        except Exception as e:  # noqa: BLE001
            print(f"[QwenClient] FAIL {e}")
            ok = False
    if "local" in (dg.backend, dg.judge_backend):
        try:
            import ollama

            ollama.list()
            print("[ollama] reachable")
        except Exception as e:  # noqa: BLE001
            print(f"[ollama] FAIL {e}")
            ok = False

    print("RESULT:", "OK" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(check())
```

- [ ] **Step 6: 运行自检**

Run: `../.venv/Scripts/python.exe scripts/check_interfaces.py`
Expected: 各 `[...] ok/exists=True`，末行 `RESULT: OK`，退出码 0。若任一 FAIL，先修环境再继续 Task 11。

- [ ] **Step 7: Commit**

```bash
git add lora_gen/compat.py scripts/check_interfaces.py tests/lora_gen/test_compat.py
git commit -m "feat(lora_gen): config 接口探测与 smoke 前自检脚本"
```

---

### Task 11: CLI 与 provenance

**Files:**
- Create: `scripts/build_lora_dataset.py`
- Modify: `docs/dataset_provenance.md`

- [ ] **Step 1: 实现 CLI**

`scripts/build_lora_dataset.py`：
```python
"""LoRA 语料生成 CLI。

用法：
  ../.venv/Scripts/python.exe scripts/build_lora_dataset.py --target 100
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config_loader import load_config
from retrieve.pipeline import Retriever
from lora_gen.compat import probe_config
from lora_gen.dgconfig import load_dg_config
from lora_gen.backends import make_backend
from lora_gen.export import build_report, stratified_split, write_dataset, write_meta, write_rejected
from lora_gen.pipeline import run


def corpus_fingerprint(index_path: Path) -> str:
    import hashlib

    p = index_path / "meta.json"
    data = p.read_bytes() if p.exists() else b""
    return hashlib.sha256(data).hexdigest()[:12]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=None)
    ap.add_argument("--out", default="data/lora_out")
    ap.add_argument("--manual-check-ratio", type=float, default=0.1)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    config = probe_config(load_config())  # 接口探测 + 缺失则 fallback adapter
    dg = load_dg_config(target_override=args.target)

    index_path = config.index_path()
    retriever = Retriever(config, index_path)
    retriever.load()

    q_backend = make_backend(dg.backend, config, dg)
    answer_backend = make_backend(dg.backend, config, dg)
    judge_backend = make_backend(dg.judge_backend, config, dg)

    out_dir = Path(args.out)
    rng = random.Random(dg.raw.get("seed", 0))
    res = run(
        config=config, dg=dg, retriever=retriever,
        q_backend=q_backend, answer_backend=answer_backend, judge_backend=judge_backend,
        out_dir=out_dir, rng=rng,
    )

    # 所有产物统一写入 out_dir，不散落项目根目录
    write_dataset(res.accepted, out_dir / "ito_lora_dataset.json")
    write_meta(res.metas, out_dir / "ito_lora_dataset.meta.jsonl")
    write_rejected(res.rejected, out_dir / "ito_lora_dataset_rejected.json")

    train, dev = stratified_split(res.task_pairs, train_ratio=dg.raw["export"]["train_dev_split"], rng=rng)
    write_dataset(train, out_dir / "ito_lora_dataset.train.json")
    write_dataset(dev, out_dir / "ito_lora_dataset.dev.json")

    report = build_report(
        accepted_tiers=[m.accept_tier for m in res.metas],
        task_types=[m.task_type for m in res.metas],
        vehicles=[m.model_id for m in res.metas],
        reject_reasons=[r.reject_reason for r in res.rejected],
        corpus_fingerprint=corpus_fingerprint(index_path),
        backend=dg.backend, config_hash=dg.config_hash, manual_check_ratio=args.manual_check_ratio,
    )
    (out_dir / "generation_report.md").write_text(report, encoding="utf-8")
    print(f"accepted={len(res.accepted)} rejected={len(res.rejected)} → {out_dir}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 接口自检（smoke 前置门）**

Run: `../.venv/Scripts/python.exe scripts/check_interfaces.py`
Expected: 末行 `RESULT: OK`，退出码 0。FAIL 则先修环境，不进行 smoke。

- [ ] **Step 3: 冒烟运行（小目标，真实 LLM + 检索）**

Run: `../.venv/Scripts/python.exe scripts/build_lora_dataset.py --target 5 --out data/lora_smoke`
Expected: 进程结束打印 `accepted=N rejected=M`；全部产物在 `data/lora_smoke/` 下：`ito_lora_dataset.json`（≥1 条且字段严格）、`ito_lora_dataset.meta.jsonl`、`ito_lora_dataset_rejected.json`、`ito_lora_dataset.train.json`/`.dev.json`、`generation_report.md`、`checkpoint.jsonl`。项目根目录不应新增数据文件。若 N=0，看 `data/lora_smoke/ito_lora_dataset_rejected.json` 的 reason 分布定位瓶颈（chunk/问题/检索）。

- [ ] **Step 4: 断点续传验证**

再次运行同一命令，确认日志显示已完成 qid 被跳过、不重复消耗 LLM。
Run: `../.venv/Scripts/python.exe scripts/build_lora_dataset.py --target 5 --out data/lora_smoke`
Expected: 快速结束，accepted/rejected 数与上次一致（无重复生成）。

- [ ] **Step 5: 更新 provenance**

将 `docs/dataset_provenance.md` 改写为新流水线版本，至少包含：生成方式（chunk 反推 + 单车型 RAG 回检门 + 质检 + 人工抽修）、复现命令（上面的 CLI）、字段说明（含 meta sidecar）、记录 corpus 指纹 / backend / `dataset_gen.yaml` hash / accepted-rejected 统计 / 人工抽检比例（引用 `generation_report.md`）。

- [ ] **Step 6: Commit**

```bash
git add scripts/build_lora_dataset.py docs/dataset_provenance.md
git commit -m "feat(lora_gen): CLI 串联 + 更新数据来源说明"
```

---

### Task 11: 试点验收（人工抽查）

**Files:** 无新增代码；产出数据与报告。

- [ ] **Step 1: 跑 100 条试点**

Run: `../.venv/Scripts/python.exe scripts/build_lora_dataset.py --target 100 --out data/lora_pilot`
Expected: 生成 `ito_lora_dataset.json` 等；`data/lora_pilot/generation_report.md` 给出通过率与拒绝原因分布。

- [ ] **Step 2: 校验最终格式严格性**

Run: `../.venv/Scripts/python.exe -c "import json;d=json.load(open('data/lora_pilot/ito_lora_dataset.json',encoding='utf-8'));assert all(set(x)=={'instruction','input','output'} for x in d);print('ok',len(d))"`
Expected: `ok <N>`（N 接近 100；task 分布≥4 类，可另行统计）。

- [ ] **Step 3: 人工抽查 meta**

抽查 `data/lora_pilot/ito_lora_dataset.meta.jsonl` 中 `accept_tier=partial_ok` 与 `seed_score` 偏低样本，确认 output 与 evidence 一致、无车型串味。记录抽检比例到报告。

- [ ] **Step 4: 决定是否扩量**

依据报告通过率与抽查结果，决定扩到 200–400（调整 `config/dataset_gen.yaml: target_size` 或 `--target`），或先修阈值/规则再扩。

---

## Self-Review

**Spec coverage：**
- §1 交付物（dataset/train-dev/meta/rejected/report/provenance）→ Task 8、11、12 ✓
- §2 模块结构 → Task 0–11 一一对应 ✓
- §3 数据流 + §3.1 round-robin + §3.2 generation reject → Task 3、9 ✓
- §4 answerability 分档/single_chunk_full/partial 配额/evidence_conflict → Task 6 ✓
- §5 quality 全部规则 + input&output 车型检查 → Task 7 ✓
- §6 task 映射 + §6.1 模板池 + §6.2 黑名单 → Task 2、4 ✓
- §7 schema → Task 1 ✓
- §8 配置键 → Task 0 ✓
- §9 测试策略 → 各 Task TDD ✓

**用户追加 7 项 coverage：**
1. CLI config 接口探测 + fallback adapter → Task 10（`compat.probe_config`/`ConfigAdapter`）、Task 11 CLI 调用 ✓
2. vehicle_conflict 归一化（detect_models 返回 id，别名解析回同一 id）→ Task 7（`foreign_vehicles` 注释 + `test_foreign_vehicles_bound_alias_not_flagged`）✓
3. 全部产物写入 --out 目录 → Task 11 CLI（dataset/meta/rejected/train/dev/report/checkpoint 均 out_dir）✓
4. oversample_factor / max_attempts，以 accepted 达 target 为目标 → Task 0 配置 + Task 9（`plan_target`、循环早停）✓
5. extract_json fenced 优先 + brace-balanced fallback → Task 5（`_FENCE_RE`/`_balanced_object` + 2 新测试）✓
6. normalized exact question 去重 → Task 7（`normalize_question`）+ Task 9（检索前 `seen_norm` 拦截 `duplicate`）✓
7. smoke 前接口检查脚本 → Task 10（`scripts/check_interfaces.py`）+ Task 11 Step 2 前置门 ✓

**Placeholder scan：** 无 TBD/TODO；所有代码步骤含完整实现与可运行命令。

**Type consistency：** `evaluate(...)` 返回 `Verdict(accept,tier,reason,signals)`，pipeline 按此读取 ✓；`run_quality(...)` 返回 `QualityVerdict(ok,reason,detail,cleaned_output)`，pipeline 按此读取 ✓；`make_backend(name, config, dg)` 三参一致（定义/CLI）✓；`build_plan(...)` 关键字参数与 registry/pipeline 一致 ✓；`normalize_question` 定义于 quality、被 pipeline import ✓；`probe_config` 定义于 compat、被 CLI/check_interfaces import ✓；`extract_json` 签名不变（fenced+balanced 内部实现）✓；`Sample/SampleMeta/Rejected.to_record` 与 export 一致 ✓。
