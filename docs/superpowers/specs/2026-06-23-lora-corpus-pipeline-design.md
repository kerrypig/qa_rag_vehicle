# LoRA 语料生成流水线设计（RAG 证据驱动 / 单车型绑定 / chunk 反推）

日期：2026-06-23
状态：已批准设计，待生成实现计划

## 1. 目标与背景

为「汽车技术文档助手」生成 LoRA 微调语料。每条样本必须**先绑定单一车型与 chunk 证据，再生成 instruction/input/output**，以避免车型混淆、事实幻觉、重复问答和检索失败。

交付物：
- `ito_lora_dataset.json`：严格 JSON 数组，每元素 `{instruction, input, output}`。
- `ito_lora_dataset.train.json` / `ito_lora_dataset.dev.json`：train/dev split（默认 9:1，按 task_type 分层），便于后续 LoRA 训练与评估。
- `ito_lora_dataset.meta.jsonl`：每条样本完整中间字段，供质检与溯源。
- `*_rejected.json`：被拒样本 + 拒绝阶段/原因，用于诊断（chunk 差 / 问题差 / 检索差）。
- `generation_report.md`：按拒绝原因、task_type、车型统计的生成报告。
- `docs/dataset_provenance.md`：数据来源说明（更新版）；记录 corpus 索引指纹、backend、`dataset_gen.yaml` 配置 hash、accepted/rejected 统计、人工抽检比例。

### 1.1 已确认决策
- **代码基线**：全新重建（旧 `aito_inputs/`、`dataset_gen/` 仅剩 `.pyc`，不复用）。
- **场景范围**：仅限手册可溯源内容。语料库 `doc_type` 当前只有 `owner_manual`（车主手册）；保养/故障/术语为手册内章节（`section_path`）。**不做保险条款**（不在 RAG 内）。
- **规模**：先跑 **100 条试点**，人工抽查无误后再扩到 200–400。
- **生成后端**：cloud DashScope `qwen3.6-plus`（问题生成 + 答案生成）。证据充分性 judge 与 `vehicle_conflict` 检测可用本地 `qwen2.5:7b`。

### 1.2 现有基础（复用，不重写）
- **车型注册表**：`config.yaml → vehicle.models`，18 个车型（`id / name / file / aliases`）。
- **合并索引 `corpus`**：`indexes/hierarchy/corpus/`，5201 chunk，覆盖 18 车型。chunk 元数据：`chunk_id, vehicle_model, doc_type, source_file, page, section_path, image_refs`。
- **单车型检索**：`retrieve.pipeline.Retriever.retrieve_stateless(question, trace=True, force_models=[model_id])`，向量 + BM25 均按 `vehicle_models` / `doc_types` 过滤；返回 `RetrievalResult(docs, scores{chunk_id:cosine}, trace)`。
- **chunk 直查**：`retrieve.chunk_lookup.find_chunk_ids_by_text`；`bm25_corpus.json` 含 `chunk_ids/texts/metadatas`。
- **车型识别**：`retrieve.model_router.detect_models(question, rewritten, config)`（别名规则匹配，用于 `vehicle_conflict`）。

## 2. 架构（新包 `lora_gen/`）

旧 `aito_inputs/`、`dataset_gen/` 仅含过期 `.pyc`，保留不动，全部新建于 `lora_gen/`。

| 模块 | 单一职责 |
|---|---|
| `lora_gen/registry.py` | 从 `config.yaml` 读 18 车型；构建采样计划 `model_id → section → chunk_ids → task_type`；**round-robin 采样 + 每车型 min/max 配额**（§3.1） |
| `lora_gen/chunks.py` | 加载 corpus；剔除不可用 chunk（过短 / 目录页 / 仅图片）；**`section_blacklist` 过滤低价值章节**；`section_path → task_type` 映射 |
| `lora_gen/prompts.py` | 集中管理全部 prompt：问题生成、答案生成、证据充分性 judge、validator；**各 task_type 的 instruction 模板池**（§6.1） |
| `lora_gen/backends.py` | 统一接口，两实现：cloud `QwenClient` / 本地 `ollama`；用于问题生成与答案生成 |
| `lora_gen/answerability.py` | RAG 回检门（§4，多信号） |
| `lora_gen/quality.py` | 质检拒绝规则（§5） |
| `lora_gen/schema.py` | `Sample` / `SampleMeta` / `Rejected` dataclass + 序列化 |
| `lora_gen/pipeline.py` | 编排 + 按 qid 断点续传；记录拒绝原因 |
| `lora_gen/export.py` | 导出最终 JSON、train/dev split、`*_rejected.json`、`generation_report.md` |
| `scripts/build_lora_dataset.py` | CLI 入口 |
| `config/dataset_gen.yaml` | 阈值、task 分布、目标规模、车型子集 |

## 3. 数据流（单条样本）

```
registry 采样计划
  → 选 chunk（chunks.py 过滤 + section→task_type）
  → [prompts] 由 chunk 反推「带车型名」的具体问题（仅一个车型）
  → RAG 回检 retrieve_stateless(force_models=[单车型], trace=True)
  → answerability 门（§4）
      ├─ 通过 → [prompts] 用「回检得到的 evidence」（非原始 seed chunk）生成 output
      │         → quality 拒绝规则（§5）
      │             ├─ 通过 → 写入 accepted + meta
      │             └─ 拒绝 → 写入 rejected(stage=quality, reason)
      └─ 不通过 → 写入 rejected(stage=answerability, reason)
```

关键原则：
- **一个样本只绑定一个车型**（`force_models=[model_id]`，单元素）。不跨车型混合。
- **答案以回检 evidence 为准**，而非原始 seed chunk，贴近真实 RAG 工作流（用户要求 #5）。
- **断点续传**：已完成 qid 跳过；rejected 也持久化原因，便于诊断瓶颈在 chunk / 问题 / 检索哪一环。

### 3.1 车型采样策略（round-robin + 配额）

`registry.py` 按 round-robin 轮转车型分配样本，避免集中在少数车型：
- `per_vehicle_min` / `per_vehicle_max`（默认 min=3 / max=10，按 `target_size` 与车型数自适应）。
- 轮转顺序遍历 `vehicle_subset`（空=全部 18）；每轮每车型补 1 条候选，达到 `per_vehicle_max` 后跳过。
- 收尾时检查未达 `per_vehicle_min` 的车型并优先补齐；若某车型可用 chunk 不足，记入 report 而非静默忽略。

### 3.2 generation 阶段拒绝

问题生成 / 答案生成的 LLM 输出若解析失败，写入 rejected(stage=generation)：
- `json_parse_failed`：模型输出非合法 JSON / 无法抽取目标字段。
- `missing_required_field`：缺 instruction/input/output 任一字段。
- `empty_generation`：字段为空白。
generation 阶段对单条样本最多重试 N 次（默认 2），仍失败则落 rejected。

## 4. Answerability 门（`answerability.py`）

对每个候选问题运行 `retrieve_stateless(question, trace=True, force_models=[model_id])`，用 `RetrievalResult.docs` 与 `scores`（余弦，bge-large-zh，已归一化）计算下列信号：

| 信号 | 默认阈值 | 含义 |
|---|---|---|
| `min_retrieved` | ≥ 2（**有例外**，见下） | 召回 chunk 数 |
| `seed_in_topk` | 必需，k=5 | seed chunk 是否回检命中 |
| `seed_rank` | strong 档 ≤ 3 | seed 在召回中的名次 |
| `seed_score` | ≥ 0.30（单 chunk 例外档 ≥ 0.35） | seed 自身分数 |
| `max_score` | ≥ 0.35（strong 档 ≥ 0.45） | 最佳 chunk 相关度 |
| `same_section_count` | ≥ 1（strong 档 ≥ 2） | 围绕 seed `section_path` 的主题聚焦度 |
| `evidence_sufficiency` | LLM judge：`full` / `partial` / `no` + `conflict(bool)` | 给定问题+evidence 能否完整回答；`conflict=true` 表示 evidence 内部相互矛盾 |

### 4.1 接受分档（accept_tier）

| 档位 | 条件 | 入库 |
|---|---|---|
| `strong` | seed 在 top-3 ∧ max_score≥0.45 ∧ judge=full | 是 |
| `ok` | seed 在 top-5 ∧ seed_score≥0.30 ∧ retrieved≥2 ∧ judge=full | 是 |
| `single_chunk_full` | seed_in_topk ∧ judge=full ∧ seed_score≥0.35（仅 1 chunk 也接受，min_retrieved 例外） | 是 |
| `partial_ok` | seed 在 top-5 ∧ seed_score≥0.30 ∧ judge=partial | **仅在配额内**（默认 ≤ 8% 目标量），超配额拒绝 |
| reject | 其它 | 否 |

拒绝原因枚举：`seed_not_returned | low_score | too_few_chunks | insufficient_evidence | partial_quota_full | evidence_conflict`（judge `conflict=true` 直接拒绝）。

`partial_ok` 配额限制理由：避免 LoRA 混入过多「根据当前资料只能确认……」的保守回答，使模型过度拒答。所有阈值与配额置于 `dataset_gen.yaml`。

## 5. Quality 拒绝规则（`quality.py`）

**先 strip 后保留**：穿帮短语（`根据手册 / 根据上述 / 作为AI / 参考第X页 / [1] 类引用标记`）。

**硬拒绝（带 reason 枚举）：**

| 规则 | 触发条件 |
|---|---|
| `vehicle_conflict` | `detect_models(input)` **或** `detect_models(output)` 命中 ≠ 绑定 `model_id` 的车型（跨车型泄漏，input 与 output 同时检查） |
| `ungrounded_number` | output 中规格数字（容量/压力/电压/扭矩/km/时间）未出现在 evidence 文本 |
| `insurance_warranty_mix` | output 断言 evidence 中不存在的 保险/质保/三包 条款 |
| `unsafe_danger_advice` | 高压/电池/救援/拖车/起火 主题 + DIY 操作建议，且**缺少**「联系授权服务中心/专业人员」护栏 |
| `over_promise` | output 含 `免费/一定/永久/保证/保险全赔/绝对/100%` 且 evidence 无支撑 |
| `field_incomplete` / `length_out_of_bounds` / `duplicate` | 字段缺失 / 长度越界（按 task_type 配置，见 §8）/ 问题近重复 |

`vehicle_conflict` 与 `ungrounded_number` 为最高价值门，均为 `(output, evidence, model_id)` 的纯函数，便于 TDD。

## 6. Task-type ↔ section 映射（≥4 类指令）

`section_path` 关键词 → 偏置 `task_type`：
- 故障/警示/报警/异常 → 故障分析
- 名词/定义/简介/说明（术语性）→ 术语解释
- 检查/保养/操作/更换/安装 → 步骤指导
- 安全/警告/儿童/安全带/气囊 → 安全提醒
- 其它 → 直接问答

目标分布（`dataset_gen.yaml` 可调）：直接问答 30% / 步骤指导 25% / 故障分析 20% / 术语解释 15% / 安全提醒 10%。至少覆盖 4 类。

### 6.1 instruction 模板池

`prompts.py` 为每个 task_type 维护一个 instruction 模板池（每类 ≥4 条变体），生成样本时轮转/随机抽取，避免 instruction 过于单一。示例（步骤指导）：「请给出……的检查步骤」「作为汽车助手，分步说明如何……」「列出……的操作流程」「指导车主完成……」。模板池为静态常量，确定性可测。

### 6.2 section_blacklist

pilot 阶段 `chunks.py` 过滤低价值章节（`section_path` 关键词命中即剔除）：前言、目录、免责声明、版权、隐私保护、单位/计量、术语与缩略语对照、修订记录、联系方式。黑名单置于 `dataset_gen.yaml`，命中数计入 report。

## 7. Schema

**最终**（`ito_lora_dataset.json`）：严格 `[{instruction, input, output}]`。

**meta sidecar**（`ito_lora_dataset.meta.jsonl`，每条 accepted 一行）：
```
qid, model_id, model_display, doc_type, section_path, task_type,
seed_chunk_id, seed_preview(前50字), seed_score,
evidence_chunk_ids[], evidence_previews[各前50字], retrieval_scores[],
max_score, seed_rank, same_section_count, evidence_sufficiency(full|partial),
accept_tier(strong|ok|single_chunk_full|partial_ok), backend, gen_question_raw
```

**rejected**（`*_rejected.json`）：同上下文 + `reject_stage(generation|answerability|quality)`、`reject_reason(枚举)`、`reject_detail`。

## 8. 配置 `config/dataset_gen.yaml`（草案键）
```yaml
target_size: 100
backend: cloud           # cloud=qwen3.6-plus | local=qwen2.5:7b
judge_backend: local
vehicle_subset: []       # 空=全部 18 车型；试点可填子集
per_vehicle_min: 3
per_vehicle_max: 10
generation_retries: 2
task_distribution: {直接问答: 0.30, 步骤指导: 0.25, 故障分析: 0.20, 术语解释: 0.15, 安全提醒: 0.10}
answerability:
  topk: 5
  min_retrieved: 2
  seed_score_min: 0.30
  seed_score_min_single: 0.35
  max_score_min: 0.35
  strong: {seed_rank_max: 3, max_score_min: 0.45, same_section_min: 2}
  partial_ok_quota: 0.08
chunks:
  min_chars: 200
  drop_toc: true
  drop_image_only: true
  section_blacklist: [前言, 目录, 免责声明, 版权, 隐私, 单位, 术语, 缩略语, 修订记录, 联系方式]
quality:
  max_output_chars:      # 按 task_type 配置长度上限
    直接问答: 400
    步骤指导: 800
    故障分析: 700
    术语解释: 400
    安全提醒: 500
  dup_threshold: 0.90
export:
  train_dev_split: 0.9   # train 占比，按 task_type 分层
```

## 9. 测试策略（TDD）

纯函数 / 确定性逻辑优先 TDD，LLM 调用打桩：
- `registry`：采样计划生成、单车型约束、round-robin 轮转与 per-vehicle min/max 配额。
- `chunks`：不可用 chunk 过滤、`section_blacklist` 命中、section→task_type 映射。
- `answerability`：给定 mock 的 `RetrievalResult`，验证分档逻辑、partial 配额、`conflict=true` 拒绝。
- `quality`：`vehicle_conflict`（input 与 output 同测）/ `ungrounded_number` / `over_promise` / 危险建议护栏，针对构造样本逐条断言。
- `export`：JSON / rejected / report 结构、train/dev 分层 split 比例。
- `schema`：序列化往返。

LLM 集成（问题生成 / 答案生成 / judge）用小样本冒烟测试，不进 CI 断言。

## 10. 范围外（YAGNI）
- 不做保险条款语料（不在 RAG）。
- 不做真实问题脱敏种子（方案 C 混合），试点阶段不引入。
- 不重写检索 / 索引 / config 车型注册表（已有且可用）。
