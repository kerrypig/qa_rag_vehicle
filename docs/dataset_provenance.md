# ito_lora_dataset 数据来源说明

## 生成方式
本数据集由 `lora_gen` 流水线从问界（AITO）官方车主手册语料自动生成，不依赖外部种子问题，
而是「以手册 chunk 反推问题」，并经单车型 RAG 回检与质检后落盘。整条流水线分五步：

1. **Plan（按车型抽 chunk）**：从合并 `corpus` 索引加载 chunk，按 `min_chars` 与
   `section_blacklist` 过滤掉过短/无关章节，再按车型分组，依 `per_vehicle_min/max`、
   `vehicle_subset` 配置抽样出候选 (model_id, chunk_id, task_type) 计划（过采样系数
   `oversample_factor`，以 accepted 达到 `target_size` 为目标）。
2. **问题生成（chunk 反推问题）**：以手册 chunk 为依据，按 5 类任务
   （直接问答/步骤指导/故障分析/术语解释/安全提醒）让 LLM 生成一条用户问句；
   生成前对 normalized 问句做精确去重，拦截重复以省检索/LLM 开销。
3. **单车型 RAG 回检门**：用生成的问题在该车型范围内（`force_models`）做 hybrid 检索，
   结合 LLM judge（证据是否充分 / 是否冲突）与回检信号（seed 命中、最高分、同章节数等）
   做 answerability 裁决；不通过则丢弃，避免无依据编造。
4. **证据驱动答案**：仅以回检到的 evidence 为准让 LLM 生成 output（而非凭模型记忆）。
5. **质检 + 人工抽修**：自动质检（字段完整、长度上限、穿帮语/引用标记清洗等），通过后写入；
   交付前按 `--manual-check-ratio` 比例人工抽检并修订。

## 字段
最终数据集 `ito_lora_dataset.json` 为严格三字段（不含其它键）：
- `instruction`：给汽车助手的任务说明（按 task_type 选取）
- `input`：规范化的用户问句
- `output`：基于该车型手册回检证据生成的准确回答

逐条溯源信息写在旁车 `ito_lora_dataset.meta.jsonl`（每行一条 SampleMeta），含：
`qid`、`model_id`/`model_display`、`doc_type`、`section_path`、`task_type`、
`seed_chunk_id`/`seed_preview`/`seed_score`、回检的 `evidence_chunk_ids`/`evidence_previews`/
`retrieval_scores`/`max_score`、`seed_rank`、`same_section_count`、`evidence_sufficiency`
（judge 标签）、`accept_tier`、`backend`、`gen_question_raw` 等，可逐条抽查 output 与来源章节是否一致。

被拒样本记录在 `ito_lora_dataset_rejected.json`（含 reject_stage / reject_reason / reject_detail，
便于诊断流失环节）。同时按 task_type 分层切分为
`ito_lora_dataset.train.json` / `ito_lora_dataset.dev.json`（比例见 `dataset_gen.yaml` 的
`export.train_dev_split`）。

## 复现命令
```bash
../.venv/Scripts/python.exe scripts/build_lora_dataset.py --target 100 --out data/lora_pilot
```
- `--target`：accepted 目标条数（覆盖 `dataset_gen.yaml` 的 `target_size`）。
- `--out`：所有产物统一写入该目录，不散落项目根目录。
- `--manual-check-ratio`：人工抽检比例（默认 0.1），记入生成报告。

## 断点续传
运行时每处理一个候选 qid 即追加写入 `<out>/checkpoint.jsonl`（记录 qid + 状态）。
中断后用相同命令重跑会自动跳过 checkpoint 中已完成的 qid，仅补做剩余候选；
因此续跑时打印的 `accepted` 只统计本次新增工作量，可能为 0（属正常）。

## 可复现性记录
每次运行在 `<out>/generation_report.md` 记录：
- **语料指纹**（corpus fingerprint，索引 `meta.json` 的 sha256 前缀），
- **backend**（生成/质检所用模型后端），
- **dataset_gen.yaml 配置哈希**（config_hash），
- **accepted / rejected 统计**（按 accept_tier、task_type、车型、reject_reason 分布），
- **人工抽检比例**（manual_check_ratio）。
凭以上信息可复核数据规模与配置，确保生成过程可追溯、可复现。
