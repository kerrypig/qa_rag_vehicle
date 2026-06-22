# LoRA 微调数据集构建器 — 设计文档

日期：2026-06-22
分支：mvp1.1
状态：已确认，待实现

## 目标

将通用汽车问答数据集 `doc/AutoMaster_TrainSet.csv` 对齐为符合问界（AITO）车主手册规范的 LoRA 微调数据集，输出严格 JSON 数组 `ito_lora_dataset.json`，每条为 `{"instruction": "...", "input": "...", "output": "..."}`。

数据规模目标：200–400 条（默认 300），覆盖 5 种指令类型。质量底线：每条 output 必须有问界手册依据，无事实错误。

## 交付物

| 文件 | 说明 |
|---|---|
| `scripts/build_lora_dataset.py` | 主脚本（新增） |
| `dataset_gen.yaml` | 数据集生成专用配置（新增，独立于 `config.yaml`） |
| `ito_lora_dataset.json` | 输出训练集，合法 JSON 数组 |
| `ito_lora_dataset.meta.jsonl` | 溯源 sidecar：每行 `{QID, task_type, source_sections, scores, backend}` |
| `logs/dataset_gen_*.log` | 运行日志（跳过/失败记录） |

## 复用边界（不修改既有代码）

| 复用件 | 调用方式 |
|---|---|
| `config_loader.load_config` | 载入 `config.yaml`，再用 `dataset_gen.yaml` 覆盖检索开关 |
| `retrieve.pipeline.Retriever` | `retrieve_stateless(question, force_models=None)`，对合并 `corpus` 索引检索 |
| `generate.prompt_builder.format_context` | 把检索 chunk 格式化为 grounding 块 |
| `generate.qwen_client.QwenClient` | 云端生成后端（`backend: cloud` 时） |
| `ollama`（库） | 本地生成后端（`backend: local`，`format='json'`） |

新增代码仅限：CSV 清洗/采样、生成后端抽象（local/cloud）、数据集 Prompt、断点续传、CLI 主循环。**既有 pipeline 文件零改动。**

## 配置（`dataset_gen.yaml`）

```yaml
dataset_gen:
  input_csv: "./doc/AutoMaster_TrainSet.csv"
  csv_encoding: "gb18030"          # CSV 为 GBK 系编码，UTF-8 读取会乱码
  output_json: "./ito_lora_dataset.json"
  output_meta: "./ito_lora_dataset.meta.jsonl"

  sample_size: 300                 # 从过滤后池中随机抽取的候选行数
  target_size: 300                 # 目标成功条数；不足则继续从池中补抽，直至池尽
  random_seed: 42

  # 新能源通用场景关键词（命中任一即保留）
  keep_keywords: [胎压, 空调, 异响, 蓝牙, 车机, 雷达, 钥匙, 充电, 刹车,
                  座椅, 车门, 后视镜, 灯光, 仪表, 续航, 充电桩, 中控]

  # 弱检索判定（任一不满足即跳过该行）
  min_chunks: 2                    # 检索到的有效 chunk 下限
  min_score: 0.30                  # 最高向量相似度下限

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

  # 覆盖到 config.yaml，实现「轻量检索」（单条问题无需指代消解）
  # 注意：keyword_search 也走 Ollama（extract_keyword），必须一并关闭，
  # 否则 _prepare_queries 仍会调 Ollama。两者全关时检索零 Ollama 调用。
  overrides:
    query_rewrite.enabled: false
    retrieval.keyword_search.enabled: false
    retrieval.bookmark_match.enabled: false
    verification.enabled: false
```

覆盖机制：`load_config` 后，在 `config.raw` 上按点路径写入 `overrides`，再构造 `Retriever`。检索因此只跑 hybrid 向量+BM25（rewritten 路径，query=原问题），检索阶段零 Ollama 调用，生成另算。

## 数据流（每行）

```
读 CSV(gb18030) → keep_keywords 过滤 → 按 seed 随机打散
  → 遍历候选行（跳过 meta 中已完成 QID）：
      retrieve_stateless(Question)                  # 轻量 hybrid
      → 弱检索？(max vector_sim < min_score 或 有效 chunk < min_chunks) → 跳过 + 记日志
      → 随机指定 task_type（5 选 1）
      → 构造数据集 Prompt（context + Question + task_type）
      → 生成后端（local Ollama format=json | cloud QwenClient）
      → json.loads → 校验三字段非空、命中禁语则判失败
      → 追加结果；每 checkpoint_every 条 → 原子重写 JSON + 追加 meta
  → 达到 target_size 或池尽即停
```

候选行清洗细节：
- `Question` 为主，去空白/截断超长；`Dialogue` 因多为脏多轮文本，默认不进 input（仅在 `Question` 为空时回退取首句）。
- 去重：相同 `Question` 文本只保留一条。

## 生成 Prompt 设计

单段中文 Prompt，由 `task_type` 参数化。核心要求：

1. 把车主口语 `Question` 转写为清晰的 `input`（贴合汽车助手口吻，无歧义）。
2. `output` **只能**依据检索到的问界 context，准确专业。
3. **严禁**出现："根据手册…"、"根据资料…"、"资料显示…"、"作为AI…"、"作为助手…" 及引用标记如 `[资料1]`。
4. 按 `task_type` 调整 instruction 与 output 风格（问答/步骤/故障归因/术语/安全提醒）。
5. 严格输出 JSON 对象，仅含 instruction/input/output 三键。

**本地 qwen2.5:7b 专属：Prompt 内置 1–2 个完整 Few-Shot 示例**（每个示例展示 context→{instruction,input,output} 的完整映射），用于锁定 JSON 结构与 AITO 语气；并配合 `ollama.generate(..., format='json')` 双重约束。云端后端用同一份指令，但以「只返回 JSON、不要任何额外文字」的强约束 + 解析重试替代 `format='json'`。

## 健壮性

- **编码**：`pd.read_csv(..., encoding='gb18030')`；失败回退 `gbk` / `utf-8`。
- **断点续传**：`ito_lora_dataset.json` 始终为合法 JSON 数组，每 10 条成功后写临时文件再 `os.replace` 原子替换；`meta.jsonl` 同步追加。重启时读 meta 收集已完成 QID 跳过。
- **异常隔离**：每行 try/except 包住「检索 + 生成 + json.loads」；LLM 调用含超时 + N 次重试；任何失败 → 写日志并跳过，绝不让整脚本崩溃。
- **进度**：`tqdm` 遍历候选，postfix 显示 `ok/skip/fail` 实时计数。
- **校验**：三字段缺失或空、命中禁语 → 计为失败（可触发一次重试），不写入。

## 达标计数

`sample_size` 为单批候选；`target_size` 为目标成功数。若候选耗尽仍不足 target，从过滤池剩余行继续补抽，直至 target 或池尽，并在结束时打印实际产出与跳过率。默认 `target_size=300` 落在 mentor 的 200–400 区间。

## 默认决策（已与用户确认）

- 脚本置于 `scripts/`，输出文件置于仓库根目录。
- 检索使用合并 `corpus` 索引，不做车型过滤（CSV 问题与品牌无关）。
- output 措辞面向「问界车辆」通用表述，不绑定具体车型/配置。
- 后端 local/cloud 可切换，默认 local。

## 数据来源说明（交付附注）

数据生成方式：以真实车主问题（AutoMaster 数据集，脱敏取 `Question`）为种子 → RAG 检索问界官方车主手册获取规范依据 → 本地/云端大模型按 5 类指令改写为 {instruction,input,output}，并经禁语/字段/JSON 校验过滤。每条样本的来源章节与检索分数记录于 `ito_lora_dataset.meta.jsonl` 供人工抽查与复核。

## 不做（YAGNI）

- 不做并发/批处理（本地 7B 串行即可，复杂度不值）。
- 不做自动事实校验模型（靠弱检索跳过 + 禁语校验 + meta 人工抽查保证质量）。
- 不修改既有检索/生成代码。
