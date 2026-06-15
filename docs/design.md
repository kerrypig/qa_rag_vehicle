# 车载手册 RAG 问答 — 设计规格

> 日期：2026-06-15  
> 状态：已批准，MVP 实现中

## 1. 目标

单场景车载问答：购车、驾驶、故障、保养、保险等问题，**严格基于**上传的车辆文档回答，支持多轮对话与会话级 chunk 缓存。

## 2. 已确认决策

| 项 | 决策 |
|----|------|
| 交互 | CLI（`python main.py chat`） |
| 多轮记忆 | C：会话 chunk 缓存 + API 对话链 |
| 车型 | B：单库 + metadata 过滤 |
| 图片 MVP | 跳过，metadata 预留 `image_refs` |
| 会话持久化 | MVP 仅内存 |
| 主 LLM | DashScope `qwen3.6-plus` Responses API |
| Query Rewrite | Ollama `qwen2.5:7b` |
| Embedding | `BAAI/bge-large-zh-v1.5` |

## 3. 架构

```
qa_rag_vehicle/
├── main.py              # build | chat | info
├── config.yaml
├── ingest/              # PDF → 切分 → 索引
├── retrieve/            # Rewrite + Hybrid + 缓存
├── generate/            # Qwen API + Prompt
├── session/             # 内存会话
└── prompts/             # 模板
```

### 建库流程

PDF → PyMuPDF 解析 → 切分策略 → metadata → FAISS + BM25 → `indexes/{strategy}/{vehicle_model}/`

### 问答流程

用户问题 → [Rewrite] → 缓存重排 → [Hybrid/向量] → metadata 过滤 → Prompt → Qwen API → 更新缓存

## 4. Metadata 字段

- `vehicle_model`, `doc_type`, `source_file`, `page`, `section_path`, `chunk_id`, `image_refs`

## 5. 切分策略

| 策略 | 适用 |
|------|------|
| `hierarchy` | 有章节的车主手册（默认） |
| `semantic` | 无清晰目录的文档 |
| `fixed_size` | baseline 对比 |

## 6. 配置

完整字段见根目录 `config.yaml`。

## 7. MVP 验收

- [ ] `build` 成功建库
- [ ] 三种 strategy 可切换
- [ ] 多轮追问有缓存命中
- [ ] Hybrid / Rewrite 开关有效
- [ ] 手册外问题拒答

## 8. 二期

- 图片描述（方案 A + Qwen-VL）
- 会话 `--resume`
- Web / API
