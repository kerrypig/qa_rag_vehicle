# qa_rag_vehicle — 车载手册 RAG 问答

基于车主手册 PDF 的多轮问答系统。回答严格依据上传文档，支持可配置的切分策略与检索组件。

## 功能

- **多轮对话**：DashScope Responses API（`previous_response_id`）+ 会话 chunk 缓存
- **切分策略**（`config.yaml` 切换）：层级目录 / 语义 / 字符数
- **检索组件**（可开关）：Query Rewrite（Ollama `qwen2.5:7b`）、Hybrid Search（FAISS + BM25）
- **车型 metadata**：单库多车型扩展，检索按 `vehicle_model` 过滤

## 快速开始

### 1. 环境

```bash
cd qa_rag_vehicle
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### 2. API 密钥（不会上传 GitHub）

```bash
copy .env.example .env
# 编辑 .env，填入 DASHSCOPE_API_KEY
```

### 3. 放入手册 PDF

将 `m9-2025-evr-product-manual-20260415.pdf` 放入 `data/manuals/`。

### 4. 启动 Ollama（Query Rewrite 需要）

```bash
ollama run qwen2.5:7b
```

若关闭改写，可在 `config.yaml` 设置 `query_rewrite.enabled: false`。

### 5. 建库 & 问答

```bash
python main.py build
python main.py build --strategy semantic
python main.py chat
python main.py info
```

## CLI 命令

| 命令 | 说明 |
|------|------|
| `python main.py build` | PDF → FAISS + BM25 索引 |
| `python main.py build --strategy hierarchy` | 指定切分策略 |
| `python main.py chat` | 多轮终端问答 |
| `python main.py info` | 查看配置与建库状态 |

对话中：`/quit` 退出 · `/clear` 清空会话 · `/log` 检索详情 · `/config` 配置摘要

## 配置说明

见 `config.yaml`。关键项：

- `chunking.strategy`：`hierarchy` | `semantic` | `fixed_size`
- `retrieval.hybrid_search.enabled`：Hybrid 开关
- `query_rewrite.enabled`：Ollama 改写开关
- `vehicle.model`：当前车型，与建库 metadata 一致

索引路径：`indexes/{strategy}/{vehicle_model}/`

## 安全说明

- `.env` 已在 `.gitignore` 中，**切勿**将 API Key 提交到 Git
- PDF 手册默认不入库（体积大），请自行放置

## 设计文档

详见 [docs/design.md](docs/design.md)。

## 二期规划

- 图片描述入库（Qwen-VL）
- 会话持久化 `--resume`
- Web UI / FastAPI
