#!/usr/bin/env python3
"""车载手册 RAG 问答 — CLI 入口。"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config_loader import load_config
from generate.prompt_builder import build_prompt
from generate.qwen_client import QwenClient
from ingest.pipeline import run_build
from retrieve.pipeline import Retriever, format_retrieved_chunks
from session.memory import SessionState

log = logging.getLogger(__name__)


def setup_logging(log_dir: Path, verbose: bool) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"run_{ts}.log"
    handlers: list[logging.Handler] = [logging.FileHandler(log_file, encoding="utf-8")]
    if verbose:
        handlers.append(logging.StreamHandler(sys.stdout))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )


def cmd_build(args) -> None:
    config = load_config(args.config, strategy_override=args.strategy)
    setup_logging(config.log_dir, verbose=True)
    pdf = Path(args.pdf) if args.pdf else None
    count = run_build(config, pdf_path=pdf)
    print(f"建库完成：{count} chunks → {config.index_path()}")


def cmd_info(args) -> None:
    config = load_config(args.config, strategy_override=args.strategy)
    idx = config.index_path()
    meta = idx / "meta.json"
    print("=== 车载 RAG 配置 ===")
    print(f"车型:       {config.vehicle_model}")
    print(f"切分策略:   {config.chunk_strategy}")
    print(f"索引路径:   {idx}")
    print(f"Hybrid:     {config.get('retrieval', 'hybrid_search', 'enabled')}")
    print(f"Rewrite:    {config.get('query_rewrite', 'enabled')}")
    if meta.exists():
        import json

        with open(meta, encoding="utf-8") as f:
            m = json.load(f)
        print(f"已建库:     是 ({m.get('chunk_count', '?')} chunks)")
    else:
        print("已建库:     否 — 请先运行 python main.py build")


def cmd_chat(args) -> None:
    config = load_config(args.config, strategy_override=args.strategy)
    setup_logging(config.log_dir, verbose=config.get("logging", "verbose", default=True))

    index_path = config.index_path()
    if not (index_path / "faiss").exists():
        print(f"索引不存在: {index_path}\n请先运行: python main.py build")
        sys.exit(1)

    retriever = Retriever(config, index_path)
    retriever.load()
    llm = QwenClient(config)
    session = SessionState()

    qa_log: list[str] = []
    log_cfg = config.raw.get("logging", {})

    print(f"\n=== 车载智能问答 ({config.vehicle_model}) ===")
    print("输入问题开始对话，/quit 退出，/clear 清空会话，/log 查看上轮检索，/config 查看配置\n")

    while True:
        try:
            question = input("你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n会话已结束。")
            break

        if not question:
            continue
        if question == "/quit":
            print("会话已结束。")
            break
        if question == "/clear":
            session.clear()
            print("会话已清空。")
            continue
        if question == "/config":
            cmd_info(args)
            continue
        if question == "/log" and session.last_retrieval:
            r = session.last_retrieval
            print(f"  改写: {r['rewritten']}")
            print(
                format_retrieved_chunks(
                    r["docs"],
                    r["scores"],
                    cache_hits=r["cache_hits"],
                    new_retrieved=r["new_retrieved"],
                )
            )
            continue
        if question == "/log":
            print("  暂无可展示的检索记录。")
            continue

        result = retriever.retrieve(question, session)
        session.last_retrieval = {
            "rewritten": result.rewritten_query,
            "cache_hits": result.cache_hits,
            "new_retrieved": result.new_retrieved,
            "docs": result.docs,
            "scores": result.scores,
        }

        chunks_log = format_retrieved_chunks(
            result.docs,
            result.scores,
            cache_hits=result.cache_hits,
            new_retrieved=result.new_retrieved,
        )
        log.info("问题: %s", question)
        if result.rewritten_query != question:
            log.info("改写: %s", result.rewritten_query)
        log.info("\n%s", chunks_log)

        if config.get("logging", "verbose"):
            hs = "hybrid" if config.get("retrieval", "hybrid_search", "enabled") else "vector"
            print(
                f"[检索] 缓存 {result.cache_hits} | 新检索 {result.new_retrieved} | "
                f"{config.chunk_strategy}+{hs}（chunk 全文见 logs/）"
            )
            if result.rewritten_query != question:
                print(f"[改写] {result.rewritten_query}")

        prompt = build_prompt(question, result.docs)
        answer, rid = llm.chat(prompt, previous_response_id=session.response_id)
        session.response_id = rid
        session.add_turn(question, answer)

        print(f"助手: {answer}\n")

        if log_cfg.get("save_qa_log"):
            qa_log.append(
                f"Q: {question}\n"
                f"改写: {result.rewritten_query}\n"
                f"资料:\n{chunks_log}\n"
                f"A: {answer}\n---"
            )

    if qa_log and log_cfg.get("save_qa_log"):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = config.log_dir / f"qa_{ts}.txt"
        path.write_text("\n\n".join(qa_log), encoding="utf-8")
        print(f"问答日志: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="车载手册 RAG 问答系统")
    parser.add_argument("--config", default=str(ROOT / "config.yaml"), help="配置文件路径")
    sub = parser.add_subparsers(dest="command", required=True)

    p_build = sub.add_parser("build", help="从 PDF 建库")
    p_build.add_argument("--strategy", choices=["hierarchy", "semantic", "fixed_size"])
    p_build.add_argument("--pdf", help="指定单个 PDF 路径")
    p_build.set_defaults(func=cmd_build)

    p_chat = sub.add_parser("chat", help="多轮 CLI 问答")
    p_chat.add_argument("--strategy", choices=["hierarchy", "semantic", "fixed_size"])
    p_chat.set_defaults(func=cmd_chat)

    p_info = sub.add_parser("info", help="查看配置与索引状态")
    p_info.add_argument("--strategy", choices=["hierarchy", "semantic", "fixed_size"])
    p_info.set_defaults(func=cmd_info)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
