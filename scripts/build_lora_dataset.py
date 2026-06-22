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
