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
    config = probe_config(load_config())
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
