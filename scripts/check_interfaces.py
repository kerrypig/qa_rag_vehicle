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
