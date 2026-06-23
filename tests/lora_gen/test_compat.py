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
