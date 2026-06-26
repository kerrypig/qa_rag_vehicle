from lora_gen.dgconfig import load_dg_config

def test_loads_defaults_and_hash():
    cfg = load_dg_config()
    assert cfg.target_size == 100
    assert cfg.answerability["partial_ok_quota"] == 0.08
    assert cfg.quality["max_output_chars"]["步骤指导"] == 800
    assert isinstance(cfg.config_hash, str) and len(cfg.config_hash) == 12

def test_override_target(tmp_path):
    cfg = load_dg_config(target_override=50)
    assert cfg.target_size == 50
