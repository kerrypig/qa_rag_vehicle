from dataset_gen.config_overrides import apply_overrides


def test_sets_existing_nested_key():
    raw = {"query_rewrite": {"enabled": True}, "retrieval": {"keyword_search": {"enabled": True}}}
    apply_overrides(raw, {
        "query_rewrite.enabled": False,
        "retrieval.keyword_search.enabled": False,
    })
    assert raw["query_rewrite"]["enabled"] is False
    assert raw["retrieval"]["keyword_search"]["enabled"] is False


def test_creates_missing_path():
    raw = {}
    apply_overrides(raw, {"a.b.c": 1})
    assert raw["a"]["b"]["c"] == 1


def test_overwrites_non_dict_intermediate():
    raw = {"a": 5}
    apply_overrides(raw, {"a.b": 2})
    assert raw["a"] == {"b": 2}
