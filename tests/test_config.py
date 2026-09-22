import pytest
import yaml

from specheads.utils.config import ConfigError, deep_merge, load_config, require


def write(tmp_path, name, data):
    path = tmp_path / name
    path.write_text(yaml.safe_dump(data))
    return path


def test_load_plain_config(tmp_path):
    path = write(tmp_path, "a.yaml", {"model": {"name": "qwen"}, "seed": 7})
    assert load_config(path) == {"model": {"name": "qwen"}, "seed": 7}


def test_extends_merges_parent_without_losing_nested_keys(tmp_path):
    """The point of `extends`: inherit the frozen protocol, override one field."""
    write(tmp_path, "base.yaml", {"bench": {"repeats": 3, "max_new_tokens": 256}, "seed": 0})
    child = write(tmp_path, "child.yaml", {"extends": "base.yaml", "bench": {"repeats": 5}})
    merged = load_config(child)
    assert merged["bench"] == {"repeats": 5, "max_new_tokens": 256}
    assert merged["seed"] == 0
    assert "extends" not in merged


def test_nested_extends_is_rejected(tmp_path):
    write(tmp_path, "grand.yaml", {"a": 1})
    write(tmp_path, "base.yaml", {"extends": "grand.yaml", "b": 2})
    child = write(tmp_path, "child.yaml", {"extends": "base.yaml"})
    with pytest.raises(ConfigError, match="nested extends"):
        load_config(child)


def test_missing_config_raises(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.yaml")


def test_non_mapping_config_raises(tmp_path):
    path = tmp_path / "list.yaml"
    path.write_text(yaml.safe_dump([1, 2, 3]))
    with pytest.raises(ConfigError, match="must be a mapping"):
        load_config(path)


def test_deep_merge_does_not_mutate_inputs():
    base = {"a": {"b": 1}}
    override = {"a": {"c": 2}}
    merged = deep_merge(base, override)
    assert merged == {"a": {"b": 1, "c": 2}}
    assert base == {"a": {"b": 1}}  # untouched


def test_require_names_every_missing_key():
    with pytest.raises(ConfigError) as excinfo:
        require({"model": {"name": "x"}}, "model.name", "bench.repeats", "seed")
    message = str(excinfo.value)
    assert "bench.repeats" in message and "seed" in message
    assert "model.name" not in message
