import os
import textwrap
import pytest
import inference


def write_yaml(tmp_path, content: str):
    p = tmp_path / "models.yaml"
    p.write_text(textwrap.dedent(content))
    return p


def test_load_config_returns_list_of_models(tmp_path):
    path = write_yaml(tmp_path, """
        models:
          - model: m1
            base_url: http://a/v1
            api_key: k1
          - model: m2
            base_url: http://b/v1
            api_key: k2
    """)
    models = inference.load_config(path)
    assert len(models) == 2
    assert models[0]["model"] == "m1"
    assert models[1]["base_url"] == "http://b/v1"


def test_load_config_expands_env_var_in_api_key(tmp_path, monkeypatch):
    monkeypatch.setenv("MY_KEY", "secret-123")
    path = write_yaml(tmp_path, """
        models:
          - model: m
            base_url: http://a/v1
            api_key: ${MY_KEY}
    """)
    models = inference.load_config(path)
    assert models[0]["api_key"] == "secret-123"


def test_load_config_missing_env_var_raises(tmp_path, monkeypatch):
    monkeypatch.delenv("UNSET_VAR", raising=False)
    path = write_yaml(tmp_path, """
        models:
          - model: m
            base_url: http://a/v1
            api_key: ${UNSET_VAR}
    """)
    with pytest.raises(inference.ConfigError) as exc:
        inference.load_config(path)
    assert "UNSET_VAR" in str(exc.value)


def test_load_config_missing_file_returns_empty(tmp_path):
    path = tmp_path / "nope.yaml"
    assert inference.load_config(path) == []


def test_load_config_empty_file_returns_empty(tmp_path):
    path = write_yaml(tmp_path, "")
    assert inference.load_config(path) == []


def test_load_config_invalid_yaml_raises(tmp_path):
    path = tmp_path / "models.yaml"
    path.write_text("models: [unclosed")
    with pytest.raises(inference.ConfigError):
        inference.load_config(path)


def test_load_config_missing_required_field_raises(tmp_path):
    path = write_yaml(tmp_path, """
        models:
          - model: m
            base_url: http://a/v1
    """)
    with pytest.raises(inference.ConfigError) as exc:
        inference.load_config(path)
    assert "api_key" in str(exc.value)
