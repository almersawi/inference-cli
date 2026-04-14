import pytest

import inference


def test_parse_command_recognizes_known_commands():
    assert inference.parse_command("/clear") == ("clear", "")
    assert inference.parse_command("/exit") == ("exit", "")
    assert inference.parse_command("/remove") == ("remove", "")
    assert inference.parse_command("/system you are helpful") == (
        "system",
        "you are helpful",
    )


def test_parse_command_is_case_insensitive():
    assert inference.parse_command("/CLEAR") == ("clear", "")


def test_parse_command_returns_none_for_non_command():
    assert inference.parse_command("hello") is None
    assert inference.parse_command("") is None
    assert inference.parse_command("  /clear") is None  # only leading slash counts


def test_parse_command_unknown_returns_unknown_marker():
    assert inference.parse_command("/wat") == ("__unknown__", "wat")


def test_parse_command_bare_slash_returns_pick():
    assert inference.parse_command("/") == ("__pick__", "")


def test_pick_command_returns_selected_command():
    def fake_select(message, choices):
        return "/clear  — Clear conversation history"
    assert inference._pick_command(_select=fake_select) == "clear"


def test_pick_command_returns_none_on_cancel():
    def fake_select(message, choices):
        return None
    assert inference._pick_command(_select=fake_select) is None


def test_handle_clear_drops_user_and_assistant_keeps_system():
    history = [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    new_history = inference.handle_clear(history)
    assert new_history == [{"role": "system", "content": "be brief"}]


def test_handle_clear_empties_when_no_system():
    history = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    assert inference.handle_clear(history) == []


def test_handle_system_inserts_when_absent():
    history = [{"role": "user", "content": "hi"}]
    new_history = inference.handle_system(history, "be terse")
    assert new_history[0] == {"role": "system", "content": "be terse"}
    assert new_history[1] == {"role": "user", "content": "hi"}


def test_handle_system_replaces_when_present():
    history = [
        {"role": "system", "content": "old"},
        {"role": "user", "content": "hi"},
    ]
    new_history = inference.handle_system(history, "new")
    assert new_history[0]["content"] == "new"
    assert len(new_history) == 2


def test_handle_add_writes_new_model_to_yaml(tmp_path):
    config_path = tmp_path / "models.yaml"
    answers = iter([
        ("model", "new-model"),
        ("base_url", "http://x/v1"),
        ("api_key", "k"),
    ])

    def prompt(field: str) -> str:
        name, value = next(answers)
        assert name == field
        return value

    new_model = inference.handle_add(prompt=prompt, config_path=config_path)
    assert new_model == {
        "model": "new-model",
        "base_url": "http://x/v1",
        "api_key": "k",
    }
    reloaded = inference.load_config(config_path)
    assert reloaded[-1]["model"] == "new-model"


def test_handle_add_appends_to_existing_yaml(tmp_path):
    config_path = tmp_path / "models.yaml"
    config_path.write_text(
        "models:\n"
        "  - model: existing\n"
        "    base_url: http://a/v1\n"
        "    api_key: k\n"
    )

    def prompt(field: str) -> str:
        return {"model": "new", "base_url": "http://b/v1", "api_key": "k2"}[field]

    inference.handle_add(prompt=prompt, config_path=config_path)
    reloaded = inference.load_config(config_path)
    assert [m["model"] for m in reloaded] == ["existing", "new"]


def test_handle_remove_drops_named_model(tmp_path):
    config_path = tmp_path / "models.yaml"
    config_path.write_text(
        "models:\n"
        "  - model: a\n    base_url: http://x/v1\n    api_key: k\n"
        "  - model: b\n    base_url: http://y/v1\n    api_key: k\n"
    )
    remaining = inference.handle_remove(model_name="a", config_path=config_path)
    assert [m["model"] for m in remaining] == ["b"]
    reloaded = inference.load_config(config_path)
    assert [m["model"] for m in reloaded] == ["b"]


def test_handle_remove_refuses_to_remove_last_model(tmp_path):
    config_path = tmp_path / "models.yaml"
    config_path.write_text(
        "models:\n"
        "  - model: a\n    base_url: http://x/v1\n    api_key: k\n"
    )
    with pytest.raises(inference.ConfigError) as exc:
        inference.handle_remove(model_name="a", config_path=config_path)
    assert "last" in str(exc.value).lower()
    assert [m["model"] for m in inference.load_config(config_path)] == ["a"]


def test_handle_remove_unknown_model_raises(tmp_path):
    config_path = tmp_path / "models.yaml"
    config_path.write_text(
        "models:\n"
        "  - model: a\n    base_url: http://x/v1\n    api_key: k\n"
        "  - model: b\n    base_url: http://y/v1\n    api_key: k\n"
    )
    with pytest.raises(inference.ConfigError) as exc:
        inference.handle_remove(model_name="zz", config_path=config_path)
    assert "zz" in str(exc.value)


def test_handle_remove_preserves_env_var_syntax_in_api_key(tmp_path, monkeypatch):
    """Regression: removing one model must not expand ${ENV_VAR} refs in surviving
    entries and persist the secret to disk."""
    monkeypatch.setenv("MY_KEY", "super-secret-value")
    config_path = tmp_path / "models.yaml"
    config_path.write_text(
        "models:\n"
        "  - model: a\n    base_url: http://x/v1\n    api_key: ${MY_KEY}\n"
        "  - model: b\n    base_url: http://y/v1\n    api_key: plain\n"
    )
    inference.handle_remove(model_name="b", config_path=config_path)
    raw = config_path.read_text()
    assert "${MY_KEY}" in raw
    assert "super-secret-value" not in raw


def test_ask_disable_thinking_returns_true_for_y(monkeypatch):
    monkeypatch.setattr(inference, "_interactive_prompt", lambda field: "y")
    assert inference._ask_disable_thinking() is True


def test_ask_disable_thinking_returns_true_for_yes_case_insensitive(monkeypatch):
    monkeypatch.setattr(inference, "_interactive_prompt", lambda field: "YES")
    assert inference._ask_disable_thinking() is True


def test_ask_disable_thinking_returns_false_for_empty(monkeypatch):
    monkeypatch.setattr(inference, "_interactive_prompt", lambda field: "")
    assert inference._ask_disable_thinking() is False


def test_ask_disable_thinking_returns_false_for_n(monkeypatch):
    monkeypatch.setattr(inference, "_interactive_prompt", lambda field: "n")
    assert inference._ask_disable_thinking() is False


def test_ask_disable_thinking_returns_false_for_anything_else(monkeypatch):
    monkeypatch.setattr(inference, "_interactive_prompt", lambda field: "garbage")
    assert inference._ask_disable_thinking() is False
