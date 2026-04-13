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
