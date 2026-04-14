import io
import inference


def test_chat_turn_returns_assembled_text_and_uses_server_usage(
    fake_client_factory, make_chunk_fn, fake_usage_cls
):
    chunks = [
        make_chunk_fn(content="Hello"),
        make_chunk_fn(content=" world"),
        make_chunk_fn(usage=fake_usage_cls(prompt_tokens=11, completion_tokens=2)),
    ]
    client = fake_client_factory(chunks)
    history = [{"role": "user", "content": "hi"}]
    text, metrics = inference.chat_turn(
        client=client,
        model="m",
        history=history,
        out=io.StringIO(),
    )
    assert text == "Hello world"
    assert metrics.prompt_tokens == 11
    assert metrics.completion_tokens == 2
    assert metrics.prompt_tokens_estimated is False
    assert metrics.completion_tokens_estimated is False


def test_chat_turn_falls_back_to_estimation_when_no_usage(
    fake_client_factory, make_chunk_fn
):
    chunks = [make_chunk_fn(content="Hi"), make_chunk_fn(content="!")]
    client = fake_client_factory(chunks)
    history = [{"role": "user", "content": "yo"}]
    _, metrics = inference.chat_turn(
        client=client,
        model="m",
        history=history,
        out=io.StringIO(),
    )
    assert metrics.completion_tokens_estimated is True
    assert metrics.prompt_tokens_estimated is True
    assert metrics.completion_tokens >= 1
    assert metrics.prompt_tokens >= 1


def test_chat_turn_writes_streamed_content_to_out(
    fake_client_factory, make_chunk_fn
):
    chunks = [make_chunk_fn(content="abc"), make_chunk_fn(content="def")]
    client = fake_client_factory(chunks)
    out = io.StringIO()
    inference.chat_turn(
        client=client, model="m", history=[{"role": "user", "content": "x"}], out=out,
    )
    assert "abc" in out.getvalue()
    assert "def" in out.getvalue()


def test_chat_turn_passes_streaming_kwargs(
    fake_client_factory, make_chunk_fn, fake_usage_cls
):
    chunks = [make_chunk_fn(content="ok"), make_chunk_fn(usage=fake_usage_cls(1, 1))]
    client = fake_client_factory(chunks)
    inference.chat_turn(
        client=client,
        model="my-model",
        history=[{"role": "user", "content": "x"}],
        out=io.StringIO(),
    )
    kwargs = client.chat.completions.last_kwargs
    assert kwargs["model"] == "my-model"
    assert kwargs["stream"] is True
    assert kwargs["stream_options"] == {"include_usage": True}
    assert kwargs["messages"] == [{"role": "user", "content": "x"}]


def test_chat_turn_measures_ttft_before_first_content_chunk(
    fake_client_factory, make_chunk_fn, fake_usage_cls
):
    chunks = [
        make_chunk_fn(content="hi"),
        make_chunk_fn(usage=fake_usage_cls(1, 1)),
    ]
    client = fake_client_factory(chunks, delays=[0.05, 0.0])
    _, metrics = inference.chat_turn(
        client=client,
        model="m",
        history=[{"role": "user", "content": "x"}],
        out=io.StringIO(),
    )
    assert metrics.ttft_seconds >= 0.04


def test_main_runs_one_chat_turn_then_exits(
    monkeypatch, tmp_path, capsys, fake_client_factory, make_chunk_fn, fake_usage_cls
):
    config_path = tmp_path / "models.yaml"
    config_path.write_text(
        "models:\n"
        "  - model: m1\n"
        "    base_url: http://a/v1\n"
        "    api_key: k\n"
    )
    monkeypatch.setenv("INFERENCE_MODELS_CONFIG", str(config_path))

    chunks = [
        make_chunk_fn(content="hi"),
        make_chunk_fn(content="!"),
        make_chunk_fn(usage=fake_usage_cls(prompt_tokens=3, completion_tokens=2)),
    ]
    monkeypatch.setattr(
        inference,
        "make_client",
        lambda entry: fake_client_factory(chunks),
    )

    # Auto-pick the first model.
    monkeypatch.setattr(inference, "pick_model", lambda models, **kw: models[0])
    monkeypatch.setattr(inference, "_ask_disable_thinking", lambda: False)

    # Drive the REPL: one user message, then /exit.
    inputs = iter(["hello there", "/exit"])
    monkeypatch.setattr(inference, "pt_prompt", lambda *a, **kw: next(inputs))

    inference.main()

    captured = capsys.readouterr().out
    assert "hi!" in captured
    assert "TTFT:" in captured
    assert "tok/s" in captured


def test_main_ctrl_c_during_interactive_prompt_cancels_command_only(
    monkeypatch, tmp_path, capsys
):
    """Regression: Ctrl-C inside /add (or similar) must cancel that command,
    not terminate the session."""
    config_path = tmp_path / "models.yaml"
    config_path.write_text(
        "models:\n"
        "  - model: m1\n"
        "    base_url: http://a/v1\n"
        "    api_key: k\n"
    )
    monkeypatch.setenv("INFERENCE_MODELS_CONFIG", str(config_path))
    monkeypatch.setattr(inference, "pick_model", lambda models, **kw: models[0])
    monkeypatch.setattr(inference, "_ask_disable_thinking", lambda: False)

    # `_interactive_prompt` will be called from the /add handler; raise KeyboardInterrupt.
    def cancelling_prompt(field):
        raise KeyboardInterrupt

    monkeypatch.setattr(inference, "_interactive_prompt", cancelling_prompt)

    # Drive the REPL: type /add (will trigger the cancelling prompt), then /exit.
    inputs = iter(["/add", "/exit"])
    monkeypatch.setattr(inference, "pt_prompt", lambda *a, **kw: next(inputs))

    inference.main()  # should NOT raise

    captured = capsys.readouterr().out
    assert "[cancelled]" in captured


def test_main_add_then_switch_expands_env_var_in_api_key(
    monkeypatch, tmp_path, capsys, fake_client_factory, make_chunk_fn, fake_usage_cls
):
    """Regression: /add → 'switch to it now? y' must expand ${ENV_VAR} in api_key
    before constructing the OpenAI client. Otherwise the client gets the literal
    '${MY_KEY}' string and every call fails auth."""
    monkeypatch.setenv("MY_KEY", "expanded-secret")
    config_path = tmp_path / "models.yaml"
    config_path.write_text(
        "models:\n"
        "  - model: existing\n"
        "    base_url: http://x/v1\n"
        "    api_key: k\n"
    )
    monkeypatch.setenv("INFERENCE_MODELS_CONFIG", str(config_path))

    # Auto-pick the first (existing) model on initial start.
    pick_calls = {"n": 0}
    def fake_pick(models, **kw):
        pick_calls["n"] += 1
        return models[0]
    monkeypatch.setattr(inference, "pick_model", fake_pick)

    # Drive _interactive_prompt: model name, base_url, api_key (with env var ref),
    # then 'y' for "switch to it now?".
    interactive_answers = iter([
        "newmodel",       # model
        "http://y/v1",    # base_url
        "${MY_KEY}",      # api_key as env var reference
        "y",              # switch to it now?
    ])
    monkeypatch.setattr(inference, "_ask_disable_thinking", lambda: False)
    monkeypatch.setattr(
        inference, "_interactive_prompt", lambda field: next(interactive_answers)
    )

    # Capture what api_key make_client receives.
    seen = {}
    def spy_make_client(entry):
        seen["api_key"] = entry["api_key"]
        return fake_client_factory([
            make_chunk_fn(content="ok"),
            make_chunk_fn(usage=fake_usage_cls(1, 1)),
        ])
    monkeypatch.setattr(inference, "make_client", spy_make_client)

    # REPL inputs: /add, then /exit
    inputs = iter(["/add", "/exit"])
    monkeypatch.setattr(inference, "pt_prompt", lambda *a, **kw: next(inputs))

    inference.main()

    # The api_key seen by make_client during the SWITCH must be the expanded value.
    assert seen["api_key"] == "expanded-secret"
    # And the on-disk YAML must still contain the unexpanded reference.
    raw = config_path.read_text()
    assert "${MY_KEY}" in raw
    assert "expanded-secret" not in raw


def test_chat_turn_passes_extra_body_when_disable_thinking_true(
    fake_client_factory, make_chunk_fn, fake_usage_cls
):
    chunks = [
        make_chunk_fn(content="ok"),
        make_chunk_fn(usage=fake_usage_cls(prompt_tokens=1, completion_tokens=1)),
    ]
    client = fake_client_factory(chunks)
    inference.chat_turn(
        client=client,
        model="m",
        history=[{"role": "user", "content": "x"}],
        out=io.StringIO(),
        disable_thinking=True,
    )
    kwargs = client.chat.completions.last_kwargs
    assert kwargs["extra_body"] == {
        "chat_template_kwargs": {"enable_thinking": False}
    }


def test_chat_turn_omits_extra_body_when_disable_thinking_false(
    fake_client_factory, make_chunk_fn, fake_usage_cls
):
    chunks = [
        make_chunk_fn(content="ok"),
        make_chunk_fn(usage=fake_usage_cls(prompt_tokens=1, completion_tokens=1)),
    ]
    client = fake_client_factory(chunks)
    inference.chat_turn(
        client=client,
        model="m",
        history=[{"role": "user", "content": "x"}],
        out=io.StringIO(),
        disable_thinking=False,
    )
    kwargs = client.chat.completions.last_kwargs
    assert "extra_body" not in kwargs


def test_main_disable_thinking_yes_passes_extra_body_to_chat_turn(
    monkeypatch, tmp_path, capsys, fake_client_factory, make_chunk_fn, fake_usage_cls
):
    """Integration: when the post-pick prompt answers 'y', every chat turn
    sends extra_body={"chat_template_kwargs": {"enable_thinking": False}}."""
    config_path = tmp_path / "models.yaml"
    config_path.write_text(
        "models:\n"
        "  - model: m1\n"
        "    base_url: http://a/v1\n"
        "    api_key: k\n"
    )
    monkeypatch.setenv("INFERENCE_MODELS_CONFIG", str(config_path))

    chunks = [
        make_chunk_fn(content="ok"),
        make_chunk_fn(usage=fake_usage_cls(prompt_tokens=1, completion_tokens=1)),
    ]
    captured_client = fake_client_factory(chunks)
    monkeypatch.setattr(
        inference, "make_client", lambda entry: captured_client
    )
    monkeypatch.setattr(inference, "pick_model", lambda models, **kw: models[0])

    # Answer 'y' to the disable-thinking prompt.
    monkeypatch.setattr(inference, "_interactive_prompt", lambda field: "y")

    inputs = iter(["hi", "/exit"])
    monkeypatch.setattr(inference, "pt_prompt", lambda *a, **kw: next(inputs))

    inference.main()

    kwargs = captured_client.chat.completions.last_kwargs
    assert kwargs["extra_body"] == {
        "chat_template_kwargs": {"enable_thinking": False}
    }
    captured_out = capsys.readouterr().out
    assert "thinking: disabled" in captured_out


def test_main_disable_thinking_no_omits_extra_body(
    monkeypatch, tmp_path, capsys, fake_client_factory, make_chunk_fn, fake_usage_cls
):
    """Integration: when the post-pick prompt answers 'n' (or empty), no
    extra_body is sent and no 'thinking: disabled' line appears."""
    config_path = tmp_path / "models.yaml"
    config_path.write_text(
        "models:\n"
        "  - model: m1\n"
        "    base_url: http://a/v1\n"
        "    api_key: k\n"
    )
    monkeypatch.setenv("INFERENCE_MODELS_CONFIG", str(config_path))

    chunks = [
        make_chunk_fn(content="ok"),
        make_chunk_fn(usage=fake_usage_cls(prompt_tokens=1, completion_tokens=1)),
    ]
    captured_client = fake_client_factory(chunks)
    monkeypatch.setattr(
        inference, "make_client", lambda entry: captured_client
    )
    monkeypatch.setattr(inference, "pick_model", lambda models, **kw: models[0])

    # Answer empty (default N) to the disable-thinking prompt.
    monkeypatch.setattr(inference, "_interactive_prompt", lambda field: "")

    inputs = iter(["hi", "/exit"])
    monkeypatch.setattr(inference, "pt_prompt", lambda *a, **kw: next(inputs))

    inference.main()

    kwargs = captured_client.chat.completions.last_kwargs
    assert "extra_body" not in kwargs
    captured_out = capsys.readouterr().out
    assert "thinking: disabled" not in captured_out
