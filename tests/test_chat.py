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

    # Drive the REPL: one user message, then /exit.
    inputs = iter(["hello there", "/exit"])
    monkeypatch.setattr("builtins.input", lambda *a, **kw: next(inputs))

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

    # `_interactive_prompt` will be called from the /add handler; raise KeyboardInterrupt.
    def cancelling_prompt(field):
        raise KeyboardInterrupt

    monkeypatch.setattr(inference, "_interactive_prompt", cancelling_prompt)

    # Drive the REPL: type /add (will trigger the cancelling prompt), then /exit.
    inputs = iter(["/add", "/exit"])
    monkeypatch.setattr("builtins.input", lambda *a, **kw: next(inputs))

    inference.main()  # should NOT raise

    captured = capsys.readouterr().out
    assert "[cancelled]" in captured
