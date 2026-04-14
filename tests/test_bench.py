import inference


def test_parse_command_recognizes_bench():
    assert inference.parse_command("/bench") == ("bench", "")


def test_parse_command_bench_with_args():
    assert inference.parse_command("/bench 256 512") == ("bench", "256 512")


def test_bench_in_known_commands():
    assert "bench" in inference.KNOWN_COMMANDS


def test_parse_bench_args_defaults():
    input_tok, output_tok, levels = inference.parse_bench_args("")
    assert input_tok == 128
    assert output_tok == 128
    assert levels == [1, 2, 4, 8, 16, 32, 64, 128]


def test_parse_bench_args_custom_tokens():
    input_tok, output_tok, levels = inference.parse_bench_args("256 512")
    assert input_tok == 256
    assert output_tok == 512
    assert levels == [1, 2, 4, 8, 16, 32, 64, 128]


def test_parse_bench_args_custom_levels():
    input_tok, output_tok, levels = inference.parse_bench_args("128 128 1,4,16")
    assert input_tok == 128
    assert output_tok == 128
    assert levels == [1, 4, 16]


def test_parse_bench_args_invalid_raises():
    import pytest
    with pytest.raises(ValueError):
        inference.parse_bench_args("abc")


def test_generate_bench_prompt_approximate_token_count():
    prompt = inference._generate_bench_prompt(128)
    token_count = inference._estimate_tokens(prompt)
    # Allow 20% tolerance
    assert abs(token_count - 128) / 128 < 0.20


def test_generate_bench_prompt_scales_up():
    prompt_small = inference._generate_bench_prompt(64)
    prompt_large = inference._generate_bench_prompt(512)
    assert len(prompt_large) > len(prompt_small)


import io


def test_bench_single_request_returns_bench_result(
    fake_client_factory, make_chunk_fn, fake_usage_cls
):
    chunks = [
        make_chunk_fn(content="Hello "),
        make_chunk_fn(content="world"),
        make_chunk_fn(usage=fake_usage_cls(prompt_tokens=10, completion_tokens=2)),
    ]
    client = fake_client_factory(chunks)
    result = inference._bench_single_request(
        client=client, model="m", prompt="test", max_tokens=128
    )
    assert result.error is None
    assert result.ttft_seconds >= 0
    assert result.generation_seconds >= 0
    assert result.completion_tokens == 2


def test_bench_single_request_records_error_on_exception(monkeypatch):
    class BrokenClient:
        class chat:
            class completions:
                @staticmethod
                def create(**kwargs):
                    raise ConnectionError("refused")
    result = inference._bench_single_request(
        client=BrokenClient(), model="m", prompt="test", max_tokens=10
    )
    assert result.error is not None
    assert "refused" in result.error


def test_bench_single_request_measures_ttft(
    fake_client_factory, make_chunk_fn, fake_usage_cls
):
    chunks = [
        make_chunk_fn(content="hi"),
        make_chunk_fn(usage=fake_usage_cls(1, 1)),
    ]
    client = fake_client_factory(chunks, delays=[0.05, 0.0])
    result = inference._bench_single_request(
        client=client, model="m", prompt="test", max_tokens=10
    )
    assert result.ttft_seconds >= 0.04


def test_bench_single_request_sets_max_tokens(
    fake_client_factory, make_chunk_fn, fake_usage_cls
):
    chunks = [
        make_chunk_fn(content="ok"),
        make_chunk_fn(usage=fake_usage_cls(1, 1)),
    ]
    client = fake_client_factory(chunks)
    inference._bench_single_request(
        client=client, model="m", prompt="test", max_tokens=256
    )
    kwargs = client.chat.completions.last_kwargs
    assert kwargs["max_tokens"] == 256
