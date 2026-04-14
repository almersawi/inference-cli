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


def test_aggregate_level_computes_stats():
    results = [
        inference.BenchResult(ttft_seconds=0.1, generation_seconds=1.0, completion_tokens=50),
        inference.BenchResult(ttft_seconds=0.2, generation_seconds=2.0, completion_tokens=100),
        inference.BenchResult(ttft_seconds=0.3, generation_seconds=1.5, completion_tokens=75),
    ]
    stats = inference._aggregate_level(results, concurrency=3, wall_seconds=2.5)
    assert stats.concurrency == 3
    assert stats.total_count == 3
    assert stats.error_count == 0
    # Mean TTFT = (0.1 + 0.2 + 0.3) / 3 = 0.2
    assert abs(stats.ttft_mean - 0.2) < 0.01
    # Median TTFT = 0.2
    assert abs(stats.ttft_median - 0.2) < 0.01
    assert abs(stats.ttft_min - 0.1) < 0.01
    assert abs(stats.ttft_max - 0.3) < 0.01
    # Total throughput = (50 + 100 + 75) / 2.5 = 90
    assert abs(stats.total_throughput - 90.0) < 1.0
    # Requests/sec = 3 / 2.5 = 1.2
    assert abs(stats.requests_per_sec - 1.2) < 0.1


def test_aggregate_level_handles_errors():
    results = [
        inference.BenchResult(ttft_seconds=0.1, generation_seconds=1.0, completion_tokens=50),
        inference.BenchResult(error="ConnectionError: refused"),
    ]
    stats = inference._aggregate_level(results, concurrency=2, wall_seconds=1.5)
    assert stats.total_count == 2
    assert stats.error_count == 1
    # Stats computed from the 1 successful result only
    assert abs(stats.ttft_mean - 0.1) < 0.01


def test_aggregate_level_all_errors():
    results = [
        inference.BenchResult(error="err1"),
        inference.BenchResult(error="err2"),
    ]
    stats = inference._aggregate_level(results, concurrency=2, wall_seconds=1.0)
    assert stats.error_count == 2
    assert stats.ttft_mean == 0.0
    assert stats.tps_mean == 0.0


def test_run_bench_level_fires_concurrent_requests(
    fake_client_factory, make_chunk_fn, fake_usage_cls
):
    chunks = [
        make_chunk_fn(content="ok"),
        make_chunk_fn(usage=fake_usage_cls(prompt_tokens=5, completion_tokens=10)),
    ]
    client = fake_client_factory(chunks)
    stats = inference._run_bench_level(
        client=client, model="m", prompt="test", max_tokens=64, concurrency=4
    )
    assert stats.concurrency == 4
    assert stats.total_count == 4
    assert stats.error_count == 0
    assert stats.total_throughput > 0


def test_run_bench_level_records_partial_failures(monkeypatch):
    call_count = {"n": 0}

    def flaky_bench(**kwargs):
        call_count["n"] += 1
        if call_count["n"] % 2 == 0:
            return inference.BenchResult(error="boom")
        return inference.BenchResult(
            ttft_seconds=0.1, generation_seconds=0.5, completion_tokens=20
        )

    monkeypatch.setattr(inference, "_bench_single_request", lambda **kw: flaky_bench(**kw))
    stats = inference._run_bench_level(
        client=None, model="m", prompt="test", max_tokens=64, concurrency=4
    )
    assert stats.total_count == 4
    assert stats.error_count == 2


def test_render_bench_table_outputs_rich_table():
    import io as _io
    from rich.console import Console
    console = Console(file=_io.StringIO(), force_terminal=True, width=120)
    stats = [
        inference.LevelStats(
            concurrency=1, ttft_mean=0.1, ttft_p99=0.15,
            tps_mean=50.0, total_throughput=50.0, e2e_mean=1.0,
            requests_per_sec=1.0, error_count=0, total_count=1,
        ),
        inference.LevelStats(
            concurrency=4, ttft_mean=0.2, ttft_p99=0.35,
            tps_mean=45.0, total_throughput=170.0, e2e_mean=1.2,
            requests_per_sec=3.5, error_count=1, total_count=4,
        ),
    ]
    inference._render_bench_table(console, stats)
    output = console.file.getvalue()
    assert "1" in output
    assert "4" in output
    assert "Benchmark Results" in output


def test_render_bench_charts_does_not_raise():
    """Smoke test: rendering charts with valid data should not error."""
    stats = [
        inference.LevelStats(
            concurrency=1, ttft_mean=0.1, ttft_p99=0.15,
            tps_mean=50.0,
            total_throughput=50.0, e2e_mean=1.0,
            requests_per_sec=1.0, error_count=0, total_count=1,
        ),
        inference.LevelStats(
            concurrency=4, ttft_mean=0.2, ttft_p99=0.35,
            tps_mean=45.0,
            total_throughput=170.0, e2e_mean=1.2,
            requests_per_sec=3.5, error_count=0, total_count=4,
        ),
    ]
    # Should run without error
    inference._render_bench_charts(stats)
