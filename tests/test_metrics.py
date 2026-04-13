import inference


def test_format_metrics_with_server_usage():
    m = inference.Metrics(
        ttft_seconds=0.234,
        generation_seconds=3.687,
        prompt_tokens=87,
        completion_tokens=156,
        prompt_tokens_estimated=False,
        completion_tokens_estimated=False,
    )
    line = inference.format_metrics(m)
    assert "TTFT: 234ms" in line
    assert "42.3 tok/s" in line
    assert "in: 87" in line
    assert "out: 156" in line
    assert "*" not in line


def test_format_metrics_marks_estimated_counts():
    m = inference.Metrics(
        ttft_seconds=0.1,
        generation_seconds=1.0,
        prompt_tokens=10,
        completion_tokens=20,
        prompt_tokens_estimated=True,
        completion_tokens_estimated=True,
    )
    line = inference.format_metrics(m)
    assert "in: 10*" in line
    assert "out: 20*" in line


def test_format_metrics_zero_generation_time_does_not_divide_by_zero():
    m = inference.Metrics(
        ttft_seconds=0.5,
        generation_seconds=0.0,
        prompt_tokens=5,
        completion_tokens=0,
        prompt_tokens_estimated=False,
        completion_tokens_estimated=False,
    )
    line = inference.format_metrics(m)
    assert "0.0 tok/s" in line


def test_format_metrics_rounds_throughput_to_one_decimal():
    m = inference.Metrics(
        ttft_seconds=0.1,
        generation_seconds=2.0,
        prompt_tokens=1,
        completion_tokens=85,
        prompt_tokens_estimated=False,
        completion_tokens_estimated=False,
    )
    assert "42.5 tok/s" in inference.format_metrics(m)
