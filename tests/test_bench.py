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
