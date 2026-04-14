# Benchmark Feature Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `/bench` slash command that stress-tests the current model across configurable concurrency levels and renders metrics as a Rich table + plotext charts.

**Architecture:** Register `/bench` in `KNOWN_COMMANDS`, add a `handle_bench()` function that uses `ThreadPoolExecutor` to fire concurrent streaming requests, aggregates per-level stats (TTFT, tok/s, latency, throughput, errors), then renders a Rich summary table and 4 plotext bar charts. All code lives in `inference.py` (single-file project).

**Tech Stack:** `plotext` (new dependency), `concurrent.futures.ThreadPoolExecutor`, `statistics` module, existing `OpenAI` sync client, Rich `Table`.

---

### Task 1: Add `plotext` dependency and register `/bench` command

**Files:**
- Modify: `inference.py:4-10` (PEP 723 dependencies)
- Modify: `inference.py:119-122` (`_help_line`)
- Modify: `inference.py:329` (`KNOWN_COMMANDS`)

- [ ] **Step 1: Write the failing test**

Create `tests/test_bench.py`:

```python
import inference


def test_parse_command_recognizes_bench():
    assert inference.parse_command("/bench") == ("bench", "")


def test_parse_command_bench_with_args():
    assert inference.parse_command("/bench 256 512") == ("bench", "256 512")


def test_bench_in_known_commands():
    assert "bench" in inference.KNOWN_COMMANDS
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/islam/Desktop/inference && uv run pytest tests/test_bench.py -v`
Expected: FAIL — "bench" not in KNOWN_COMMANDS

- [ ] **Step 3: Add plotext dependency and register bench command**

In `inference.py`, update the PEP 723 block (lines 4-10) to add plotext:

```python
# dependencies = [
#     "openai>=1.40",
#     "pyyaml>=6.0",
#     "questionary>=2.0",
#     "rich>=13.7",
#     "tiktoken>=0.7",
#     "plotext>=5.2",
# ]
```

Update `KNOWN_COMMANDS` (line 329):

```python
KNOWN_COMMANDS = {"clear", "exit", "model", "system", "add", "remove", "bench"}
```

Update `_help_line` (lines 119-122):

```python
def _help_line(console: Console) -> None:
    console.print(
        "Commands: /clear /model /system /add /remove /bench /exit", style="dim"
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/islam/Desktop/inference && uv run pytest tests/test_bench.py -v`
Expected: PASS

- [ ] **Step 5: Run full test suite to check for regressions**

Run: `cd /Users/islam/Desktop/inference && uv run pytest -v`
Expected: All tests pass

- [ ] **Step 6: Commit**

```bash
git add inference.py tests/test_bench.py
git commit -m "feat(bench): register /bench command and add plotext dependency"
```

---

### Task 2: Implement `_generate_bench_prompt()` and `parse_bench_args()`

**Files:**
- Modify: `inference.py` (add functions after `_estimate_prompt_tokens`, around line 247)
- Modify: `tests/test_bench.py` (add tests)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_bench.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/islam/Desktop/inference && uv run pytest tests/test_bench.py::test_parse_bench_args_defaults tests/test_bench.py::test_generate_bench_prompt_approximate_token_count -v`
Expected: FAIL — functions not defined

- [ ] **Step 3: Implement the functions**

Add to `inference.py` after `_estimate_prompt_tokens` (after line 246):

```python
def parse_bench_args(args: str) -> tuple[int, int, list[int]]:
    """Parse /bench arguments: [input_tokens] [output_tokens] [concurrency_levels].
    Returns (input_tokens, output_tokens, concurrency_levels)."""
    _DEFAULT_LEVELS = [1, 2, 4, 8, 16, 32, 64, 128]
    parts = args.strip().split()
    if not parts:
        return 128, 128, _DEFAULT_LEVELS
    try:
        input_tok = int(parts[0])
    except ValueError:
        raise ValueError(f"invalid input_tokens: {parts[0]!r}")
    output_tok = int(parts[1]) if len(parts) > 1 else 128
    if len(parts) > 2:
        levels = [int(x) for x in parts[2].split(",")]
    else:
        levels = _DEFAULT_LEVELS
    return input_tok, output_tok, levels


def _generate_bench_prompt(target_tokens: int) -> str:
    """Generate a filler prompt of approximately `target_tokens` tokens."""
    filler = (
        "The quick brown fox jumps over the lazy dog. "
        "Pack my box with five dozen liquor jugs. "
    )
    # Estimate tokens per repetition
    filler_tok = _estimate_tokens(filler)
    if filler_tok == 0:
        filler_tok = 1
    reps = max(1, target_tokens // filler_tok)
    prompt = filler * reps
    # Trim to get closer to target
    actual = _estimate_tokens(prompt)
    while actual > target_tokens and len(prompt) > len(filler):
        prompt = prompt[: -len(filler)]
        actual = _estimate_tokens(prompt)
    return prompt.strip()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/islam/Desktop/inference && uv run pytest tests/test_bench.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add inference.py tests/test_bench.py
git commit -m "feat(bench): add parse_bench_args and _generate_bench_prompt"
```

---

### Task 3: Implement `BenchResult` dataclass and `_bench_single_request()`

**Files:**
- Modify: `inference.py` (add after `_generate_bench_prompt`)
- Modify: `tests/test_bench.py` (add tests)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_bench.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/islam/Desktop/inference && uv run pytest tests/test_bench.py::test_bench_single_request_returns_bench_result -v`
Expected: FAIL — BenchResult and _bench_single_request not defined

- [ ] **Step 3: Implement BenchResult and _bench_single_request**

Add to `inference.py` after `_generate_bench_prompt`:

```python
@dataclass
class BenchResult:
    ttft_seconds: float = 0.0
    generation_seconds: float = 0.0
    completion_tokens: int = 0
    error: str | None = None


def _bench_single_request(
    *, client: Any, model: str, prompt: str, max_tokens: int
) -> BenchResult:
    """Run a single streaming request for benchmarking. Returns timing metrics."""
    t0 = time.perf_counter()
    t_first: float | None = None
    server_completion: int | None = None
    pieces: list[str] = []

    try:
        stream = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            max_tokens=max_tokens,
            stream_options={"include_usage": True},
        )
        try:
            for chunk in stream:
                if getattr(chunk, "usage", None) is not None:
                    server_completion = getattr(chunk.usage, "completion_tokens", None)
                if not getattr(chunk, "choices", None):
                    continue
                delta = chunk.choices[0].delta
                content = getattr(delta, "content", None)
                if content:
                    if t_first is None:
                        t_first = time.perf_counter()
                    pieces.append(content)
        finally:
            close = getattr(stream, "close", None)
            if callable(close):
                close()
    except Exception as e:
        return BenchResult(error=f"{type(e).__name__}: {e}")

    t_end = time.perf_counter()
    if t_first is None:
        t_first = t_end
    ttft = t_first - t0
    gen_seconds = max(0.0, t_end - t_first)

    if server_completion is not None:
        completion_tokens = server_completion
    else:
        completion_tokens = _estimate_tokens("".join(pieces))

    return BenchResult(
        ttft_seconds=ttft,
        generation_seconds=gen_seconds,
        completion_tokens=completion_tokens,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/islam/Desktop/inference && uv run pytest tests/test_bench.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add inference.py tests/test_bench.py
git commit -m "feat(bench): add BenchResult dataclass and _bench_single_request"
```

---

### Task 4: Implement `_aggregate_level()` stats computation

**Files:**
- Modify: `inference.py` (add after `_bench_single_request`)
- Modify: `tests/test_bench.py` (add tests)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_bench.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/islam/Desktop/inference && uv run pytest tests/test_bench.py::test_aggregate_level_computes_stats -v`
Expected: FAIL — _aggregate_level not defined

- [ ] **Step 3: Implement LevelStats and _aggregate_level**

Add to `inference.py` after `_bench_single_request`:

```python
@dataclass
class LevelStats:
    concurrency: int
    ttft_mean: float = 0.0
    ttft_median: float = 0.0
    ttft_min: float = 0.0
    ttft_max: float = 0.0
    ttft_p95: float = 0.0
    ttft_p99: float = 0.0
    tps_mean: float = 0.0
    tps_median: float = 0.0
    tps_min: float = 0.0
    tps_max: float = 0.0
    tps_p95: float = 0.0
    tps_p99: float = 0.0
    e2e_mean: float = 0.0
    e2e_median: float = 0.0
    e2e_min: float = 0.0
    e2e_max: float = 0.0
    total_throughput: float = 0.0
    requests_per_sec: float = 0.0
    error_count: int = 0
    total_count: int = 0


def _percentile(sorted_vals: list[float], p: float) -> float:
    """Compute the p-th percentile (0-100) from a sorted list."""
    if not sorted_vals:
        return 0.0
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = int(k)
    c = f + 1
    if c >= len(sorted_vals):
        return sorted_vals[-1]
    return sorted_vals[f] + (k - f) * (sorted_vals[c] - sorted_vals[f])


def _aggregate_level(
    results: list[BenchResult], *, concurrency: int, wall_seconds: float
) -> LevelStats:
    """Compute aggregate statistics for one concurrency level."""
    import statistics as st

    ok = [r for r in results if r.error is None]
    total_count = len(results)
    error_count = total_count - len(ok)

    if not ok:
        return LevelStats(
            concurrency=concurrency,
            error_count=error_count,
            total_count=total_count,
        )

    ttfts = sorted(r.ttft_seconds for r in ok)
    tps_list = sorted(
        r.completion_tokens / r.generation_seconds
        if r.generation_seconds > 0 else 0.0
        for r in ok
    )
    e2e_list = sorted(r.ttft_seconds + r.generation_seconds for r in ok)
    total_tokens = sum(r.completion_tokens for r in ok)

    return LevelStats(
        concurrency=concurrency,
        ttft_mean=st.mean(ttfts),
        ttft_median=st.median(ttfts),
        ttft_min=ttfts[0],
        ttft_max=ttfts[-1],
        ttft_p95=_percentile(ttfts, 95),
        ttft_p99=_percentile(ttfts, 99),
        tps_mean=st.mean(tps_list),
        tps_median=st.median(tps_list),
        tps_min=tps_list[0],
        tps_max=tps_list[-1],
        tps_p95=_percentile(tps_list, 95),
        tps_p99=_percentile(tps_list, 99),
        e2e_mean=st.mean(e2e_list),
        e2e_median=st.median(e2e_list),
        e2e_min=e2e_list[0],
        e2e_max=e2e_list[-1],
        total_throughput=total_tokens / wall_seconds if wall_seconds > 0 else 0.0,
        requests_per_sec=len(ok) / wall_seconds if wall_seconds > 0 else 0.0,
        error_count=error_count,
        total_count=total_count,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/islam/Desktop/inference && uv run pytest tests/test_bench.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add inference.py tests/test_bench.py
git commit -m "feat(bench): add LevelStats dataclass and _aggregate_level"
```

---

### Task 5: Implement `_run_bench_level()` concurrency driver

**Files:**
- Modify: `inference.py` (add after `_aggregate_level`)
- Modify: `tests/test_bench.py` (add tests)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_bench.py`:

```python
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
    original_bench = inference._bench_single_request

    def flaky_bench(**kwargs):
        call_count["n"] += 1
        if call_count["n"] % 2 == 0:
            return inference.BenchResult(error="boom")
        return inference.BenchResult(
            ttft_seconds=0.1, generation_seconds=0.5, completion_tokens=20
        )

    monkeypatch.setattr(inference, "_bench_single_request", lambda **kw: flaky_bench(**kw))
    # Use a dummy client since _bench_single_request is monkeypatched
    stats = inference._run_bench_level(
        client=None, model="m", prompt="test", max_tokens=64, concurrency=4
    )
    assert stats.total_count == 4
    assert stats.error_count == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/islam/Desktop/inference && uv run pytest tests/test_bench.py::test_run_bench_level_fires_concurrent_requests -v`
Expected: FAIL — _run_bench_level not defined

- [ ] **Step 3: Implement _run_bench_level**

Add to `inference.py` after `_aggregate_level`. Add `from concurrent.futures import ThreadPoolExecutor` to the imports at the top (around line 17):

```python
from concurrent.futures import ThreadPoolExecutor
```

Then add the function:

```python
def _run_bench_level(
    *, client: Any, model: str, prompt: str, max_tokens: int, concurrency: int
) -> LevelStats:
    """Fire `concurrency` parallel requests and return aggregated stats."""
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [
            pool.submit(
                _bench_single_request,
                client=client,
                model=model,
                prompt=prompt,
                max_tokens=max_tokens,
            )
            for _ in range(concurrency)
        ]
        results = [f.result() for f in futures]
    wall_seconds = time.perf_counter() - t0
    return _aggregate_level(results, concurrency=concurrency, wall_seconds=wall_seconds)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/islam/Desktop/inference && uv run pytest tests/test_bench.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add inference.py tests/test_bench.py
git commit -m "feat(bench): add _run_bench_level concurrency driver"
```

---

### Task 6: Implement `_render_bench_table()` Rich summary table

**Files:**
- Modify: `inference.py` (add after `_run_bench_level`)
- Modify: `tests/test_bench.py` (add tests)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_bench.py`:

```python
def test_render_bench_table_outputs_rich_table(capsys):
    from rich.console import Console
    console = Console(file=io.StringIO(), force_terminal=True, width=120)
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/islam/Desktop/inference && uv run pytest tests/test_bench.py::test_render_bench_table_outputs_rich_table -v`
Expected: FAIL — _render_bench_table not defined

- [ ] **Step 3: Implement _render_bench_table**

Add to `inference.py` after `_run_bench_level`. Add `from rich.table import Table` to the imports at the top (around line 27):

```python
from rich.table import Table
```

Then add the function:

```python
def _render_bench_table(console: Console, all_stats: list[LevelStats]) -> None:
    """Print a Rich table summarizing benchmark results."""
    table = Table(title="Benchmark Results", border_style="cyan")
    table.add_column("Concurrency", justify="right", style="bold")
    table.add_column("TTFT mean", justify="right")
    table.add_column("TTFT p99", justify="right")
    table.add_column("Tok/s/user", justify="right")
    table.add_column("Total tok/s", justify="right")
    table.add_column("E2E lat mean", justify="right")
    table.add_column("Req/s", justify="right")
    table.add_column("Errors", justify="right")

    for s in all_stats:
        err_style = "red" if s.error_count > 0 else ""
        table.add_row(
            str(s.concurrency),
            f"{s.ttft_mean * 1000:.0f}ms",
            f"{s.ttft_p99 * 1000:.0f}ms",
            f"{s.tps_mean:.1f}",
            f"{s.total_throughput:.1f}",
            f"{s.e2e_mean * 1000:.0f}ms",
            f"{s.requests_per_sec:.1f}",
            f"[{err_style}]{s.error_count}/{s.total_count}[/{err_style}]"
            if err_style else f"{s.error_count}/{s.total_count}",
        )

    console.print()
    console.print(table)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/islam/Desktop/inference && uv run pytest tests/test_bench.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add inference.py tests/test_bench.py
git commit -m "feat(bench): add _render_bench_table Rich summary"
```

---

### Task 7: Implement `_render_bench_charts()` plotext charts

**Files:**
- Modify: `inference.py` (add after `_render_bench_table`)
- Modify: `tests/test_bench.py` (add tests)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_bench.py`:

```python
def test_render_bench_charts_does_not_raise():
    """Smoke test: rendering charts with valid data should not error."""
    stats = [
        inference.LevelStats(
            concurrency=1, ttft_mean=0.1, ttft_p99=0.15,
            tps_mean=50.0, tps_p99=55.0,
            total_throughput=50.0, e2e_mean=1.0, e2e_p99=1.1,
            requests_per_sec=1.0, error_count=0, total_count=1,
        ),
        inference.LevelStats(
            concurrency=4, ttft_mean=0.2, ttft_p99=0.35,
            tps_mean=45.0, tps_p99=48.0,
            total_throughput=170.0, e2e_mean=1.2, e2e_p99=1.5,
            requests_per_sec=3.5, error_count=0, total_count=4,
        ),
    ]
    # Should run without error
    inference._render_bench_charts(stats)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/islam/Desktop/inference && uv run pytest tests/test_bench.py::test_render_bench_charts_does_not_raise -v`
Expected: FAIL — _render_bench_charts not defined

- [ ] **Step 3: Implement _render_bench_charts**

Add to `inference.py` after `_render_bench_table`:

```python
def _render_bench_charts(all_stats: list[LevelStats]) -> None:
    """Render 4 plotext bar charts to the terminal."""
    import plotext as plt

    labels = [str(s.concurrency) for s in all_stats]

    # 1. TTFT vs Concurrency (mean and p99)
    plt.clear_figure()
    plt.theme("dark")
    plt.multiple_bar(
        labels,
        [
            [s.ttft_mean * 1000 for s in all_stats],
            [s.ttft_p99 * 1000 for s in all_stats],
        ],
        labels=["mean", "p99"],
    )
    plt.title("TTFT vs Concurrency (ms)")
    plt.xlabel("Concurrency")
    plt.ylabel("TTFT (ms)")
    plt.show()
    print()

    # 2. Token/s per user vs Concurrency
    plt.clear_figure()
    plt.theme("dark")
    plt.bar(labels, [s.tps_mean for s in all_stats])
    plt.title("Token/s per User vs Concurrency")
    plt.xlabel("Concurrency")
    plt.ylabel("Tok/s")
    plt.show()
    print()

    # 3. Total throughput vs Concurrency
    plt.clear_figure()
    plt.theme("dark")
    plt.bar(labels, [s.total_throughput for s in all_stats])
    plt.title("Total Throughput vs Concurrency (tok/s)")
    plt.xlabel("Concurrency")
    plt.ylabel("Tok/s")
    plt.show()
    print()

    # 4. E2E Latency vs Concurrency (mean and p99)
    plt.clear_figure()
    plt.theme("dark")
    plt.multiple_bar(
        labels,
        [
            [s.e2e_mean * 1000 for s in all_stats],
            [s.e2e_p99 * 1000 for s in all_stats],
        ],
        labels=["mean", "p99"],
    )
    plt.title("E2E Latency vs Concurrency (ms)")
    plt.xlabel("Concurrency")
    plt.ylabel("Latency (ms)")
    plt.show()
    print()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/islam/Desktop/inference && uv run pytest tests/test_bench.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add inference.py tests/test_bench.py
git commit -m "feat(bench): add _render_bench_charts plotext output"
```

---

### Task 8: Implement `handle_bench()` orchestrator and wire into REPL

**Files:**
- Modify: `inference.py` (add `handle_bench` after `_render_bench_charts`, update `main()`)
- Modify: `tests/test_bench.py` (add integration test)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_bench.py`:

```python
def test_handle_bench_runs_all_levels(monkeypatch):
    """Integration: handle_bench calls _run_bench_level for each concurrency level."""
    from rich.console import Console
    console = Console(file=io.StringIO(), force_terminal=True, width=120)
    levels_seen = []

    def fake_run_level(*, client, model, prompt, max_tokens, concurrency):
        levels_seen.append(concurrency)
        return inference.LevelStats(
            concurrency=concurrency,
            ttft_mean=0.1, ttft_p99=0.15, ttft_median=0.1,
            ttft_min=0.05, ttft_max=0.2, ttft_p95=0.14,
            tps_mean=50.0, tps_p99=55.0, tps_median=50.0,
            tps_min=40.0, tps_max=60.0, tps_p95=54.0,
            e2e_mean=1.0, e2e_median=1.0, e2e_min=0.8, e2e_max=1.2,
            e2e_p99=1.1,
            total_throughput=50.0 * concurrency,
            requests_per_sec=float(concurrency),
            error_count=0, total_count=concurrency,
        )

    monkeypatch.setattr(inference, "_run_bench_level", fake_run_level)
    monkeypatch.setattr(inference, "_render_bench_charts", lambda stats: None)

    inference.handle_bench(
        client=None, model="m", args="64 64 1,4,8", console=console
    )
    assert levels_seen == [1, 4, 8]


def test_main_bench_command_integration(
    monkeypatch, tmp_path, capsys, fake_client_factory, make_chunk_fn, fake_usage_cls
):
    """Integration: /bench in the REPL dispatches handle_bench."""
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

    bench_called = {"called": False}

    def fake_handle_bench(*, client, model, args, console):
        bench_called["called"] = True

    monkeypatch.setattr(inference, "handle_bench", fake_handle_bench)

    inputs = iter(["/bench 64 64 1,2", "/exit"])
    monkeypatch.setattr("builtins.input", lambda *a, **kw: next(inputs))

    inference.main()
    assert bench_called["called"] is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/islam/Desktop/inference && uv run pytest tests/test_bench.py::test_handle_bench_runs_all_levels -v`
Expected: FAIL — handle_bench not defined

- [ ] **Step 3: Implement handle_bench**

Add to `inference.py` after `_render_bench_charts`:

```python
def handle_bench(
    *, client: Any, model: str, args: str, console: Console
) -> None:
    """Run the benchmark across concurrency levels and display results."""
    try:
        input_tokens, output_tokens, levels = parse_bench_args(args)
    except ValueError as e:
        _error(console, f"invalid /bench arguments: {e}")
        return

    prompt = _generate_bench_prompt(input_tokens)
    all_stats: list[LevelStats] = []

    with console.status("") as status:
        for i, conc in enumerate(levels, 1):
            status.update(
                f"Benchmarking: {conc} concurrent ({i}/{len(levels)} levels)"
            )
            stats = _run_bench_level(
                client=client,
                model=model,
                prompt=prompt,
                max_tokens=output_tokens,
                concurrency=conc,
            )
            all_stats.append(stats)

    _render_bench_table(console, all_stats)
    _render_bench_charts(all_stats)
```

- [ ] **Step 4: Wire /bench into the REPL main loop**

In `inference.py`, in the `main()` function, add the `/bench` handler inside the command dispatch block. After the `/remove` handler (around line 553) and before the `# unknown` comment (line 554), add:

```python
                if name == "bench":
                    handle_bench(
                        client=client,
                        model=current["model"],
                        args=args,
                        console=_console,
                    )
                    continue
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd /Users/islam/Desktop/inference && uv run pytest tests/test_bench.py -v`
Expected: PASS

- [ ] **Step 6: Run full test suite**

Run: `cd /Users/islam/Desktop/inference && uv run pytest -v`
Expected: All tests pass (existing + new)

- [ ] **Step 7: Commit**

```bash
git add inference.py tests/test_bench.py
git commit -m "feat(bench): add handle_bench orchestrator and wire into REPL"
```

---

### Task 9: End-to-end manual verification

- [ ] **Step 1: Verify /bench help line shows**

Run: `cd /Users/islam/Desktop/inference && uv run python -c "import inference; inference._help_line(inference._console)"`
Expected: Output includes `/bench`

- [ ] **Step 2: Run the full test suite one final time**

Run: `cd /Users/islam/Desktop/inference && uv run pytest -v`
Expected: All tests pass

- [ ] **Step 3: Final commit if any cleanup needed**

Only commit if there were fixups. Otherwise skip.
