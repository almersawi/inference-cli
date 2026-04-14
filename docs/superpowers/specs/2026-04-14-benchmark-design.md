# Benchmark Feature Design

## Overview

A `/bench` slash command that stress-tests the currently selected model across configurable concurrency levels, measures latency and throughput metrics, and renders results as a Rich table and plotext charts in the terminal.

## Command Interface

**Syntax:** `/bench [input_tokens] [output_tokens] [concurrency_levels]`

| Argument | Default | Description |
|----------|---------|-------------|
| `input_tokens` | `128` | Approximate input prompt size in tokens |
| `output_tokens` | `128` | `max_tokens` cap for output |
| `concurrency_levels` | `1,2,4,8,16,32,64,128` | Comma-separated list of concurrency levels to test |

**Examples:**
- `/bench` — defaults: 128 in, 128 out, full ramp
- `/bench 256 512` — custom token sizes, full ramp
- `/bench 128 128 1,4,16,64` — custom concurrency levels
- `/bench 256 512 1,2,4,8` — quick benchmark with lower concurrency

## Execution Engine

### Prompt Generation

Generate a padded prompt to hit the desired input token count:
- Use a repeated filler phrase (e.g., "The quick brown fox jumps over the lazy dog.")
- Measure with tiktoken (`cl100k_base`) and trim/pad to reach the target
- Wrap in a single user message

### Per-Request Worker

A stripped-down function (not `chat_turn`) focused purely on timing:
- Creates a streaming completion with `max_tokens` set
- Measures TTFT (time to first content chunk)
- Measures generation time (first chunk to last chunk)
- Counts completion tokens (from server usage or estimation)
- Catches exceptions and records them as errors
- Returns a `BenchResult` dataclass

### Concurrency Driver

For each concurrency level in order:
1. Create a `ThreadPoolExecutor(max_workers=N)`
2. Submit all N requests simultaneously
3. Measure wall-clock time for the entire batch
4. Collect all `BenchResult` instances
5. Aggregate into `LevelStats`
6. Show progress via Rich status spinner (e.g., "Benchmarking: 8 concurrent (4/8 levels)")

Error handling:
- Individual request failures are recorded, not fatal
- Error rate reported per concurrency level
- If ALL requests at a level fail, note it and continue

## Data Structures

### BenchResult (per request)

```
BenchResult:
    ttft_seconds: float
    generation_seconds: float
    completion_tokens: int
    error: str | None
```

### LevelStats (per concurrency level)

```
LevelStats:
    concurrency: int
    ttft_mean, ttft_median, ttft_min, ttft_max, ttft_p95, ttft_p99: float
    tps_mean, tps_median, tps_min, tps_max, tps_p95, tps_p99: float
    e2e_mean, e2e_median, e2e_min, e2e_max: float
    total_throughput: float       # aggregate tok/s across all requests
    requests_per_sec: float       # completed requests / wall-clock time
    error_count: int
    total_count: int
```

## Metrics

### Per-request measurements

| Metric | Calculation |
|--------|-------------|
| TTFT | `t_first_chunk - t_start` |
| Token/s per user | `completion_tokens / generation_seconds` |
| E2E latency | `t_end - t_start` |

### Aggregation per concurrency level

| Metric | Stats |
|--------|-------|
| TTFT | mean, median, min, max, p95, p99 |
| Token/s per user | mean, median, min, max, p95, p99 |
| E2E latency | mean, median, min, max |
| Total throughput | sum(all completion_tokens) / wall_clock_time |
| Requests/sec | completed_requests / wall_clock_time |
| Error rate | error_count / total_count |

Percentiles computed with sorted-index approach using the `statistics` module. No numpy dependency.

## Terminal Output

### 1. Summary Table (Rich)

One row per concurrency level:

| Concurrency | TTFT mean | TTFT p99 | Tok/s/user mean | Total tok/s | E2E lat mean | Req/s | Errors |
|-------------|-----------|----------|-----------------|-------------|--------------|-------|--------|

### 2. Charts (plotext)

Four bar charts rendered sequentially:

1. **TTFT vs Concurrency** — mean and p99 bars side by side
2. **Token/s per user vs Concurrency** — mean throughput per user
3. **Total throughput vs Concurrency** — aggregate tok/s scaling
4. **E2E Latency vs Concurrency** — mean and p99 bars side by side

Each chart: labeled axes, title, concurrency levels as x-axis categories.

Table prints first (raw numbers), then charts below (visual story).

## Dependencies

New dependency: `plotext` — added to the PEP 723 script metadata in `inference.py`.

## Integration Points

- Registered in `KNOWN_COMMANDS` alongside existing commands
- Handled in `_handle_command()` with the same pattern as `/clear`, `/system`, etc.
- Uses the existing `client` and `model` from the REPL session
- Progress shown via `console.status()` (Rich, already available)
- No changes to existing code paths — purely additive
