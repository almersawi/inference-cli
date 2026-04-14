#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "openai>=1.40",
#     "pyyaml>=6.0",
#     "questionary>=2.0",
#     "rich>=13.7",
#     "tiktoken>=0.7",
#     "plotext>=5.2",
# ]
# ///
"""CLI for chatting with locally deployed OpenAI-compatible LLMs."""

from __future__ import annotations

import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, TextIO

import yaml
from openai import OpenAI
from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


_console = Console()

import questionary as _questionary

_picker_style = _questionary.Style([
    ("qmark", "fg:#00afff bold"),
    ("question", "bold"),
    ("pointer", "fg:#00afff bold"),
    ("highlighted", "fg:#00afff bold"),
    ("selected", "fg:#00afff"),
])


class _LiveMarkdownOut:
    """TextIO-duck-typed adapter that renders accumulated writes as
    Markdown via rich.live.Live. Starts the Live region lazily on the
    first non-empty write so an empty response renders nothing."""

    def __init__(self, console: Console) -> None:
        self._console = console
        self._buffer: list[str] = []
        self._live: Live | None = None

    def __enter__(self) -> "_LiveMarkdownOut":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._live is not None:
            self._live.stop()
            self._live = None

    def write(self, s: str) -> int:
        if not s:
            return 0
        self._buffer.append(s)
        renderable = Markdown("".join(self._buffer))
        if self._live is None:
            self._live = Live(
                renderable,
                console=self._console,
                refresh_per_second=10,
                vertical_overflow="visible",
            )
            self._live.start()
        else:
            self._live.update(renderable)
        return len(s)

    def flush(self) -> None:
        pass


def _welcome_banner(console: Console, model: str, *, disable_thinking: bool) -> None:
    """Render the 'Chatting with <model>' welcome panel. Shown at every
    model-switch site. When thinking is disabled, adds a yellow status line."""
    body = Text()
    body.append("Chatting with ")
    body.append(model, style="bold cyan")
    if disable_thinking:
        body.append("\nthinking: disabled", style="yellow")
    console.print(Panel(body, title="inference", border_style="cyan", expand=False))


def _success(console: Console, message: str) -> None:
    console.print(f"✓ {message}", style="green")


def _error(console: Console, message: str) -> None:
    text = Text(f"[error] {message}", style="red")
    console.print(text)


def _cancelled(console: Console) -> None:
    text = Text("[cancelled]", style="yellow")
    console.print(text)


def _interrupted(console: Console) -> None:
    text = Text("[interrupted]", style="yellow")
    console.print(text)


def _info(console: Console, message: str) -> None:
    console.print(message, style="cyan")


def _help_line(console: Console) -> None:
    console.print(
        "Commands: /clear /model /system /add /remove /bench /exit", style="dim"
    )


class ConfigError(Exception):
    """Raised for problems loading or validating models.yaml."""


_ENV_VAR_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
_REQUIRED_FIELDS = ("model", "base_url", "api_key")


def _expand_env(value: str) -> str:
    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in os.environ:
            raise ConfigError(f"environment variable ${{{name}}} is not set")
        return os.environ[name]
    return _ENV_VAR_RE.sub(replace, value)


def load_config(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    try:
        data = yaml.safe_load(p.read_text()) or {}
    except yaml.YAMLError as e:
        raise ConfigError(f"invalid YAML in {p}: {e}") from e
    if not data:
        return []
    if not isinstance(data, dict) or "models" not in data:
        raise ConfigError(f"{p}: top-level 'models' key required")
    models = data["models"] or []
    if not isinstance(models, list):
        raise ConfigError(f"{p}: 'models' must be a list")
    for i, m in enumerate(models):
        if not isinstance(m, dict):
            raise ConfigError(f"{p}: models[{i}] must be a mapping")
        for field in _REQUIRED_FIELDS:
            if field not in m or m[field] in (None, ""):
                raise ConfigError(f"{p}: models[{i}] missing required field '{field}'")
        m["api_key"] = _expand_env(str(m["api_key"]))
    return models


def _read_raw_models(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    """Load models.yaml WITHOUT expanding ${ENV_VAR} references in api_key.
    Returns [] for missing/empty files. Used by writers that round-trip the
    file so env-var references are preserved on disk."""
    p = Path(path)
    if not p.exists():
        return []
    text = p.read_text()
    if not text.strip():
        return []
    try:
        data = yaml.safe_load(text) or {}
    except yaml.YAMLError as e:
        raise ConfigError(f"invalid YAML in {p}: {e}") from e
    if not isinstance(data, dict):
        return []
    models = data.get("models") or []
    if not isinstance(models, list):
        raise ConfigError(f"{p}: 'models' must be a list")
    return models


def write_config(path: str | os.PathLike[str], models: list[dict[str, Any]]) -> None:
    """Overwrite models.yaml with the given list of model dicts. Each entry
    is projected to the three required fields in canonical order."""
    p = Path(path)
    payload = [
        {"model": m["model"], "base_url": m["base_url"], "api_key": m["api_key"]}
        for m in models
    ]
    p.write_text(yaml.safe_dump({"models": payload}, sort_keys=False))


def save_config(path: str | os.PathLike[str], new_model: dict[str, Any]) -> None:
    """Append `new_model` to models.yaml. Validates required fields."""
    for field in _REQUIRED_FIELDS:
        if field not in new_model or new_model[field] in (None, ""):
            raise ConfigError(f"new model missing required field '{field}'")
    models = _read_raw_models(path)
    models.append(new_model)
    write_config(path, models)


@dataclass
class Metrics:
    ttft_seconds: float
    generation_seconds: float
    prompt_tokens: int
    completion_tokens: int
    prompt_tokens_estimated: bool
    completion_tokens_estimated: bool


def format_metrics(m: Metrics) -> str:
    ttft_ms = int(round(m.ttft_seconds * 1000))
    if m.generation_seconds > 0:
        tps = m.completion_tokens / m.generation_seconds
    else:
        tps = 0.0
    in_str = f"{m.prompt_tokens}{'*' if m.prompt_tokens_estimated else ''}"
    out_str = f"{m.completion_tokens}{'*' if m.completion_tokens_estimated else ''}"
    return f"⏱ TTFT: {ttft_ms}ms · {tps:.1f} tok/s · in: {in_str} · out: {out_str}"


def _estimate_tokens(text: str) -> int:
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        # Crude fallback: ~4 chars per token
        return max(1, len(text) // 4)


def _estimate_prompt_tokens(history: list[dict[str, Any]]) -> int:
    total = 0
    for msg in history:
        total += _estimate_tokens(str(msg.get("content", "")))
        total += 4  # rough per-message overhead
    return total


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
    filler_tok = _estimate_tokens(filler)
    if filler_tok == 0:
        filler_tok = 1
    reps = max(1, target_tokens // filler_tok)
    prompt = filler * reps
    actual = _estimate_tokens(prompt)
    while actual > target_tokens and len(prompt) > len(filler):
        prompt = prompt[: -len(filler)]
        actual = _estimate_tokens(prompt)
    return prompt.strip()


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

    # 4. E2E Latency vs Concurrency (mean and max)
    plt.clear_figure()
    plt.theme("dark")
    plt.multiple_bar(
        labels,
        [
            [s.e2e_mean * 1000 for s in all_stats],
            [s.e2e_max * 1000 for s in all_stats],
        ],
        labels=["mean", "max"],
    )
    plt.title("E2E Latency vs Concurrency (ms)")
    plt.xlabel("Concurrency")
    plt.ylabel("Latency (ms)")
    plt.show()
    print()


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


def chat_turn(
    *,
    client: Any,
    model: str,
    history: list[dict[str, Any]],
    out: TextIO,
    disable_thinking: bool = False,
) -> tuple[str, Metrics]:
    """Run one streaming chat completion. Writes content to `out` as it arrives.
    Returns the assembled assistant text and a populated Metrics."""
    t0 = time.perf_counter()
    t_first: float | None = None
    pieces: list[str] = []
    server_prompt: int | None = None
    server_completion: int | None = None

    create_kwargs: dict[str, Any] = {
        "model": model,
        "messages": history,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if disable_thinking:
        create_kwargs["extra_body"] = {
            "chat_template_kwargs": {"enable_thinking": False}
        }
    stream = client.chat.completions.create(**create_kwargs)
    try:
        for chunk in stream:
            if getattr(chunk, "usage", None) is not None:
                server_prompt = getattr(chunk.usage, "prompt_tokens", None)
                server_completion = getattr(chunk.usage, "completion_tokens", None)
            if not getattr(chunk, "choices", None):
                continue
            delta = chunk.choices[0].delta
            content = getattr(delta, "content", None)
            if content:
                if t_first is None:
                    t_first = time.perf_counter()
                pieces.append(content)
                out.write(content)
                out.flush()
    finally:
        close = getattr(stream, "close", None)
        if callable(close):
            close()

    t_end = time.perf_counter()
    text = "".join(pieces)

    if t_first is None:
        t_first = t_end
    ttft = t_first - t0
    gen_seconds = max(0.0, t_end - t_first)

    if server_prompt is not None:
        prompt_tokens = server_prompt
        prompt_estimated = False
    else:
        prompt_tokens = _estimate_prompt_tokens(history)
        prompt_estimated = True

    if server_completion is not None:
        completion_tokens = server_completion
        completion_estimated = False
    else:
        completion_tokens = _estimate_tokens(text)
        completion_estimated = True

    metrics = Metrics(
        ttft_seconds=ttft,
        generation_seconds=gen_seconds,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        prompt_tokens_estimated=prompt_estimated,
        completion_tokens_estimated=completion_estimated,
    )
    return text, metrics


KNOWN_COMMANDS = {"clear", "exit", "model", "system", "add", "remove", "bench"}


def parse_command(line: str) -> tuple[str, str] | None:
    """Returns (command_name, args) for known commands,
    ('__unknown__', raw_word) for unknown /commands,
    or None for non-command input."""
    if not line or not line.startswith("/"):
        return None
    body = line[1:].strip()
    if not body:
        return ("__unknown__", "")
    parts = body.split(maxsplit=1)
    name = parts[0].lower()
    args = parts[1] if len(parts) > 1 else ""
    if name in KNOWN_COMMANDS:
        return (name, args)
    return ("__unknown__", parts[0])


def handle_clear(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [m for m in history if m.get("role") == "system"]


def handle_system(history: list[dict[str, Any]], content: str) -> list[dict[str, Any]]:
    new_msg = {"role": "system", "content": content}
    if history and history[0].get("role") == "system":
        return [new_msg, *history[1:]]
    return [new_msg, *history]


def handle_add(
    *,
    prompt: Callable[[str], str],
    config_path: str | os.PathLike[str],
) -> dict[str, Any]:
    """Collect the three required fields via `prompt(field_name)` and append to yaml.
    Returns the new model dict."""
    new_model = {
        "model": prompt("model").strip(),
        "base_url": prompt("base_url").strip(),
        "api_key": prompt("api_key").strip(),
    }
    save_config(config_path, new_model)
    return new_model


def handle_remove(
    *,
    model_name: str,
    config_path: str | os.PathLike[str],
) -> list[dict[str, Any]]:
    """Remove the entry whose `model` field matches `model_name`. Returns the
    remaining list. Raises ConfigError if the name is unknown or removing
    would leave zero models. Preserves ${ENV_VAR} references in surviving
    entries by using `_read_raw_models` (not `load_config`)."""
    models = _read_raw_models(config_path)
    if not any(m.get("model") == model_name for m in models):
        raise ConfigError(f"no model named '{model_name}' in config")
    if len(models) <= 1:
        raise ConfigError("refusing to remove the last model in config")
    remaining = [m for m in models if m.get("model") != model_name]
    write_config(config_path, remaining)
    return remaining


ADD_SENTINEL = "__add__"
_ADD_LABEL = "+ add new model"


def pick_model(
    models: list[dict[str, Any]],
    *,
    _select: Callable[[str, list[str]], str] | None = None,
) -> dict[str, Any] | str:
    """Show the arrow-key picker. Returns the selected model dict, or
    ADD_SENTINEL ('__add__') if the user chose to add a new model."""
    if _select is None:
        import questionary
        def _select(message: str, choices: list[str]) -> str:
            return questionary.select(
                message, choices=choices, style=_picker_style
            ).ask()

    labels = [m["model"] for m in models] + [_ADD_LABEL]
    choice = _select("Select a model:", labels)
    if choice is None or choice == _ADD_LABEL:
        return ADD_SENTINEL
    for m in models:
        if m["model"] == choice:
            return m
    return ADD_SENTINEL


def make_client(model_entry: dict[str, Any]) -> Any:
    return OpenAI(base_url=model_entry["base_url"], api_key=model_entry["api_key"])


def _config_path() -> Path:
    override = os.environ.get("INFERENCE_MODELS_CONFIG")
    if override:
        return Path(override)
    return Path(__file__).parent / "models.yaml"


def _interactive_prompt(field: str) -> str:
    import questionary
    answer = questionary.text(f"{field}:").ask()
    if answer is None:
        raise KeyboardInterrupt
    return answer


def _ask_disable_thinking() -> bool:
    """Prompt 'Disable thinking for this session? (y/N)'. Returns True only for
    y/yes (case-insensitive). Empty input, n, no, or anything else returns False."""
    answer = _interactive_prompt(
        "Disable thinking for this session? (y/N)"
    ).strip().lower()
    return answer in ("y", "yes")


def _select_or_bootstrap(config_path: Path) -> dict[str, Any]:
    """Load config, run picker. If empty or user picks +add, run /add and re-pick."""
    while True:
        try:
            models = load_config(config_path)
        except ConfigError as e:
            print(f"config error: {e}", file=sys.stderr)
            sys.exit(1)
        if not models:
            print("No models configured. Let's add one.")
            handle_add(prompt=_interactive_prompt, config_path=config_path)
            continue
        choice = pick_model(models)
        if choice == ADD_SENTINEL:
            handle_add(prompt=_interactive_prompt, config_path=config_path)
            continue
        return choice  # dict


def main() -> None:
    config_path = _config_path()
    current = _select_or_bootstrap(config_path)
    client = make_client(current)
    disable_thinking = _ask_disable_thinking()
    history: list[dict[str, Any]] = []

    _welcome_banner(_console, current["model"], disable_thinking=disable_thinking)
    while True:
        try:
            line = input("You ▸ ")
        except (EOFError, KeyboardInterrupt):
            print()
            return

        cmd = parse_command(line)
        if cmd is not None:
            try:
                name, args = cmd
                if name == "exit":
                    return
                if name == "clear":
                    history = handle_clear(history)
                    _success(_console, "History cleared.")
                    continue
                if name == "system":
                    content = args or _interactive_prompt("system prompt").strip()
                    if content:
                        history = handle_system(history, content)
                        _success(_console, "System prompt set.")
                    continue
                if name == "model":
                    current = _select_or_bootstrap(config_path)
                    client = make_client(current)
                    disable_thinking = _ask_disable_thinking()
                    history = []
                    _welcome_banner(_console, current["model"], disable_thinking=disable_thinking)
                    _info(_console, "History cleared.")
                    continue
                if name == "add":
                    new = handle_add(prompt=_interactive_prompt, config_path=config_path)
                    _success(_console, f"Added {new['model']} to {config_path}")
                    ans = _interactive_prompt("switch to it now? (y/N)").strip().lower()
                    if ans in ("y", "yes"):
                        current = {**new, "api_key": _expand_env(new["api_key"])}
                        client = make_client(current)
                        disable_thinking = _ask_disable_thinking()
                        history = []
                        _welcome_banner(_console, current["model"], disable_thinking=disable_thinking)
                        _info(_console, "History cleared.")
                    continue
                if name == "remove":
                    try:
                        models = load_config(config_path)
                    except ConfigError as e:
                        _error(_console, str(e))
                        continue
                    if len(models) <= 1:
                        _error(_console, "refusing to remove the last model in config")
                        continue
                    choice = pick_model(models)
                    if choice == ADD_SENTINEL:
                        continue  # user picked '+ add new model' — treat as cancel
                    target_name = choice["model"]
                    confirm = _interactive_prompt(
                        f"remove '{target_name}' from {config_path}? (y/N)"
                    ).strip().lower()
                    if confirm not in ("y", "yes"):
                        _cancelled(_console)
                        continue
                    try:
                        handle_remove(model_name=target_name, config_path=config_path)
                    except ConfigError as e:
                        _error(_console, str(e))
                        continue
                    _success(_console, f"Removed {target_name}.")
                    if target_name == current["model"]:
                        _info(_console, "That was the active model. Returning to picker.")
                        current = _select_or_bootstrap(config_path)
                        client = make_client(current)
                        disable_thinking = _ask_disable_thinking()
                        history = []
                        _welcome_banner(_console, current["model"], disable_thinking=disable_thinking)
                    continue
                if name == "bench":
                    handle_bench(
                        client=client,
                        model=current["model"],
                        args=args,
                        console=_console,
                    )
                    continue
                # unknown
                _help_line(_console)
            except KeyboardInterrupt:
                print()  # newline to clear the half-typed prompt
                _cancelled(_console)
            continue

        if not line.strip():
            continue

        history.append({"role": "user", "content": line})
        _console.print(Text("Assistant ▸", style="bold green"))
        try:
            with _LiveMarkdownOut(_console) as live_out:
                text, metrics = chat_turn(
                    client=client,
                    model=current["model"],
                    history=history,
                    out=live_out,
                    disable_thinking=disable_thinking,
                )
        except KeyboardInterrupt:
            print()
            _interrupted(_console)
            continue
        except Exception as e:  # API errors, connection errors
            history.pop()  # roll back the user turn
            print()  # ensure we start on a fresh line after partial stream
            # Surface the root cause — openai.APIConnectionError's default
            # str() is just "Connection error." which hides the underlying
            # httpx exception (DNS, timeout, refused, TLS, etc.).
            parts = [f"{type(e).__name__}: {e}"]
            cause = e.__cause__ or e.__context__
            while cause is not None:
                parts.append(f"  caused by {type(cause).__name__}: {cause}")
                cause = cause.__cause__ or cause.__context__
            _error(_console, "\n".join(parts))
            continue
        _console.print(format_metrics(metrics), style="dim")
        history.append({"role": "assistant", "content": text})


if __name__ == "__main__":
    main()
