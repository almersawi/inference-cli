# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "openai>=1.40",
#     "pyyaml>=6.0",
#     "questionary>=2.0",
#     "rich>=13.7",
#     "tiktoken>=0.7",
# ]
# ///
"""CLI for chatting with locally deployed OpenAI-compatible LLMs."""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, TextIO

import yaml


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


def write_config(path: str | os.PathLike[str], models: list[dict[str, Any]]) -> None:
    """Overwrite models.yaml with the given list of model dicts."""
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
    p = Path(path)
    if p.exists() and p.read_text().strip():
        try:
            data = yaml.safe_load(p.read_text()) or {}
        except yaml.YAMLError as e:
            raise ConfigError(f"invalid YAML in {p}: {e}") from e
        if not isinstance(data, dict):
            data = {}
        models = data.get("models") or []
        if not isinstance(models, list):
            raise ConfigError(f"{p}: 'models' must be a list")
    else:
        models = []
    models.append(new_model)
    write_config(p, models)


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


def chat_turn(
    *,
    client: Any,
    model: str,
    history: list[dict[str, Any]],
    out: TextIO,
) -> tuple[str, Metrics]:
    """Run one streaming chat completion. Writes content to `out` as it arrives.
    Returns the assembled assistant text and a populated Metrics."""
    t0 = time.perf_counter()
    t_first: float | None = None
    pieces: list[str] = []
    server_prompt: int | None = None
    server_completion: int | None = None

    stream = client.chat.completions.create(
        model=model,
        messages=history,
        stream=True,
        stream_options={"include_usage": True},
    )
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


KNOWN_COMMANDS = {"clear", "exit", "model", "system", "add", "remove"}


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
    remaining list. Raises ConfigError if it would leave zero models or the
    name is unknown."""
    models = load_config(config_path)
    if not any(m["model"] == model_name for m in models):
        raise ConfigError(f"no model named '{model_name}' in config")
    if len(models) <= 1:
        raise ConfigError("refusing to remove the last model in config")
    remaining = [m for m in models if m["model"] != model_name]
    write_config(config_path, remaining)
    return remaining


def main() -> None:
    print("inference.py: bootstrap")


if __name__ == "__main__":
    main()
