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
from pathlib import Path
from typing import Any

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


def main() -> None:
    print("inference.py: bootstrap")


if __name__ == "__main__":
    main()
