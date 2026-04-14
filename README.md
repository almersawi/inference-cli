# Inference CLI

A frictionless CLI for chatting with locally deployed OpenAI-compatible LLMs. Single file, zero config, real-time performance metrics.

Works with any OpenAI-compatible endpoint: **vLLM**, **llama.cpp**, **Ollama**, **LM Studio**, **TGI**, and more.

## Features

- **Interactive chat** with streaming Markdown rendering
- **Multi-model management** via YAML config with arrow-key picker
- **Real-time metrics** — TTFT, token/s, token counts on every response
- **Benchmark mode** — stress-test models across concurrency levels with terminal charts
- **Thinking mode toggle** — disable reasoning for models that support it (e.g., Qwen3)
- **Slash commands with autocomplete** — type `/` to see all commands in a dropdown
- **Environment variable secrets** — use `${ENV_VAR}` in `api_key` fields

## Requirements

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

## Installation

### Quick start (with uv)

```bash
# Clone and run directly — uv handles all dependencies automatically
git clone https://github.com/almersawi/inference-cli.git
cd inference-cli
uv run inference.py
```

### Install as a command

```bash
# Symlinks to ~/.local/bin/inference
./install.sh
```

After installation, run from anywhere:

```bash
inference
```

> If `~/.local/bin` is not on your PATH, add `export PATH="$HOME/.local/bin:$PATH"` to your `~/.zshrc` or `~/.bashrc`.

### Manual (with pip)

```bash
pip install openai pyyaml questionary rich tiktoken plotext
python inference.py
```

## Configuration

On first launch, you'll be prompted to add a model. Models are stored in `models.yaml` next to the script:

```yaml
models:
  - model: "meta-llama/Llama-3-8B-Instruct"
    base_url: "http://localhost:8000/v1"
    api_key: "sk-local"

  - model: "qwen2.5-coder-7b"
    base_url: "http://localhost:11434/v1"
    api_key: "ollama"

  # Environment variable for api_key:
  - model: "my-model"
    base_url: "http://localhost:9000/v1"
    api_key: "${MY_API_KEY}"
```

Override the config path with:

```bash
export INFERENCE_MODELS_CONFIG=/path/to/models.yaml
```

## Usage

### Chat

```
$ inference
? Select a model:
> meta-llama/Llama-3-8B-Instruct
  qwen2.5-coder-7b
  + add new model

╭─ inference ─────────────────────────╮
│ Chatting with meta-llama/Llama-3-8B │
│ thinking: disabled                  │
╰─────────────────────────────────────╯

You ▸ What is the capital of France?
Assistant ▸ The capital of France is **Paris**.
⏱ TTFT: 45ms · 82.3 tok/s · in: 12 · out: 8
```

### Commands

Type `/` to see all available commands in a real-time autocomplete dropdown, or type the full command directly:

| Command | Description |
|---------|-------------|
| `/clear` | Clear conversation history (keeps system prompt) |
| `/model` | Switch to a different model |
| `/system [prompt]` | Set or replace the system prompt |
| `/add` | Add a new model to config |
| `/remove` | Remove a model from config |
| `/bench [in] [out] [levels]` | Benchmark the current model |
| `/exit` | Exit (or press Ctrl-D) |

### Benchmark

Stress-test your model across concurrency levels:

```
You ▸ /bench
```

With custom parameters:

```
You ▸ /bench 256 512              # 256 input tokens, 512 output tokens
You ▸ /bench 128 128 1,4,16,64   # Custom concurrency levels
```

**Default:** 128 input tokens, 128 output tokens, concurrency levels 1, 2, 4, 6, 8, 10, ..., 128 (stepping by 2).

Outputs a summary table and 4 bar charts showing:
- **TTFT** (time to first token) — mean and p99
- **Token/s per user** — mean throughput per request
- **Total throughput** — aggregate tok/s across all concurrent requests
- **E2E latency** — mean and max end-to-end response time

## License

MIT
