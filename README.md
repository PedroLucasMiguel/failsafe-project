# Failsafe

Multi-agent system for automated codebase discovery and knowledge base generation.

## Quick Start

```bash
uv sync
uv run failsafe discover ./path/to/codebase --provider openai --api-key YOUR_KEY
```

## Supported LLM Providers

| Provider  | Flag                   | Default Model             |
| --------- | ---------------------- | ------------------------- |
| OpenAI    | `--provider openai`    | `gpt-4o`                  |
| Anthropic | `--provider anthropic` | `claude-3-5-haiku-latest` |
| Google    | `--provider google`    | `gemini-3-flash-preview`  |
