Skills Lab — Docker Guide

## Quick Start

```bash
# Build and run dashboard
docker compose up --build

# Open dashboard
# http://localhost:7788
```

## MCP Server (for Claude / Cursor / Copilot)

```bash
docker compose run mcp
```

## Download Model (pre-cache during build)

Model is downloaded during `docker build`. If skipped, it downloads on first search.

## Volumes

- `./workspace` → Skills data (SKILL.md files + SQLite DB)
- Model cache → Inside container (`~/.cache/huggingface/`)

## Rebuild

```bash
docker compose build --no-cache
```
