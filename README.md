# Skills Lab — MCP Server for AI Coding Agents

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Version: 2.0.0](https://img.shields.io/badge/version-2.0.0-green.svg)
![Status: Beta](https://img.shields.io/badge/status-4--Beta-orange.svg)
![MCP](https://img.shields.io/badge/MCP-Server-purple.svg)

> **Save, search, and evolve technical knowledge as Skills** — a structured knowledge base that plugs directly into AI coding agents (Claude, GitHub Copilot, Cursor) via the [Model Context Protocol](https://modelcontextprotocol.io/).

---

## Overview

Skills Lab is a knowledge-management system designed from the ground up for AI coding agents. Every piece of technical knowledge — a bug fix, a workflow, an architectural decision, a project convention — is stored as a **Skill** following the open **SKILL.md standard**. Skills are versioned, searchable, and evolvable over time, so the AI agent always has access to the latest, validated solutions for the problems it encounters in your codebase.

The project addresses a fundamental problem: AI coding agents are powerful, but they lack persistent memory. When you solve a tricky CORS configuration issue today, the agent forgets it tomorrow. Skills Lab closes that loop by providing a persistent, structured knowledge layer that the agent can read from and write to. When the agent encounters a problem it has seen before (or something similar), it searches the Skills database, retrieves the relevant solution, and applies it — saving time and reducing errors across every session.

At its core, Skills Lab is an **MCP (Model Context Protocol) server** that exposes three tools to the agent: `search_skills` for finding relevant knowledge, `get_skill` for reading full solutions, and `save_skill` for recording new insights or evolving existing ones. It also ships with a **web dashboard** for human inspection, management, and browsing of the skill library. The system uses a hybrid search engine (BM25 lexical + semantic embeddings + Reciprocal Rank Fusion) to deliver accurate, ranked results, and supports a four-phase evolution model (ARCHIVE, FIX, DERIVE, MERGE) so skills improve organically as your codebase matures.

---

## Architecture

### SKILL.md Standard

Every skill is stored as a `SKILL.md` file — a Markdown document with YAML frontmatter that follows a well-defined specification. The file format is human-readable, version-control-friendly, and machine-parseable. Each skill lives in its own directory under `workspace/skills/{skill-name}/SKILL.md`, and optionally may include a `references/` subdirectory for supplementary materials such as screenshots, logs, or linked documents.

The SKILL.md standard defines strict frontmatter keys (`name`, `description`, `metadata`) and metadata sub-keys (`skill-type`, `repo`, `version`, `tags`, `created`, `last-modified`, `ttl-days`, `author`, `source`). Any deviation from these keys triggers validation warnings. The markdown body is expected to contain structured sections such as "When to Use", "Solution", "Root Causes", and "Lessons Learned", though the system does not enforce a rigid body schema — the AI agent and developers are free to structure the body to best convey the knowledge.

### 3-Tier Progressive Disclosure

Skills Lab uses a three-tier information architecture to minimize context-window usage while maximizing knowledge retrieval accuracy:

- **Tier 1 — Metadata Search (`search_skills`):** Returns lightweight metadata only — skill name, type, tags, description, relevance score, and lineage chain. The agent scans these results to decide which skills are worth examining in detail. This keeps the initial response small and fast.
- **Tier 2 — Full Content (`get_skill`):** Returns the complete SKILL.md body for a single skill. The agent calls this only after identifying a relevant skill from Tier 1, ensuring that context is consumed efficiently.
- **Tier 3 — Skill Evolution (`save_skill`):** Enables the agent (or developer) to record new knowledge, correct existing skills, derive variants, or merge redundant skills. Every evolution operation writes a changelog entry and updates the skill's version, creating a complete audit trail.

This design ensures that the AI agent never loads more context than necessary, which is critical when working within the token limits of large language models.

### Hybrid Search: BM25 + Semantic + RRF

The search engine combines two complementary retrieval strategies:

1. **BM25 Lexical Search:** Indexes the concatenated text of `skill.id + description + tags + repo_name` using a smart tokenizer that handles camelCase boundaries, hyphens, dots, and mixed-case input (e.g., "NextJs" becomes ["next", "js"]). The BM25 index is cached in memory and only rebuilt when skills change, not on every search call.
2. **Semantic Search:** Encodes skill descriptions and tags into dense vector embeddings using a sentence-transformer model (default: `BAAI/bge-small-en-v1.5`, 384 dimensions, 512 tokens). Embeddings are cached to disk (pickle format) and batch-computed for uncached skills.
3. **Reciprocal Rank Fusion (RRF):** Fuses the two ranked result lists using the RRF formula `score = Σ 1/(k + rank + 1)` with a configurable constant `k=60`. This naturally balances lexical and semantic relevance without requiring tuned weights.

### Evolution Model: ARCHIVE / FIX / DERIVE / MERGE

Skills follow a versioned lifecycle governed by four evolution operations:

| Operation | Description | Version | Parent |
|-----------|-------------|---------|--------|
| **ARCHIVE** | Create a brand-new skill | V1 | None |
| **FIX** | Update existing skill in-place | V+1 | Self (same row updated) |
| **DERIVE** | Create a variant from a parent skill | V1 (new row) | Parent remains active |
| **MERGE** | Combine multiple skills into one | max(V)+1 | All sources deactivated |

Every evolution operation writes a `SkillChangelog` entry recording `from_version`, `to_version`, `trigger`, `reason`, and `source_skill_id`. The FIX operation preserves accumulated Lessons Learned by merging new lessons with existing ones, deduplicating version markers. The MERGE operation collects lessons from all source skills and unions their tags. After every save, the system runs a deduplication check — if the new skill's embedding has cosine similarity ≥ 0.85 with any existing active skill, a warning is returned suggesting a potential merge.

---

## Quick Start

### 1. Install

```bash
cd skills_lab
pip install -e .
```

For semantic search support (recommended):
```bash
pip install -e ".[semantic]"
```

For development:
```bash
pip install -e ".[dev]"
```

### 2. Initialize Workspace

```bash
skills-lab init
```

This creates the workspace directory (`./workspace`), initializes the SQLite database (`brain.db`), and generates three sample skills so you can immediately explore the system.

### 3. Run MCP Server

```bash
skills-lab run-mcp
```

The MCP server communicates over **stdio**, making it compatible with Claude Desktop, GitHub Copilot, Cursor, and any MCP-compliant agent. Configure your agent to launch the `skills-lab` binary with the `run-mcp` subcommand.

### 4. Run Dashboard

```bash
skills-lab run-dashboard
```

Open [http://localhost:7788](http://localhost:7788) to access the web dashboard for browsing, searching, creating, and managing skills through a visual interface.

---

## SKILL.md Format

A SKILL.md file consists of two parts: a **YAML frontmatter block** (delimited by `---`) and a **Markdown body**. Here is the complete specification:

### YAML Frontmatter

```yaml
---
name: cors-fix-nextjs-api                    # Required. Kebab-case, 2-64 chars.
description: Fix CORS errors on Next.js API  # Required. Short description for matching.
  routes with credentials
license: MIT                                  # Optional. License identifier.
metadata:
  skill-type: TROUBLESHOOTING                # IMPLEMENTATION | WORKFLOW | TROUBLESHOOTING | ARCHITECTURE | RULE
  repo: my-webapp                            # Repository name or "global"
  version: "2"                               # String representation of version number
  tags: [cors, nextjs, api, credentials]     # List of tags for search/filtering
  created: "2025-01-15T10:30:00Z"           # ISO 8601 creation timestamp
  last-modified: "2025-01-20T14:00:00Z"     # ISO 8601 last modification timestamp
  ttl-days: 90                               # Optional. Time-to-live in days (0 or absent = no expiry)
  author: developer-name                     # Optional. Author attribution
  source: https://example.com/reference      # Optional. Reference URL or identifier
---
```

### Markdown Body

```markdown
# CORS Fix for Next.js API Routes

## When to Use
- Access-Control-Allow-Origin error in browser console
- CORS preflight (OPTIONS) request fails
- Cookies not sent with cross-origin requests

## Root Causes
1. Missing `credentials: true` in CORS config
2. Using wildcard `*` origin with `credentials: true`
3. Missing `Access-Control-Allow-Credentials: true` header

## Solution
\```typescript
import { NextApiRequest, NextApiResponse } from 'next';

const corsOptions = {
  origin: (origin, callback) => {
    const allowed = process.env.ALLOWED_ORIGINS?.split(',') || ['http://localhost:3000'];
    if (!origin || allowed.includes(origin)) {
      callback(null, true);
    } else {
      callback(new Error('Not allowed by CORS'));
    }
  },
  credentials: true,
  methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
};

export default function handler(req: NextApiRequest, res: NextApiResponse) {
  res.setHeader('Access-Control-Allow-Origin', req.headers.origin || '*');
  res.setHeader('Access-Control-Allow-Credentials', 'true');
  res.setHeader('Access-Control-Allow-Methods', 'GET,POST,PUT,DELETE,OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization');

  if (req.method === 'OPTIONS') {
    return res.status(200).end();
  }
  // ... your handler
}
\```

## Lessons Learned
- **V1** (2025-01-15): Next.js API routes don't enable CORS by default.
- **V2** (2025-01-20): Wildcard origin `*` + `credentials: true` = browser blocks. Use dynamic origin callback.
```

### Allowed Frontmatter Keys

| Key | Level | Required | Description |
|-----|-------|----------|-------------|
| `name` | Top | Yes | Kebab-case skill identifier |
| `description` | Top | Yes | Short description for search matching |
| `license` | Top | No | License identifier |
| `metadata.skill-type` | Nested | No | Skill type enum value |
| `metadata.repo` | Nested | No | Repository name or "global" |
| `metadata.version` | Nested | No | Version number (string) |
| `metadata.tags` | Nested | No | List of tag strings |
| `metadata.created` | Nested | No | ISO 8601 creation timestamp |
| `metadata.last-modified` | Nested | No | ISO 8601 modification timestamp |
| `metadata.ttl-days` | Nested | No | Time-to-live in days |
| `metadata.author` | Nested | No | Author attribution |
| `metadata.source` | Nested | No | Reference URL or identifier |

---

## MCP Tools

Skills Lab exposes three MCP tools that AI agents can invoke during coding sessions.

### `search_skills` — Tier 1: Lightweight Metadata Search

Searches the skill database and returns metadata with relevance scores. The agent calls this first to identify candidate skills before fetching full content.

```python
search_skills(
    query: str,           # Free-text description of the problem or technique
    repo_scope: str = "all",       # "current" (current repo + global) or "all"
    current_repo: str = "",        # Auto-detected from WORKSPACE_FOLDER env
    tags_filter: str = "",         # Comma-separated tags: "cors,nextjs,api"
)
```

**Returns:** Formatted markdown with skill ID, display name, type, repo, version, tags, description, RRF relevance score, and lineage chain for each matching skill. Expired skills are flagged with a warning icon.

**When the agent should call this:**
- Starting a new task and looking for known patterns
- Encountering an error or unfamiliar configuration issue
- Needing to apply a convention or rule specific to the current codebase

### `get_skill` — Tier 2: Full Skill Content

Retrieves the complete SKILL.md body for a specific skill by name. The agent calls this after identifying a relevant skill from `search_skills`.

```python
get_skill(
    name: str,   # Kebab-case skill name, e.g. "cors-fix-nextjs-api"
)
```

**Returns:** Full skill including metadata (type, repo, tags, version, lineage, timestamps) and the complete markdown body with all sections (When to Use, Solution, Root Causes, Lessons Learned, etc.).

### `save_skill` — Tier 3: Knowledge Evolution

Records new knowledge or evolves existing skills. The agent should only call this **after confirming that the solution works** (code compiles, tests pass, or the fix resolves the issue).

```python
save_skill(
    action: str,              # "ARCHIVE" | "FIX" | "DERIVE" | "MERGE"
    name: str,                # Kebab-case skill name
    description: str,         # Short description for agent matching
    body: str,                # Markdown body (When to Use, Solution, etc.)
    skill_type: str = "IMPLEMENTATION",  # Skill type enum value
    lesson: str = "",         # Lesson learned (for FIX and DERIVE)
    repo_name: str = "",      # Repo name (auto-detected if empty)
    tags: str = "",           # Comma-separated tags
    reason: str = "",         # Explanation for the evolution action
    target_skill_name: str = "",      # Required for FIX and DERIVE
    source_skill_names: str = "",     # Required for MERGE (comma-separated)
    ttl_days: int = 0,               # Time-to-live in days (0 = no expiry)
    display_name: str = "",          # Display name (auto-generated if empty)
    author: str = "",                # Author attribution
)
```

**Action Details:**

| Action | `target_skill_name` | `source_skill_names` | Behavior |
|--------|---------------------|---------------------|----------|
| `ARCHIVE` | Not used | Not used | Creates new skill at V1 |
| `FIX` | **Required** | Not used | Updates skill in-place, V+1, lessons accumulated |
| `DERIVE` | **Required** | Not used | Creates new variant V1, parent stays active |
| `MERGE` | Not used | **Required** | Merges sources, deactivates sources, lessons united |

After ARCHIVE or DERIVE, the system automatically checks for potential duplicates using cosine similarity (threshold: 0.85) and returns warnings if similar skills already exist, suggesting a MERGE if appropriate.

---

## CLI Reference

The `skills-lab` CLI provides all management commands for the system.

### `run-mcp`

Starts the MCP server over stdio for integration with AI coding agents (Claude, GitHub Copilot, Cursor).

```bash
skills-lab run-mcp
```

### `run-dashboard`

Starts the FastAPI-based web dashboard server. By default listens on `0.0.0.0:7788`.

```bash
skills-lab run-dashboard
```

Customize host and port via environment variables `SKILLS_LAB_HOST` and `SKILLS_LAB_PORT`.

### `init`

Initializes a new workspace with sample skills. Creates the workspace directory, SQLite database, and three demo skills: `cors-fix-nextjs-api`, `docker-multi-stage-build`, and `git-conventional-commits`. Skips skills that already exist.

```bash
skills-lab init
```

### `test`

Runs the integration test suite (`tests/test_skills_lab.py`), which exercises all core functionality: ARCHIVE, SEARCH, GET, FIX, DERIVE, MERGE, lineage tracking, TTL, SKILL.md format validation, tag filtering, and repo filtering. Tests use a temporary directory that is automatically cleaned up.

```bash
skills-lab test
```

### `sync`

Scans the `workspace/skills/` directory for SKILL.md files and imports them into the database. Skips skills already present in the database, validates frontmatter, and reports import statistics.

```bash
skills-lab sync
```

### `version`

Displays the current version and a brief project description.

```bash
skills-lab version
# Output: Skills Lab v2.0.0 — SKILL.md Standard
#         MCP Server for AI coding agents
```

### `export`

*(Planned)* Export skills from the database to SKILL.md files or a portable archive format.

### `import`

*(Planned)* Import skills from an external archive or directory of SKILL.md files.

### `stats`

*(Available via Dashboard API at `GET /api/stats`)* Returns aggregate statistics: total skills, active/inactive/expired counts, breakdown by repo and type, top 10 most-used skills, and 5 most recently created skills.

### `search`

*(Available via MCP tool `search_skills` or Dashboard API at `GET /api/skills?search=...`)* Searches the skill database using the hybrid BM25+semantic engine with RRF fusion.

---

## Dashboard

The Skills Lab Dashboard is a web-based UI served by a FastAPI application at `http://localhost:7788`. It provides a visual interface for browsing, searching, creating, and managing skills without interacting with the MCP server directly.

### Features

- **Overview Statistics:** At-a-glance cards showing total skills, active count, inactive count, and expired count. Breakdown by repository and skill type is displayed as well.
- **Skill Browsing Table:** A sortable, filterable table listing all skills with columns for name, type, repo, version, tags, last-used timestamp, and active status. Supports text search across name, display name, and description.
- **Filtering:** Filter skills by repository name, skill type, active status, and expired status using the filter bar above the table.
- **Skill Detail Modal:** Click any skill name to open a full detail view showing metadata, version history (lineage chain with trigger type, version transitions, and reasons), and the complete SKILL.md body rendered with syntax highlighting for code blocks.
- **Create New Skill:** A form-based interface for creating new skills with fields for name, description, body, skill type, repo, tags, and TTL. Validates the kebab-case name requirement before submission.
- **Deprecate & Delete:** Mark skills as inactive (deprecated) with an optional reason, or permanently delete a skill from both the database and filesystem.
- **Extend TTL:** Extend the time-to-live for skills with an expiry date by a configurable number of additional days.
- **Dark Theme UI:** The dashboard uses a polished dark theme optimized for developer workflows, with color-coded badges for skill types and statuses.

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/skills` | List all skills (supports `repo`, `skill_type`, `active_only`, `search`, `expired` query params) |
| `GET` | `/api/skills/{name}` | Get skill metadata with lineage chain |
| `GET` | `/api/skills/{name}/content` | Get full SKILL.md content |
| `GET` | `/api/skills/{name}/lineage` | Get lineage tree |
| `POST` | `/api/skills` | Create a new skill |
| `DELETE` | `/api/skills/{name}` | Delete a skill entirely |
| `PATCH` | `/api/skills/{name}/deprecate` | Deactivate a skill |
| `POST` | `/api/skills/{name}/extend-ttl` | Extend TTL |
| `GET` | `/api/stats` | Dashboard statistics |
| `GET` | `/api/repos` | List all repositories |

---

## Search Architecture

### BM25 Lexical Search

BM25 (Best Matching 25) is a probabilistic ranking function that scores documents based on term frequency and inverse document frequency. Skills Lab uses the `rank-bm25` library with the `BM25Okapi` variant.

**Indexed Text:** For each skill, the BM25 index is built over the concatenated string: `skill.id + " " + description + " " + tags.join(" ") + " " + repo_name`. This ensures that exact name matches, tag matches, and repo-scoped queries are all handled by the lexical search path.

**Smart Tokenizer:** The custom `_smart_tokenize()` function performs two passes:
1. Splits on all non-alphanumeric characters (hyphens, dots, underscores, whitespace).
2. Re-splits surviving tokens on camelCase boundaries using uppercase character detection (e.g., "NextJsApi" becomes ["next", "js", "api"]).

This means a search for "nextjs" will match a skill named "NextJsApi" because the tokenizer normalizes both to the same token set.

**Caching:** The BM25 index is built once and cached in memory. A dirty flag (`_bm25_dirty`) is set whenever a skill is created, updated, derived, or merged, causing the index to be lazily rebuilt on the next search call. This avoids rebuilding the index on every query.

### Semantic Search

Semantic search encodes text into dense vector embeddings and computes cosine similarity to find conceptually related skills, even when the exact keywords do not match.

**Model:** The default model is `BAAI/bge-small-en-v1.5` (512 max tokens, 384 dimensions). If unavailable, the system falls back to `sentence-transformers/all-MiniLM-L6-v2`. If neither model can be loaded, semantic search is gracefully disabled and the system operates in BM25-only mode.

**Encoding:** The text encoded for each skill is `description + " " + tags.join(" ")`. This fits comfortably within the 512-token limit of the default model. Query text is encoded with the same model.

**Embedding Cache:** All skill embeddings are cached in a dictionary keyed by skill name and persisted to disk as a pickle file at `workspace/.cache/embedding_cache.pkl`. When skills change, affected entries are evicted from the cache and recomputed. Uncached skills encountered during search are batch-encoded and added to the cache automatically.

**Batch Encoding:** Uses the `SentenceTransformer.encode()` method with batch processing for efficiency. If batch encoding fails, it falls back to sequential single-text encoding.

### Reciprocal Rank Fusion (RRF)

RRF is a rank-fusion algorithm that combines multiple ranked lists without requiring relevance scores to be on the same scale. The formula used is:

```
RRF_score(d) = Σ_{r in rankings}  1 / (k + rank(d, r) + 1)
```

where `k` is a constant (default: 60) that controls how much the fusion flattens rank differences. A higher `k` gives more weight to items that appear in both lists regardless of their exact rank positions, while a lower `k` favors items with higher absolute ranks.

Each skill's final score is the sum of its RRF contributions from the BM25 ranking and the semantic ranking. This means a skill that appears at position 1 in both rankings will receive a higher score than one that appears at position 1 in only one ranking, naturally promoting consensus results.

### Dedup Detection

After every `ARCHIVE` or `DERIVE` operation, the system computes the cosine similarity between the new skill's description embedding and all other active skills. Any skill with similarity ≥ 0.85 (configurable via `DEDUP_THRESHOLD`) is flagged as a potential duplicate in the response, along with its name, similarity score, and a truncated description. This encourages users and agents to consider merging redundant skills rather than creating near-duplicates.

---

## Configuration

All configuration is handled through environment variables, optionally loaded from a `.env` file. Values are resolved in priority order: explicit constructor argument > environment variable > `.env` file > default value.

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SKILLS_LAB_WORKSPACE` | `./workspace` | Root directory for all persisted data (database, skill files, cache) |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | Primary sentence-transformer model for semantic search embeddings |
| `SEARCH_TOP_K` | `5` | Maximum number of results returned from search queries (range: 1–50) |
| `SKILLS_LAB_HOST` | `0.0.0.0` | Dashboard server bind host |
| `SKILLS_LAB_PORT` | `7788` | Dashboard server bind port |
| `SKILLS_LAB_LOG_LEVEL` | `INFO` | Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL) |
| `WORKSPACE_FOLDER` | *(auto-detect)* | Set by MCP host (Claude/Copilot/Cursor) to enable repo name auto-detection |
| `DEDUP_THRESHOLD` | `0.85` | Cosine similarity threshold for duplicate detection (range: 0.5–1.0) |

### Configuration Validation

The `SkillsLabConfig` class provides a `validate()` method that checks for common misconfigurations:
- Workspace path is empty (skills will not be persisted)
- `SEARCH_TOP_K` is outside the recommended range [1, 50]
- `DEDUP_THRESHOLD` is outside the recommended range [0.5, 1.0]

### Workspace Structure

```
workspace/
├── brain.db                # SQLite database (WAL mode)
├── skills/
│   ├── cors-fix-nextjs-api/
│   │   ├── SKILL.md
│   │   └── references/      # Optional supplementary files
│   ├── docker-multi-stage-build/
│   │   └── SKILL.md
│   └── ...
└── .cache/
    └── embedding_cache.pkl  # Persisted embedding vectors
```

---

## Development

### Running Tests

```bash
# Run the integration test suite
skills-lab test

# Or directly with Python
python tests/test_skills_lab.py
```

The test suite covers 14 test cases: ARCHIVE (repo-scoped and global), SEARCH (BM25, repo-scoped, tag-filtered), GET SKILL content, FIX with lesson accumulation and changelog, DERIVE with parent-preservation, LINEAGE chain tracking, MERGE with source deactivation, NAME VALIDATION, TTL expiration, SKILL.MD format verification, skill listing, and deduplication checks.

### Project Structure

```
skills_lab/
├── pyproject.toml              # Package configuration (setuptools)
├── cli.py                      # CLI entry point (skills-lab command)
├── server.py                   # MCP server (FastMCP, 3 tools)
├── core/
│   ├── __init__.py
│   ├── config.py               # Centralized configuration (SkillsLabConfig)
│   ├── models.py               # SQLAlchemy ORM models (Skill, SkillChangelog)
│   ├── manager.py              # SKILL.md file I/O (read, write, parse, validate)
│   ├── retriever.py            # Hybrid search engine (BM25 + semantic + RRF)
│   ├── evolver.py              # Evolution engine (ARCHIVE, FIX, DERIVE, MERGE)
│   └── exceptions.py           # Exception hierarchy
├── dashboard/
│   ├── __init__.py
│   └── server.py               # FastAPI dashboard (REST API + React UI)
├── tests/
│   └── test_skills_lab.py      # Integration tests (14 test cases)
├── ui/
│   └── skills-lab-ui.jsx       # React dashboard UI (served via Babel transpilation)
├── workspace/                  # Runtime workspace (gitignored)
└── .env                        # Local environment overrides
```

### Database

Skills Lab uses **SQLite** with **WAL (Write-Ahead Logging)** journal mode for safe concurrent access. The database is initialized automatically on startup by `init_db()`, which creates the workspace directory, configures the engine, and runs `CREATE TABLE IF NOT EXISTS` for all models. Two tables are used:
- `skills` — stores metadata, versioning, TTL, and usage statistics
- `skill_changelog` — stores version transition history (audit log)

### Key Design Decisions

- **Skill name as primary key:** The kebab-case skill name serves as the primary key in the database and the directory name on disk. FIX operations update the existing row in-place (incrementing `version_number`) rather than creating a new row, keeping the namespace clean and avoiding proliferation of skill versions.
- **Filesystem-first architecture:** Actual skill content lives in SKILL.md files on disk, not in the database. The database stores only metadata and indexes. This means skills can be edited directly in a text editor, version-controlled with Git, and shared across systems by copying files.
- **Lazy model loading:** The embedding model is loaded only when the first search is performed, not at startup. This keeps initialization fast for use cases that only need BM25 search or CRUD operations.

---

## License

This project is licensed under the **MIT License**.

```
MIT License

Copyright (c) 2025 Skills Lab

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
