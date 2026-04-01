"""
Skills Lab — CLI Entry Point

Easy-to-use command line interface for the Skills Lab platform.

Usage:
    skills-lab run-mcp          Start MCP server (stdio) for Claude/Copilot/Cursor
    skills-lab run-dashboard    Start web dashboard on http://localhost:7788
    skills-lab test             Run integration tests
    skills-lab init             Create workspace with sample skills
    skills-lab sync             Import existing SKILL.md files into DB
    skills-lab export [opts]    Export skills to JSON file (default: stdout)
    skills-lab import <file>    Import skills from JSON file
    skills-lab stats            Show workspace statistics
    skills-lab search <query>   Search skills by name, description, tags
    skills-lab download-model   Download embedding model for semantic search
    skills-lab version          Show version

Install:
    cd skills_lab && pip install -e .

Or just run directly:
    python -m cli run-mcp
    python -m cli run-dashboard
"""

import argparse
import json
import os
import subprocess
import sys


def _ensure_project_root():
    """Ensure we can import core modules by adding the project root to sys.path."""
    cli_dir = os.path.dirname(os.path.abspath(__file__))
    if cli_dir not in sys.path:
        sys.path.insert(0, cli_dir)


def _get_workspace() -> str:
    """Return the resolved workspace path from env or default location."""
    return os.environ.get(
        "SKILLS_LAB_WORKSPACE",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "workspace"),
    )


def _parse_known_args(args: list[str], parser: argparse.ArgumentParser) -> argparse.Namespace:
    """Parse known args, ignoring unknown flags (for backward compatibility)."""
    parsed, _ = parser.parse_known_args(args)
    return parsed


# ---------------------------------------------------------------------------
# Existing commands
# ---------------------------------------------------------------------------

def cmd_run_mcp(args):
    """Start MCP server (stdio)."""
    _ensure_project_root()
    os.environ.setdefault("SKILLS_LAB_WORKSPACE", _get_workspace())
    from server import mcp
    mcp.run()


def cmd_run_dashboard(args):
    """Start web dashboard."""
    _ensure_project_root()
    os.environ.setdefault("SKILLS_LAB_WORKSPACE", _get_workspace())
    import uvicorn
    from dashboard.server import app
    port = int(os.environ.get("SKILLS_LAB_PORT", "7788"))
    host = os.environ.get("SKILLS_LAB_HOST", "0.0.0.0")
    print(f"\n  Skills Lab Dashboard: http://localhost:{port}\n")
    uvicorn.run(app, host=host, port=port)


def cmd_test(args):
    """Run integration tests."""
    _ensure_project_root()
    test_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "tests", "test_skills_lab.py"
    )
    if os.path.exists(test_path):
        subprocess.run(
            [sys.executable, test_path],
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )
    else:
        print("  Test file not found: tests/test_skills_lab.py")


def cmd_init(args):
    """Initialize workspace with sample skills."""
    _ensure_project_root()
    from core.models import init_db, get_session
    from core.manager import SKILLManager
    from core.evolver import EvolutionEngine
    from core.retriever import HybridRetriever

    workspace = _get_workspace()

    init_db(workspace)
    mgr = SKILLManager(workspace)
    session = get_session()
    ret = HybridRetriever(
        session_factory=get_session,
        manager=mgr,
        workspace_path=workspace,
    )
    engine = EvolutionEngine(
        session=session,
        manager=mgr,
        on_embedding_cache_clear=ret.clear_cache,
        on_embedding_compute=ret.compute_and_cache_embedding,
    )

    # Sample skills
    samples = [
        {
            "name": "cors-fix-nextjs-api",
            "description": "Fix CORS errors on Next.js API routes with credentials",
            "body": """# CORS Fix for Next.js API Routes

## When to Use
- Access-Control-Allow-Origin error in browser console
- CORS preflight (OPTIONS) request fails
- Cookies not sent with cross-origin requests

## Root Causes
1. Missing `credentials: true` in CORS config
2. Using wildcard `*` origin with `credentials: true` (browser blocks this)
3. Missing `Access-Control-Allow-Credentials: true` header

## Solution
```typescript
// pages/api/[...path].ts or app/api/[...path]/route.ts
import { NextApiRequest, NextApiResponse } from 'next';

// Dynamic origin check - NEVER use wildcard with credentials
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
```

## Lessons Learned
- **V1**: Next.js API routes don't enable CORS by default. You must set headers manually.
- **V2**: Wildcard origin `*` + `credentials: true` = browser blocks. Use dynamic origin callback.""",
            "skill_type": "TROUBLESHOOTING",
            "repo_name": "global",
            "tags": ["cors", "nextjs", "api", "credentials", "security"],
        },
        {
            "name": "docker-multi-stage-build",
            "description": "Optimized Dockerfile with multi-stage build for Node.js apps",
            "body": """# Docker Multi-Stage Build for Node.js

## When to Use
- Production Docker images need to be small
- Building from source (TypeScript to JavaScript)
- Dependencies differ between build and runtime

## Solution
```dockerfile
# Stage 1: Build
FROM node:20-alpine AS builder
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production
COPY . .
RUN npm run build

# Stage 2: Production
FROM node:20-alpine AS runner
WORKDIR /app
ENV NODE_ENV=production
COPY --from=builder /app/dist ./dist
COPY --from=builder /app/node_modules ./node_modules
COPY --from=builder /app/package.json ./
EXPOSE 3000
USER node
CMD ["node", "dist/index.js"]
```

## Lessons Learned
- **V1**: Multi-stage reduces image from ~1.2GB to ~150MB for Node.js apps.
- **V2**: Always use `npm ci` instead of `npm install` for reproducible builds.""",
            "skill_type": "IMPLEMENTATION",
            "repo_name": "global",
            "tags": ["docker", "nodejs", "build", "optimization"],
        },
        {
            "name": "git-conventional-commits",
            "description": "Use conventional commits format for clean changelog",
            "body": """# Git Conventional Commits

## When to Use
- Every commit message in every project
- Team projects that need automated changelog

## Format
```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

## Types
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `style`: Formatting (no code change)
- `refactor`: Code restructuring
- `perf`: Performance improvement
- `test`: Adding tests
- `chore`: Maintenance tasks

## Examples
```
feat(auth): add OAuth2 login flow
fix(api): handle null response from user service
docs(readme): update installation instructions
```

## Lessons Learned
- **V1**: Conventional commits enable automated semantic versioning and changelog generation.""",
            "skill_type": "RULE",
            "repo_name": "global",
            "tags": ["git", "convention", "commit", "changelog"],
        },
    ]

    created = 0
    for sample in samples:
        try:
            engine.archive(**sample)
            created += 1
        except ValueError:
            print(f"  [skip] '{sample['name']}' already exists, skipping")

    try:
        session.commit()
    except Exception as e:
        print(f"  [err] Failed to commit: {e}")
        session.rollback()
    finally:
        session.close()

    print(f"\n  Skills Lab initialized!")
    print(f"   Workspace:  {workspace}")
    print(f"   Skills created: {created}")
    print(f"   SKILL.md files: {os.path.join(workspace, 'skills/')}")

    # Auto-download embedding model
    print(f"\n  Downloading embedding model...")
    print(f"   (this may take a minute on first run)")
    try:
        from core.model_manager import download_with_fallback, detect_backend, check_onnx_deps

        # Show backend status before trying
        backend = detect_backend()
        if backend == "onnx":
            print(f"   Backend: ONNX (CPU-only, no GPU needed)")
        elif backend == "torch":
            print(f"   Backend: PyTorch (warning: may crash without GPU)")
            print(f"   TIP: For no-GPU machines, install ONNX deps:")
            print(f"         pip install onnxruntime tokenizers huggingface_hub")
        else:
            deps = check_onnx_deps()
            missing = [k for k, v in deps.items() if not v]
            print(f"   [warn] No embedding backend found!")
            print(f"          Missing: {', '.join(missing)}")
            print(f"          Install with: pip install onnxruntime tokenizers huggingface_hub")

        success, loaded_model = download_with_fallback(workspace)
        if success:
            print(f"   Semantic search model ready: {loaded_model}")
        else:
            print(f"   [warn] Could not download embedding model.")
            print(f"          Semantic search will use BM25-only mode.")
            print(f"          To fix, install ONNX deps and retry:")
            print(f"            pip install onnxruntime tokenizers huggingface_hub")
            print(f"            skills-lab download-model")
    except Exception as e:
        print(f"   [warn] Model download skipped: {e}")
        print(f"          Run 'skills-lab download-model' to download manually.")

    print(f"\n   Try: skills-lab run-mcp")
    print(f"        skills-lab run-dashboard")


def cmd_sync(args):
    """Import existing SKILL.md files from workspace/skills/ into DB."""
    _ensure_project_root()
    from core.models import init_db, get_session, Skill, SkillType
    from core.manager import SKILLManager
    from core.evolver import EvolutionEngine
    from core.retriever import HybridRetriever

    workspace = _get_workspace()

    init_db(workspace)
    mgr = SKILLManager(workspace)
    session = get_session()
    ret = HybridRetriever(
        session_factory=get_session,
        manager=mgr,
        workspace_path=workspace,
    )
    engine = EvolutionEngine(
        session=session,
        manager=mgr,
        on_embedding_cache_clear=ret.clear_cache,
        on_embedding_compute=ret.compute_and_cache_embedding,
    )

    # Find all SKILL.md files
    skill_names = mgr.list_skills()
    if not skill_names:
        print("No SKILL.md files found in workspace/skills/")
        session.close()
        return

    imported = 0
    skipped = 0
    errors = 0

    print(f"\n  Scanning {len(skill_names)} SKILL.md files...\n")

    for name in skill_names:
        try:
            # Check if already in DB
            existing = session.query(Skill).filter_by(id=name).first()
            if existing:
                skipped += 1
                print(f"  [skip] {name} - already in DB (V{existing.version_number})")
                continue

            # Parse SKILL.md
            data = mgr.read_skill(name)
            fm = data["frontmatter"]
            meta = fm.get("metadata", {}) or {}

            skill_name = fm.get("name", name)
            description = fm.get("description", "")
            body = data["body"]
            skill_type = meta.get("skill-type", "IMPLEMENTATION")
            repo = meta.get("repo", "global")
            tags = meta.get("tags", [])
            if isinstance(tags, list):
                tags = [str(t) for t in tags]
            else:
                tags = []

            if not description:
                print(f"  [warn] {name} - missing description, using name")
                description = name.replace("-", " ")

            # Validate
            validation = SKILLManager.validate_frontmatter(fm)
            if validation:
                print(f"  [warn] {name} - validation warnings: {'; '.join(validation)}")

            engine.archive(
                name=skill_name,
                description=description,
                body=body,
                skill_type=skill_type,
                repo_name=repo,
                tags=tags if tags else None,
            )
            imported += 1
            print(f"  [ok] {name} - imported ({skill_type}, {repo})")

        except Exception as e:
            errors += 1
            print(f"  [err] {name} - error: {e}")

    print(f"\n  Sync complete!")
    try:
        session.commit()
    except Exception as e:
        print(f"  [err] Failed to commit: {e}")
        session.rollback()
    finally:
        session.close()

    print(f"\n  Imported: {imported}")
    print(f"   Skipped (already in DB): {skipped}")
    print(f"   Errors: {errors}")


def cmd_download_model(args):
    """Download the embedding model for semantic search.

    Usage:
        skills-lab download-model              # Download default model
        skills-lab download-model <model_name> # Download specific model
    """
    _ensure_project_root()
    from core.model_manager import (
        DEFAULT_MODEL,
        FALLBACK_MODEL,
        download_embedding_model,
        download_with_fallback,
        detect_backend,
        check_onnx_deps,
    )

    workspace = _get_workspace()

    # Show backend info
    backend = detect_backend()
    print(f"\n  Detected backend: {backend or 'none'}")
    if backend != "onnx":
        deps = check_onnx_deps()
        missing = [k for k, v in deps.items() if not v]
        if missing:
            print(f"  Missing ONNX deps: {', '.join(missing)}")
        print(f"  For best compatibility (no GPU needed):")
        print(f"    pip install onnxruntime tokenizers huggingface_hub")
        print()

    parser = argparse.ArgumentParser(prog="skills-lab download-model")
    parser.add_argument(
        "model_name",
        nargs="?",
        default=None,
        help=f"Model name to download (default: {DEFAULT_MODEL})",
    )
    parsed = _parse_known_args(args, parser)

    model_name = parsed.model_name

    print()
    if model_name:
        # Download a specific model
        print(f"  Downloading embedding model: {model_name}")
        success = download_embedding_model(model_name, workspace)
        if success:
            print(f"  Model '{model_name}' downloaded and ready.")
        else:
            print(f"  Failed to download '{model_name}'.")
            print(f"  Make sure sentence-transformers is installed:")
            print(f"    pip install 'skills-lab[semantic-onnx]'  (recommended, no GPU)")
            print(f"    pip install 'skills-lab[semantic]'      (with GPU support)")
    else:
        # Default: try primary, fallback to backup
        print(f"  Downloading embedding model...")
        print(f"  Primary:   {DEFAULT_MODEL}")
        print(f"  Fallback:  {FALLBACK_MODEL}")
        print()
        success, loaded_model = download_with_fallback(workspace)
        if success:
            print(f"  Model '{loaded_model}' downloaded and ready.")
        else:
            print(f"  Failed to download any embedding model.")
            print(f"  Semantic search will use BM25-only mode.")
            print(f"  Make sure sentence-transformers is installed:")
            print(f"    pip install 'skills-lab[semantic-onnx]'  (recommended, no GPU)")
            print(f"    pip install 'skills-lab[semantic]'      (with GPU support)")
    print()


def cmd_version(args):
    """Show version information."""
    print("Skills Lab v3.0.0 - SKILL.md Standard")
    print("MCP Server for AI coding agents")


# ---------------------------------------------------------------------------
# New commands: export, import, stats, search
# ---------------------------------------------------------------------------

def cmd_export(args):
    """Export skills to a JSON file or stdout.

    Usage:
        skills-lab export [output.json]
        skills-lab export --all -o backup.json
        skills-lab export --all            # prints JSON to stdout
    """
    _ensure_project_root()
    from core.exporter import SkillExporter

    workspace = _get_workspace()
    parser = argparse.ArgumentParser(prog="skills-lab export")
    parser.add_argument(
        "output",
        nargs="?",
        default=None,
        help="Path to write JSON output (default: stdout)",
    )
    parser.add_argument(
        "--all",
        dest="active_only",
        action="store_false",
        default=True,
        help="Include inactive skills in the export",
    )
    parser.add_argument(
        "-o", "--output",
        dest="output_flag",
        default=None,
        help="Path to write JSON output (alternative positional arg)",
    )
    parsed = _parse_known_args(args, parser)

    output_path = parsed.output or parsed.output_flag

    exporter = SkillExporter(workspace)
    result = exporter.export_all(
        output_path=output_path,
        active_only=parsed.active_only,
    )

    if output_path is None:
        # Print to stdout
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(f"  Exported {result['count']} skills to {output_path}")


def cmd_import(args):
    """Import skills from a JSON file.

    Usage:
        skills-lab import backup.json
        skills-lab import backup.json --overwrite
        skills-lab import backup.json --force
    """
    _ensure_project_root()
    from core.exporter import SkillExporter

    workspace = _get_workspace()
    parser = argparse.ArgumentParser(prog="skills-lab import")
    parser.add_argument(
        "input",
        help="Path to the JSON file to import",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Overwrite existing skills during import",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Do not skip existing skills (equivalent to --overwrite)",
    )
    parsed = _parse_known_args(args, parser)

    if not os.path.isfile(parsed.input):
        print(f"  [err] File not found: {parsed.input}")
        return

    overwrite = parsed.overwrite or parsed.force
    skip_existing = not overwrite

    exporter = SkillExporter(workspace)
    result = exporter.import_skills(
        data=parsed.input,
        overwrite=overwrite,
        skip_existing=skip_existing,
    )

    print(f"  Import complete!")
    print(f"   Imported: {result['imported']}")
    print(f"   Skipped:  {result['skipped']}")
    print(f"   Errors:   {result['errors']}")


def cmd_stats(args):
    """Show workspace statistics.

    Usage:
        skills-lab stats
    """
    _ensure_project_root()
    from core.exporter import get_workspace_stats

    workspace = _get_workspace()
    stats = get_workspace_stats(workspace)

    print(f"\n  Skills Lab Workspace Statistics\n")
    print(f"   Total skills:  {stats['total']}")
    print(f"   Active:        {stats['active']}")
    print(f"   Inactive:      {stats['inactive']}")
    print(f"   Expired:       {stats['expired']}")

    if stats["repos"]:
        print(f"\n  Per-Repo Counts:")
        for repo, count in sorted(stats["repos"].items(), key=lambda x: -x[1]):
            print(f"    {repo}: {count}")

    if stats["types"]:
        print(f"\n  Per-Type Counts:")
        for stype, count in sorted(stats["types"].items(), key=lambda x: -x[1]):
            print(f"    {stype}: {count}")

    if stats["top_used"]:
        print(f"\n  Top Used Skills:")
        for entry in stats["top_used"]:
            print(f"    {entry['name']} ({entry['use_count']} uses)")

    if stats["recent"]:
        print(f"\n  Recently Created:")
        for entry in stats["recent"]:
            created = entry.get("created_at", "unknown")
            print(f"    {entry['name']} ({created})")

    print()


def cmd_search(args):
    """Search skills from the CLI.

    Usage:
        skills-lab search <query>
        skills-lab search "docker build" --type IMPLEMENTATION
        skills-lab search cors --repo global --limit 5
        skills-lab search api --tags security,auth
    """
    _ensure_project_root()
    from core.exporter import SkillExporter

    workspace = _get_workspace()
    parser = argparse.ArgumentParser(prog="skills-lab search")
    parser.add_argument(
        "query",
        nargs="?",
        default="",
        help="Search query (matched against name and description)",
    )
    parser.add_argument(
        "--repo",
        default=None,
        help="Filter by repository name",
    )
    parser.add_argument(
        "--type",
        dest="skill_type",
        default=None,
        help="Filter by skill type (IMPLEMENTATION, WORKFLOW, etc.)",
    )
    parser.add_argument(
        "--tags",
        default=None,
        help="Comma-separated list of tags to filter by",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Maximum number of results (default: 20)",
    )
    parsed = _parse_known_args(args, parser)

    # Parse tags from comma-separated string
    tags = None
    if parsed.tags:
        tags = [t.strip() for t in parsed.tags.split(",") if t.strip()]

    exporter = SkillExporter(workspace)
    results = exporter.search_skills(
        query=parsed.query,
        repo=parsed.repo,
        skill_type=parsed.skill_type,
        tags=tags,
        limit=parsed.limit,
    )

    if not results:
        print(f"  No skills found matching: {parsed.query or '(all)'}")
        return

    print(f"\n  Found {len(results)} skill(s)\n")
    for skill in results:
        status = "active" if skill["is_active"] else "inactive"
        tags_str = ", ".join(skill["tags"]) if skill["tags"] else "-"
        print(f"  [{status}] {skill['name']}")
        print(f"    Type: {skill['skill_type']}  |  Repo: {skill['repo_name']}  |  Tags: {tags_str}")
        print(f"    {skill['description'][:100]}")
        print()

    print(f"  Total: {len(results)} result(s)")


# ---------------------------------------------------------------------------
# Version diff command
# ---------------------------------------------------------------------------

def cmd_diff(args):
    """Show diff between two versions of a skill.

    Usage:
        skills-lab diff <skill-name> [--v1 1] [--v2 2]
        skills-lab diff <skill-name> --v1 1          # Diff V1 vs current
    """
    _ensure_project_root()
    from core.manager import SKILLManager

    workspace = _get_workspace()
    parser = argparse.ArgumentParser(prog="skills-lab diff")
    parser.add_argument(
        "skill_name",
        help="Kebab-case skill name to diff",
    )
    parser.add_argument(
        "--v1",
        default="1",
        help="First version to compare (default: 1)",
    )
    parser.add_argument(
        "--v2",
        default="current",
        help="Second version to compare (default: current)",
    )
    parsed = _parse_known_args(args, parser)

    mgr = SKILLManager(workspace)
    try:
        result = mgr.get_version_diff(parsed.skill_name, parsed.v1, parsed.v2)
    except FileNotFoundError as e:
        print(f"  [err] {e}")
        return

    print(f"\n  Diff: {result['skill_name']} (v{result['v1']} → v{result['v2']})")
    print(f"{'=' * 60}")
    if result["diff"]:
        for line in result["diff"].split("\n"):
            if line.startswith("+++") or line.startswith("---"):
                print(f"\033[36m{line}\033[0m")  # cyan for headers
            elif line.startswith("@@"):
                print(f"\033[33m{line}\033[0m")  # yellow for hunks
            elif line.startswith("+"):
                print(f"\033[32m{line}\033[0m")  # green for additions
            elif line.startswith("-"):
                print(f"\033[31m{line}\033[0m")  # red for deletions
            else:
                print(line)
    else:
        print("  No differences found.")
    print()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    """Main CLI entry point.

    Parses the first positional argument as the command name and dispatches
    to the appropriate handler function.  Unrecognised commands trigger a
    usage help message.
    """
    if len(sys.argv) < 2:
        print("""
Skills Lab v3.0.0 - MCP Server for AI coding agents

Usage:
    skills-lab run-mcp          Start MCP server (stdio)
    skills-lab run-dashboard    Start web dashboard (http://localhost:7788)
    skills-lab test             Run integration tests
    skills-lab init             Create workspace with sample skills
    skills-lab sync             Import existing SKILL.md files into DB
    skills-lab export [opts]    Export skills to JSON file (default: stdout)
    skills-lab import <file>    Import skills from JSON file
    skills-lab stats            Show workspace statistics
    skills-lab search <query>   Search skills by name, description, tags
    skills-lab download-model   Download embedding model for semantic search
    skills-lab version          Show version

Export Options:
    --all                       Include inactive skills
    -o, --output <path>         Write JSON to a file instead of stdout

Import Options:
    --overwrite                 Overwrite existing skills
    --force                     Do not skip existing skills

Search Options:
    --repo <name>               Filter by repository
    --type <type>               Filter by skill type
    --tags <tags>               Filter by tags (comma-separated)
    --limit <n>                 Max results (default: 20)

Quick Start:
    cd skills_lab
    pip install -e .              # Install (one time)
    skills-lab init               # Create sample workspace
    skills-lab run-dashboard      # Open dashboard
    skills-lab run-mcp            # Start MCP for AI agent
    skills-lab export             # Export all skills to stdout
    skills-lab stats              # View workspace stats
    skills-lab search "docker"    # Search skills

Environment Variables:
    SKILLS_LAB_WORKSPACE    Path to workspace (default: ./workspace)
    EMBEDDING_MODEL         Model name (default: BAAI/bge-small-en-v1.5)
    SKILLS_LAB_PORT         Dashboard port (default: 7788)
    SKILLS_LAB_HOST         Dashboard host (default: 0.0.0.0)
    WORKSPACE_FOLDER        Auto-detect repo name for MCP tools
    SKILLS_LAB_SEMANTIC     Enable semantic search (default: auto-detect)
                           Set to 0 to explicitly disable, 1 to force enable
        """)
        sys.exit(0)

    cmd = sys.argv[1].lower().replace("-", "_")
    args = sys.argv[2:]

    commands = {
        "run_mcp": cmd_run_mcp,
        "run-dashboard": cmd_run_dashboard,
        "run_dashboard": cmd_run_dashboard,
        "test": cmd_test,
        "init": cmd_init,
        "sync": cmd_sync,
        "export": cmd_export,
        "import": cmd_import,
        "stats": cmd_stats,
        "search": cmd_search,
        "version": cmd_version,
        "download-model": cmd_download_model,
        "download_model": cmd_download_model,
        "diff": cmd_diff,
    }

    handler = commands.get(cmd)
    if handler:
        handler(args)
    else:
        print(f"  Unknown command: {cmd}")
        print(f"   Run 'skills-lab' for usage help.")


if __name__ == "__main__":
    main()
