"""
Skills Lab — MCP Server (SKILL.md Standard)

Entry point. Communicates via stdio with AI agents (GitHub Copilot, Claude, Cursor).

3 MCP tools:
  1. search_skills  — Tier 1: search metadata + scores
  2. get_skill      — Tier 2: retrieve full SKILL.md content
  3. save_skill     — ARCHIVE / FIX / DERIVE / MERGE

Run with: python server.py
"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timezone

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("skills_lab")

# ---------------------------------------------------------------------------
# Load .env
# ---------------------------------------------------------------------------
_server_dir = os.path.dirname(os.path.abspath(__file__))
_env_path = os.path.join(_server_dir, ".env")
if os.path.exists(_env_path):
    load_dotenv(_env_path)
else:
    _workspace_env = os.environ.get("SKILLS_LAB_WORKSPACE", "")
    if _workspace_env:
        _alt_env = os.path.join(_workspace_env, ".env")
        if os.path.exists(_alt_env):
            load_dotenv(_alt_env)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
WORKSPACE_PATH = os.environ.get(
    "SKILLS_LAB_WORKSPACE",
    os.path.join(_server_dir, "workspace"),
)
EMBEDDING_MODEL = os.environ.get(
    "EMBEDDING_MODEL",
    "BAAI/bge-small-en-v1.5",
)
SEARCH_TOP_K = int(os.environ.get("SEARCH_TOP_K", "5"))

from core.model_manager import is_model_cached

# Semantic search auto-detection:
#   SKILLS_LAB_SEMANTIC=0 → explicitly disabled
#   SKILLS_LAB_SEMANTIC=1 → explicitly enabled
#   Default (empty)     → auto-detect: enable if model is cached
env_semantic = os.environ.get("SKILLS_LAB_SEMANTIC", "").strip().lower()
if env_semantic in ("0", "false", "no"):
    SEMANTIC_ENABLED = False
    _semantic_reason = "explicitly disabled via SKILLS_LAB_SEMANTIC=0"
elif env_semantic in ("1", "true", "yes"):
    SEMANTIC_ENABLED = True
    _semantic_reason = "explicitly enabled via SKILLS_LAB_SEMANTIC=1"
else:
    _model_cached = is_model_cached(EMBEDDING_MODEL, WORKSPACE_PATH)
    SEMANTIC_ENABLED = _model_cached
    _semantic_reason = (
        "AUTO-ENABLED (model cached)" if _model_cached
        else "DISABLED (model not cached)"
    )

logger.info(f"Workspace: {WORKSPACE_PATH}")
logger.info(f"Embedding model: {EMBEDDING_MODEL}")
logger.info(f"Search Top-K: {SEARCH_TOP_K}")
logger.info(f"Semantic search: {_semantic_reason}")
if not SEMANTIC_ENABLED:
    if env_semantic in ("0", "false", "no"):
        logger.info("  → To enable: unset SKILLS_LAB_SEMANTIC or set SKILLS_LAB_SEMANTIC=1")
    else:
        logger.info("  → To enable: run 'skills-lab download-model' or set SKILLS_LAB_SEMANTIC=1")

# ---------------------------------------------------------------------------
# Init
# ---------------------------------------------------------------------------
from core.models import init_db, get_session, Skill, SkillChangelog, SkillType, EvolutionTrigger
from core.manager import SKILLManager
from core.retriever import HybridRetriever
from core.evolver import EvolutionEngine

init_db(WORKSPACE_PATH)
manager = SKILLManager(WORKSPACE_PATH)
retriever = HybridRetriever(
    session_factory=get_session,
    manager=manager,
    workspace_path=WORKSPACE_PATH,
    model_name=EMBEDDING_MODEL,
    top_k=SEARCH_TOP_K,
)

mcp = FastMCP("SkillsLab")


def detect_repo_name() -> str:
    """Detect the current developer's repository name."""
    workspace = os.environ.get("WORKSPACE_FOLDER", "")
    if workspace:
        return os.path.basename(workspace.rstrip("/\\"))
    cwd = os.getcwd()
    return os.path.basename(cwd)


def format_lineage_chain(skill_name: str) -> str:
    """Format changelog entries into readable text."""
    try:
        chain = retriever.get_lineage_chain(skill_name)
        if len(chain) <= 1:
            return "Root skill"
        parts = []
        for item in chain:
            trigger = item.get("trigger") or "FIX"
            reason = item.get("reason") or ""
            if len(reason) > 50:
                reason = reason[:50] + "..."
            parts.append(f"{trigger} V{item['to_version']} ({reason})")
        return " → ".join(parts)
    except Exception as e:
        logger.warning(f"Lineage chain error for {skill_name}: {e}")
        return "Lineage unavailable"


def _parse_tags(tags_str: str) -> list[str]:
    """Parse tags from comma-separated or JSON array string."""
    if not tags_str:
        return []
    tags_str = tags_str.strip()
    # Try JSON array
    if tags_str.startswith("["):
        try:
            parsed = json.loads(tags_str)
            if isinstance(parsed, list):
                return [str(t).strip().lower() for t in parsed if str(t).strip()]
        except json.JSONDecodeError:
            pass
    # Fallback: comma-separated
    return [t.strip().lower() for t in tags_str.split(",") if t.strip()]


# ---------------------------------------------------------------------------
# Tool 1: search_skills (Tier 1)
# ---------------------------------------------------------------------------

@mcp.tool()
async def search_skills(
    query: str,
    repo_scope: str = "all",
    current_repo: str = "",
    tags_filter: str = "",
) -> str:
    """
    Search Skills by query. Returns METADATA ONLY (Tier 1 — lightweight).

    CALL THIS TOOL when starting a new task, encountering an error, or needing a known pattern.
    After receiving results, call get_skill(name) to retrieve the full content.

    Auto-determine repo_scope:
      - "current": repo-specific issues (config, conventions)
      - "all": general technical issues (auth, docker, CORS...)

    Args:
        query: Description of the problem or technique to search for.
        repo_scope: "current" (current repo + global) or "all".
        current_repo: Name of the currently open repo (from WORKSPACE_FOLDER env).
        tags_filter: Comma-separated tags to filter, e.g. "cors,nextjs".
    """
    try:
        if not current_repo:
            current_repo = detect_repo_name()

        tags = _parse_tags(tags_filter) if tags_filter else None

        # Run synchronous retriever.search() in a thread to avoid blocking the event loop
        results = await asyncio.to_thread(
            retriever.search,
            query=query,
            repo_scope=repo_scope,
            current_repo=current_repo,
            tags_filter=tags,
        )

        if not results:
            return (
                f"No skills found for: \"{query}\"\n\n"
                f"Suggestions:\n"
                f"- Try repo_scope='all'\n"
                f"- Change the query\n"
                f"- Add tags_filter (e.g. \"cors,docker\")"
            )

        output = [f"## Search Results — {len(results)} skills found\n"]

        for idx, item in enumerate(results, 1):
            skill = item["skill"]
            score = item["rrf_score"]
            tags = skill.get_tags()
            expired_flag = " ⚠️ EXPIRED" if skill.is_expired() else ""
            lineage = format_lineage_chain(skill.id)

            tag_str = ", ".join(tags) if tags else "(none)"
            output.append(f"### [{idx}] `{skill.id}` ⭐ {score:.4f}{expired_flag}")
            output.append(f"**Name:** {skill.display_name}")
            output.append(f"**Type:** {skill.skill_type} | **Repo:** {skill.repo_name} | **V{skill.version_number}**")
            output.append(f"**Tags:** {tag_str}")
            output.append(f"**Description:** {skill.description}")
            output.append(f"**Lineage:** {lineage}")

        output.append(f"\n→ Call `get_skill(name=\"skill-name\")` to view the full content.")

        # Update use_count
        session = get_session()
        try:
            skill_ids = [item["skill"].id for item in results]
            now = datetime.now(timezone.utc)
            session.query(Skill).filter(Skill.id.in_(skill_ids)).update(
                {
                    Skill.use_count: Skill.use_count + 1,
                    Skill.last_used_at: now,
                },
                synchronize_session="fetch",
            )
            session.commit()
        except Exception as db_err:
            session.rollback()
            logger.error(f"DB update error: {db_err}")
        finally:
            session.close()

        return "\n".join(output)

    except Exception as e:
        logger.error(f"search_skills error: {e}", exc_info=True)
        return f"Search error: {str(e)}"


# ---------------------------------------------------------------------------
# Tool 2: get_skill (Tier 2)
# ---------------------------------------------------------------------------

@mcp.tool()
async def get_skill(name: str) -> str:
    """
    Retrieve the FULL CONTENT of a Skill (Tier 2 — full SKILL.md).

    CALL AFTER search_skills() once you have decided which skill to inspect.
    Returns the complete SKILL.md content: When to Use, Solution, Lessons Learned.

    Args:
        name: Skill name (kebab-case), e.g. "cors-fix-nextjs-api".
    """
    try:
        data = await asyncio.to_thread(retriever.get_skill_content, name)
        if not data:
            return f"Skill '{name}' does not exist.\n\nCall search_skills() to find a suitable skill."

        skill = data["skill"]
        body = data["body"]

        tags = skill.get("tags", [])
        tag_str = ", ".join(tags) if tags else "(none)"
        expired_flag = " ⚠️ EXPIRED" if skill.get("is_expired") else ""
        lineage = format_lineage_chain(name)

        output = [
            f"# {skill['display_name']} `{name}` V{skill['version_number']}{expired_flag}",
            f"**Type:** {skill['skill_type']} | **Repo:** {skill['repo_name']}",
            f"**Tags:** {tag_str} | **Description:** {skill['description']}",
            f"**Lineage:** {lineage}",
            f"**Created:** {skill['created_at']} | **Modified:** {skill['last_modified_at']}",
            "",
            "---",
            "",
            body,
        ]

        return "\n".join(output)

    except Exception as e:
        logger.error(f"get_skill error: {e}", exc_info=True)
        return f"Error reading skill: {str(e)}"


# ---------------------------------------------------------------------------
# Tool 3: save_skill (ARCHIVE / FIX / DERIVE / MERGE)
# ---------------------------------------------------------------------------

@mcp.tool()
async def save_skill(
    action: str,
    name: str,
    description: str,
    body: str,
    skill_type: str = "IMPLEMENTATION",
    lesson: str = "",
    repo_name: str = "",
    tags: str = "",
    reason: str = "",
    target_skill_name: str = "",
    source_skill_names: str = "",
    ttl_days: int = 0,
    display_name: str = "",
    author: str = "",
) -> str:
    """
    Save knowledge into Skills Lab (SKILL.md standard).

    ONLY CALL when the developer has confirmed the code/process WORKS (Green Light).

    Actions:
      - ARCHIVE: Create a brand new skill (V1, no parent).
      - FIX: Update an existing skill (V+1, parent becomes inactive, lessons accumulated).
        Requires: target_skill_name.
      - DERIVE: Create a variant from a source skill (new V1, parent stays active).
        Requires: target_skill_name.
      - MERGE: Combine multiple skills into one (all sources become inactive).
        Requires: source_skill_names (comma-separated).

    Args:
        action: "ARCHIVE" | "FIX" | "DERIVE" | "MERGE"
        name: Skill name (kebab-case). For FIX: same as target. For DERIVE/MERGE: new name.
        description: Short description for agent matching.
        body: Markdown body (When to Use, Solution, Root Causes, ...).
        skill_type: "IMPLEMENTATION" | "WORKFLOW" | "TROUBLESHOOTING" | "ARCHITECTURE" | "RULE"
        lesson: New lesson learned (for FIX, also appended to DERIVE).
        repo_name: Repository name or "global" (auto-detected if empty).
        tags: Comma-separated tags, e.g. "cors,nextjs,api".
        reason: Reason for creating/updating (for lineage record).
        target_skill_name: Required for FIX/DERIVE (parent skill name).
        source_skill_names: Required for MERGE (comma-separated source names).
        ttl_days: Number of days the skill is valid (0 = never expires).
        display_name: Display name (auto-generated if empty).
        author: Author.
    """
    try:
        # Auto-detect repo
        if not repo_name:
            repo_name = detect_repo_name()

        # Validate action
        valid_actions = ["ARCHIVE", "FIX", "DERIVE", "MERGE"]
        if action not in valid_actions:
            return f"Error: invalid action '{action}'. Must be one of: {', '.join(valid_actions)}"

        # Validate type
        valid_types = [t.value for t in SkillType]
        if skill_type not in valid_types:
            return f"Error: invalid type '{skill_type}'. Must be one of: {', '.join(valid_types)}"

        # Parse tags
        parsed_tags = _parse_tags(tags) if tags else []

        # Validate per-action requirements
        if action == "FIX" and not target_skill_name:
            return "Error: action='FIX' requires target_skill_name."
        if action == "DERIVE" and not target_skill_name:
            return "Error: action='DERIVE' requires target_skill_name."
        if action == "MERGE" and not source_skill_names:
            return "Error: action='MERGE' requires source_skill_names (comma-separated)."

        ttl = int(ttl_days) if ttl_days and int(ttl_days) > 0 else None

        # IMPORTANT: Session must be created AND used entirely within the worker
        # thread. SQLAlchemy sessions are NOT thread-safe — creating the session
        # in the async context and committing in a thread causes data corruption.
        def _execute_evolution():
            session = get_session()
            try:
                engine = EvolutionEngine(
                    session=session,
                    manager=manager,
                    on_embedding_cache_clear=retriever.clear_cache,
                    on_embedding_compute=retriever.compute_and_cache_embedding,
                )

                if action == "ARCHIVE":
                    skill = engine.archive(
                        name=name,
                        description=description,
                        body=body,
                        skill_type=skill_type,
                        repo_name=repo_name,
                        display_name=display_name,
                        tags=parsed_tags if parsed_tags else None,
                        ttl_days=ttl,
                        author=author or None,
                    )
                    action_desc = "Created new Skill (ARCHIVE)"

                elif action == "FIX":
                    skill = engine.fix(
                        target_skill_name=target_skill_name,
                        body=body,
                        lesson=lesson,
                        reason=reason,
                        description=description or None,
                        tags=parsed_tags or None,
                    )
                    action_desc = f"Fixed Skill → V{skill.version_number}"

                elif action == "DERIVE":
                    skill = engine.derive(
                        target_skill_name=target_skill_name,
                        new_name=name,
                        body=body,
                        description=description,
                        lesson=lesson,
                        repo_name=repo_name,
                        reason=reason,
                        skill_type=skill_type,
                        tags=parsed_tags or None,
                    )
                    action_desc = f"Derived Skill → {name} V1"

                elif action == "MERGE":
                    sources = [s.strip() for s in source_skill_names.split(",") if s.strip()]
                    skill = engine.merge(
                        target_skill_name=name,
                        source_skill_names=sources,
                        new_body=body,
                        reason=reason,
                        description=description or None,
                        tags=parsed_tags or None,
                    )
                    action_desc = f"Merged {len(sources)} skills → {name} V{skill.version_number}"

                session.commit()
                return skill, action_desc

            except ValueError as ve:
                session.rollback()
                raise ve
            except Exception as e:
                session.rollback()
                raise e
            finally:
                session.close()

        try:
            skill, action_desc = await asyncio.to_thread(_execute_evolution)
        except ValueError as ve:
            return f"Error: {str(ve)}"
        except Exception as e:
            logger.error(f"save_skill error: {e}", exc_info=True)
            return f"Error saving knowledge: {str(e)}"

        # Flush embedding cache to disk
        retriever.flush_cache()

        # Dedup check (after commit)
        dup_warnings = ""
        if action in ("ARCHIVE", "DERIVE"):
            dups = retriever.check_duplicates(skill.id, skill.description)
            if dups:
                dup_lines = [f"  - `{d['name']}` (similarity: {d['similarity']}) — {d['description'][:60]}" for d in dups[:3]]
                dup_warnings = f"\n\n⚠️ **Potential duplicates detected:**\n" + "\n".join(dup_lines)
                dup_warnings += "\nConsider MERGE if these skills are redundant."

        result_msg = (
            f"✅ {action_desc}\n\n"
            f"- **Name:** `{skill.id}`\n"
            f"- **Display:** {skill.display_name}\n"
            f"- **Version:** V{skill.version_number}\n"
            f"- **Type:** {skill.skill_type}\n"
            f"- **Repo:** {skill.repo_name}\n"
            f"- **Tags:** {', '.join(skill.get_tags())}\n"
            f"- **Active:** {skill.is_active}\n"
        )

        if action in ("FIX", "DERIVE", "MERGE"):
            result_msg += f"- **Parent(s):** {target_skill_name or source_skill_names}\n"
            result_msg += f"- **Trigger:** {action}\n"
            result_msg += f"- **Reason:** {reason}\n"

        result_msg += dup_warnings

        logger.info(f"save_skill({action}): {skill.id} V{skill.version_number}")
        return result_msg

    except Exception as e:
        logger.error(f"save_skill outer error: {e}", exc_info=True)
        return f"System error: {str(e)}"


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logger.info("Skills Lab MCP Server starting (SKILL.md Standard)...")
    mcp.run()