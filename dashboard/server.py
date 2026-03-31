"""
Skills Lab — Dashboard REST API (SKILL.md Standard)

FastAPI server: statistics, browse skills, view lineage, deprecate, search, export, edit.
Serves React UI at /.

Endpoints:
  GET  /api/skills             — List all skills (with filters)
  GET  /api/skills/{name}      — Get single skill detail
  GET  /api/skills/{name}/lineage — Get lineage chain
  GET  /api/skills/{name}/content — Get full SKILL.md content
  GET  /api/skills/{name}/references — Get references for a skill
  PUT  /api/skills/{name}      — Update/edit a skill
  PATCH /api/skills/{name}/deprecate — Deactivate skill
  POST /api/skills/{name}/extend-ttl — Extend TTL
  POST /api/skills             — Create a new skill
  DELETE /api/skills/{name}    — Delete a skill
  POST /api/search             — Hybrid BM25+semantic+RRF search
  GET  /api/export             — Export all skills as JSON
  GET  /api/stats              — Dashboard statistics
  GET  /api/repos              — List repos
  GET  /api/health             — Health check
  GET  /api/analytics/summary  — Overall usage statistics
  GET  /api/analytics/trending — Trending (recently used) skills
  GET  /api/analytics/stale    — Stale (not recently used) skills
  GET  /api/analytics/types    — Skill type distribution
  GET  /api/analytics/tags     — Tag frequency / tag-cloud data
  GET  /api/analytics/activity — Recent changelog entries
  GET  /api/analytics/network  — Skill relationship graph
  GET  /api/analytics/gaps     — Coverage-gap analysis
  GET  /                        — Serve React UI
"""

import logging
import os
import threading
import time
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import func

from core.analytics import SkillsAnalytics
from core.models import init_db, get_session, Skill, SkillChangelog, SkillType

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Init
# ---------------------------------------------------------------------------
WORKSPACE_PATH = os.environ.get(
    "SKILLS_LAB_WORKSPACE",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "workspace"),
)

init_db(WORKSPACE_PATH)

# ---------------------------------------------------------------------------
# Singleton instances for SKILLManager and HybridRetriever
# ---------------------------------------------------------------------------

from core.manager import SKILLManager
from core.retriever import HybridRetriever

_mgr: SKILLManager | None = None
_ret: HybridRetriever | None = None
_singleton_lock = threading.Lock()
_singleton_ready = False


def _get_manager() -> SKILLManager:
    """Lazily create the SKILLManager singleton (thread-safe)."""
    global _mgr
    if _mgr is None:
        with _singleton_lock:
            if _mgr is None:
                _mgr = SKILLManager(WORKSPACE_PATH)
                logger.info("SKILLManager singleton created")
    return _mgr


def _get_retriever() -> HybridRetriever:
    """Lazily create the HybridRetriever singleton (thread-safe)."""
    global _ret, _singleton_ready
    if _ret is None:
        with _singleton_lock:
            if _ret is None:
                t0 = time.time()
                _ret = HybridRetriever(
                    session_factory=get_session,
                    manager=_get_manager(),
                    workspace_path=WORKSPACE_PATH,
                )
                _singleton_ready = True
                logger.info(f"HybridRetriever singleton created ({time.time() - t0:.2f}s)")
    return _ret

app = FastAPI(title="Skills Lab Dashboard", version="2.1 (SKILL.md Standard)")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class DeprecateRequest(BaseModel):
    reason: str = ""


class ExtendTTLRequest(BaseModel):
    additional_days: int = 90


class CreateSkillRequest(BaseModel):
    name: str = ""
    description: str = ""
    body: str = ""
    skill_type: str = "IMPLEMENTATION"
    repo_name: str = "global"
    tags: str = ""
    ttl_days: int = 0


class SearchRequest(BaseModel):
    query: str = ""
    repo_scope: str = "all"
    current_repo: str = ""
    tags_filter: str = ""
    top_k: int = 5


class UpdateSkillRequest(BaseModel):
    description: str = ""
    body: str = ""
    tags: str = ""
    display_name: str = ""
    references: str = ""  # comma-separated skill names


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/skills")
def list_skills(
    repo: str = Query("", description="Filter by repo name"),
    skill_type: str = Query("", description="Filter by skill type"),
    active_only: bool = Query(True, description="Only active skills"),
    search: str = Query("", description="Search in name/description/tags"),
    expired: bool = Query(False, description="Include expired skills"),
):
    """List all skills with optional filters."""
    session = get_session()
    try:
        query = session.query(Skill)

        if active_only:
            query = query.filter(Skill.is_active == True)

        if not expired:
            query = query.filter(
                (Skill.expires_at == None) | (Skill.expires_at > datetime.now(timezone.utc))
            )

        if repo:
            query = query.filter(Skill.repo_name == repo)

        if skill_type:
            query = query.filter(Skill.skill_type == skill_type)

        if search:
            like = f"%{search}%"
            query = query.filter(
                (Skill.id.ilike(like)) |
                (Skill.display_name.ilike(like)) |
                (Skill.description.ilike(like))
            )

        skills = query.order_by(Skill.last_used_at.desc().nullslast(), Skill.created_at.desc()).all()

        # Enrich with changelog count — use single batch query instead of N+1
        all_skill_ids = [s.id for s in skills]
        changelog_counts = {}
        if all_skill_ids:
            count_rows = (
                session.query(
                    SkillChangelog.skill_id,
                    func.count(SkillChangelog.id),
                )
                .filter(SkillChangelog.skill_id.in_(all_skill_ids))
                .group_by(SkillChangelog.skill_id)
                .all()
            )
            changelog_counts = dict(count_rows)

        result = []
        for skill in skills:
            result.append({
                **skill.to_dict(),
                "lineage_count": changelog_counts.get(skill.id, 0),
            })

        return result
    finally:
        session.close()


@app.get("/api/skills/{name}")
def get_skill(name: str):
    """Get skill metadata by name."""
    session = get_session()
    try:
        skill = session.query(Skill).filter_by(id=name).first()
        if not skill:
            raise HTTPException(status_code=404, detail=f"Skill '{name}' not found")

        # Changelog history
        entries = session.query(SkillChangelog).filter(
            SkillChangelog.skill_id == name
        ).order_by(SkillChangelog.created_at.asc()).all()

        chain = []
        for entry in entries:
            chain.append({
                "id": entry.id,
                "trigger": entry.trigger,
                "from_version": entry.from_version,
                "to_version": entry.to_version,
                "reason": entry.reason,
                "source_skill_id": entry.source_skill_id,
                "created_at": entry.created_at.isoformat() if entry.created_at else None,
            })

        return {
            **skill.to_dict(),
            "lineage_chain": chain,
        }
    finally:
        session.close()


@app.get("/api/skills/{name}/content")
def get_skill_content(name: str):
    """Get full SKILL.md content for a skill."""
    session = get_session()
    try:
        skill = session.query(Skill).filter_by(id=name).first()
        if not skill:
            raise HTTPException(status_code=404, detail=f"Skill '{name}' not found")

        mgr = _get_manager()
        try:
            data = mgr.read_skill(name)
            return {
                "name": name,
                "frontmatter": data["frontmatter"],
                "body": data["body"],
                "raw": data["raw"],
            }
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail=f"SKILL.md not found for '{name}'")
    finally:
        session.close()


@app.get("/api/skills/{name}/lineage")
def get_skill_lineage(name: str):
    """Get lineage tree for a skill."""
    ret = _get_retriever()
    tree = ret.get_lineage_tree(name)
    if not tree:
        raise HTTPException(status_code=404, detail=f"Skill '{name}' not found")
    return tree


@app.get("/api/skills/{name}/references")
def get_references(name: str):
    """Get skills referenced by this skill and skills that reference it."""
    import re

    session = get_session()
    try:
        skill = session.query(Skill).filter_by(id=name).first()
        if not skill:
            raise HTTPException(status_code=404, detail=f"Skill '{name}' not found")

        mgr = _get_manager()

        # --- Outgoing references (1 file read) ---
        try:
            data = mgr.read_skill(name)
            raw_body = data.get("body", "")
        except FileNotFoundError:
            raw_body = ""

        all_skill_ids = {s.id for s in session.query(Skill.id).all()}

        bracket_refs = set(re.findall(r"\[\[([a-z0-9-]+)\]\]", raw_body))

        plain_refs = set()
        for sid in all_skill_ids:
            if sid != name and re.search(r"\b" + re.escape(sid) + r"\b", raw_body, re.IGNORECASE):
                plain_refs.add(sid)

        outgoing = sorted(bracket_refs | plain_refs)

        # --- Incoming references (DB-based: check metadata.references column) ---
        # Avoid O(N) file reads by using the DB-stored references field first,
        # then only scan bodies of candidates for plain/bracket references.
        incoming = []

        # Strategy 1: Check metadata.references in other skills' frontmatter
        all_skills = session.query(Skill).all()
        for other in all_skills:
            if other.id == name:
                continue
            other_tags = other.get_tags()  # just to check references stored in DB
            # Check if this skill's description or tags mention the target
            # (lightweight DB-only check before expensive file reads)
            other_refs = []
            try:
                fm = mgr.read_frontmatter(other.id)
                meta = fm.get("metadata", {}) or {}
                other_refs = meta.get("references", [])
                if isinstance(other_refs, list):
                    if name in [str(r) for r in other_refs]:
                        incoming.append(other.id)
                        continue
            except (FileNotFoundError, Exception):
                pass

            # Strategy 2: Only if DB check didn't match, scan body for [[name]]
            try:
                other_body = mgr.read_body(other.id)
                if f"[[{name}]]" in other_body:
                    incoming.append(other.id)
                elif re.search(r"\b" + re.escape(name) + r"\b", other_body, re.IGNORECASE):
                    # Verify it's a genuine reference (at least 3 chars of context)
                    incoming.append(other.id)
            except (FileNotFoundError, Exception):
                pass

        return {
            "skill_name": name,
            "outgoing_references": outgoing,
            "incoming_references": sorted(incoming),
        }
    finally:
        session.close()


def bracket_refs_pattern_match(text: str, target: str) -> set:
    """Check if text contains a [[target]] bracket reference."""
    import re
    matches = re.findall(r"\[\[([a-z0-9-]+)\]\]", text)
    return set(matches)


def plain_refs_match(text: str, target: str, all_skill_ids: set) -> bool:
    """Check if text contains a plain reference to the target skill."""
    import re
    return bool(re.search(r"\b" + re.escape(target) + r"\b", text, re.IGNORECASE))


@app.patch("/api/skills/{name}/deprecate")
def deprecate_skill(name: str, req: DeprecateRequest):
    """Deactivate a skill."""
    session = get_session()
    try:
        skill = session.query(Skill).filter_by(id=name).first()
        if not skill:
            raise HTTPException(status_code=404, detail=f"Skill '{name}' not found")

        skill.is_active = False
        skill.last_modified_at = datetime.now(timezone.utc)
        session.commit()

        # Invalidate BM25 index (active skill list changed)
        _get_retriever().invalidate_all()

        return {"status": "deprecated", "name": name, "reason": req.reason}
    except HTTPException:
        raise
    except Exception as e:
        session.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        session.close()


@app.post("/api/skills/{name}/extend-ttl")
def extend_ttl(name: str, req: ExtendTTLRequest):
    """Extend TTL for a skill."""
    session = get_session()
    try:
        skill = session.query(Skill).filter_by(id=name).first()
        if not skill:
            raise HTTPException(status_code=404, detail=f"Skill '{name}' not found")

        skill.ttl_days = (skill.ttl_days or 0) + req.additional_days
        skill.compute_expires_at()
        skill.last_modified_at = datetime.now(timezone.utc)
        session.commit()

        return {
            "status": "extended",
            "name": name,
            "new_ttl_days": skill.ttl_days,
            "new_expires_at": skill.expires_at.isoformat() if skill.expires_at else None,
        }
    except HTTPException:
        raise
    except Exception as e:
        session.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        session.close()


@app.post("/api/skills")
def create_skill(req: CreateSkillRequest):
    """Create a new skill via dashboard."""
    import json as _json
    from core.evolver import EvolutionEngine
    from core.models import Skill as SkillModel

    name = req.name.strip().lower()
    if not name or not SkillModel.validate_name(name):
        raise HTTPException(status_code=400, detail=f"Invalid name '{name}'. Must be kebab-case, 2-64 chars.")

    if not req.description.strip():
        raise HTTPException(status_code=400, detail="Description is required.")
    if not req.body.strip():
        raise HTTPException(status_code=400, detail="Body is required.")

    session = get_session()
    try:
        existing = session.query(Skill).filter_by(id=name).first()
        if existing:
            raise HTTPException(status_code=409, detail=f"Skill '{name}' already exists.")

        mgr = _get_manager()
        ret = _get_retriever()
        engine = EvolutionEngine(
            session=session,
            manager=mgr,
            on_embedding_cache_clear=ret.clear_cache,
            on_embedding_compute=ret.compute_and_cache_embedding,
        )

        parsed_tags = [t.strip().lower() for t in req.tags.split(",") if t.strip()] if req.tags else []
        ttl = int(req.ttl_days) if req.ttl_days and int(req.ttl_days) > 0 else None

        skill = engine.archive(
            name=name,
            description=req.description.strip(),
            body=req.body,
            skill_type=req.skill_type,
            repo_name=req.repo_name or "global",
            tags=parsed_tags if parsed_tags else None,
            ttl_days=ttl,
        )
        session.commit()
        ret.flush_cache()
        return {"status": "created", "skill": skill.to_dict()}
    except HTTPException:
        raise
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        session.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        session.close()


@app.delete("/api/skills/{name}")
def delete_skill(name: str):
    """Delete a skill entirely (DB + filesystem)."""

    session = get_session()
    try:
        skill = session.query(Skill).filter_by(id=name).first()
        if not skill:
            raise HTTPException(status_code=404, detail=f"Skill '{name}' not found")

        # Delete changelog entries first
        session.query(SkillChangelog).filter(SkillChangelog.skill_id == name).delete()
        # Delete the skill
        session.delete(skill)
        session.commit()

        # Delete filesystem
        _get_manager().delete_skill_dir(name)

        # Invalidate BM25 index and embedding cache
        ret = _get_retriever()
        ret.clear_cache(name)
        ret.flush_cache()

        return {"status": "deleted", "name": name}
    except HTTPException:
        raise
    except Exception as e:
        session.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        session.close()


@app.put("/api/skills/{name}")
def update_skill(name: str, req: UpdateSkillRequest):
    """Update an existing skill's content."""

    session = get_session()
    try:
        skill = session.query(Skill).filter_by(id=name).first()
        if not skill:
            raise HTTPException(status_code=404, detail=f"Skill '{name}' not found")

        mgr = _get_manager()

        # Read current SKILL.md
        try:
            data = mgr.read_skill(name)
            current_frontmatter = data.get("frontmatter", {})
            current_body = data.get("body", "")
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail=f"SKILL.md not found for '{name}'")

        # Apply updates
        updated_frontmatter = dict(current_frontmatter)
        updated_body = current_body

        if req.description:
            updated_frontmatter["description"] = req.description
        if req.display_name:
            updated_frontmatter["display_name"] = req.display_name
        if req.tags:
            parsed_tags = [t.strip().lower() for t in req.tags.split(",") if t.strip()]
            updated_frontmatter["tags"] = parsed_tags
        if req.body:
            updated_body = req.body

        # Handle references: append to body as [[ref]] links if specified
        if req.references:
            ref_names = [r.strip() for r in req.references.split(",") if r.strip()]
            for ref_name in ref_names:
                ref_pattern = f"[[{ref_name}]]"
                if ref_pattern not in updated_body:
                    updated_body = updated_body.rstrip() + "\n\n## References\n"
                    for rn in ref_names:
                        updated_body += f"- [[{rn}]]\n"
                    break

        # Extract metadata from merged frontmatter for write_skill call
        metadata = updated_frontmatter.get("metadata", {}) or {}
        updated_description = updated_frontmatter.get("description", "") or ""
        display_name = updated_frontmatter.get("display_name", "") or ""
        skill_type = metadata.get("skill-type", "IMPLEMENTATION") if metadata else "IMPLEMENTATION"
        repo = metadata.get("repo", "global") if metadata else "global"
        version = int(metadata.get("version", 1)) if metadata else 1
        tags = metadata.get("tags", []) if metadata else None
        ttl_days = metadata.get("ttl-days") if metadata else None
        if ttl_days is not None:
            ttl_days = int(ttl_days)
        author = metadata.get("author") if metadata else None
        source = metadata.get("source") if metadata else None
        references = metadata.get("references") if metadata else None

        # Write updated skill file with correct write_skill signature
        mgr.write_skill(
            skill_name=name,
            description=updated_description,
            body=updated_body,
            display_name=display_name,
            skill_type=skill_type,
            repo=repo,
            version=version,
            tags=tags,
            ttl_days=ttl_days,
            author=author,
            source=source,
            references=references,
        )

        # Update DB fields
        if req.description:
            skill.description = req.description
        if req.display_name:
            skill.display_name = req.display_name
        if req.tags:
            parsed_tags = [t.strip().lower() for t in req.tags.split(",") if t.strip()]
            skill.set_tags(parsed_tags)

        skill.last_modified_at = datetime.now(timezone.utc)
        session.commit()

        # Invalidate BM25 index and embedding cache (content changed)
        ret = _get_retriever()
        ret.clear_cache(name)
        ret.flush_cache()

        return {
            "status": "updated",
            "name": name,
            "updated_fields": {
                k: v for k, v in {
                    "description": req.description,
                    "display_name": req.display_name,
                    "tags": req.tags,
                    "body": "(updated)" if req.body else None,
                    "references": req.references,
                }.items() if v
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        session.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        session.close()


@app.post("/api/search")
def search_skills(request: SearchRequest):
    """Search skills using hybrid BM25+semantic+RRF search.

    Request body:
        query: str (required)
        repo_scope: str = "all"
        current_repo: str = ""
        tags_filter: str = ""
        top_k: int = 5
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query parameter is required.")

    ret = _get_retriever()

    # Parse tags filter
    tags = [t.strip().lower() for t in request.tags_filter.split(",") if t.strip()] if request.tags_filter else None

    try:
        results = ret.search(
            query=request.query.strip(),
            top_k=request.top_k,
            repo_scope=request.repo_scope,
            current_repo=request.current_repo if request.current_repo else None,
            tags_filter=tags,
        )

        # CRITICAL FIX: Properly serialize Skill objects to dicts.
        # Raw SQLAlchemy objects cause jsonable_encoder to traverse lazy
        # relationships (e.g. changelog), which triggers DB queries on closed
        # sessions → request hangs indefinitely.
        serializable = []
        for r in results:
            if isinstance(r, dict):
                skill_obj = r.get("skill")
                if skill_obj is not None and hasattr(skill_obj, "to_dict"):
                    serializable.append({
                        "skill": skill_obj.to_dict(),
                        "rrf_score": r.get("rrf_score", 0),
                    })
                else:
                    serializable.append(r)
            elif hasattr(r, "to_dict"):
                serializable.append(r.to_dict())
            else:
                serializable.append({"result": str(r)})

        return {
            "query": request.query.strip(),
            "total": len(serializable),
            "results": serializable,
        }
    except Exception as e:
        logger.error(f"Search failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@app.get("/api/export")
def export_skills(active_only: bool = Query(True, description="Only active skills"), repo: str = Query("", description="Filter by repo")):
    """Export skills as JSON for download."""
    session = get_session()
    try:
        query = session.query(Skill)

        if active_only:
            query = query.filter(Skill.is_active == True)

        if repo:
            query = query.filter(Skill.repo_name == repo)

        skills = query.order_by(Skill.created_at.desc()).all()

        export_data = []
        for skill in skills:
            export_data.append({
                **skill.to_dict(),
            })

        return JSONResponse(
            content={
                "exported_at": datetime.now(timezone.utc).isoformat(),
                "total": len(export_data),
                "filters": {
                    "active_only": active_only,
                    "repo": repo,
                },
                "skills": export_data,
            },
            media_type="application/json",
            headers={
                "Content-Disposition": "attachment; filename=skills-export.json"
            },
        )
    finally:
        session.close()


@app.get("/api/stats")
def get_stats():
    """Dashboard statistics."""
    session = get_session()
    try:
        now = datetime.now(timezone.utc)

        total = session.query(func.count(Skill.id)).scalar() or 0
        active = session.query(func.count(Skill.id)).filter(Skill.is_active == True).scalar() or 0
        inactive = total - active
        expired = session.query(func.count(Skill.id)).filter(
            Skill.is_active == True,
            Skill.expires_at != None,
            Skill.expires_at < now,
        ).scalar() or 0

        repos = session.query(Skill.repo_name, func.count(Skill.id)).group_by(Skill.repo_name).all()
        types = session.query(Skill.skill_type, func.count(Skill.id)).group_by(Skill.skill_type).all()

        # Top used skills
        top_used = session.query(Skill).filter(Skill.is_active == True).order_by(
            Skill.use_count.desc().nullslast()
        ).limit(10).all()

        # Recent skills
        recent = session.query(Skill).order_by(Skill.created_at.desc()).limit(5).all()

        return {
            "total": total,
            "active": active,
            "inactive": inactive,
            "expired": expired,
            "repos": [{"name": r[0], "count": r[1]} for r in repos],
            "types": [{"type": t[0], "count": t[1]} for t in types],
            "top_used": [s.to_dict() for s in top_used],
            "recent": [s.to_dict() for s in recent],
        }
    finally:
        session.close()


@app.get("/api/repos")
def list_repos():
    """List all repository names."""
    session = get_session()
    try:
        repos = session.query(Skill.repo_name, func.count(Skill.id)).group_by(
            Skill.repo_name
        ).order_by(func.count(Skill.id).desc()).all()
        return [{"name": r[0], "skill_count": r[1]} for r in repos]
    finally:
        session.close()


@app.get("/api/health")
def health_check():
    """Health check endpoint."""
    try:
        session = get_session()
        try:
            # Quick DB connectivity check
            session.query(func.count(Skill.id)).scalar()
            db_ok = True
            db_error = None
        except Exception as e:
            db_ok = False
            db_error = str(e)
        finally:
            session.close()

        # Workspace check
        workspace_ok = os.path.isdir(WORKSPACE_PATH)

        status = "ok" if (db_ok and workspace_ok) else "degraded"
        issues = []
        if not db_ok:
            issues.append(f"database: {db_error}")
        if not workspace_ok:
            issues.append(f"workspace not found: {WORKSPACE_PATH}")

        return {
            "status": status,
            "version": "2.1",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "checks": {
                "database": db_ok,
                "workspace": workspace_ok,
            },
            "issues": issues,
        }
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "version": "2.1",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "error": str(e),
            },
        )


# ---------------------------------------------------------------------------
# Analytics Endpoints — singleton instance for this process
# ---------------------------------------------------------------------------

_analytics: SkillsAnalytics | None = None


def _get_analytics() -> SkillsAnalytics:
    """Lazily create the analytics engine singleton (thread-safe)."""
    global _analytics
    if _analytics is None:
        with _singleton_lock:
            if _analytics is None:
                _analytics = SkillsAnalytics(WORKSPACE_PATH)
                logger.info("SkillsAnalytics singleton created")
    return _analytics


@app.get("/api/analytics/summary")
def analytics_summary():
    """Get overall usage statistics."""
    return _get_analytics().get_usage_summary()


@app.get("/api/analytics/trending")
def analytics_trending(days: int = Query(7, description="Look-back window in days"), limit: int = Query(10, description="Max results")):
    """Get trending skills (most recently used)."""
    return _get_analytics().get_trending_skills(days=days, limit=limit)


@app.get("/api/analytics/stale")
def analytics_stale(days: int = Query(30, description="Stale threshold in days"), limit: int = Query(10, description="Max results")):
    """Get stale skills (not used recently)."""
    return _get_analytics().get_stale_skills(days=days, limit=limit)


@app.get("/api/analytics/types")
def analytics_types():
    """Get skill type distribution."""
    return _get_analytics().get_type_distribution()


@app.get("/api/analytics/tags")
def analytics_tags(limit: int = Query(50, description="Max number of tags")):
    """Get tag frequency data for tag cloud."""
    return _get_analytics().get_tag_cloud(limit=limit)


@app.get("/api/analytics/activity")
def analytics_activity(limit: int = Query(20, description="Max changelog entries")):
    """Get recent changelog entries."""
    return _get_analytics().get_recent_activity(limit=limit)


@app.get("/api/analytics/network")
def analytics_network():
    """Get skill relationship graph for visualization."""
    return _get_analytics().get_skill_network()


@app.get("/api/analytics/gaps")
def analytics_gaps():
    """Identify potential coverage gaps."""
    return _get_analytics().get_coverage_gaps()


# ---------------------------------------------------------------------------
# Version Diff Endpoint
# ---------------------------------------------------------------------------

@app.get("/api/skills/{name}/diff")
def get_skill_diff(name: str, v1: str = Query("1", description="First version"), v2: str = Query("current", description="Second version")):
    """Get unified diff between two versions of a skill."""
    try:
        result = _get_manager().get_version_diff(name, v1, v2)
        return result
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Serve React UI
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def serve_ui():
    """Serve the React dashboard UI."""
    ui_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ui", "skills-lab-ui.jsx")
    if not os.path.exists(ui_path):
        return "<h1>Skills Lab Dashboard</h1><p>UI file not found. Place skills-lab-ui.jsx in ui/ directory.</p>"

    with open(ui_path, "r", encoding="utf-8") as f:
        jsx_content = f.read()

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Skills Lab Dashboard</title>
    <script src="https://unpkg.com/react@18/umd/react.production.min.js" crossorigin></script>
    <script src="https://unpkg.com/react-dom@18/umd/react-dom.production.min.js" crossorigin></script>
    <script src="https://unpkg.com/@babel/standalone/babel.min.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #0f172a; color: #e2e8f0; }}
        .container {{ max-width: 1400px; margin: 0 auto; padding: 20px; }}
        header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 24px; padding: 16px; background: #1e293b; border-radius: 12px; }}
        header h1 {{ font-size: 20px; color: #38bdf8; }}
        .badge {{ display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 600; margin: 0 2px; }}
        .badge-active {{ background: #065f46; color: #6ee7b7; }}
        .badge-inactive {{ background: #7f1d1d; color: #fca5a5; }}
        .badge-expired {{ background: #78350f; color: #fcd34d; }}
        .badge-rule {{ background: #1e3a5f; color: #93c5fd; }}
        .badge-troubleshooting {{ background: #3b1f2b; color: #f9a8d4; }}
        .badge-implementation {{ background: #1a3329; color: #86efac; }}
        .badge-workflow {{ background: #2d1b4e; color: #c4b5fd; }}
        .badge-architecture {{ background: #3b2f1a; color: #fde68a; }}
        .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; margin-bottom: 24px; }}
        .stat-card {{ background: #1e293b; padding: 16px; border-radius: 10px; text-align: center; }}
        .stat-card .number {{ font-size: 28px; font-weight: 700; color: #38bdf8; }}
        .stat-card .label {{ font-size: 12px; color: #94a3b8; margin-top: 4px; }}
        .filters {{ display: flex; gap: 12px; margin-bottom: 20px; flex-wrap: wrap; align-items: center; }}
        .filters input, .filters select {{ padding: 8px 12px; border: 1px solid #334155; border-radius: 6px; background: #1e293b; color: #e2e8f0; font-size: 14px; }}
        .filters input:focus, .filters select:focus {{ outline: none; border-color: #38bdf8; }}
        .tag {{ display: inline-block; padding: 1px 6px; background: #334155; border-radius: 4px; font-size: 11px; color: #94a3b8; margin: 1px; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ text-align: left; padding: 10px 12px; border-bottom: 1px solid #1e293b; font-size: 13px; }}
        th {{ color: #94a3b8; font-weight: 600; font-size: 12px; text-transform: uppercase; letter-spacing: 0.5px; }}
        tr:hover {{ background: #1e293b; }}
        .skill-name {{ color: #38bdf8; cursor: pointer; font-weight: 500; }}
        .skill-name:hover {{ text-decoration: underline; }}
        .modal-overlay {{ position: fixed; top: 0; left: 0; right: 0; bottom: 0; background: rgba(0,0,0,0.7); display: flex; justify-content: center; align-items: center; z-index: 1000; }}
        .modal {{ background: #1e293b; border-radius: 12px; padding: 24px; max-width: 850px; width: 90%; max-height: 85vh; overflow-y: auto; border: 1px solid #334155; }}
        .modal h2 {{ margin-bottom: 16px; color: #38bdf8; font-size: 18px; }}
        .modal h3 {{ font-size: 14px; color: #94a3b8; margin: 16px 0 8px; }}
        .modal pre {{ background: #0f172a; padding: 16px; border-radius: 8px; overflow-x: auto; font-size: 13px; line-height: 1.5; white-space: pre-wrap; word-break: break-word; }}
        .modal .close {{ float: right; cursor: pointer; font-size: 24px; color: #94a3b8; line-height: 1; }}
        .modal .close:hover {{ color: #fff; }}
        .meta-line {{ font-size: 13px; color: #94a3b8; margin-bottom: 8px; }}
        .description {{ font-size: 13px; margin-bottom: 12px; color: #cbd5e1; }}
        .empty {{ text-align: center; padding: 40px; color: #64748b; }}
        button {{ padding: 6px 14px; border: 1px solid #334155; border-radius: 6px; background: #334155; color: #e2e8f0; cursor: pointer; font-size: 13px; }}
        button:hover {{ background: #475569; }}
        button.danger {{ border-color: #dc2626; color: #fca5a5; }}
        button.danger:hover {{ background: #7f1d1d; }}
        button.btn-primary {{ background: #0ea5e9; border-color: #0ea5e9; color: #fff; font-weight: 600; }}
        button.btn-primary:hover {{ background: #0284c7; }}
        .error-text {{ color: #fca5a5; font-size: 13px; margin-bottom: 12px; }}
        .version-history {{ background: #0f172a; border-radius: 8px; padding: 12px 16px; margin-bottom: 16px; }}
        .version-entry {{ display: flex; align-items: center; gap: 8px; padding: 4px 0; font-size: 12px; }}
        .version-badge {{ background: #334155; padding: 2px 8px; border-radius: 4px; color: #38bdf8; font-weight: 600; white-space: nowrap; }}
        .version-reason {{ color: #cbd5e1; flex: 1; }}
        .version-source {{ color: #64748b; font-style: italic; }}
        .skill-md-body {{ background: #0f172a; padding: 16px; border-radius: 8px; overflow-x: auto; font-size: 13px; line-height: 1.6; }}
        .skill-md-body h1 {{ color: #38bdf8; font-size: 18px; margin: 16px 0 8px; }}
        .skill-md-body h2 {{ color: #38bdf8; font-size: 16px; margin: 14px 0 6px; }}
        .skill-md-body h3 {{ color: #7dd3fc; font-size: 14px; margin: 12px 0 4px; }}
        .skill-md-body code {{ background: #334155; padding: 1px 4px; border-radius: 3px; font-size: 12px; color: #86efac; }}
        .skill-md-body pre {{ background: #0b1120; padding: 12px; border-radius: 6px; margin: 8px 0; overflow-x: auto; }}
        .skill-md-body pre code {{ background: none; padding: 0; color: #e2e8f0; }}
        .skill-md-body strong {{ color: #f0f9ff; }}
        .skill-md-body blockquote {{ border-left: 3px solid #38bdf8; padding-left: 12px; margin: 8px 0; color: #94a3b8; }}
        .skill-md-body ul {{ padding-left: 20px; margin: 4px 0; }}
        .skill-md-body li {{ margin: 2px 0; }}
        .form-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 12px; }}
        .form-grid label {{ display: flex; flex-direction: column; gap: 4px; font-size: 13px; color: #94a3b8; }}
        .full-width {{ grid-column: 1 / -1; margin-bottom: 12px; }}
        .full-width label {{ display: flex; flex-direction: column; gap: 4px; font-size: 13px; color: #94a3b8; }}
        .form-grid input, .form-grid select, .full-width input, .full-width textarea {{
            padding: 8px 12px; border: 1px solid #334155; border-radius: 6px; background: #0f172a; color: #e2e8f0; font-size: 14px;
        }}
        .full-width textarea {{ font-family: 'Fira Code', 'Consolas', monospace; resize: vertical; }}
        .form-actions {{ display: flex; justify-content: flex-end; gap: 8px; margin-top: 16px; }}
        button.btn-secondary {{ background: #1e293b; border-color: #475569; color: #cbd5e1; font-weight: 500; }}
        button.btn-secondary:hover {{ background: #334155; border-color: #64748b; }}
        button.btn-secondary:disabled {{ opacity: 0.5; cursor: not-allowed; }}
        button.btn-edit {{ background: #1e3a5f; border-color: #2563eb; color: #93c5fd; padding: 2px 8px; font-size: 14px; }}
        button.btn-edit:hover {{ background: #1e40af; }}
        /* Search bar */
        .search-bar-container {{ margin-bottom: 16px; }}
        .search-bar {{ display: flex; align-items: center; gap: 10px; padding: 0 16px; background: #1e293b; border: 1px solid #334155; border-radius: 10px; transition: border-color 0.2s; }}
        .search-bar:focus-within {{ border-color: #38bdf8; box-shadow: 0 0 0 2px rgba(56,189,248,0.15); }}
        .search-icon {{ font-size: 16px; color: #64748b; }}
        .search-input {{ flex: 1; background: transparent; border: none; outline: none; color: #e2e8f0; font-size: 15px; padding: 12px 0; }}
        .search-input::placeholder {{ color: #64748b; }}
        .search-clear {{ background: none; border: none; color: #64748b; cursor: pointer; font-size: 16px; padding: 4px 8px; border-radius: 4px; }}
        .search-clear:hover {{ color: #fca5a5; background: rgba(252,165,165,0.1); }}
        .search-spinner {{ font-size: 16px; animation: spin 1s linear infinite; }}
        @keyframes spin {{ from {{ transform: rotate(0deg); }} to {{ transform: rotate(360deg); }} }}
        .search-status {{ display: flex; align-items: center; justify-content: space-between; padding: 6px 16px; margin-top: 6px; }}
        .search-result-count {{ font-size: 13px; color: #38bdf8; }}
        .search-clear-btn {{ background: none; border: 1px solid #334155; color: #94a3b8; padding: 2px 10px; border-radius: 4px; cursor: pointer; font-size: 12px; }}
        .search-clear-btn:hover {{ border-color: #64748b; color: #e2e8f0; }}
        /* Sortable table headers */
        th.sortable {{ cursor: pointer; user-select: none; }}
        th.sortable:hover {{ color: #38bdf8; }}
        .sort-arrow {{ font-size: 10px; margin-left: 4px; opacity: 0.5; }}
        th.sortable:hover .sort-arrow {{ opacity: 1; }}
        /* Wide modal variant */
        .modal-wide {{ max-width: 1000px; }}
        /* References display */
        .references-section-wrapper {{ margin-top: 8px; }}
        .references-section {{ background: #0f172a; border-radius: 8px; padding: 16px; margin-bottom: 16px; }}
        .ref-group {{ margin-bottom: 12px; }}
        .ref-group:last-child {{ margin-bottom: 0; }}
        .ref-group h4 {{ font-size: 13px; color: #cbd5e1; margin-bottom: 8px; display: flex; align-items: center; gap: 6px; }}
        .ref-icon {{ color: #38bdf8; font-size: 14px; }}
        .ref-count {{ background: #334155; padding: 1px 6px; border-radius: 10px; font-size: 11px; color: #94a3b8; }}
        .ref-list {{ list-style: none; padding: 0; }}
        .ref-item {{ display: flex; align-items: center; gap: 8px; padding: 4px 0; font-size: 12px; }}
        .ref-item code {{ background: #334155; padding: 1px 6px; border-radius: 3px; color: #38bdf8; font-size: 11px; cursor: pointer; }}
        .ref-item code:hover {{ background: #2563eb; }}
        .ref-empty {{ font-size: 12px; color: #475569; font-style: italic; }}
    </style>
</head>
<body>
    <div class="container" id="root"></div>
    <script type="text/babel">{jsx_content}</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7788)
