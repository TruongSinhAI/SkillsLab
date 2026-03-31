"""
Skills Lab — Skill Exporter / Importer

Handles serialisation of skills to/from JSON format for backup, sharing,
and migration between workspaces.

Usage (programmatic)::

    from core.exporter import SkillExporter

    exporter = SkillExporter("/path/to/workspace")
    data = exporter.export_all()
    result = exporter.import_skills(data)

Usage (CLI)::

    skills-lab export [output.json]
    skills-lab import <input.json>
"""

import json
import os
from datetime import datetime, timezone
from typing import Any

from core.manager import SKILLManager
from core.models import Skill, get_session, init_db


class SkillExporter:
    """Handles export/import of skills to/from JSON format.

    Reads skill metadata from the SQLite database and body content from
    SKILL.md files on disk.  When importing, writes both the SKILL.md
    file and the corresponding database row (via ``EvolutionEngine``).

    Attributes:
        workspace_path: Root workspace directory path.
        manager: ``SKILLManager`` instance for SKILL.md I/O.
    """

    EXPORT_FORMAT_VERSION = "3.0.0"

    def __init__(self, workspace_path: str) -> None:
        """Initialize the exporter.

        Ensures the database is initialised so that ORM queries work,
        and creates a ``SKILLManager`` for reading SKILL.md body content.

        Args:
            workspace_path: Absolute or relative path to the workspace root.
        """
        self.workspace_path: str = workspace_path
        self.manager: SKILLManager = SKILLManager(workspace_path)
        init_db(workspace_path)

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _skill_to_export_dict(skill: Skill, body: str = "") -> dict[str, Any]:
        """Convert a single ``Skill`` ORM row into an export-friendly dict.

        The returned dict includes the skill body (when available) and a
        ``references`` key listing files found in the skill's
        ``references/`` subdirectory.

        Args:
            skill: A ``Skill`` ORM instance.
            body: Markdown body content from the SKILL.md file.

        Returns:
            A dictionary ready for JSON serialisation.
        """
        return {
            "name": skill.id,
            "description": skill.description,
            "body": body,
            "skill_type": skill.skill_type,
            "repo_name": skill.repo_name,
            "tags": skill.get_tags(),
            "version_number": skill.version_number,
            "display_name": skill.display_name,
            "created_at": (
                skill.created_at.isoformat() if skill.created_at else None
            ),
            "last_modified_at": (
                skill.last_modified_at.isoformat()
                if skill.last_modified_at
                else None
            ),
            "references": [],
            "is_active": skill.is_active,
        }

    @staticmethod
    def _get_references(manager: SKILLManager, skill_name: str) -> list[str]:
        """List reference files in a skill's ``references/`` subdirectory.

        Args:
            manager: The ``SKILLManager`` instance.
            skill_name: Kebab-case skill identifier.

        Returns:
            A list of filenames found under ``skills/{name}/references/``.
        """
        ref_dir = os.path.join(manager._skill_dir(skill_name), "references")
        if not os.path.isdir(ref_dir):
            return []
        return sorted(os.listdir(ref_dir))

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_all(
        self,
        output_path: str | None = None,
        active_only: bool = True,
    ) -> dict[str, Any]:
        """Export all skills to a JSON-serializable dict (or file).

        Queries the database for skills matching the ``active_only`` filter,
        reads each skill's body from its SKILL.md file, and assembles a
        structured dict with metadata.

        Args:
            output_path: If provided, write JSON to this file path.
                         Otherwise return the dict directly.
            active_only: Only include skills where ``is_active`` is True.

        Returns:
            Dict with ``"version"``, ``"exported_at"``, ``"count"``, and
            ``"skills"`` keys.  Each skill dict contains: name, description,
            body, skill_type, repo_name, tags, version_number, display_name,
            created_at, last_modified_at, references, is_active.
        """
        session = get_session()
        try:
            query = session.query(Skill)
            if active_only:
                query = query.filter(Skill.is_active.is_(True))
            skills = query.order_by(Skill.created_at).all()

            skill_dicts: list[dict[str, Any]] = []
            for skill in skills:
                # Read body from SKILL.md on disk
                body = ""
                try:
                    body = self.manager.read_body(skill.id)
                except (FileNotFoundError, OSError, ValueError):
                    pass  # body stays empty if SKILL.md is missing or unreadable

                export = self._skill_to_export_dict(skill, body=body)
                export["references"] = self._get_references(
                    self.manager, skill.id
                )
                skill_dicts.append(export)

            result: dict[str, Any] = {
                "version": self.EXPORT_FORMAT_VERSION,
                "exported_at": datetime.now(timezone.utc).isoformat(),
                "count": len(skill_dicts),
                "skills": skill_dicts,
            }

            if output_path:
                os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
                with open(output_path, "w", encoding="utf-8") as fh:
                    json.dump(result, fh, indent=2, ensure_ascii=False)

            return result
        finally:
            session.close()

    def export_skill(self, skill_name: str) -> dict | None:
        """Export a single skill as a dict.

        Args:
            skill_name: Kebab-case skill identifier.

        Returns:
            An export dict for the skill, or ``None`` if the skill does
            not exist in the database.
        """
        session = get_session()
        try:
            skill = session.query(Skill).filter_by(id=skill_name).first()
            if skill is None:
                return None

            body = ""
            try:
                body = self.manager.read_body(skill.id)
            except (FileNotFoundError, OSError, ValueError):
                pass

            export = self._skill_to_export_dict(skill, body=body)
            export["references"] = self._get_references(
                self.manager, skill.id
            )
            return export
        finally:
            session.close()

    # ------------------------------------------------------------------
    # Import
    # ------------------------------------------------------------------

    def import_skills(
        self,
        data: dict | str,
        overwrite: bool = False,
        skip_existing: bool = True,
    ) -> dict[str, int]:
        """Import skills from a JSON dict or file path.

        If *data* is a string and looks like a file path (ends with
        ``.json`` or the file exists on disk), it is read as JSON.
        Otherwise it is treated as a JSON string or already-parsed dict.

        Each skill is imported via :meth:`import_skill`.  Results are
        tallied and returned as a summary dict.

        Args:
            data: JSON dict, JSON string, or file path to import from.
            overwrite: When True, overwrite existing skills (DB + SKILL.md).
            skip_existing: When True, skip skills that already exist without
                           error (takes precedence over *overwrite*).

        Returns:
            Dict with ``"imported"``, ``"skipped"``, and ``"errors"`` counts.
        """
        # Resolve input data
        if isinstance(data, str):
            if os.path.isfile(data):
                with open(data, "r", encoding="utf-8") as fh:
                    payload = json.load(fh)
            else:
                payload = json.loads(data)
        else:
            payload = data

        skills_list = payload.get("skills", [])
        if isinstance(payload, list):
            skills_list = payload

        imported = 0
        skipped = 0
        errors = 0

        for skill_data in skills_list:
            if not isinstance(skill_data, dict):
                errors += 1
                continue
            try:
                result = self.import_skill(
                    skill_data, overwrite=overwrite, skip_existing=skip_existing
                )
                if result:
                    imported += 1
                else:
                    skipped += 1
            except Exception:
                errors += 1

        return {"imported": imported, "skipped": skipped, "errors": errors}

    def import_skill(
        self,
        skill_data: dict,
        overwrite: bool = False,
        skip_existing: bool = True,
    ) -> bool:
        """Import a single skill from a dict.

        Creates a new skill via ``EvolutionEngine.archive`` (which writes
        both the SKILL.md file and the DB row).  If the skill already
        exists, behaviour depends on the flags.

        Args:
            skill_data: Dict with at minimum ``name``, ``description``, and
                        ``body`` keys.  Optional keys: ``skill_type``,
                        ``repo_name``, ``tags``, ``version_number``,
                        ``display_name``, ``is_active``.
            overwrite: Overwrite an existing skill (re-creates DB row and
                       SKILL.md file).
            skip_existing: Skip if skill already exists (no-op, return
                           ``False``).

        Returns:
            ``True`` if the skill was imported, ``False`` if it was
            skipped.

        Raises:
            ValueError: If required fields are missing.
        """
        from core.evolver import EvolutionEngine

        name = skill_data.get("name", "").strip()
        description = skill_data.get("description", "").strip()
        body = skill_data.get("body", "").strip()

        if not name:
            raise ValueError("Skill 'name' is required for import")
        if not description:
            raise ValueError("Skill 'description' is required for import")
        if not body:
            raise ValueError("Skill 'body' is required for import")

        session = get_session()
        try:
            existing = session.query(Skill).filter_by(id=name).first()

            if existing is not None:
                if skip_existing:
                    return False
                if not overwrite:
                    return False
                # Delete the existing skill so we can re-create it
                session.delete(existing)
                session.flush()

            mgr = self.manager
            engine = EvolutionEngine(session=session, manager=mgr)

            engine.archive(
                name=name,
                description=description,
                body=body,
                skill_type=skill_data.get("skill_type", "IMPLEMENTATION"),
                repo_name=skill_data.get("repo_name", "global"),
                display_name=skill_data.get("display_name", ""),
                tags=skill_data.get("tags"),
            )

            session.commit()
            return True
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    # ------------------------------------------------------------------
    # Search (convenience)
    # ------------------------------------------------------------------

    def search_skills(
        self,
        query: str = "",
        repo: str | None = None,
        skill_type: str | None = None,
        tags: list[str] | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Search skills by text query, repo, type, or tags.

        Performs simple substring matching on the skill name and description.
        Additional filters for repo, type, and tags narrow the result set.

        Args:
            query: Free-text search term (matched against name + description).
            repo: Filter by ``repo_name``.
            skill_type: Filter by ``skill_type``.
            tags: Filter skills that have at least one of the given tags.
            limit: Maximum number of results to return.

        Returns:
            A list of skill export dicts matching the criteria.
        """
        session = get_session()
        try:
            results = session.query(Skill).filter(Skill.is_active.is_(True))

            if repo:
                results = results.filter(Skill.repo_name == repo)
            if skill_type:
                results = results.filter(Skill.skill_type == skill_type.upper())
            if tags:
                # Filter in Python for JSON-encoded tags column
                tag_set = {t.lower() for t in tags}

            if query:
                q_lower = query.lower()
                results = results.filter(
                    (Skill.id.ilike(f"%{q_lower}%"))
                    | (Skill.description.ilike(f"%{q_lower}%"))
                )

            results = results.order_by(Skill.use_count.desc()).limit(limit)
            skills = results.all()

            # Apply tag filtering in Python (tags are JSON in the DB)
            out: list[dict[str, Any]] = []
            for skill in skills:
                if tags:
                    skill_tags = {t.lower() for t in skill.get_tags()}
                    if not skill_tags.intersection(tag_set):
                        continue

                body = ""
                try:
                    body = self.manager.read_body(skill.id)
                except Exception:
                    pass

                out.append(self._skill_to_export_dict(skill, body=body))

            return out
        finally:
            session.close()


# ------------------------------------------------------------------
# Standalone utility functions
# ------------------------------------------------------------------


def get_workspace_stats(workspace_path: str) -> dict[str, Any]:
    """Return aggregate statistics for a workspace.

    Counts total, active, inactive, and expired skills.  Also computes
    per-repo counts, per-type counts, top most-used skills, and the
    most recently created skills.

    Args:
        workspace_path: Absolute or relative path to the workspace root.

    Returns:
        A dict with the following top-level keys:

        - ``total``: Total number of skills.
        - ``active``: Skills where ``is_active`` is True.
        - ``inactive``: Skills where ``is_active`` is False.
        - ``expired``: Skills where ``is_expired()`` returns True.
        - ``repos``: Dict mapping repo name to skill count.
        - ``types``: Dict mapping skill type to skill count.
        - ``top_used``: List of top-10 most-used skills (name + use_count).
        - ``recent``: List of 10 most recently created skills.
    """
    init_db(workspace_path)
    session = get_session()

    try:
        all_skills = session.query(Skill).all()

        total = len(all_skills)
        active = sum(1 for s in all_skills if s.is_active)
        inactive = total - active
        expired = sum(1 for s in all_skills if s.is_expired())

        # Per-repo counts
        repos: dict[str, int] = {}
        for s in all_skills:
            repos[s.repo_name] = repos.get(s.repo_name, 0) + 1

        # Per-type counts
        types: dict[str, int] = {}
        for s in all_skills:
            types[s.skill_type] = types.get(s.skill_type, 0) + 1

        # Top used skills (sorted descending by use_count, take top 10)
        top_used = sorted(all_skills, key=lambda s: s.use_count, reverse=True)[
            :10
        ]
        top_used_list = [
            {"name": s.id, "use_count": s.use_count} for s in top_used
        ]

        # Most recently created skills (take top 10)
        recent = sorted(
            all_skills,
            key=lambda s: s.created_at or datetime.min.replace(tzinfo=timezone.utc),
            reverse=True,
        )[:10]
        recent_list = [
            {
                "name": s.id,
                "display_name": s.display_name,
                "created_at": (
                    s.created_at.isoformat() if s.created_at else None
                ),
            }
            for s in recent
        ]

        return {
            "total": total,
            "active": active,
            "inactive": inactive,
            "expired": expired,
            "repos": repos,
            "types": types,
            "top_used": top_used_list,
            "recent": recent_list,
        }
    finally:
        session.close()
