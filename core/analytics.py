"""
Skills Lab — Analytics Module

Comprehensive analytics engine for the Skills Lab workspace. Provides insights
about skill usage patterns, trending/stale skills, coverage gaps, skill type
and repository distributions, tag clouds, version distributions, recent activity
feeds, orphan detection, and skill relationship networks.

Data sources:
    - **Database** (SQLAlchemy): Skills, SkillChangelog tables — used for
      usage statistics, trending/stale detection, distributions, activity feed,
      coverage gaps, and skill network derived-from edges.
    - **Filesystem** (SKILLManager): SKILL.md files — used for reference
      parsing, orphan detection, and skill network reference edges.
"""

import logging
from collections import Counter
from datetime import datetime, timedelta, timezone

from sqlalchemy import Integer, func

from core.manager import SKILLManager, SKILLParseError
from core.models import Skill, SkillChangelog, get_session

logger = logging.getLogger(__name__)


class SkillsAnalytics:
    """Comprehensive analytics engine for Skills Lab workspace.

    Provides insights about skill usage patterns, trending/stale skills,
    coverage gaps, and skill relationships.
    """

    def __init__(self, workspace_path: str):
        """Initialize the analytics engine.

        Args:
            workspace_path: Path to the workspace root directory that contains
                the ``brain.db`` database and the ``skills/`` directory.
        """
        self.workspace_path = workspace_path
        self.manager = SKILLManager(workspace_path)

    # ------------------------------------------------------------------
    # 1. Overall usage summary
    # ------------------------------------------------------------------

    def get_usage_summary(self) -> dict:
        """Overall usage statistics.

        Returns:
            {
                "total_skills": int,
                "active_skills": int,
                "inactive_skills": int,
                "expired_skills": int,
                "total_searches": int (estimated from use_count),
                "avg_version": float,
                "most_used_type": str,
                "coverage_score": float (0-100, based on tag/repo diversity),
            }
        """
        session = get_session()
        try:
            now = datetime.now(timezone.utc)

            # Aggregate counts
            total = session.query(func.count(Skill.id)).scalar() or 0
            active = (
                session.query(func.count(Skill.id))
                .filter(Skill.is_active.is_(True))
                .scalar()
                or 0
            )
            inactive = total - active

            # Expired: skills whose expires_at is in the past (or naive comparison)
            all_skills = session.query(Skill).all()
            expired = sum(1 for s in all_skills if s.is_expired())

            # Total searches (sum of use_count)
            total_searches = (
                session.query(func.coalesce(func.sum(Skill.use_count), 0)).scalar()
                or 0
            )

            # Average version
            avg_version = (
                session.query(func.avg(Skill.version_number)).scalar() or 0.0
            )

            # Most-used type
            type_counts: dict[str, int] = Counter()
            for s in all_skills:
                type_counts[s.skill_type or "UNKNOWN"] += 1
            most_used_type = (
                max(type_counts, key=type_counts.get) if type_counts else "N/A"
            )

            # Coverage score (0–100) based on tag diversity and repo diversity
            coverage_score = self._compute_coverage_score(all_skills)

            return {
                "total_skills": total,
                "active_skills": active,
                "inactive_skills": inactive,
                "expired_skills": expired,
                "total_searches": total_searches,
                "avg_version": round(float(avg_version), 2),
                "most_used_type": most_used_type,
                "coverage_score": round(coverage_score, 1),
            }
        finally:
            session.close()

    def _compute_coverage_score(self, all_skills: list[Skill]) -> float:
        """Compute a coverage score based on tag and repository diversity.

        The score is calculated as:
            - Tag diversity component (50%): ratio of unique tags to total skills,
              capped at 1.0.
            - Repo diversity component (50%): ratio of unique repos to total skills,
              capped at 1.0.

        If there are no skills the score is 0.

        Args:
            all_skills: List of all Skill ORM instances.

        Returns:
            A float between 0 and 100.
        """
        if not all_skills:
            return 0.0

        unique_tags: set[str] = set()
        unique_repos: set[str] = set()
        for s in all_skills:
            unique_tags.update(s.get_tags())
            unique_repos.add(s.repo_name or "unknown")

        total = len(all_skills)

        # Tag diversity: how many tags per skill on average vs. ideal
        tag_ratio = min(len(unique_tags) / max(total, 1), 1.0)
        # Repo diversity: inverse concentration — more repos = higher score
        repo_ratio = min(len(unique_repos) / max(total, 1), 1.0)

        # Blend the two components (equal weight)
        raw = (tag_ratio + repo_ratio) / 2.0
        return raw * 100.0

    # ------------------------------------------------------------------
    # 2. Trending skills
    # ------------------------------------------------------------------

    def get_trending_skills(self, days: int = 7, limit: int = 10) -> list[dict]:
        """Get trending skills (most recently used).

        Returns skills sorted by ``last_used_at`` DESC, only active skills.

        Args:
            days: Only consider skills used within this many days.  Set to ``0``
                to disable the time filter.
            limit: Maximum number of results to return.

        Returns:
            List of dicts with keys:
            ``{name, display_name, use_count, last_used_at, skill_type, tags}``
        """
        session = get_session()
        try:
            query = session.query(Skill).filter(Skill.is_active.is_(True))

            cutoff = datetime.now(timezone.utc) - timedelta(days=days)
            if days > 0:
                query = query.filter(Skill.last_used_at >= cutoff)

            query = query.order_by(Skill.last_used_at.desc().nullslast())
            skills = query.limit(limit).all()

            result = []
            for s in skills:
                result.append(
                    {
                        "name": s.id,
                        "display_name": s.display_name,
                        "use_count": s.use_count,
                        "last_used_at": (
                            s.last_used_at.isoformat() if s.last_used_at else None
                        ),
                        "skill_type": s.skill_type,
                        "tags": s.get_tags(),
                    }
                )
            return result
        finally:
            session.close()

    # ------------------------------------------------------------------
    # 3. Stale skills
    # ------------------------------------------------------------------

    def get_stale_skills(self, days: int = 30, limit: int = 10) -> list[dict]:
        """Get stale skills (not used recently but still active).

        Returns active skills where ``last_used_at`` is ``NULL`` or older
        than *days*.

        Args:
            days: Threshold in days. Skills not used within this window are
                considered stale.
            limit: Maximum number of results to return.

        Returns:
            List of dicts with keys:
            ``{name, display_name, last_used_at, created_at, days_since_used, skill_type}``
        """
        session = get_session()
        try:
            now = datetime.now(timezone.utc)
            cutoff = now - timedelta(days=days)

            query = session.query(Skill).filter(Skill.is_active.is_(True))

            # Stale: last_used_at IS NULL or older than cutoff
            query = query.filter(
                (Skill.last_used_at.is_(None)) | (Skill.last_used_at < cutoff)
            )

            # Order by most stale first (NULLs first, then oldest)
            query = query.order_by(Skill.last_used_at.asc().nullsfirst())
            skills = query.limit(limit).all()

            result = []
            for s in skills:
                if s.last_used_at:
                    # Handle both tz-aware and naive datetimes
                    used_at = s.last_used_at
                    if used_at.tzinfo is None:
                        delta = (now.replace(tzinfo=None) - used_at).days
                    else:
                        delta = (now - used_at).days
                    last_used_iso = s.last_used_at.isoformat()
                else:
                    # Never used — days since creation
                    created = s.created_at or now
                    if created.tzinfo is None:
                        delta = (now.replace(tzinfo=None) - created).days
                    else:
                        delta = (now - created).days
                    last_used_iso = None

                result.append(
                    {
                        "name": s.id,
                        "display_name": s.display_name,
                        "last_used_at": last_used_iso,
                        "created_at": (
                            s.created_at.isoformat() if s.created_at else None
                        ),
                        "days_since_used": max(delta, 0),
                        "skill_type": s.skill_type,
                    }
                )
            return result
        finally:
            session.close()

    # ------------------------------------------------------------------
    # 4. Type distribution
    # ------------------------------------------------------------------

    def get_type_distribution(self) -> list[dict]:
        """Get skill type distribution.

        Returns:
            ``[{type, count, percentage}]`` sorted by count descending.
        """
        session = get_session()
        try:
            rows = (
                session.query(
                    Skill.skill_type,
                    func.count(Skill.id).label("cnt"),
                )
                .group_by(Skill.skill_type)
                .order_by(func.count(Skill.id).desc())
                .all()
            )

            total = sum(r.cnt for r in rows) or 1
            return [
                {
                    "type": r.skill_type,
                    "count": r.cnt,
                    "percentage": round((r.cnt / total) * 100, 1),
                }
                for r in rows
            ]
        finally:
            session.close()

    # ------------------------------------------------------------------
    # 5. Repository distribution
    # ------------------------------------------------------------------

    def get_repo_distribution(self) -> list[dict]:
        """Get repository distribution.

        Returns:
            ``[{repo, count, percentage, active_count}]`` sorted by count
            descending.
        """
        session = get_session()
        try:
            rows = (
                session.query(
                    Skill.repo_name,
                    func.count(Skill.id).label("cnt"),
                    func.sum(
                        func.cast(Skill.is_active, Integer)  # type: ignore[arg-type]
                    ).label(
                        "active_cnt"
                    ),
                )
                .group_by(Skill.repo_name)
                .order_by(func.count(Skill.id).desc())
                .all()
            )

            total = sum(r.cnt for r in rows) or 1
            return [
                {
                    "repo": r.repo_name,
                    "count": r.cnt,
                    "percentage": round((r.cnt / total) * 100, 1),
                    "active_count": int(r.active_cnt or 0),
                }
                for r in rows
            ]
        finally:
            session.close()

    # ------------------------------------------------------------------
    # 6. Tag cloud
    # ------------------------------------------------------------------

    def get_tag_cloud(self, limit: int = 50) -> list[dict]:
        """Get tag frequency for tag cloud visualization.

        Parses the JSON-encoded ``tags`` column for every skill and counts
        occurrences of each individual tag.

        Args:
            limit: Maximum number of tags to return (most frequent first).

        Returns:
            ``[{tag, count}]`` sorted by count descending.
        """
        session = get_session()
        try:
            skills = session.query(Skill).all()

            tag_counter: Counter = Counter()
            for s in skills:
                for tag in s.get_tags():
                    tag_counter[tag] += 1

            return [
                {"tag": tag, "count": count}
                for tag, count in tag_counter.most_common(limit)
            ]
        finally:
            session.close()

    # ------------------------------------------------------------------
    # 7. Version distribution
    # ------------------------------------------------------------------

    def get_version_distribution(self) -> list[dict]:
        """Get version number distribution.

        Returns:
            ``[{version, count}]`` sorted by version descending.
        """
        session = get_session()
        try:
            rows = (
                session.query(
                    Skill.version_number,
                    func.count(Skill.id).label("cnt"),
                )
                .group_by(Skill.version_number)
                .order_by(Skill.version_number.desc())
                .all()
            )

            return [
                {"version": r.version_number, "count": r.cnt} for r in rows
            ]
        finally:
            session.close()

    # ------------------------------------------------------------------
    # 8. Recent activity (changelog)
    # ------------------------------------------------------------------

    def get_recent_activity(self, limit: int = 20) -> list[dict]:
        """Get recent changelog entries across all skills.

        Returns:
            ``[{skill_name, trigger, from_version, to_version, reason, created_at}]``
            sorted by ``created_at`` descending.
        """
        session = get_session()
        try:
            rows = (
                session.query(SkillChangelog)
                .order_by(SkillChangelog.created_at.desc())
                .limit(limit)
                .all()
            )

            return [
                {
                    "skill_name": r.skill_id,
                    "trigger": r.trigger,
                    "from_version": r.from_version,
                    "to_version": r.to_version,
                    "reason": r.reason,
                    "created_at": (
                        r.created_at.isoformat() if r.created_at else None
                    ),
                }
                for r in rows
            ]
        finally:
            session.close()

    # ------------------------------------------------------------------
    # 9. Orphan skills
    # ------------------------------------------------------------------

    def get_orphan_skills(self) -> list[dict]:
        """Get skills that have no references and are not referenced by others.

        Uses ``SKILLManager.get_references`` and ``find_referencing_skills``
        to determine whether a skill is connected to any other skill.

        Optimized: builds a single reference map in O(N) file reads instead of O(N²).
        """
        session = get_session()
        try:
            skills = session.query(Skill).all()

            # Build reference map in a single pass: skill_id → set of referenced skill_ids
            # This avoids calling find_referencing_skills() per skill (which does O(N) reads each)
            ref_map: dict[str, set[str]] = {}
            for s in skills:
                try:
                    refs = self.manager.get_references(s.id)
                    ref_map[s.id] = set(refs)
                except (FileNotFoundError, SKILLParseError):
                    ref_map[s.id] = set()

            # Build reverse map: skill_id → set of skills that reference it
            referenced_by: dict[str, set[str]] = {s.id: set() for s in skills}
            for src, targets in ref_map.items():
                for tgt in targets:
                    if tgt in referenced_by:
                        referenced_by[tgt].add(src)

            result = []
            for s in skills:
                has_outgoing = bool(ref_map.get(s.id))
                has_incoming = bool(referenced_by.get(s.id))
                if not has_outgoing and not has_incoming:
                    result.append(
                        {
                            "name": s.id,
                            "display_name": s.display_name,
                            "skill_type": s.skill_type,
                            "created_at": (
                                s.created_at.isoformat()
                                if s.created_at
                                else None
                            ),
                        }
                    )

            return result
        finally:
            session.close()

    # ------------------------------------------------------------------
    # 10. Skill network graph
    # ------------------------------------------------------------------

    def get_skill_network(self) -> dict:
        """Build a skill relationship graph for visualization.

        Combines two types of edges:
            - **references**: from ``SKILLManager.get_references`` (file-based).
            - **derived_from**: from ``SkillChangelog.source_skill_id`` where
              trigger is ``DERIVE`` (database-based).

        Returns:
            {
                "nodes": [{id, name, type, group, use_count}],
                "edges": [{source, target, type}],
            }
            where *type* is ``"references"`` or ``"derived_from"``.
        """
        session = get_session()
        try:
            skills = session.query(Skill).all()

            # Build nodes
            nodes = []
            for s in skills:
                nodes.append(
                    {
                        "id": s.id,
                        "name": s.display_name,
                        "type": s.skill_type,
                        "group": s.repo_name or "global",
                        "use_count": s.use_count,
                    }
                )

            # Collect all skill names for validation
            skill_ids = {s.id for s in skills}

            # Build edges — references (file-based)
            edges: list[dict] = []
            seen_edges: set[tuple[str, str, str]] = set()

            for s in skills:
                try:
                    refs = self.manager.get_references(s.id)
                except (FileNotFoundError, SKILLParseError):
                    refs = []

                for ref in refs:
                    if ref != s.id and ref in skill_ids:
                        edge_key = (s.id, ref, "references")
                        if edge_key not in seen_edges:
                            seen_edges.add(edge_key)
                            edges.append(
                                {
                                    "source": s.id,
                                    "target": ref,
                                    "type": "references",
                                }
                            )

            # Build edges — derived_from (database-based)
            derivations = (
                session.query(SkillChangelog)
                .filter(SkillChangelog.trigger == "DERIVE")
                .filter(SkillChangelog.source_skill_id.isnot(None))
                .all()
            )

            for cl in derivations:
                src = cl.source_skill_id
                tgt = cl.skill_id
                if src and tgt and src != tgt and src in skill_ids and tgt in skill_ids:
                    edge_key = (tgt, src, "derived_from")
                    if edge_key not in seen_edges:
                        seen_edges.add(edge_key)
                        edges.append(
                            {
                                "source": tgt,
                                "target": src,
                                "type": "derived_from",
                            }
                        )

            return {"nodes": nodes, "edges": edges}
        finally:
            session.close()

    # ------------------------------------------------------------------
    # 11. Coverage gaps
    # ------------------------------------------------------------------

    def get_coverage_gaps(self) -> list[dict]:
        """Identify potential coverage gaps.

        Analyses the workspace for:
            - Tags that only have a single skill (potential single point of
              failure).
            - Repositories with very few skills (under-represented).
            - Active skills that share identical tag sets (possible duplicates).

        Returns:
            ``[{type, description, severity}]`` where *severity* is
            ``"low"``, ``"medium"``, or ``"high"``.
        """
        session = get_session()
        try:
            all_skills = session.query(Skill).all()
            gaps: list[dict] = []

            # --- Tag analysis: tags with only 1 skill ---
            tag_to_skills: dict[str, list[str]] = {}
            for s in all_skills:
                for tag in s.get_tags():
                    tag_to_skills.setdefault(tag, []).append(s.id)

            single_skill_tags = [
                tag for tag, skills in tag_to_skills.items() if len(skills) == 1
            ]
            if single_skill_tags:
                gaps.append(
                    {
                        "type": "single_skill_tag",
                        "description": (
                            f"{len(single_skill_tags)} tag(s) are only used by a "
                            f"single skill, creating potential coverage gaps: "
                            f"{', '.join(single_skill_tags[:5])}"
                            f"{'...' if len(single_skill_tags) > 5 else ''}"
                        ),
                        "severity": "medium",
                    }
                )

            # --- Repo analysis: repos with very few skills ---
            repo_counts: dict[str, int] = Counter()
            for s in all_skills:
                repo_counts[s.repo_name or "unknown"] += 1

            low_skill_repos = [
                repo
                for repo, cnt in repo_counts.items()
                if cnt <= 1 and repo != "global"
            ]
            if low_skill_repos:
                gaps.append(
                    {
                        "type": "low_skill_repo",
                        "description": (
                            f"{len(low_skill_repos)} repository/repositor"
                            f"{'ies' if len(low_skill_repos) != 1 else 'y'} "
                            f"have only 1 skill: "
                            f"{', '.join(low_skill_repos[:5])}"
                            f"{'...' if len(low_skill_repos) > 5 else ''}"
                        ),
                        "severity": "low",
                    }
                )

            # --- Duplicate tag sets: active skills with identical tags ---
            tag_set_to_skills: dict[tuple[str, ...], list[str]] = {}
            for s in all_skills:
                if not s.is_active:
                    continue
                tag_tuple = tuple(sorted(s.get_tags()))
                tag_set_to_skills.setdefault(tag_tuple, []).append(s.id)

            duplicate_groups = [
                skills
                for skills in tag_set_to_skills.values()
                if len(skills) > 1
            ]
            if duplicate_groups:
                desc_parts = []
                for group in duplicate_groups[:3]:
                    desc_parts.append(
                        f"{', '.join(group[:3])}"
                        f"{'...' if len(group) > 3 else ''}"
                    )
                gaps.append(
                    {
                        "type": "duplicate_tag_sets",
                        "description": (
                            f"{len(duplicate_groups)} group(s) of active skills "
                            f"share identical tag sets (possible duplicates): "
                            + "; ".join(desc_parts)
                        ),
                        "severity": "high",
                    }
                )

            # --- No skills at all ---
            if not all_skills:
                gaps.append(
                    {
                        "type": "empty_workspace",
                        "description": (
                            "The workspace contains no skills. Consider "
                            "archiving new skills to build coverage."
                        ),
                        "severity": "high",
                    }
                )

            # --- Untagged skills ---
            untagged = [s for s in all_skills if not s.get_tags()]
            if untagged:
                gaps.append(
                    {
                        "type": "untagged_skills",
                        "description": (
                            f"{len(untagged)} skill(s) have no tags, making them "
                            f"harder to discover: "
                            f"{', '.join(s.id for s in untagged[:5])}"
                            f"{'...' if len(untagged) > 5 else ''}"
                        ),
                        "severity": "medium",
                    }
                )

            return gaps
        finally:
            session.close()

    # ------------------------------------------------------------------
    # 12. Full analytics (dashboard)
    # ------------------------------------------------------------------

    def get_full_analytics(self) -> dict:
        """Get all analytics data in one call (for dashboard).

        Aggregates the output of every individual analytics method into a
        single dictionary.  Useful for rendering a comprehensive dashboard
        in a single API response.

        Returns:
            A dictionary with the following top-level keys:

            - ``usage_summary`` — from :meth:`get_usage_summary`
            - ``trending_skills`` — from :meth:`get_trending_skills`
            - ``stale_skills`` — from :meth:`get_stale_skills`
            - ``type_distribution`` — from :meth:`get_type_distribution`
            - ``repo_distribution`` — from :meth:`get_repo_distribution`
            - ``tag_cloud`` — from :meth:`get_tag_cloud`
            - ``version_distribution`` — from :meth:`get_version_distribution`
            - ``recent_activity`` — from :meth:`get_recent_activity`
            - ``orphan_skills`` — from :meth:`get_orphan_skills`
            - ``skill_network`` — from :meth:`get_skill_network`
            - ``coverage_gaps`` — from :meth:`get_coverage_gaps`
        """
        return {
            "usage_summary": self.get_usage_summary(),
            "trending_skills": self.get_trending_skills(),
            "stale_skills": self.get_stale_skills(),
            "type_distribution": self.get_type_distribution(),
            "repo_distribution": self.get_repo_distribution(),
            "tag_cloud": self.get_tag_cloud(),
            "version_distribution": self.get_version_distribution(),
            "recent_activity": self.get_recent_activity(),
            "orphan_skills": self.get_orphan_skills(),
            "skill_network": self.get_skill_network(),
            "coverage_gaps": self.get_coverage_gaps(),
        }
