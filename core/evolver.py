"""
Skills Lab — Evolution Engine (SKILL.md Standard)

Four evolution actions: ARCHIVE, FIX, DERIVE, MERGE.

Design: skill name is the primary key. FIX = UPDATE existing row (version++).
Changelog table tracks version history for every evolution operation.
"""

import logging
import re
from datetime import datetime, timezone
from typing import Callable, Optional

from sqlalchemy.orm import Session

from core.manager import LESSONS_HEADING
from core.models import Skill, SkillChangelog, SkillType, EvolutionTrigger

logger = logging.getLogger(__name__)


class EvolutionEngine:
    """Engine that drives the lifecycle evolution of skills.

    Each skill follows a versioned evolution model where the skill name is
    the primary key. The engine supports four operations:

    - **ARCHIVE**: Create a brand-new skill at version 1.
    - **FIX**: Update an existing skill in-place, incrementing its version.
    - **DERIVE**: Create a new child skill derived from a parent skill.
    - **MERGE**: Combine multiple source skills into a single target skill,
      deactivating the sources.

    Every operation writes a changelog entry and (optionally) recomputes
    the search embedding for affected skills.
    """

    def __init__(
        self,
        session: Session,
        manager,
        on_embedding_cache_clear: Optional[Callable] = None,
        on_embedding_compute: Optional[Callable] = None,
    ) -> None:
        """Initialize the evolution engine.

        Args:
            session: SQLAlchemy database session for persistence.
            manager: Skill file-system manager used to read/write SKILL.md files.
            on_embedding_cache_clear: Optional callback invoked with a skill name
                to clear its cached embedding before recompute.
            on_embedding_compute: Optional callback invoked with a skill name
                and its search text to compute (or schedule) a new embedding.
        """
        self.session = session
        self.manager = manager
        self._on_cache_clear = on_embedding_cache_clear
        self._on_compute = on_embedding_compute

    def _clear_cache(self, name: str) -> None:
        """Clear the embedding cache for a skill, if a cache-clear callback is registered."""
        if self._on_cache_clear:
            self._on_cache_clear(name)

    def _compute_embedding(self, name: str, text: str) -> None:
        """Compute (or schedule) an embedding for a skill, if a compute callback is registered."""
        if self._on_compute:
            self._on_compute(name, text)

    def _add_changelog(
        self,
        skill_id: str,
        from_v: int,
        to_v: int,
        trigger: str,
        reason: str,
        source_id: Optional[str] = None,
    ) -> SkillChangelog:
        """Add a changelog entry recording a version transition.

        Args:
            skill_id: The skill's primary key (name).
            from_v: Previous version number (0 for newly created skills).
            to_v: New version number after the evolution operation.
            trigger: The evolution trigger type (e.g. ARCHIVE, FIX, DERIVE, MERGE).
            reason: Human-readable explanation for the evolution.
            source_id: Optional ID of the source skill (used for DERIVE/MERGE).

        Returns:
            The newly created SkillChangelog instance (not yet flushed).
        """
        cl = SkillChangelog(
            skill_id=skill_id,
            from_version=from_v,
            to_version=to_v,
            trigger=trigger,
            reason=reason,
            source_skill_id=source_id,
            created_at=datetime.now(timezone.utc),
        )
        self.session.add(cl)
        return cl

    # -----------------------------------------------------------------------
    # ARCHIVE — INSERT new row (V1)
    # -----------------------------------------------------------------------

    def archive(
        self,
        name: str,
        description: str,
        body: str,
        skill_type: str = "IMPLEMENTATION",
        repo_name: str = "global",
        display_name: str = "",
        tags: Optional[list[str]] = None,
        ttl_days: Optional[int] = None,
        author: Optional[str] = None,
    ) -> Skill:
        """Archive a new skill into the system at version 1.

        Creates a new SKILL.md file on disk, inserts a new database row,
        records a changelog entry, and computes the initial search embedding.

        Args:
            name: Kebab-case skill identifier (2–64 characters). Must be unique.
            description: Short human-readable description of the skill.
            body: Full Markdown body for the SKILL.md file.
            skill_type: The type of skill (e.g. ``IMPLEMENTATION``, ``PATTERN``).
                Defaults to ``"IMPLEMENTATION"``.
            repo_name: Repository or namespace the skill belongs to.
                Defaults to ``"global"``.
            display_name: User-facing display name. Auto-generated from *name*
                (title-cased) when empty.
            tags: Optional list of tag strings for categorisation.
            ttl_days: Optional time-to-live in days; ``None`` means no expiry.
            author: Optional author attribution stored in frontmatter.

        Returns:
            The newly created ``Skill`` ORM instance.

        Raises:
            ValueError: If *name* fails validation or a skill with the same
                name already exists.
        """
        if not Skill.validate_name(name):
            raise ValueError(
                f"Invalid name '{name}'. Must be kebab-case, 2-64 characters."
            )

        existing = self.session.query(Skill).filter_by(id=name).first()
        if existing:
            raise ValueError(
                f"Skill '{name}' already exists (V{existing.version_number}). "
                f"Use FIX to update."
            )

        now = datetime.now(timezone.utc)
        if not display_name:
            display_name = name.replace("-", " ").title()

        body = self._ensure_lessons_section(body, 1)

        skill = Skill(
            id=name,
            display_name=display_name,
            description=description,
            skill_type=skill_type,
            repo_name=repo_name,
            version_number=1,
            is_active=True,
            use_count=0,
            ttl_days=ttl_days,
            created_at=now,
            last_modified_at=now,
        )
        skill.set_tags(tags or [])
        skill.compute_expires_at()

        self.manager.write_skill(
            skill_name=name,
            description=description,
            body=body,
            display_name=display_name,
            skill_type=skill_type,
            repo=repo_name,
            version=1,
            tags=tags,
            ttl_days=ttl_days,
            author=author,
        )

        self._add_changelog(
            name, 0, 1, EvolutionTrigger.ARCHIVE.value, "Initial version"
        )
        self.session.add(skill)
        self.session.flush()

        search_text = self.manager.get_description_for_search(name)
        self._compute_embedding(name, search_text)

        logger.info(f"ARCHIVE: {name} V1 ({skill_type})")
        return skill

    # -----------------------------------------------------------------------
    # FIX — UPDATE existing row, version++
    # -----------------------------------------------------------------------

    def fix(
        self,
        target_skill_name: str,
        body: str,
        lesson: str,
        reason: str,
        description: Optional[str] = None,
        tags: Optional[list[str]] = None,
    ) -> Skill:
        """Fix / update an existing skill in-place (version increment).

        Workflow:
            1. Append the new lesson to the existing SKILL.md.
            2. Overwrite SKILL.md with the new body + accumulated lessons.
            3. Update the database row: increment version, update metadata.
            4. Record a changelog entry.
            5. Recompute the search embedding.

        Args:
            target_skill_name: Name of the skill to update.
            body: New Markdown body content (without the Lessons Learned section).
            lesson: A lesson-learned string to record for this version.
            reason: Human-readable explanation for the fix / update.
            description: Optional new description. When ``None``, the existing
                description is preserved.
            tags: Optional new tag list. When ``None``, the existing tags are
                preserved.

        Returns:
            The updated ``Skill`` ORM instance.

        Raises:
            ValueError: If the target skill does not exist or is inactive.
        """
        skill = (
            self.session.query(Skill).filter_by(id=target_skill_name).first()
        )
        if not skill:
            raise ValueError(f"Skill does not exist: {target_skill_name}")
        if not skill.is_active:
            raise ValueError(f"Skill '{target_skill_name}' is already inactive.")

        old_version = skill.version_number
        new_version = old_version + 1
        now = datetime.now(timezone.utc)

        # 1. Append lesson to the existing SKILL.md
        self.manager.append_lesson(target_skill_name, new_version, lesson)

        # 2. Read current SKILL.md (now contains accumulated lessons)
        current_body = self.manager.read_body(target_skill_name)

        # 3. Replace body sections but keep the Lessons Learned section
        new_body = self._replace_body_keep_lessons(current_body, body, new_version)

        # 4. Overwrite SKILL.md with merged content
        self.manager.write_skill(
            skill_name=target_skill_name,
            description=description or skill.description,
            body=new_body,
            display_name=skill.display_name,
            skill_type=skill.skill_type,
            repo=skill.repo_name,
            version=new_version,
            tags=tags if tags is not None else skill.get_tags(),
            ttl_days=skill.ttl_days,
        )

        # 5. Update the database row
        skill.version_number = new_version
        skill.last_modified_at = now
        if description:
            skill.description = description
        if tags is not None:
            skill.set_tags(tags)

        self._add_changelog(
            target_skill_name,
            old_version,
            new_version,
            EvolutionTrigger.FIX.value,
            reason,
        )
        self.session.flush()

        # 6. Recompute embedding
        self._clear_cache(target_skill_name)
        search_text = self.manager.get_description_for_search(target_skill_name)
        self._compute_embedding(target_skill_name, search_text)

        logger.info(f"FIX: {target_skill_name} V{old_version} → V{new_version}")
        return skill

    # -----------------------------------------------------------------------
    # DERIVE — INSERT new row, different name
    # -----------------------------------------------------------------------

    def derive(
        self,
        target_skill_name: str,
        new_name: str,
        body: str,
        description: str,
        lesson: str,
        repo_name: str,
        reason: str,
        skill_type: Optional[str] = None,
        tags: Optional[list[str]] = None,
    ) -> Skill:
        """Derive a new child skill from an existing parent skill.

        The new skill starts at version 1 and includes a reference back to
        the parent in its SKILL.md body.

        Args:
            target_skill_name: Name of the parent skill to derive from.
            new_name: Kebab-case name for the new child skill. Must be unique.
            body: Markdown body for the new skill.
            description: Short description of the new skill.
            lesson: Initial lesson-learned to record.
            repo_name: Repository / namespace for the new skill.
            reason: Human-readable explanation for the derivation.
            skill_type: Skill type for the child. When ``None``, inherits from
                the parent.
            tags: Tags for the child. When ``None``, inherits from the parent.

        Returns:
            The newly created child ``Skill`` ORM instance.

        Raises:
            ValueError: If the parent does not exist, *new_name* is invalid,
                or a skill with *new_name* already exists.
        """
        parent = (
            self.session.query(Skill).filter_by(id=target_skill_name).first()
        )
        if not parent:
            raise ValueError(f"Skill does not exist: {target_skill_name}")
        if not Skill.validate_name(new_name):
            raise ValueError(f"Invalid name '{new_name}'.")
        if self.session.query(Skill).filter_by(id=new_name).first():
            raise ValueError(f"Skill '{new_name}' already exists.")

        now = datetime.now(timezone.utc)
        display_name = new_name.replace("-", " ").title()

        # Ensure the body has a Lessons Learned section and add a parent reference
        body = self._ensure_lessons_section(body, 1)
        ref = (
            f"> Derived from [{parent.display_name}] "
            f"V{parent.version_number} ({target_skill_name})\n"
        )
        if LESSONS_HEADING in body:
            body = body.replace(LESSONS_HEADING, f"{ref}\n{LESSONS_HEADING}", 1)
        else:
            body = (
                f"{body}\n\n{ref}\n{LESSONS_HEADING}\n"
                f"- **V1** ({now.strftime('%Y-%m-%d')}): {lesson}"
            )

        child = Skill(
            id=new_name,
            display_name=display_name,
            description=description,
            skill_type=skill_type or parent.skill_type,
            repo_name=repo_name,
            version_number=1,
            is_active=True,
            use_count=0,
            ttl_days=parent.ttl_days,
            created_at=now,
            last_modified_at=now,
        )
        child.set_tags(tags if tags is not None else parent.get_tags())
        child.compute_expires_at()

        self.manager.write_skill(
            skill_name=new_name,
            description=description,
            body=body,
            display_name=display_name,
            skill_type=child.skill_type,
            repo=repo_name,
            version=1,
            tags=child.get_tags(),
            ttl_days=child.ttl_days,
        )

        self._add_changelog(
            new_name,
            0,
            1,
            EvolutionTrigger.DERIVE.value,
            reason,
            source_id=target_skill_name,
        )
        self.session.add(child)
        self.session.flush()

        search_text = self.manager.get_description_for_search(new_name)
        self._compute_embedding(new_name, search_text)

        logger.info(f"DERIVE: {target_skill_name} → {new_name} V1")
        return child

    # -----------------------------------------------------------------------
    # MERGE — UPDATE target, deactivate sources
    # -----------------------------------------------------------------------

    def merge(
        self,
        target_skill_name: str,
        source_skill_names: list[str],
        new_body: str,
        reason: str,
        description: Optional[str] = None,
        tags: Optional[list[str]] = None,
    ) -> Skill:
        """Merge multiple source skills into a single target skill.

        All lessons learned from the source skills are collected and merged
        into the target. Source skills (other than the target itself) are
        deactivated. Tags from all sources are unioned together.

        If the target skill already exists it is updated in-place (version
        bumped). Otherwise a new skill row is created.

        Args:
            target_skill_name: Name of the target (merged) skill.
            source_skill_names: List of source skill names to merge.
            new_body: New Markdown body for the merged skill.
            reason: Human-readable explanation for the merge.
            description: Optional new description. When ``None`` and the target
                is new, the first source's description is used.
            tags: Optional tag list. When ``None`` only the sources' tags are
                unioned.

        Returns:
            The updated or newly created target ``Skill`` ORM instance.

        Raises:
            ValueError: If no source names are provided or any source does not exist.
        """
        if not source_skill_names:
            raise ValueError("At least one source skill is required.")

        sources: list[Skill] = []
        for sname in source_skill_names:
            s = self.session.query(Skill).filter_by(id=sname).first()
            if not s:
                raise ValueError(f"Source skill does not exist: {sname}")
            sources.append(s)

        now = datetime.now(timezone.utc)
        display_name = target_skill_name.replace("-", " ").title()

        # Collect lessons from all sources
        all_lessons: list[str] = []
        for src in sources:
            try:
                src_body = self.manager.read_body(src.id)
                if LESSONS_HEADING in src_body:
                    for line in src_body.split(LESSONS_HEADING)[1].strip().split("\n"):
                        line = line.strip()
                        if line.startswith("- **"):
                            all_lessons.append(f"{line} (from {src.display_name})")
            except FileNotFoundError:
                pass

        new_body = self._ensure_lessons_section(new_body, 1)
        if all_lessons and LESSONS_HEADING in new_body:
            lessons_text = "\n".join(all_lessons)
            new_body = new_body.replace(
                LESSONS_HEADING, f"{LESSONS_HEADING}\n{lessons_text}", 1
            )

        # Merge tags from all sources
        merged_tags: set[str] = set(tags) if tags else set()
        for src in sources:
            merged_tags.update(src.get_tags())
        merged_tags = sorted(merged_tags)

        max_ver = max(s.version_number for s in sources)
        new_version = max_ver + 1

        # Check whether the target already exists
        target = (
            self.session.query(Skill).filter_by(id=target_skill_name).first()
        )

        if target:
            # Update existing target
            old_version = target.version_number
            target.version_number = new_version
            target.description = description or target.description
            target.last_modified_at = now
            target.set_tags(list(merged_tags))
            target.compute_expires_at()
            self._add_changelog(
                target_skill_name,
                old_version,
                new_version,
                EvolutionTrigger.MERGE.value,
                reason,
            )
        else:
            # Create a brand-new target skill
            target = Skill(
                id=target_skill_name,
                display_name=display_name,
                description=description or sources[0].description,
                skill_type=sources[0].skill_type,
                repo_name=sources[0].repo_name,
                version_number=new_version,
                is_active=True,
                use_count=0,
                ttl_days=max(
                    (s.ttl_days for s in sources if s.ttl_days), default=None
                ),
                created_at=now,
                last_modified_at=now,
            )
            target.set_tags(list(merged_tags))
            target.compute_expires_at()
            self._add_changelog(
                target_skill_name,
                0,
                new_version,
                EvolutionTrigger.MERGE.value,
                reason,
            )
            self.session.add(target)

        self.manager.write_skill(
            skill_name=target_skill_name,
            description=target.description,
            body=new_body,
            display_name=display_name,
            skill_type=target.skill_type,
            repo=target.repo_name,
            version=new_version,
            tags=list(merged_tags),
            ttl_days=target.ttl_days,
        )

        # Deactivate all source skills (except the target itself)
        for src in sources:
            if src.id != target_skill_name:
                src.is_active = False
                src.last_modified_at = now
                self._clear_cache(src.id)

        self.session.flush()

        self._clear_cache(target_skill_name)
        search_text = self.manager.get_description_for_search(target_skill_name)
        self._compute_embedding(target_skill_name, search_text)

        logger.info(
            f"MERGE: {', '.join(s.id for s in sources)} → "
            f"{target_skill_name} V{new_version}"
        )
        return target

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _ensure_lessons_section(body: str, version: int) -> str:
        """Ensure the body contains a Lessons Learned section.

        If the heading is already present the body is returned unchanged.
        Otherwise a default Lessons Learned section with an initial version
        entry is appended.

        Args:
            body: The Markdown body to check / modify.
            version: The version number for the initial lesson entry.

        Returns:
            The body guaranteed to contain a Lessons Learned section.
        """
        if LESSONS_HEADING in body:
            return body
        return f"{body}\n\n{LESSONS_HEADING}\n- **V{version}**: Initial version."

    @staticmethod
    def _replace_body_keep_lessons(
        current_body: str, new_body: str, new_version: int
    ) -> str:
        """Replace body content while preserving Lessons Learned entries.

        Strategy:
            - Extract lessons from *current_body* (the on-disk file).
            - Extract any lessons already in *new_body* (provided by caller).
            - Merge them so that newer lessons come first and duplicate
              version entries are de-duplicated.

        Args:
            current_body: The existing SKILL.md body (contains accumulated
                lessons from previous versions).
            new_body: The new body content supplied for the update.
            new_version: The version number being applied (unused in
                deduplication but kept for API symmetry).

        Returns:
            The merged body with a single Lessons Learned section.
        """
        # Extract lessons from the current body
        current_lessons = ""
        if LESSONS_HEADING in current_body:
            parts = current_body.split(LESSONS_HEADING, 1)
            current_lessons = parts[1] if len(parts) > 1 else ""

        # Extract lessons from the new body (if any)
        new_lessons = ""
        new_body_without_lessons = new_body
        if LESSONS_HEADING in new_body:
            parts = new_body.split(LESSONS_HEADING, 1)
            new_lessons = parts[1] if len(parts) > 1 else ""
            new_body_without_lessons = parts[0]

        # Merge: new lessons come first (more recent), then old lessons
        merged_lessons = ""
        if new_lessons.strip():
            merged_lessons = new_lessons.strip()
        if current_lessons.strip():
            # De-duplicate — only keep old lessons whose version marker is
            # not already present in the new lessons.
            existing_entries: set[str] = set()
            for line in new_lessons.strip().split("\n"):
                m = re.match(r"- \*\*V(\d+)\*\*", line)
                if m:
                    existing_entries.add(f"V{m.group(1)}")

            for line in current_lessons.strip().split("\n"):
                m = re.match(r"- \*\*V(\d+)\*\*", line)
                if m and f"V{m.group(1)}" in existing_entries:
                    continue  # Skip duplicate version entry
                merged_lessons += "\n" + line

        if merged_lessons.strip():
            return (
                f"{new_body_without_lessons.strip()}\n\n"
                f"{LESSONS_HEADING}\n{merged_lessons.strip()}\n"
            )
        else:
            return new_body
