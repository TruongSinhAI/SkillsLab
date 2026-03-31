"""
Unit tests for EvolutionEngine — ARCHIVE, FIX, DERIVE, MERGE operations.
"""

import pytest

from core.evolver import EvolutionEngine
from core.models import Skill, SkillChangelog


class TestArchive:
    """Tests for the ARCHIVE (create new skill) operation."""

    def test_archive_creates_skill_v1(self, db_session, skill_manager, sample_skill_data):
        """ARCHIVE should create a new skill at version 1."""
        engine = EvolutionEngine(session=db_session, manager=skill_manager)
        skill = engine.archive(**sample_skill_data)
        db_session.commit()

        assert skill.id == sample_skill_data["name"]
        assert skill.version_number == 1
        assert skill.is_active is True
        assert skill.description == sample_skill_data["description"]

    def test_archive_writes_skill_md(self, db_session, skill_manager, sample_skill_data):
        """ARCHIVE should create a SKILL.md file on disk."""
        engine = EvolutionEngine(session=db_session, manager=skill_manager)
        engine.archive(**sample_skill_data)
        db_session.commit()

        data = skill_manager.read_skill(sample_skill_data["name"])
        assert "frontmatter" in data
        assert "body" in data
        assert sample_skill_data["name"] in data["frontmatter"].get("name", "")

    def test_archive_creates_changelog_entry(self, db_session, skill_manager, sample_skill_data):
        """ARCHIVE should record a changelog entry from version 0 to 1."""
        engine = EvolutionEngine(session=db_session, manager=skill_manager)
        engine.archive(**sample_skill_data)
        db_session.commit()

        entries = (
            db_session.query(SkillChangelog)
            .filter_by(skill_id=sample_skill_data["name"])
            .all()
        )
        assert len(entries) == 1
        assert entries[0].from_version == 0
        assert entries[0].to_version == 1
        assert entries[0].trigger == "ARCHIVE"

    def test_archive_duplicate_raises(self, db_session, skill_manager, sample_skill_data):
        """ARCHIVE should raise ValueError if the skill name already exists."""
        engine = EvolutionEngine(session=db_session, manager=skill_manager)
        engine.archive(**sample_skill_data)
        db_session.commit()

        with pytest.raises(ValueError, match="already exists"):
            engine.archive(**sample_skill_data)

    def test_archive_invalid_name_raises(self, db_session, skill_manager):
        """ARCHIVE should raise ValueError for invalid skill names."""
        engine = EvolutionEngine(session=db_session, manager=skill_manager)
        with pytest.raises(ValueError, match="Invalid name"):
            engine.archive(
                name="A",
                description="Too short name",
                body="# Short Name\n\nBody here.",
            )


class TestFix:
    """Tests for the FIX (update existing skill) operation."""

    def test_fix_increments_version(self, db_session, skill_manager, sample_skill_data):
        """FIX should increment the version number."""
        engine = EvolutionEngine(session=db_session, manager=skill_manager)
        engine.archive(**sample_skill_data)
        db_session.commit()

        fixed = engine.fix(
            target_skill_name=sample_skill_data["name"],
            body=sample_skill_data["body"] + "\n## Updated\nNew content.",
            lesson="Fixed CORS headers",
            reason="Bug fix: added missing headers",
        )
        db_session.commit()

        assert fixed.version_number == 2

    def test_fix_appends_lesson(self, db_session, skill_manager, sample_skill_data):
        """FIX should append a new lesson to the SKILL.md."""
        engine = EvolutionEngine(session=db_session, manager=skill_manager)
        engine.archive(**sample_skill_data)
        db_session.commit()

        engine.fix(
            target_skill_name=sample_skill_data["name"],
            body=sample_skill_data["body"],
            lesson="Fixed CORS headers",
            reason="Bug fix",
        )
        db_session.commit()

        data = skill_manager.read_body(sample_skill_data["name"])
        assert "Fixed CORS headers" in data

    def test_fix_nonexistent_raises(self, db_session, skill_manager):
        """FIX should raise ValueError for a non-existent skill."""
        engine = EvolutionEngine(session=db_session, manager=skill_manager)
        with pytest.raises(ValueError, match="does not exist"):
            engine.fix(
                target_skill_name="nonexistent-skill",
                body="New body",
                lesson="Test lesson",
                reason="Test",
            )


class TestDerive:
    """Tests for the DERIVE (create child skill) operation."""

    def test_derive_creates_child_skill(self, db_session, skill_manager, sample_skill_data):
        """DERIVE should create a new skill at version 1."""
        engine = EvolutionEngine(session=db_session, manager=skill_manager)
        engine.archive(**sample_skill_data)
        db_session.commit()

        child = engine.derive(
            target_skill_name=sample_skill_data["name"],
            new_name="cors-fix-express",
            body="# CORS Fix for Express\n\n## Solution\n",
            description="Fix CORS errors on Express.js API routes",
            lesson="Derived from Next.js version",
            repo_name="global",
            reason="Create Express.js variant",
        )
        db_session.commit()

        assert child.id == "cors-fix-express"
        assert child.version_number == 1
        assert child.is_active is True

    def test_derive_records_source_in_changelog(self, db_session, skill_manager, sample_skill_data):
        """DERIVE should record the source skill in the changelog."""
        engine = EvolutionEngine(session=db_session, manager=skill_manager)
        engine.archive(**sample_skill_data)
        db_session.commit()

        engine.derive(
            target_skill_name=sample_skill_data["name"],
            new_name="cors-fix-express",
            body="# CORS Fix for Express\n\n## Solution\n",
            description="Fix CORS errors on Express.js API routes",
            lesson="Derived",
            repo_name="global",
            reason="Create Express.js variant",
        )
        db_session.commit()

        entries = (
            db_session.query(SkillChangelog)
            .filter_by(skill_id="cors-fix-express")
            .all()
        )
        assert len(entries) == 1
        assert entries[0].trigger == "DERIVE"
        assert entries[0].source_skill_id == sample_skill_data["name"]


class TestMerge:
    """Tests for the MERGE operation."""

    def test_merge_creates_target(self, db_session, skill_manager, sample_skill_data):
        """MERGE should create a target skill and deactivate sources."""
        engine = EvolutionEngine(session=db_session, manager=skill_manager)
        engine.archive(**sample_skill_data)
        db_session.commit()

        # Use a new session/engine for merge to pick up committed data
        engine2 = EvolutionEngine(session=db_session, manager=skill_manager)
        merged = engine2.merge(
            target_skill_name="cors-fix-universal",
            source_skill_names=[sample_skill_data["name"]],
            new_body="# Universal CORS Fix\n\n## Solution\n",
            reason="Merge into universal skill",
        )
        db_session.commit()

        assert merged.id == "cors-fix-universal"
        assert merged.is_active is True

        # Source should be deactivated
        s1 = db_session.query(Skill).filter_by(id=sample_skill_data["name"]).first()
        assert s1.is_active is False

    def test_merge_empty_sources_raises(self, db_session, skill_manager):
        """MERGE should raise ValueError when no source skills are provided."""
        engine = EvolutionEngine(session=db_session, manager=skill_manager)
        with pytest.raises(ValueError, match="At least one source"):
            engine.merge(
                target_skill_name="merged-skill",
                source_skill_names=[],
                new_body="# Merged\n",
                reason="Test",
            )
