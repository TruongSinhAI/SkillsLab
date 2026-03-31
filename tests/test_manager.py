"""
Skills Lab — Unit tests for core/manager.py

Tests cover: write_skill, read_skill, parse_frontmatter, validate_frontmatter,
append_lesson, update_frontmatter_version, list/delete skills, references.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.manager import SKILLManager, LESSONS_HEADING, FRONTMATTER_DELIMITER
from core.exceptions import SKILLParseError


class TestParseFrontmatter:
    """Tests for SKILLManager._parse_frontmatter()."""

    def test_valid_frontmatter(self):
        raw = "---\nname: test-skill\ndescription: A test\n---\n\n# Body\nSome content\n"
        fm, body = SKILLManager._parse_frontmatter(raw)
        assert fm["name"] == "test-skill"
        assert fm["description"] == "A test"
        assert "# Body" in body

    def test_missing_open_delimiter(self):
        raw = "name: test\ndescription: desc\n---\n\nBody"
        with pytest.raises(SKILLParseError, match="must start with"):
            SKILLManager._parse_frontmatter(raw)

    def test_missing_close_delimiter(self):
        raw = "---\nname: test\ndescription: desc\n\nBody content"
        with pytest.raises(SKILLParseError, match="not closed"):
            SKILLManager._parse_frontmatter(raw)

    def test_empty_frontmatter(self):
        raw = "---\n---\n\nBody"
        with pytest.raises(SKILLParseError, match="empty"):
            SKILLManager._parse_frontmatter(raw)

    def test_invalid_yaml(self):
        raw = "---\nname: [\nbroken yaml\n---\n\nBody"
        with pytest.raises(SKILLParseError, match="Invalid YAML"):
            SKILLManager._parse_frontmatter(raw)

    def test_non_mapping_frontmatter(self):
        raw = "---\njust a string\n---\n\nBody"
        with pytest.raises(SKILLParseError, match="must be a YAML mapping"):
            SKILLManager._parse_frontmatter(raw)

    def test_with_metadata(self):
        raw = (
            "---\n"
            "name: test-skill\n"
            "description: Test\n"
            "metadata:\n"
            "  skill-type: IMPLEMENTATION\n"
            "  tags: [cors, api]\n"
            "---\n\nBody content\n"
        )
        fm, body = SKILLManager._parse_frontmatter(raw)
        assert fm["metadata"]["skill-type"] == "IMPLEMENTATION"
        assert fm["metadata"]["tags"] == ["cors", "api"]


class TestValidateFrontmatter:
    """Tests for SKILLManager.validate_frontmatter()."""

    def test_valid_complete(self):
        fm = {
            "name": "cors-fix",
            "description": "Fix CORS",
            "metadata": {"skill-type": "TROUBLESHOOTING", "tags": ["cors"]},
        }
        errors = SKILLManager.validate_frontmatter(fm)
        assert errors == []

    def test_missing_name(self):
        fm = {"description": "No name"}
        errors = SKILLManager.validate_frontmatter(fm)
        assert any("name" in e for e in errors)

    def test_missing_description(self):
        fm = {"name": "test"}
        errors = SKILLManager.validate_frontmatter(fm)
        assert any("description" in e for e in errors)

    def test_unknown_key(self):
        fm = {"name": "test", "description": "d", "unknown_key": "val"}
        errors = SKILLManager.validate_frontmatter(fm)
        assert any("unknown_key" in e for e in errors)

    def test_unknown_metadata_key(self):
        fm = {"name": "test", "description": "d", "metadata": {"bad_key": "val"}}
        errors = SKILLManager.validate_frontmatter(fm)
        assert any("bad_key" in e for e in errors)

    def test_invalid_name_format(self):
        fm = {"name": "Not-Kebab", "description": "d"}
        errors = SKILLManager.validate_frontmatter(fm)
        assert any("kebab-case" in e for e in errors)

    def test_invalid_tags_type(self):
        fm = {"name": "test", "description": "d", "metadata": {"tags": "not-a-list"}}
        errors = SKILLManager.validate_frontmatter(fm)
        assert any("JSON array" in e for e in errors)

    def test_empty_name(self):
        fm = {"name": "", "description": "d"}
        errors = SKILLManager.validate_frontmatter(fm)
        assert any("'name'" in e for e in errors)


class TestWriteAndReadSkill:
    """Tests for SKILLManager write_skill / read_skill."""

    def test_write_and_read(self, skill_manager):
        skill_manager.write_skill(
            skill_name="test-write",
            description="A test skill",
            body="# Test\n\n## Solution\nCode here\n",
        )
        data = skill_manager.read_skill("test-write")
        assert data["frontmatter"]["name"] == "test-write"
        assert data["frontmatter"]["description"] == "A test skill"
        assert "## Solution" in data["body"]
        assert data["raw"] is not None

    def test_write_with_all_params(self, skill_manager):
        skill_manager.write_skill(
            skill_name="test-full",
            description="Full params",
            body="# Full\n\nContent\n",
            display_name="Full Params Skill",
            skill_type="ARCHITECTURE",
            repo="my-repo",
            version=3,
            tags=["tag1", "tag2"],
            ttl_days=60,
            author="tester",
            source="https://example.com",
        )
        fm = skill_manager.read_frontmatter("test-full")
        assert fm["metadata"]["skill-type"] == "ARCHITECTURE"
        assert fm["metadata"]["repo"] == "my-repo"
        assert fm["metadata"]["version"] == "3"
        assert fm["metadata"]["ttl-days"] == 60
        assert fm["metadata"]["author"] == "tester"
        assert fm["metadata"]["source"] == "https://example.com"

    def test_write_empty_description_raises(self, skill_manager):
        with pytest.raises(ValueError, match="description"):
            skill_manager.write_skill("test", "", "body")

    def test_write_empty_body_raises(self, skill_manager):
        with pytest.raises(ValueError, match="body"):
            skill_manager.write_skill("test", "desc", "")

    def test_read_nonexistent_raises(self, skill_manager):
        with pytest.raises(FileNotFoundError):
            skill_manager.read_skill("does-not-exist")

    def test_read_body_only(self, skill_manager):
        skill_manager.write_skill(
            skill_name="test-body",
            description="desc",
            body="# Title\n\nBody content here\n",
        )
        body = skill_manager.read_body("test-body")
        assert "Body content here" in body

    def test_read_raw(self, skill_manager):
        skill_manager.write_skill(
            skill_name="test-raw",
            description="desc",
            body="# Raw\n\nContent\n",
        )
        raw = skill_manager.read_raw("test-raw")
        assert raw.startswith("---")
        assert "# Raw" in raw


class TestAppendLesson:
    """Tests for SKILLManager.append_lesson()."""

    def test_append_to_existing_section(self, skill_manager):
        skill_manager.write_skill(
            skill_name="test-lesson",
            description="desc",
            body="# Title\n\n## Solution\nFix\n\n## Lessons Learned\n- **V1**: First lesson\n",
        )
        skill_manager.append_lesson("test-lesson", 2, "Second lesson")
        body = skill_manager.read_body("test-lesson")
        assert "V1" in body
        assert "V2" in body
        assert "Second lesson" in body

    def test_append_creates_section(self, skill_manager):
        skill_manager.write_skill(
            skill_name="test-lesson-new",
            description="desc",
            body="# Title\n\n## Solution\nFix\n",
        )
        skill_manager.append_lesson("test-lesson-new", 1, "First lesson")
        body = skill_manager.read_body("test-lesson-new")
        assert LESSONS_HEADING in body
        assert "V1" in body

    def test_append_nonexistent_raises(self, skill_manager):
        with pytest.raises(FileNotFoundError):
            skill_manager.append_lesson("nope", 1, "lesson")


class TestUpdateFrontmatterVersion:
    """Tests for SKILLManager.update_frontmatter_version()."""

    def test_update_version(self, skill_manager):
        skill_manager.write_skill(
            skill_name="test-ver",
            description="desc",
            body="# Body\n\n## Lessons Learned\n- V1\n",
        )
        skill_manager.update_frontmatter_version("test-ver", 5)
        fm = skill_manager.read_frontmatter("test-ver")
        assert fm["metadata"]["version"] == "5"
        assert fm["metadata"]["last-modified"] is not None


class TestSkillLifecycle:
    """Tests for list_skills, skill_dir_exists, delete_skill_dir."""

    def test_list_empty(self, skill_manager):
        assert skill_manager.list_skills() == []

    def test_list_after_write(self, skill_manager):
        skill_manager.write_skill("skill-a", "desc a", "# A\n\nBody\n")
        skill_manager.write_skill("skill-b", "desc b", "# B\n\nBody\n")
        result = skill_manager.list_skills()
        assert "skill-a" in result
        assert "skill-b" in result

    def test_dir_exists(self, skill_manager):
        skill_manager.write_skill("test-dir", "desc", "# Body\n")
        assert skill_manager.skill_dir_exists("test-dir") is True
        assert skill_manager.skill_dir_exists("no-dir") is False

    def test_delete_dir(self, skill_manager):
        skill_manager.write_skill("test-del", "desc", "# Body\n")
        assert skill_manager.skill_dir_exists("test-del")
        skill_manager.delete_skill_dir("test-del")
        assert skill_manager.skill_dir_exists("test-del") is False

    def test_delete_nonexistent_noop(self, skill_manager):
        skill_manager.delete_skill_dir("no-exist")  # Should not raise


class TestReferences:
    """Tests for get_references, set_references, find_referencing_skills."""

    def test_references_from_body_at_mentions(self, skill_manager):
        skill_manager.write_skill("skill-a", "desc", "# A\n\n## References\n- @skill-b - Referenced\n")
        skill_manager.write_skill("skill-b", "desc", "# B\n\nContent\n")
        refs = skill_manager.get_references("skill-a")
        assert "skill-b" in refs

    def test_references_from_frontmatter(self, skill_manager):
        skill_manager.write_skill("skill-x", "desc", "# X\n\nContent\n")
        skill_manager.write_skill("skill-y", "desc", "# Y\n\nContent\n")
        skill_manager.set_references("skill-x", ["skill-y"])
        refs = skill_manager.get_references("skill-x")
        assert "skill-y" in refs

    def test_find_referencing_skills(self, skill_manager):
        skill_manager.write_skill("target-skill", "desc", "# Target\n\nContent\n")
        skill_manager.write_skill(
            "referring-skill",
            "desc",
            "# Ref\n\n## References\n- @target-skill - Link\n",
        )
        incoming = skill_manager.find_referencing_skills("target-skill")
        assert "referring-skill" in incoming

    def test_no_references(self, skill_manager):
        skill_manager.write_skill("no-refs", "desc", "# No refs\n\nContent\n")
        refs = skill_manager.get_references("no-refs")
        assert refs == []


class TestGetSearchText:
    """Tests for get_description_for_search and get_bm25_text."""

    def test_get_description_for_search(self, skill_manager):
        skill_manager.write_skill(
            "search-test",
            description="Fix CORS errors",
            body="# CORS\n\nContent\n",
            tags=["cors", "api"],
        )
        text = skill_manager.get_description_for_search("search-test")
        assert "Fix CORS errors" in text
        assert "cors" in text
        assert "api" in text

    def test_get_search_text_nonexistent(self, skill_manager):
        # Should return empty string gracefully
        text = skill_manager.get_description_for_search("nope")
        assert text == ""

    def test_get_search_text_fallback(self, skill_manager):
        # Create a file that's not valid SKILL.md
        os.makedirs(os.path.join(skill_manager.skills_dir, "bad-skill"), exist_ok=True)
        path = os.path.join(skill_manager.skills_dir, "bad-skill", "SKILL.md")
        with open(path, "w") as f:
            f.write("not valid yaml but has some words here for search\n")
        text = skill_manager.get_description_for_search("bad-skill")
        assert isinstance(text, str)
