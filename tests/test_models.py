"""
Skills Lab — Unit tests for core/models.py

Tests cover: Skill.validate_name, to_kebab_case, get/set tags,
compute_expires_at, is_expired, to_dict, init_db, get_session.
"""

import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.models import Skill, SkillChangelog, SkillType, EvolutionTrigger
from core.models import init_db, get_session, Base
from core.config import reset_config


class TestValidateName:
    """Tests for Skill.validate_name()."""

    def test_valid_short(self):
        assert Skill.validate_name("ab") is True

    def test_valid_with_hyphens(self):
        assert Skill.validate_name("cors-fix-nextjs-api") is True

    def test_valid_with_numbers(self):
        assert Skill.validate_name("test-123") is True

    def test_valid_single_char_rejected(self):
        assert Skill.validate_name("a") is False

    def test_invalid_empty(self):
        assert Skill.validate_name("") is False

    def test_invalid_none(self):
        assert Skill.validate_name(None) is False

    def test_invalid_uppercase(self):
        assert Skill.validate_name("My-Skill") is False

    def test_invalid_spaces(self):
        assert Skill.validate_name("has space") is False

    def test_invalid_starts_with_hyphen(self):
        assert Skill.validate_name("-starts") is False

    def test_invalid_ends_with_hyphen(self):
        assert Skill.validate_name("ends-") is False

    def test_invalid_too_long(self):
        name = "a" * 65
        assert Skill.validate_name(name) is False

    def test_valid_max_length(self):
        name = "a" * 64
        assert Skill.validate_name(name) is True

    def test_invalid_special_chars(self):
        assert Skill.validate_name("hello.world") is False
        assert Skill.validate_name("hello_world") is False

    def test_valid_numbers_only(self):
        assert Skill.validate_name("123") is True


class TestToKebabCase:
    """Tests for Skill.to_kebab_case()."""

    def test_basic(self):
        assert Skill.to_kebab_case("My CORS Fix!") == "my-cors-fix"

    def test_camel_case(self):
        assert Skill.to_kebab_case("NextJSAPIRouter") == "nextjsapirouter"

    def test_with_spaces_and_special(self):
        assert Skill.to_kebab_case("Docker Multi-Stage Build!") == "docker-multi-stage-build"

    def test_empty(self):
        assert Skill.to_kebab_case("") == ""

    def test_truncation(self):
        long_input = "a " * 40  # 80 chars
        result = Skill.to_kebab_case(long_input)
        assert len(result) <= 64

    def test_consecutive_hyphens(self):
        assert Skill.to_kebab_case("hello---world") == "hello-world"

    def test_leading_trailing_hyphens_stripped(self):
        result = Skill.to_kebab_case("--hello--")
        assert not result.startswith("-")
        assert not result.endswith("-")


class TestTags:
    """Tests for Skill.get_tags() / Skill.set_tags()."""

    def test_set_and_get(self):
        skill = Skill(id="test", display_name="Test", description="desc")
        skill.set_tags(["cors", "api", "nextjs"])
        assert skill.get_tags() == ["cors", "api", "nextjs"]

    def test_empty_tags(self):
        skill = Skill(id="test", display_name="Test", description="desc")
        skill.set_tags([])
        assert skill.get_tags() == []

    def test_malformed_json(self):
        skill = Skill(id="test", display_name="Test", description="desc")
        skill.tags = "not-json"
        assert skill.get_tags() == []

    def test_non_list_json(self):
        skill = Skill(id="test", display_name="Test", description="desc")
        skill.tags = '"just-a-string"'
        assert skill.get_tags() == []

    def test_unicode_tags(self):
        skill = Skill(id="test", display_name="Test", description="desc")
        skill.set_tags(["đăng-nhập", "api"])
        assert skill.get_tags() == ["đăng-nhập", "api"]

    def test_numeric_tags_coerced_to_string(self):
        skill = Skill(id="test", display_name="Test", description="desc")
        skill.set_tags([1, 2, 3])
        assert skill.get_tags() == ["1", "2", "3"]


class TestExpiresAt:
    """Tests for compute_expires_at() / is_expired()."""

    def test_with_ttl(self):
        skill = Skill(id="test", display_name="Test", description="desc")
        skill.ttl_days = 30
        from datetime import datetime, timezone, timedelta
        skill.created_at = datetime(2025, 1, 1, tzinfo=timezone.utc)
        skill.compute_expires_at()
        assert skill.expires_at is not None
        assert skill.expires_at == skill.created_at + timedelta(days=30)

    def test_without_ttl(self):
        skill = Skill(id="test", display_name="Test", description="desc")
        skill.ttl_days = None
        skill.compute_expires_at()
        assert skill.expires_at is None

    def test_zero_ttl(self):
        skill = Skill(id="test", display_name="Test", description="desc")
        skill.ttl_days = 0
        skill.compute_expires_at()
        assert skill.expires_at is None

    def test_is_expired_true(self):
        from datetime import datetime, timezone, timedelta
        skill = Skill(id="test", display_name="Test", description="desc")
        skill.expires_at = datetime(2020, 1, 1, tzinfo=timezone.utc)
        assert skill.is_expired() is True

    def test_is_expired_false(self):
        from datetime import datetime, timezone, timedelta
        skill = Skill(id="test", display_name="Test", description="desc")
        skill.expires_at = datetime(2030, 1, 1, tzinfo=timezone.utc)
        assert skill.is_expired() is False

    def test_is_expired_no_expiry(self):
        skill = Skill(id="test", display_name="Test", description="desc")
        skill.expires_at = None
        assert skill.is_expired() is False

    def test_is_expired_naive_datetime(self):
        from datetime import datetime
        skill = Skill(id="test", display_name="Test", description="desc")
        skill.expires_at = datetime(2020, 1, 1)  # naive
        assert skill.is_expired() is True


class TestToDict:
    """Tests for Skill.to_dict()."""

    def test_all_fields_present(self):
        from datetime import datetime, timezone
        skill = Skill(
            id="test-skill", display_name="Test Skill", description="A test skill",
            skill_type="TROUBLESHOOTING", repo_name="my-repo", version_number=2,
            is_active=True, use_count=5, ttl_days=30,
        )
        skill.set_tags(["tag1", "tag2"])
        skill.created_at = datetime(2025, 1, 1, tzinfo=timezone.utc)
        skill.last_modified_at = datetime(2025, 1, 15, tzinfo=timezone.utc)
        skill.compute_expires_at()

        d = skill.to_dict()
        assert d["id"] == "test-skill"
        assert d["display_name"] == "Test Skill"
        assert d["description"] == "A test skill"
        assert d["skill_type"] == "TROUBLESHOOTING"
        assert d["repo_name"] == "my-repo"
        assert d["version_number"] == 2
        assert d["is_active"] is True
        assert d["use_count"] == 5
        assert d["tags"] == ["tag1", "tag2"]
        assert d["ttl_days"] == 30
        assert d["is_expired"] is not None
        assert d["created_at"] is not None
        assert d["last_modified_at"] is not None
        assert d["expires_at"] is not None

    def test_nullable_fields(self):
        skill = Skill(id="t", display_name="T", description="d")
        d = skill.to_dict()
        assert d["last_used_at"] is None
        assert d["expires_at"] is None
