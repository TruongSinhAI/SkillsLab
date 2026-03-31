"""
Skills Lab — Unit tests for core/config.py

Tests cover: SkillsLabConfig defaults, env overrides, validate(),
get_config/reset_config singleton.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import SkillsLabConfig, get_config, reset_config


class TestSkillsLabConfig:
    """Tests for SkillsLabConfig dataclass."""

    def test_defaults(self):
        cfg = SkillsLabConfig()
        assert cfg.search_top_k == 5
        assert cfg.rrf_k == 60
        assert cfg.dedup_threshold == 0.85
        assert cfg.dashboard_host == "0.0.0.0"
        assert cfg.dashboard_port == 7788
        assert cfg.log_level == "INFO"
        assert cfg.embedding_model == "BAAI/bge-small-en-v1.5"

    def test_workspace_path_computed(self):
        cfg = SkillsLabConfig(workspace_path="/tmp/test-ws")
        assert cfg.db_path.endswith("brain.db")
        assert cfg.cache_dir.endswith(".cache")
        assert cfg.skills_dir.endswith("skills")

    def test_empty_workspace(self):
        cfg = SkillsLabConfig(workspace_path="")
        assert cfg.db_path == ""
        assert cfg.cache_dir == ""

    def test_validate_clean(self):
        cfg = SkillsLabConfig(workspace_path="/tmp/test")
        warnings = cfg.validate()
        assert warnings == []

    def test_validate_no_workspace(self):
        cfg = SkillsLabConfig()
        warnings = cfg.validate()
        assert any("workspace_path" in w for w in warnings)

    def test_validate_bad_top_k(self):
        cfg = SkillsLabConfig(search_top_k=100)
        warnings = cfg.validate()
        assert any("search_top_k" in w for w in warnings)

    def test_validate_bad_dedup_threshold(self):
        cfg = SkillsLabConfig(dedup_threshold=1.5)
        warnings = cfg.validate()
        assert any("dedup_threshold" in w for w in warnings)

    def test_detect_repo_name_from_cwd(self, monkeypatch):
        monkeypatch.chdir("/tmp")
        cfg = SkillsLabConfig()
        name = cfg.detect_repo_name()
        assert name == "tmp"

    def test_detect_repo_name_from_workspace_folder(self):
        cfg = SkillsLabConfig(workspace_folder="/home/user/my-project")
        name = cfg.detect_repo_name()
        assert name == "my-project"

    def test_env_override_top_k(self, monkeypatch):
        monkeypatch.setenv("SEARCH_TOP_K", "20")
        cfg = SkillsLabConfig()
        assert cfg.search_top_k == 20

    def test_env_override_embedding_model(self, monkeypatch):
        monkeypatch.setenv("EMBEDDING_MODEL", "test-model")
        cfg = SkillsLabConfig()
        assert cfg.embedding_model == "test-model"

    def test_env_override_dashboard_port(self, monkeypatch):
        monkeypatch.setenv("SKILLS_LAB_PORT", "9000")
        cfg = SkillsLabConfig()
        assert cfg.dashboard_port == 9000


class TestConfigSingleton:
    """Tests for get_config / reset_config."""

    def test_get_config_returns_config(self):
        reset_config()
        cfg = get_config()
        assert isinstance(cfg, SkillsLabConfig)

    def test_get_config_same_instance(self):
        reset_config()
        a = get_config()
        b = get_config()
        assert a is b

    def test_reset_config_new_instance(self):
        reset_config()
        a = get_config()
        reset_config()
        b = get_config()
        assert a is not b
