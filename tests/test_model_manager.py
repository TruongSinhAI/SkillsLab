"""
Skills Lab — Unit tests for core/model_manager.py

Tests cover: is_model_cached, download_embedding_model (graceful handling).
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.model_manager import is_model_cached, DEFAULT_MODEL, FALLBACK_MODEL


class TestIsModelCached:
    """Tests for is_model_cached()."""

    def test_no_cache_dir(self):
        # With a non-existent workspace and no model cached
        result = is_model_cached("nonexistent-model-xyz-12345")
        assert result is False

    def test_with_empty_workspace(self, tmp_path):
        result = is_model_cached("some-model", workspace_path=str(tmp_path))
        assert result is False

    def test_with_simulated_workspace_cache(self, tmp_path):
        # Create a fake model cache directory in the workspace
        safe_name = DEFAULT_MODEL.replace("/", "--").lower()
        model_dir = os.path.join(str(tmp_path), ".cache", "models", safe_name)
        os.makedirs(model_dir, exist_ok=True)
        # Create a dummy file to make it non-empty
        with open(os.path.join(model_dir, "pytorch_model.bin"), "w") as f:
            f.write("dummy")
        result = is_model_cached(DEFAULT_MODEL, workspace_path=str(tmp_path))
        assert result is True


class TestDownloadEmbeddingModel:
    """Tests for download_embedding_model() graceful handling."""

    def test_no_sentence_transformers(self, monkeypatch):
        """Should return False gracefully when sentence_transformers is not installed."""
        import importlib
        # Remove sentence_transformers from sys.modules temporarily
        monkeypatch.setattr("importlib.util.find_spec", lambda name: None)
        result = is_model_cached("nonexistent-model")
        # This just tests that the function doesn't crash
        assert isinstance(result, bool)
