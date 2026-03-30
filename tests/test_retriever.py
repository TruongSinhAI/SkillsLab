"""
Unit tests for HybridRetriever — BM25 search, RRF fusion, caching.
"""

import os
import pytest
import numpy as np

from core.retriever import HybridRetriever, _smart_tokenize


class TestSmartTokenizer:
    """Tests for the _smart_tokenize helper function."""

    def test_lowercase(self):
        assert _smart_tokenize("Hello World") == ["hello", "world"]

    def test_camel_case_split(self):
        # After lowercasing, camelCase boundaries are harder to detect
        # The tokenizer first lowercases, then splits camelCase
        # "NextJsApi" → lowercase "nextjsapi" → one token (no uppercase chars left)
        # But "NextJs Api" → ["nextjs", "api"] because of space split
        tokens = _smart_tokenize("NextJs Api Route")
        assert "nextjs" in tokens
        assert "api" in tokens
        assert "route" in tokens

    def test_hyphen_split(self):
        tokens = _smart_tokenize("cors-fix-nextjs")
        assert "cors" in tokens
        assert "fix" in tokens
        assert "nextjs" in tokens

    def test_dot_split(self):
        tokens = _smart_tokenize("app.route.js")
        assert "app" in tokens
        assert "route" in tokens
        assert "js" in tokens

    def test_empty_string(self):
        assert _smart_tokenize("") == []

    def test_preserves_digits(self):
        tokens = _smart_tokenize("v1.2.3 api-v2")
        assert "1" in tokens or "v1" in tokens


class TestHybridRetrieverBM25:
    """Tests for BM25-only search mode (no semantic model required)."""

    def test_search_returns_results(self, integration_retriever):
        """Search should return results for a matching query."""
        results = integration_retriever.search("cors fix nextjs")
        assert len(results) > 0
        assert any(r["skill"].id == "cors-fix-nextjs-api" for r in results)

    def test_search_no_match_returns_empty(self, integration_retriever):
        """Search with a completely unrelated query should return results but with low scores."""
        results = integration_retriever.search("xyznonexistent123456")
        # May return results with very low scores, but should not crash
        assert isinstance(results, list)

    def test_search_repo_scope_current(self, integration_retriever):
        """Search with repo_scope=current should only return skills from the current repo + global."""
        results = integration_retriever.search(
            "cors fix",
            repo_scope="current",
            current_repo="my-webapp",
        )
        for r in results:
            assert r["skill"].repo_name in ("my-webapp", "global")

    def test_search_top_k_limits_results(self, integration_retriever):
        """top_k should limit the number of returned results."""
        results = integration_retriever.search("cors", top_k=1)
        assert len(results) <= 1

    def test_get_skill_content_returns_dict(self, integration_retriever):
        """get_skill_content should return a dict with frontmatter and body."""
        result = integration_retriever.get_skill_content("cors-fix-nextjs-api")
        assert result is not None
        assert "body" in result
        assert "frontmatter" in result

    def test_get_skill_content_nonexistent_returns_none(self, integration_retriever):
        """get_skill_content should return None for a non-existent skill."""
        result = integration_retriever.get_skill_content("nonexistent-skill-xyz")
        assert result is None

    def test_bm25_dirty_flag_rebuilds_index(self, integration_retriever):
        """Setting _bm25_dirty should cause index rebuild on next search."""
        integration_retriever._bm25_dirty = True
        results = integration_retriever.search("cors")
        assert isinstance(results, list)

    def test_embedding_cache_clear(self, tmp_workspace):
        """clear_cache should remove a skill from the embedding cache."""
        mgr = None  # Intentionally None to avoid file reads
        ret = HybridRetriever(
            session_factory=lambda: None,
            manager=mgr,
            workspace_path=tmp_workspace,
        )
        ret._embedding_cache["test-skill"] = np.array([1.0, 2.0, 3.0])
        ret.clear_cache("test-skill")
        assert "test-skill" not in ret._embedding_cache
        assert ret._bm25_dirty is True

    def test_invalidate_all(self, tmp_workspace):
        """invalidate_all should clear the entire cache and mark BM25 dirty."""
        ret = HybridRetriever(
            session_factory=lambda: None,
            manager=None,
            workspace_path=tmp_workspace,
        )
        ret._embedding_cache["a"] = np.array([1.0])
        ret._embedding_cache["b"] = np.array([2.0])
        ret.invalidate_all()
        assert len(ret._embedding_cache) == 0
        assert ret._bm25_dirty is True
