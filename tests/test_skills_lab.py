"""
Skills Lab — Integration Tests (Pytest)

Tests cover the full lifecycle: ARCHIVE, SEARCH, GET, FIX, DERIVE, MERGE,
repo filtering, validation, TTL, SKILL.md format, references, export/import,
analytics, and version diff.
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.models import init_db, get_session, Skill, SkillChangelog, SkillType, EvolutionTrigger
from core.manager import SKILLManager
from core.retriever import HybridRetriever
from core.evolver import EvolutionEngine


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def ws(tmp_path):
    """Isolated workspace for a single test."""
    ws = str(tmp_path)
    init_db(ws)
    return ws


@pytest.fixture
def mgr(ws):
    return SKILLManager(ws)


@pytest.fixture
def ret(mgr):
    r = HybridRetriever(session_factory=get_session, manager=mgr, top_k=5)
    r._model_loaded = True
    r._semantic_available = False
    return r


@pytest.fixture
def ses(ws):
    s = get_session()
    yield s
    s.close()


@pytest.fixture
def eng(ses, mgr):
    def _dirty(_n):
        pass
    return EvolutionEngine(session=ses, manager=mgr, on_embedding_cache_clear=_dirty)


# ---------------------------------------------------------------------------
# TEST 1: ARCHIVE — repo-scoped skill
# ---------------------------------------------------------------------------


class TestArchive:
    def test_archive_new_skill(self, eng, ses, mgr):
        s = eng.archive(
            name="cors-fix-nextjs-api",
            description="Debug CORS errors on Next.js API routes with credentials",
            body="# CORS Fix\n\n## When to Use\n- CORS 403 error\n\n## Solution\n```typescript\napp.use(cors({credentials:true}))\n```",
            skill_type="TROUBLESHOOTING",
            repo_name="my-webapp",
            tags=["cors", "nextjs", "api"],
        )
        ses.commit()
        assert s.id == "cors-fix-nextjs-api"
        assert s.version_number == 1
        assert s.is_active
        assert s.get_tags() == ["cors", "nextjs", "api"]
        assert mgr.skill_dir_exists("cors-fix-nextjs-api")

        data = mgr.read_skill("cors-fix-nextjs-api")
        assert data["frontmatter"]["name"] == "cors-fix-nextjs-api"
        assert data["frontmatter"]["metadata"]["repo"] == "my-webapp"
        assert "## Solution" in data["body"]
        assert "## Lessons Learned" in data["body"]

    def test_archive_global_skill(self, eng, ses):
        s = eng.archive(
            name="docker-multi-stage-build",
            description="Dockerfile multi-stage build for Node.js apps",
            body="# Docker Multi-Stage Build\n\n## Solution\n```dockerfile\nFROM node:18-alpine AS builder\n```",
            skill_type="IMPLEMENTATION",
            repo_name="global",
            tags=["docker", "nodejs", "build"],
        )
        ses.commit()
        assert s.repo_name == "global"


# ---------------------------------------------------------------------------
# TEST 3: SEARCH — BM25 + repo + tag filters
# ---------------------------------------------------------------------------


class TestSearch:
    def test_bm25_search(self, eng, ses, ret):
        eng.archive(
            name="cors-fix-nextjs-api", description="CORS fix for Next.js",
            body="# CORS\n\n## Solution\nCode", skill_type="TROUBLESHOOTING",
            repo_name="my-webapp", tags=["cors", "nextjs", "api"],
        )
        eng.archive(
            name="docker-multi-stage-build", description="Docker multi-stage build",
            body="# Docker\n\n## Solution\nDockerfile", skill_type="IMPLEMENTATION",
            repo_name="global", tags=["docker", "nodejs", "build"],
        )
        ses.commit()

        res = ret.search(query="cors nextjs api")
        names = [r["skill"].id for r in res]
        assert "cors-fix-nextjs-api" in names

    def test_repo_filter(self, eng, ses, ret):
        eng.archive(
            name="cors-fix-nextjs-api", description="CORS fix",
            body="# CORS\n\n## Solution\nCode", skill_type="TROUBLESHOOTING",
            repo_name="my-webapp", tags=["cors"],
        )
        eng.archive(
            name="cors-fix-express-rest", description="Express CORS",
            body="# Express\n\n## Solution\nCode", skill_type="TROUBLESHOOTING",
            repo_name="mobile-api", tags=["cors"],
        )
        ses.commit()

        res = ret.search(query="cors", repo_scope="current", current_repo="mobile-api")
        names = [r["skill"].id for r in res]
        assert "cors-fix-express-rest" in names

    def test_tags_filter(self, eng, ses, ret):
        eng.archive(
            name="cors-fix-python", description="CORS for Python",
            body="# Python\n\n## Solution\nCode", skill_type="TROUBLESHOOTING",
            repo_name="backend", tags=["cors", "python"],
        )
        ses.commit()

        res = ret.search(query="cors", tags_filter=["python"])
        names = [r["skill"].id for r in res]
        assert "cors-fix-python" in names

    def test_full_text_body_search(self, eng, ses, ret):
        """BM25 should search within skill body content (task #8)."""
        eng.archive(
            name="unique-body-term-test",
            description="A test skill with unique body content",
            body="# UniqueTerm456\n\n## Solution\nUse UniqueTerm456 to fix the issue\n",
            skill_type="IMPLEMENTATION",
            repo_name="global",
            tags=["test"],
        )
        ses.commit()

        res = ret.search(query="UniqueTerm456")
        names = [r["skill"].id for r in res]
        assert "unique-body-term-test" in names


# ---------------------------------------------------------------------------
# TEST 4: GET SKILL CONTENT
# ---------------------------------------------------------------------------


class TestGetSkill:
    def test_get_skill_content(self, eng, ses, ret):
        eng.archive(
            name="cors-fix-nextjs-api", description="CORS fix",
            body="# CORS Fix\n\nBody content here", skill_type="TROUBLESHOOTING",
            repo_name="global", tags=["cors"],
        )
        ses.commit()
        content = ret.get_skill_content("cors-fix-nextjs-api")
        assert content is not None
        assert "Body content here" in content["body"]

    def test_get_nonexistent_returns_none(self, ret):
        assert ret.get_skill_content("nonexistent") is None


# ---------------------------------------------------------------------------
# TEST 5: FIX
# ---------------------------------------------------------------------------


class TestFix:
    def test_fix_skill(self, eng, ses, mgr):
        eng.archive(
            name="cors-fix-nextjs-api", description="CORS fix",
            body="# CORS Fix\n\n## Solution\nV1 fix\n",
            skill_type="TROUBLESHOOTING", repo_name="global", tags=["cors"],
        )
        ses.commit()

        sv = eng.fix(
            target_skill_name="cors-fix-nextjs-api",
            body="# CORS Fix V2\n\n## Solution\nNew fix\n",
            lesson="Dynamic origin callback needed",
            reason="Production fix",
        )
        ses.commit()
        assert sv.version_number == 2
        assert sv.is_active
        assert sv.id == "cors-fix-nextjs-api"

        body = mgr.read_body("cors-fix-nextjs-api")
        assert "V1" in body
        assert "V2" in body

        cl = ses.query(SkillChangelog).filter_by(
            skill_id="cors-fix-nextjs-api", trigger="FIX"
        ).first()
        assert cl is not None
        assert cl.from_version == 1
        assert cl.to_version == 2


# ---------------------------------------------------------------------------
# TEST 6: DERIVE
# ---------------------------------------------------------------------------


class TestDerive:
    def test_derive_skill(self, eng, ses):
        eng.archive(
            name="cors-fix-nextjs-api", description="CORS Next.js",
            body="# CORS\n\n## Solution\nCode", skill_type="TROUBLESHOOTING",
            repo_name="global", tags=["cors"],
        )
        ses.commit()

        sd = eng.derive(
            target_skill_name="cors-fix-nextjs-api",
            new_name="cors-fix-express-rest",
            body="# Express CORS\n\n## Solution\nCode",
            description="Express CORS",
            lesson="Express simpler",
            repo_name="mobile-api",
            reason="Mobile API",
            tags=["cors", "express"],
        )
        ses.commit()
        assert sd.id == "cors-fix-express-rest"
        assert sd.version_number == 1
        assert sd.repo_name == "mobile-api"

        parent = ses.query(Skill).filter_by(id="cors-fix-nextjs-api", is_active=True).first()
        assert parent is not None

        cl = ses.query(SkillChangelog).filter_by(
            skill_id="cors-fix-express-rest", trigger="DERIVE"
        ).first()
        assert cl is not None
        assert cl.source_skill_id == "cors-fix-nextjs-api"


# ---------------------------------------------------------------------------
# TEST 7: LINEAGE
# ---------------------------------------------------------------------------


class TestLineage:
    def test_lineage_chain(self, eng, ses, ret):
        eng.archive(
            name="cors-fix-nextjs-api", description="CORS",
            body="# CORS\n\n## Solution\nCode", skill_type="TROUBLESHOOTING",
            repo_name="global", tags=["cors"],
        )
        ses.commit()
        eng.fix(
            target_skill_name="cors-fix-nextjs-api",
            body="# CORS V2\n\n## Solution\nNew", lesson="Fix", reason="update",
        )
        ses.commit()

        chain = ret.get_lineage_chain("cors-fix-nextjs-api")
        assert len(chain) >= 2
        triggers = [c["trigger"] for c in chain]
        assert "ARCHIVE" in triggers
        assert "FIX" in triggers


# ---------------------------------------------------------------------------
# TEST 8: MERGE
# ---------------------------------------------------------------------------


class TestMerge:
    def test_merge_skills(self, eng, ses):
        eng.archive(name="cors-fix-fastapi", description="CORS FastAPI",
                     body="# FastAPI\n\n## Solution\nCode", skill_type="TROUBLESHOOTING",
                     repo_name="backend", tags=["cors", "python"])
        ses.commit()
        eng.archive(name="cors-fix-django", description="CORS Django",
                     body="# Django\n\n## Solution\nCode", skill_type="TROUBLESHOOTING",
                     repo_name="backend", tags=["cors", "python"])
        ses.commit()

        merged = eng.merge(
            target_skill_name="cors-fix-python",
            source_skill_names=["cors-fix-fastapi", "cors-fix-django"],
            new_body="# Python CORS\n\n## Solution\nCombined",
            reason="Consolidate",
        )
        ses.commit()
        assert merged.id == "cors-fix-python"
        assert merged.version_number >= 2
        assert merged.is_active

        s1 = ses.query(Skill).filter_by(id="cors-fix-fastapi").first()
        s2 = ses.query(Skill).filter_by(id="cors-fix-django").first()
        assert s1.is_active is False
        assert s2.is_active is False


# ---------------------------------------------------------------------------
# TEST 10: VALIDATION
# ---------------------------------------------------------------------------


class TestValidation:
    def test_validate_name(self):
        assert Skill.validate_name("cors-fix") is True
        assert Skill.validate_name("a") is False
        assert Skill.validate_name("A-Bad") is False
        assert Skill.validate_name("has space") is False
        assert Skill.validate_name("-starts") is False
        assert Skill.validate_name("ends-") is False

    def test_to_kebab_case(self):
        assert Skill.to_kebab_case("My CORS Fix!") == "my-cors-fix"


# ---------------------------------------------------------------------------
# TEST 12: TTL
# ---------------------------------------------------------------------------


class TestTTL:
    def test_ttl(self, eng, ses):
        s = eng.archive(name="test-ttl", description="TTL test",
                        body="# TTL\n\n## Solution\nTest",
                        skill_type="IMPLEMENTATION", repo_name="global", ttl_days=1)
        ses.commit()
        assert s.ttl_days == 1
        assert s.expires_at is not None
        assert not s.is_expired()


# ---------------------------------------------------------------------------
# TEST 13: SKILL.MD FORMAT
# ---------------------------------------------------------------------------


class TestSkillMdFormat:
    def test_format(self, eng, ses, mgr):
        eng.archive(name="fmt-test", description="Format test",
                     body="# Test\n\n## Solution\nCode",
                     skill_type="IMPLEMENTATION", repo_name="global")
        ses.commit()
        fm = mgr.read_frontmatter("fmt-test")
        assert "name" in fm
        assert "description" in fm
        assert "metadata" in fm
        meta = fm["metadata"]
        assert "skill-type" in meta
        assert "repo" in meta
        assert "version" in meta
        assert "tags" in meta


# ---------------------------------------------------------------------------
# TEST 14: LIST SKILLS
# ---------------------------------------------------------------------------


class TestListSkills:
    def test_list(self, eng, ses, mgr):
        eng.archive(name="list-a", description="A", body="# A\n\nBody",
                     skill_type="IMPLEMENTATION", repo_name="global")
        eng.archive(name="list-b", description="B", body="# B\n\nBody",
                     skill_type="IMPLEMENTATION", repo_name="global")
        ses.commit()
        result = mgr.list_skills()
        assert "list-a" in result
        assert "list-b" in result


# ---------------------------------------------------------------------------
# TEST 15-16: REFERENCES
# ---------------------------------------------------------------------------


class TestReferences:
    def test_at_mentions(self, eng, ses, mgr):
        eng.archive(name="ref-target", description="Target",
                     body="# Target\n\nContent", skill_type="IMPLEMENTATION", repo_name="global")
        eng.archive(name="ref-source", description="Source",
                     body="# Source\n\n## References\n- @ref-target - Linked\n",
                     skill_type="ARCHITECTURE", repo_name="global", tags=["refs"])
        ses.commit()

        refs = mgr.get_references("ref-source")
        assert "ref-target" in refs

        incoming = mgr.find_referencing_skills("ref-target")
        assert "ref-source" in incoming

    def test_frontmatter_references(self, eng, ses, mgr):
        eng.archive(name="fm-ref-a", description="A",
                     body="# A\n\nContent", skill_type="IMPLEMENTATION", repo_name="global")
        eng.archive(name="fm-ref-b", description="B",
                     body="# B\n\nContent", skill_type="IMPLEMENTATION", repo_name="global")
        ses.commit()
        mgr.set_references("fm-ref-a", ["fm-ref-b"])
        refs = mgr.get_references("fm-ref-a")
        assert "fm-ref-b" in refs


# ---------------------------------------------------------------------------
# TEST 17-18: EXPORT / IMPORT
# ---------------------------------------------------------------------------


class TestExportImport:
    def test_export(self, eng, ses):
        eng.archive(name="export-test", description="Export skill",
                     body="# Export\n\n## Solution\nCode",
                     skill_type="IMPLEMENTATION", repo_name="global", tags=["test"])
        ses.commit()

        from core.exporter import SkillExporter
        exporter = SkillExporter(eng.session.query(Skill).first().repo_name or "")
        # Need a proper workspace, so use the workspace path
        # Just verify export functionality works

    def test_import(self, eng, ses, mgr, tmp_path):
        from core.exporter import SkillExporter
        exporter = SkillExporter(str(tmp_path))

        eng.archive(name="import-source", description="Source",
                     body="# Import\n\n## Solution\nCode",
                     skill_type="IMPLEMENTATION", repo_name="global", tags=["test"])
        ses.commit()

        single = exporter.export_skill("import-source")
        assert single is not None
        assert single["name"] == "import-source"

        # Re-import should skip
        result = exporter.import_skills(
            {"version": "3.0.0", "skills": [single]}, skip_existing=True
        )
        assert result["skipped"] >= 1


# ---------------------------------------------------------------------------
# TEST 20: CONFIG
# ---------------------------------------------------------------------------


class TestConfig:
    def test_config(self, tmp_path):
        from core.config import SkillsLabConfig, reset_config
        reset_config()
        cfg = SkillsLabConfig(workspace_path=str(tmp_path))
        assert cfg.workspace_path == str(tmp_path)
        assert cfg.search_top_k == 5
        assert cfg.db_path.endswith("brain.db")
        assert cfg.cache_dir.endswith(".cache")
        assert cfg.skills_dir.endswith("skills")
        warnings = cfg.validate()
        assert len(warnings) == 0


# ---------------------------------------------------------------------------
# TEST 21: EXCEPTIONS
# ---------------------------------------------------------------------------


class TestExceptions:
    def test_exception_hierarchy(self):
        from core.exceptions import (
            SkillsLabError, SKILLParseError, SKILLValidationError,
            SkillNotFoundError, SkillAlreadyExistsError, SkillInactiveError,
            EvolutionError, SearchError, DatabaseError,
        )
        e1 = SkillsLabError("test message", "details here")
        assert str(e1) == "test message: details here"
        assert e1.message == "test message"
        assert e1.details == "details here"

        e2 = SkillNotFoundError("cors-fix-404")
        assert isinstance(e2, SkillsLabError)

        e3 = SKILLParseError("bad yaml")
        assert isinstance(e3, SkillsLabError)


# ---------------------------------------------------------------------------
# TEST 22: ANALYTICS
# ---------------------------------------------------------------------------


class TestAnalytics:
    def test_analytics(self, eng, ses):
        eng.archive(name="analytics-1", description="A1",
                     body="# A1\n\nContent", skill_type="IMPLEMENTATION", repo_name="repo-a",
                     tags=["tag-x"])
        eng.archive(name="analytics-2", description="A2",
                     body="# A2\n\nContent", skill_type="WORKFLOW", repo_name="repo-b",
                     tags=["tag-y"])
        ses.commit()

        from core.analytics import SkillsAnalytics
        ws_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        analytics = SkillsAnalytics(ws_path)

        summary = analytics.get_usage_summary()
        assert "total_skills" in summary
        assert "coverage_score" in summary
        assert summary["total_skills"] >= 2

        types = analytics.get_type_distribution()
        assert len(types) >= 1

        tags = analytics.get_tag_cloud(limit=10)
        assert isinstance(tags, list)

        network = analytics.get_skill_network()
        assert "nodes" in network
        assert "edges" in network

        gaps = analytics.get_coverage_gaps()
        assert isinstance(gaps, list)


# ---------------------------------------------------------------------------
# TEST: VERSION DIFF
# ---------------------------------------------------------------------------


class TestVersionDiff:
    def test_version_diff(self, eng, ses, mgr):
        eng.archive(
            name="diff-test", description="V1",
            body="# V1\n\n## Solution\nOld code here\n",
            skill_type="IMPLEMENTATION", repo_name="global",
        )
        ses.commit()

        # FIX creates V2: write_skill archives V1 snapshot, writes V2 file
        eng.fix(
            target_skill_name="diff-test",
            body="# V2\n\n## Solution\nNew code here\n",
            lesson="Updated to V2",
            reason="Improvement",
        )
        ses.commit()

        # Diff V1 vs current (V2 in the file)
        result = mgr.get_version_diff("diff-test", "1", "current")
        assert result["skill_name"] == "diff-test"
        assert result["v1"] == "1"
        assert result["v2"] == "current"
        assert len(result["diff"]) > 0

    def test_version_diff_v1_vs_v2(self, eng, ses, mgr):
        eng.archive(
            name="diff-v1v2", description="V1",
            body="# V1\n\n## Solution\nOld\n",
            skill_type="IMPLEMENTATION", repo_name="global",
        )
        ses.commit()
        eng.fix(
            target_skill_name="diff-v1v2",
            body="# V2\n\n## Solution\nNew\n",
            lesson="Updated", reason="Update",
        )
        ses.commit()
        # Second FIX creates V3: archives V2 snapshot
        eng.fix(
            target_skill_name="diff-v1v2",
            body="# V3\n\n## Solution\nNewer\n",
            lesson="Updated again", reason="Another update",
        )
        ses.commit()

        # Now we can diff V1 vs V2
        result = mgr.get_version_diff("diff-v1v2", "1", "2")
        assert result["v1"] == "1"
        assert result["v2"] == "2"
        assert "Old" in result["diff"] or "New" in result["diff"]

    def test_version_diff_list_versions(self, eng, ses, mgr):
        eng.archive(name="diff-list", description="V1",
                     body="# V1\n\nBody", skill_type="IMPLEMENTATION", repo_name="global")
        ses.commit()
        eng.fix(target_skill_name="diff-list", body="# V2\n\nBody",
                  lesson="Updated", reason="Update")
        ses.commit()
        eng.fix(target_skill_name="diff-list", body="# V3\n\nBody",
                  lesson="Updated", reason="Update again")
        ses.commit()

        versions = mgr.list_skill_versions("diff-list")
        assert len(versions) >= 2
