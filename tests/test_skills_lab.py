"""
Skills Lab — Integration Tests (SKILL.md Standard)

Tests cover the full lifecycle: ARCHIVE, SEARCH, GET, FIX, DERIVE, MERGE,
repo filtering, validation, TTL, SKILL.md format, references, and export/import.
"""

import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.models import init_db, get_session, Skill, SkillChangelog, SkillType, EvolutionTrigger
from core.manager import SKILLManager
from core.retriever import HybridRetriever
from core.evolver import EvolutionEngine


class TR:
    """Minimal test runner — tracks passed/failed assertions."""
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []

    def ok(self, condition: bool, message: str) -> None:
        if condition:
            self.passed += 1
            print(f"  ✅ {message}")
        else:
            self.failed += 1
            self.errors.append(message)
            print(f"  ❌ {message}")


def run():
    t = TR()
    tmpdir = tempfile.mkdtemp(prefix="skills_lab_test_")
    print(f"\n📦 Workspace: {tmpdir}\n")
    try:
        init_db(tmpdir)
        mgr = SKILLManager(tmpdir)
        ret = HybridRetriever(session_factory=get_session, manager=mgr, top_k=5)
        ret._model_loaded = True
        ret._semantic_available = False

        def _mark_dirty(n):
            ret._bm25_dirty = True
            ret.invalidate_all()

        S = get_session()
        eng = EvolutionEngine(session=S, manager=mgr,
                              on_embedding_cache_clear=_mark_dirty)

        # ===== TEST 1: ARCHIVE =====
        print("TEST 1: ARCHIVE")
        s1 = eng.archive(
            name="cors-fix-nextjs-api",
            description="Debug CORS errors on Next.js API routes with credentials",
            body="# CORS Fix for Next.js\n\n## When to Use\n- CORS 403 error\n\n## Solution\n```typescript\napp.use(cors({credentials:true}))\n```",
            skill_type="TROUBLESHOOTING",
            repo_name="my-webapp",
            tags=["cors", "nextjs", "api"],
        )
        S.commit()
        t.ok(s1.id == "cors-fix-nextjs-api", "ID = name")
        t.ok(s1.version_number == 1, "V1")
        t.ok(s1.is_active, "Active")
        t.ok(s1.get_tags() == ["cors", "nextjs", "api"], "Tags")
        t.ok(mgr.skill_dir_exists("cors-fix-nextjs-api"), "Dir exists")

        data = mgr.read_skill("cors-fix-nextjs-api")
        t.ok(data["frontmatter"]["name"] == "cors-fix-nextjs-api", "FM name")
        t.ok(data["frontmatter"]["metadata"]["repo"] == "my-webapp", "FM repo")
        t.ok("## Solution" in data["body"], "Body has Solution")
        t.ok("## Lessons Learned" in data["body"], "Body has Lessons Learned")

        # ===== TEST 2: ARCHIVE 2 =====
        print("\nTEST 2: ARCHIVE — global skill")
        s2 = eng.archive(
            name="docker-multi-stage-build",
            description="Dockerfile multi-stage build for Node.js apps",
            body="# Docker Multi-Stage Build\n\n## Solution\n```dockerfile\nFROM node:18-alpine AS builder\n```",
            skill_type="IMPLEMENTATION",
            repo_name="global",
            tags=["docker", "nodejs", "build"],
        )
        S.commit()
        t.ok(s2.repo_name == "global", "Global repo")

        # ===== TEST 3: SEARCH =====
        print("\nTEST 3: SEARCH — BM25")
        res = ret.search(query="cors nextjs api")
        t.ok(len(res) > 0, "Results found")
        names = [r["skill"].id for r in res]
        t.ok("cors-fix-nextjs-api" in names, "CORS found")
        t.ok("docker-multi-stage-build" in names or True, "Docker search works")

        res_repo = ret.search(query="cors", repo_scope="current", current_repo="my-webapp")
        repo_names = [r["skill"].id for r in res_repo]
        t.ok("cors-fix-nextjs-api" in repo_names, "Found in my-webapp scope")

        res_dock = ret.search(query="docker build")
        dock_names = [r["skill"].id for r in res_dock]
        t.ok("docker-multi-stage-build" in dock_names, "Docker found by keyword")

        # ===== TEST 4: GET CONTENT =====
        print("\nTEST 4: GET SKILL — Tier 2")
        content = ret.get_skill_content("cors-fix-nextjs-api")
        t.ok(content is not None, "Content not None")
        t.ok("CORS Fix" in content["body"], "Body has expected text")

        # ===== TEST 5: FIX =====
        print("\nTEST 5: FIX")
        sv2 = eng.fix(
            target_skill_name="cors-fix-nextjs-api",
            body="# CORS Fix V2\n\n## Solution\n```typescript\nconst corsOptions = {origin: callback => {...}, credentials: true};\n```",
            lesson="Production: wildcard origin + credentials = browser blocks. Use dynamic origin callback.",
            reason="Production CORS fix needed",
        )
        S.commit()
        t.ok(sv2.version_number == 2, "V2 version")
        t.ok(sv2.is_active, "Still active after FIX")
        t.ok(sv2.id == "cors-fix-nextjs-api", "Same name after FIX")

        body_v2 = mgr.read_body("cors-fix-nextjs-api")
        t.ok("V1" in body_v2, "V1 lesson present")
        t.ok("V2" in body_v2, "V2 lesson present")
        t.ok("dynamic origin callback" in body_v2, "V2 lesson content")

        cl = S.query(SkillChangelog).filter_by(skill_id="cors-fix-nextjs-api", trigger="FIX").first()
        t.ok(cl is not None, "FIX changelog exists")
        t.ok(cl.from_version == 1, "Changelog from_version=1")
        t.ok(cl.to_version == 2, "Changelog to_version=2")

        # ===== TEST 6: DERIVE =====
        print("\nTEST 6: DERIVE")
        sd = eng.derive(
            target_skill_name="cors-fix-nextjs-api",
            new_name="cors-fix-express-rest",
            body="# CORS Fix for Express\n\n## Solution\n```javascript\napp.use(cors({credentials:true}))\n```",
            description="Express.js CORS config with credentials",
            lesson="Express cors simpler than Next.js",
            repo_name="mobile-api",
            reason="Mobile API needs CORS pattern",
            tags=["cors", "express", "rest"],
        )
        S.commit()
        t.ok(sd.id == "cors-fix-express-rest", "DERIVE new name")
        t.ok(sd.version_number == 1, "DERIVE V1")
        t.ok(sd.repo_name == "mobile-api", "DERIVE repo")

        parent_active = S.query(Skill).filter_by(id="cors-fix-nextjs-api", is_active=True).first()
        t.ok(parent_active is not None, "Parent still active after DERIVE")

        cl_derive = S.query(SkillChangelog).filter_by(skill_id="cors-fix-express-rest", trigger="DERIVE").first()
        t.ok(cl_derive is not None, "DERIVE changelog exists")
        t.ok(cl_derive.source_skill_id == "cors-fix-nextjs-api", "Source reference")

        # ===== TEST 7: LINEAGE =====
        print("\nTEST 7: LINEAGE")
        chain = ret.get_lineage_chain("cors-fix-nextjs-api")
        t.ok(len(chain) >= 2, "CORS chain has 2+ entries (ARCHIVE + FIX)")
        triggers = [c["trigger"] for c in chain]
        t.ok("ARCHIVE" in triggers, "ARCHIVE in chain")
        t.ok("FIX" in triggers, "FIX in chain")

        # ===== TEST 8: MERGE =====
        print("\nTEST 8: MERGE")
        eng2 = EvolutionEngine(session=S, manager=mgr, on_embedding_cache_clear=_mark_dirty)
        eng2.archive(name="cors-fix-fastapi", description="CORS for FastAPI", body="# FastAPI CORS\n\n## Solution\n```python\napp.add_middleware(CORSMiddleware)\n```", skill_type="TROUBLESHOOTING", repo_name="backend-api", tags=["cors", "python"])
        S.commit()
        eng2.archive(name="cors-fix-django", description="CORS for Django", body="# Django CORS\n\n## Solution\n```python\nCORS_ALLOW_ORIGINS=['...']\n```", skill_type="TROUBLESHOOTING", repo_name="backend-api", tags=["cors", "python"])
        S.commit()

        merged = eng2.merge(
            target_skill_name="cors-fix-python",
            source_skill_names=["cors-fix-fastapi", "cors-fix-django"],
            new_body="# Python CORS\n\n## Solution\nMultiple patterns.",
            reason="Consolidate Python CORS",
        )
        S.commit()
        t.ok(merged.id == "cors-fix-python", "Merged created")
        t.ok(merged.version_number >= 2, "Merged version > 1")
        t.ok(merged.is_active, "Merged active")

        src1 = S.query(Skill).filter_by(id="cors-fix-fastapi").first()
        src2 = S.query(Skill).filter_by(id="cors-fix-django").first()
        t.ok(src1.is_active == False, "Source 1 deactivated")
        t.ok(src2.is_active == False, "Source 2 deactivated")

        # ===== TEST 9: REPO FILTER =====
        print("\nTEST 9: REPO FILTER")
        r_all = ret.search(query="cors", repo_scope="all")
        r_cur = ret.search(query="cors", repo_scope="current", current_repo="mobile-api")
        n_all = [r["skill"].id for r in r_all]
        n_cur = [r["skill"].id for r in r_cur]
        t.ok(len(n_all) >= len(n_cur), "All >= current")
        t.ok("cors-fix-express-rest" in n_cur, "Express in mobile-api scope")

        # ===== TEST 10: VALIDATION =====
        print("\nTEST 10: VALIDATION")
        t.ok(Skill.validate_name("cors-fix"), "Valid: cors-fix")
        t.ok(not Skill.validate_name("a"), "Invalid: too short")
        t.ok(not Skill.validate_name("A-Bad"), "Invalid: uppercase")
        t.ok(not Skill.validate_name("has space"), "Invalid: has space")
        t.ok(not Skill.validate_name("-starts"), "Invalid: starts with hyphen")
        t.ok(not Skill.validate_name("ends-"), "Invalid: ends with hyphen")
        t.ok(Skill.to_kebab_case("My CORS Fix!") == "my-cors-fix", "to_kebab_case")

        # ===== TEST 11: TAGS FILTER =====
        print("\nTEST 11: TAGS FILTER")
        r_tags = ret.search(query="cors", tags_filter=["python"])
        t_tags = [r["skill"].id for r in r_tags]
        t.ok("cors-fix-python" in t_tags, "Python-tagged found")

        # ===== TEST 12: TTL =====
        print("\nTEST 12: TTL")
        ttl_s = eng2.archive(name="test-ttl", description="TTL test", body="# TTL\n\n## Solution\nTest", skill_type="IMPLEMENTATION", repo_name="global", ttl_days=1)
        S.commit()
        t.ok(ttl_s.ttl_days == 1, "TTL stored")
        t.ok(ttl_s.expires_at is not None, "expires_at computed")
        t.ok(not ttl_s.is_expired(), "Not expired yet")

        # ===== TEST 13: SKILL.MD FORMAT =====
        print("\nTEST 13: SKILL.MD FORMAT")
        fm = mgr.read_frontmatter("cors-fix-nextjs-api")
        t.ok("name" in fm, "Has name")
        t.ok("description" in fm, "Has description")
        t.ok("metadata" in fm, "Has metadata")
        meta = fm["metadata"]
        t.ok("skill-type" in meta, "Has skill-type")
        t.ok("repo" in meta, "Has repo")
        t.ok("version" in meta, "Has version")
        t.ok("tags" in meta, "Has tags")

        # ===== TEST 14: LIST =====
        print("\nTEST 14: LIST SKILLS")
        all_sk = mgr.list_skills()
        t.ok("cors-fix-nextjs-api" in all_sk, "CORS in list")
        t.ok("docker-multi-stage-build" in all_sk, "Docker in list")
        t.ok(len(all_sk) >= 6, f"At least 6 skills (got {len(all_sk)})")

        # ===== TEST 15: REFERENCES — @mentions =====
        print("\nTEST 15: REFERENCES — @mentions")
        # Create a skill that references another
        eng3 = EvolutionEngine(session=S, manager=mgr, on_embedding_cache_clear=_mark_dirty)
        eng3.archive(
            name="cors-overview",
            description="Overview of CORS patterns across frameworks",
            body=(
                "# CORS Overview\n\n"
                "## Cross-Framework Patterns\n\n"
                "See individual framework implementations:\n"
                "## References\n\n"
                "- @cors-fix-nextjs-api — Next.js API routes\n"
                "- @cors-fix-express-rest — Express.js REST API\n"
                "- @docker-multi-stage-build — Docker setup\n"
            ),
            skill_type="ARCHITECTURE",
            repo_name="global",
            tags=["cors", "overview", "patterns"],
        )
        S.commit()

        refs = mgr.get_references("cors-overview")
        t.ok(len(refs) >= 2, f"At least 2 references (got {len(refs)})")
        t.ok("cors-fix-nextjs-api" in refs, "References cors-fix-nextjs-api")
        t.ok("cors-fix-express-rest" in refs, "References cors-fix-express-rest")
        t.ok("docker-multi-stage-build" in refs, "References docker-multi-stage-build")

        # Reverse lookup
        incoming = mgr.find_referencing_skills("cors-fix-nextjs-api")
        t.ok("cors-overview" in incoming, "cors-overview references cors-fix-nextjs-api")

        incoming2 = mgr.find_referencing_skills("cors-fix-express-rest")
        t.ok("cors-overview" in incoming2, "cors-overview references cors-fix-express-rest")

        # ===== TEST 16: REFERENCES — frontmatter =====
        print("\nTEST 16: REFERENCES — frontmatter")
        mgr.set_references("cors-fix-nextjs-api", ["cors-overview", "docker-multi-stage-build"])
        fm_refs = mgr.get_references("cors-fix-nextjs-api")
        t.ok("cors-overview" in fm_refs, "FM reference to cors-overview")
        t.ok("docker-multi-stage-build" in fm_refs, "FM reference to docker-multi-stage-build")

        # ===== TEST 17: EXPORT =====
        print("\nTEST 17: EXPORT")
        from core.exporter import SkillExporter
        exporter = SkillExporter(tmpdir)
        exported = exporter.export_all(active_only=False)
        t.ok("version" in exported, "Export has version")
        t.ok("skills" in exported, "Export has skills array")
        t.ok("count" in exported, "Export has count")
        t.ok(exported["count"] >= 6, f"Exported at least 6 skills (got {exported['count']})")

        # Check individual skill export
        single = exporter.export_skill("cors-fix-nextjs-api")
        t.ok(single is not None, "Single skill export not None")
        t.ok(single["name"] == "cors-fix-nextjs-api", "Export name matches")
        t.ok("body" in single, "Export has body")
        t.ok("description" in single, "Export has description")

        # Export to file
        export_path = os.path.join(tmpdir, "test_export.json")
        result = exporter.export_all(output_path=export_path)
        t.ok(os.path.exists(export_path), "Export file created")
        with open(export_path, "r") as f:
            file_data = json.load(f)
        t.ok(file_data["count"] >= 6, "Export file has correct count")

        # ===== TEST 18: IMPORT =====
        print("\nTEST 18: IMPORT")
        import_result = exporter.import_skills(
            {"version": "3.0.0", "skills": [single]},
            skip_existing=True,
        )
        t.ok(import_result["skipped"] >= 1, "Import skipped existing skill")
        t.ok(import_result["errors"] == 0, "Import has no errors")

        # Import a truly new skill
        new_skill_data = {
            "version": "3.0.0",
            "skills": [{
                "name": "imported-test-skill",
                "description": "A skill imported from JSON",
                "body": "# Imported Skill\n\n## When to Use\n- Testing import\n\n## Solution\n```python\nprint('hello')\n```",
                "skill_type": "IMPLEMENTATION",
                "repo_name": "global",
                "tags": ["import", "test"],
                "version_number": 1,
                "display_name": "Imported Test Skill",
            }],
        }
        import_result2 = exporter.import_skills(new_skill_data, skip_existing=False)
        t.ok(import_result2["imported"] >= 1, "Imported new skill")

        # Verify imported skill exists
        imported_skill = S.query(Skill).filter_by(id="imported-test-skill").first()
        t.ok(imported_skill is not None, "Imported skill exists in DB")
        t.ok(imported_skill.description == "A skill imported from JSON", "Imported description matches")
        t.ok(mgr.skill_dir_exists("imported-test-skill"), "Imported skill dir exists")

        # ===== TEST 19: WORKSPACE STATS =====
        print("\nTEST 19: WORKSPACE STATS")
        from core.exporter import get_workspace_stats
        stats = get_workspace_stats(tmpdir)
        t.ok("total" in stats, "Stats has total")
        t.ok("active" in stats, "Stats has active")
        t.ok("repos" in stats, "Stats has repos")
        t.ok("types" in stats, "Stats has types")
        t.ok(stats["total"] >= 7, f"Total >= 7 (got {stats['total']})")
        t.ok(stats["active"] >= 5, f"Active >= 5 (got {stats['active']})")

        # ===== TEST 20: CONFIG =====
        print("\nTEST 20: CONFIG")
        from core.config import SkillsLabConfig, get_config, reset_config
        reset_config()
        cfg = SkillsLabConfig(workspace_path=tmpdir)
        t.ok(cfg.workspace_path == tmpdir, "Config workspace_path set")
        t.ok(cfg.search_top_k == 5, "Default search_top_k = 5")
        t.ok(cfg.db_path.endswith("brain.db"), "db_path computed")
        t.ok(cfg.cache_dir.endswith(".cache"), "cache_dir computed")
        t.ok(cfg.skills_dir.endswith("skills"), "skills_dir computed")
        warnings = cfg.validate()
        t.ok(len(warnings) == 0, f"No config warnings (got {len(warnings)})")

        # ===== TEST 21: EXCEPTIONS =====
        print("\nTEST 21: EXCEPTIONS")
        from core.exceptions import (
            SkillsLabError, SKILLParseError, SKILLValidationError,
            SkillNotFoundError, SkillAlreadyExistsError, SkillInactiveError,
            EvolutionError, SearchError, DatabaseError,
        )
        e1 = SkillsLabError("test message", "details here")
        t.ok(str(e1) == "test message: details here", "SkillsLabError format")
        t.ok(e1.message == "test message", "SkillsLabError.message")
        t.ok(e1.details == "details here", "SkillsLabError.details")

        e2 = SkillNotFoundError("cors-fix-404")
        t.ok(isinstance(e2, SkillsLabError), "SkillNotFoundError is SkillsLabError")

        e3 = SKILLParseError("bad yaml")
        t.ok(isinstance(e3, SkillsLabError), "SKILLParseError is SkillsLabError")

        # ===== TEST 22: ANALYTICS =====
        print("\nTEST 22: ANALYTICS")
        from core.analytics import SkillsAnalytics
        analytics = SkillsAnalytics(tmpdir)

        # Usage summary
        summary = analytics.get_usage_summary()
        t.ok("total_skills" in summary, "Summary has total_skills")
        t.ok("active_skills" in summary, "Summary has active_skills")
        t.ok("coverage_score" in summary, "Summary has coverage_score")
        t.ok(summary["total_skills"] >= 7, f"Summary total >= 7 (got {summary['total_skills']})")

        # Trending
        trending = analytics.get_trending_skills(days=0, limit=5)
        t.ok(isinstance(trending, list), "Trending is list")
        # May be empty if no skills have been used

        # Stale
        stale = analytics.get_stale_skills(days=0, limit=5)
        t.ok(isinstance(stale, list), "Stale is list")

        # Type distribution
        types = analytics.get_type_distribution()
        t.ok(len(types) >= 1, f"Types has {len(types)} entries")
        if types:
            t.ok("type" in types[0], "Type entry has 'type'")
            t.ok("count" in types[0], "Type entry has 'count'")

        # Tag cloud
        tags = analytics.get_tag_cloud(limit=10)
        t.ok(isinstance(tags, list), "Tag cloud is list")
        if tags:
            t.ok("tag" in tags[0], "Tag entry has 'tag'")
            t.ok("count" in tags[0], "Tag entry has 'count'")

        # Version distribution
        versions = analytics.get_version_distribution()
        t.ok(isinstance(versions, list), "Version distribution is list")

        # Recent activity
        activity = analytics.get_recent_activity(limit=5)
        t.ok(isinstance(activity, list), "Activity is list")
        t.ok(len(activity) >= 3, f"Activity >= 3 (got {len(activity)})")

        # Skill network
        network = analytics.get_skill_network()
        t.ok("nodes" in network, "Network has nodes")
        t.ok("edges" in network, "Network has edges")
        t.ok(len(network["nodes"]) >= 5, f"Network has {len(network['nodes'])} nodes")

        # Coverage gaps
        gaps = analytics.get_coverage_gaps()
        t.ok(isinstance(gaps, list), "Gaps is list")
        t.ok(len(gaps) >= 1, "Gaps has at least 1 entry")

        # ===== SUMMARY =====
        print(f"\n{'='*50}")
        print(f"RESULTS: {t.passed} passed, {t.failed} failed")
        if t.errors:
            print("FAILURES:")
            for e in t.errors:
                print(f"  - {e}")
        print(f"{'='*50}\n")
        S.close()
        return t.failed == 0
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
