"""
Skills Lab — Shared Pytest Fixtures

Provides reusable fixtures for all test modules:
- tmp_workspace: isolated temporary directory
- db_session: initialized database session (function-scoped)
- skill_manager: SKILLManager bound to tmp_workspace
- sample_skill_data: dict with representative test skill data

Integration test fixtures (module-scoped):
- integration_workspace: long-lived workspace with pre-archived skills
"""

import os
import sys
from datetime import datetime, timezone

import pytest

# Ensure the project root is on sys.path so "from core.*" works everywhere.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.models import Skill, SkillChangelog, init_db, get_session
from core.manager import SKILLManager
from core.retriever import HybridRetriever
from core.evolver import EvolutionEngine


# ---------------------------------------------------------------------------
# Per-function fixtures (unit tests)
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_workspace(tmp_path):
    """Return a clean temporary workspace directory (tmp_path)."""
    return str(tmp_path)


@pytest.fixture
def db_session(tmp_workspace):
    """
    Initialize an in-memory database in tmp_workspace and yield a session.

    The session is closed after each test so the global engine/factory
    is re-initialized for the next test function.
    """
    init_db(tmp_workspace)
    session = get_session()
    yield session
    session.close()


@pytest.fixture
def skill_manager(tmp_workspace):
    """Return a SKILLManager backed by tmp_workspace."""
    return SKILLManager(tmp_workspace)


@pytest.fixture
def sample_skill_data():
    """Return a dict with representative skill fields for testing."""
    return {
        "name": "test-cors-fix",
        "description": "Fix CORS errors on API routes",
        "body": (
            "# CORS Fix\n\n"
            "## When to Use\n"
            "- CORS 403 error\n\n"
            "## Solution\n"
            "```javascript\n"
            "app.use(cors({credentials: true}))\n"
            "```\n"
        ),
        "display_name": "Test CORS Fix",
        "skill_type": "TROUBLESHOOTING",
        "repo_name": "my-webapp",
        "tags": ["cors", "api", "javascript"],
        "ttl_days": None,
        "author": "tester",
    }


# ---------------------------------------------------------------------------
# Module-scoped integration fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def integration_workspace(tmp_path_factory):
    """
    Create a long-lived workspace with a pre-initialized database.

    This fixture is shared across all integration test functions in a module.
    It creates two base skills (cors-fix-nextjs-api, docker-multi-stage-build)
    so that downstream tests can search, fix, derive, merge, etc.
    """
    ws = str(tmp_path_factory.mktemp("integration_ws"))
    init_db(ws)

    mgr = SKILLManager(ws)

    def _mark_dirty(_name):
        pass

    session = get_session()
    try:
        eng = EvolutionEngine(
            session=session,
            manager=mgr,
            on_embedding_cache_clear=_mark_dirty,
        )

        # Skill 1: cors-fix-nextjs-api (repo: my-webapp)
        s1 = eng.archive(
            name="cors-fix-nextjs-api",
            description="Debug CORS errors on Next.js API routes with credentials",
            body=(
                "# CORS Fix for Next.js\n\n"
                "## When to Use\n"
                "- CORS 403 error\n\n"
                "## Solution\n"
                "```typescript\n"
                "app.use(cors({credentials:true}))\n"
                "```\n"
            ),
            skill_type="TROUBLESHOOTING",
            repo_name="my-webapp",
            tags=["cors", "nextjs", "api"],
        )

        # Skill 2: docker-multi-stage-build (repo: global)
        s2 = eng.archive(
            name="docker-multi-stage-build",
            description="Dockerfile multi-stage build for Node.js apps",
            body=(
                "# Docker Multi-Stage Build\n\n"
                "## Solution\n"
                "```dockerfile\n"
                "FROM node:18-alpine AS builder\n"
                "```\n"
            ),
            skill_type="IMPLEMENTATION",
            repo_name="global",
            tags=["docker", "nodejs", "build"],
        )

        session.commit()
    finally:
        session.close()

    yield ws


@pytest.fixture(scope="module")
def integration_engine(integration_workspace):
    """
    Provide an EvolutionEngine bound to the integration workspace.

    A new session is created for each test that uses this fixture.
    """
    ws = integration_workspace
    mgr = SKILLManager(ws)

    def _mark_dirty(_name):
        pass

    session = get_session()
    eng = EvolutionEngine(
        session=session,
        manager=mgr,
        on_embedding_cache_clear=_mark_dirty,
    )
    yield eng
    session.close()


@pytest.fixture(scope="module")
def integration_retriever(integration_workspace):
    """
    Provide a HybridRetriever in BM25-only mode (semantic disabled).
    """
    ws = integration_workspace
    mgr = SKILLManager(ws)
    ret = HybridRetriever(
        session_factory=get_session,
        manager=mgr,
        top_k=5,
    )
    # Force BM25-only mode for deterministic tests
    ret._model_loaded = True
    ret._semantic_available = False
    return ret


@pytest.fixture(scope="module")
def integration_manager(integration_workspace):
    """Return a SKILLManager bound to the integration workspace."""
    return SKILLManager(integration_workspace)
