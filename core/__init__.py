"""
Skills Lab — Core Package

Provides data models, SKILL.md file management, evolution engine,
and hybrid search (BM25 + semantic + RRF fusion).
"""

from core.models import (
    Base,
    Skill,
    SkillChangelog,
    SkillLineage,
    SkillType,
    EvolutionTrigger,
    init_db,
    get_session,
)
from core.config import SkillsLabConfig, get_config, reset_config
from core.exceptions import (
    SkillsLabError,
    SKILLParseError,
    SKILLValidationError,
    SkillNotFoundError,
    SkillAlreadyExistsError,
    SkillInactiveError,
    EvolutionError,
    SearchError,
    DatabaseError,
)

__all__ = [
    # Models
    "Base",
    "Skill",
    "SkillChangelog",
    "SkillLineage",
    "SkillType",
    "EvolutionTrigger",
    "init_db",
    "get_session",
    # Config
    "SkillsLabConfig",
    "get_config",
    "reset_config",
    # Exceptions
    "SkillsLabError",
    "SKILLParseError",
    "SKILLValidationError",
    "SkillNotFoundError",
    "SkillAlreadyExistsError",
    "SkillInactiveError",
    "EvolutionError",
    "SearchError",
    "DatabaseError",
]
