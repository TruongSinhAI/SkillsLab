"""
Skills Lab — Centralized Configuration

All configurable parameters in one place. Supports environment variables
and .env file overrides. Provides sensible defaults for local development.
"""

import os
import threading
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SkillsLabConfig:
    """Centralized configuration for Skills Lab.

    Values are resolved in priority order:
        1. Explicit constructor argument
        2. Environment variable
        3. .env file (via python-dotenv, loaded at app startup)
        4. Default value defined here
    """

    # --- Paths ---
    workspace_path: str = ""

    # --- Embedding / Search ---
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    fallback_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    search_top_k: int = 5
    rrf_k: int = 60
    dedup_threshold: float = 0.85

    # --- Database ---
    db_filename: str = "brain.db"

    # --- Server ---
    dashboard_host: str = "0.0.0.0"
    dashboard_port: int = 7788

    # --- Logging ---
    log_level: str = "INFO"
    log_format: str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

    # --- Workspace detection ---
    workspace_folder: str = ""

    def __post_init__(self):
        """Resolve values from environment variables after construction."""
        if not self.workspace_path:
            self.workspace_path = os.environ.get(
                "SKILLS_LAB_WORKSPACE", ""
            )
        self.embedding_model = os.environ.get(
            "EMBEDDING_MODEL", self.embedding_model
        )
        self.search_top_k = int(os.environ.get("SEARCH_TOP_K", str(self.search_top_k)))
        self.dashboard_host = os.environ.get("SKILLS_LAB_HOST", self.dashboard_host)
        self.dashboard_port = int(os.environ.get("SKILLS_LAB_PORT", str(self.dashboard_port)))
        self.log_level = os.environ.get("SKILLS_LAB_LOG_LEVEL", self.log_level)
        self.workspace_folder = os.environ.get("WORKSPACE_FOLDER", self.workspace_folder)

    @property
    def db_path(self) -> str:
        """Full path to the SQLite database file."""
        return os.path.join(self.workspace_path, self.db_filename) if self.workspace_path else ""

    @property
    def cache_dir(self) -> str:
        """Full path to the cache directory (embeddings, BM25, etc.)."""
        return os.path.join(self.workspace_path, ".cache") if self.workspace_path else ""

    @property
    def skills_dir(self) -> str:
        """Full path to the skills directory."""
        return os.path.join(self.workspace_path, "skills") if self.workspace_path else ""

    def detect_repo_name(self) -> str:
        """Auto-detect the current repository name from workspace or cwd."""
        if self.workspace_folder:
            return os.path.basename(self.workspace_folder.rstrip("/\\"))
        return os.path.basename(os.getcwd())

    def validate(self) -> list[str]:
        """Validate configuration and return a list of warnings."""
        warnings = []
        if not self.workspace_path:
            warnings.append("workspace_path is not set. Skills will not be persisted.")
        if self.search_top_k < 1 or self.search_top_k > 50:
            warnings.append(f"search_top_k={self.search_top_k} is out of recommended range [1, 50].")
        if self.dedup_threshold < 0.5 or self.dedup_threshold > 1.0:
            warnings.append(f"dedup_threshold={self.dedup_threshold} is out of recommended range [0.5, 1.0].")
        return warnings


# Global singleton — used by modules that don't receive config explicitly
_config: Optional[SkillsLabConfig] = None
_config_lock = threading.Lock()


def get_config() -> SkillsLabConfig:
    """Get the global configuration singleton, creating it if needed (thread-safe)."""
    global _config
    if _config is None:
        with _config_lock:
            if _config is None:
                _config = SkillsLabConfig()
    return _config


def reset_config() -> None:
    """Reset the global config (useful for testing)."""
    global _config
    with _config_lock:
        _config = None
