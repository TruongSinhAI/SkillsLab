"""
Skills Lab — Core Data Models (SKILL.md Standard)

SQLAlchemy ORM models for managing skills metadata, versioning, and changelog.
The database stores indexes and metadata for fast search. Actual skill content
lives in SKILL.md files on disk (filesystem-first architecture).

Key design decisions:
    - Skill name (kebab-case) is the primary key.
    - A FIX operation updates the existing row in-place and appends a changelog entry
      (no new skill row is created).
    - SQLite is used with WAL journal mode for safe concurrent access.
"""

import json
import os
import re
import threading
from datetime import datetime, timedelta, timezone
from enum import Enum

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import Session, declarative_base, relationship, scoped_session

Base = declarative_base()

# Compiled regex for validating kebab-case identifiers.
# Matches strings like "my-skill", "abc-123-def", or "a".
KEBAB_CASE_RE = re.compile(r"^[a-z0-9]([a-z0-9-]*[a-z0-9])?$")


class SkillType(str, Enum):
    """Enumeration of supported skill types.

    Each skill is classified into exactly one of these categories,
    which helps the agent matching engine select the most relevant
    skills for a given task.
    """

    IMPLEMENTATION = "IMPLEMENTATION"
    WORKFLOW = "WORKFLOW"
    TROUBLESHOOTING = "TROUBLESHOOTING"
    ARCHITECTURE = "ARCHITECTURE"
    RULE = "RULE"


class EvolutionTrigger(str, Enum):
    """Enumeration of events that can trigger a skill version transition.

    Used in the changelog to record why a skill was evolved:
        ARCHIVE  — Skill was archived/deactivated.
        FIX      — A bug fix or minor correction was applied.
        DERIVE   — A new skill was derived from an existing one.
        MERGE    — Two or more skills were merged into one.
    """

    ARCHIVE = "ARCHIVE"
    FIX = "FIX"
    DERIVE = "DERIVE"
    MERGE = "MERGE"


class Skill(Base):
    """ORM model for the **skills** table.

    Each row represents exactly one skill identified by its kebab-case name
    (which is also the primary key).  When a FIX is applied the existing row
    is updated in-place (the ``version_number`` is incremented and a new
    ``SkillChangelog`` entry is appended); no additional row is created.

    Attributes:
        id:               Kebab-case skill name, serves as primary key.
        display_name:     Human-readable name shown in UI/search results.
        description:      Free-text description used for agent matching.
        skill_type:       One of the ``SkillType`` enum values (stored as string).
        tags:             JSON-encoded list of tag strings.
        repo_name:        Name of the owning repository, or ``"global"``.
        version_number:   Current version (starts at 1, incremented on fixes).
        is_active:        Whether the skill is currently active and discoverable.
        use_count:        Number of times the skill has been invoked.
        last_used_at:     Timestamp of the most recent usage.
        ttl_days:         Optional time-to-live in days after creation.
        expires_at:       Computed expiry timestamp (set by ``compute_expires_at``).
        created_at:       Row creation timestamp (UTC).
        last_modified_at: Row last-modification timestamp (UTC).
    """

    __tablename__ = "skills"

    id = Column(String, primary_key=True, comment="kebab-case skill name")
    display_name = Column(String, nullable=False, comment="Human-readable name")
    description = Column(Text, nullable=False, comment="Description for agent matching")
    skill_type = Column(String, nullable=False, comment="SkillType enum value")
    tags = Column(Text, default="[]", nullable=False, comment="JSON array of tags")
    repo_name = Column(
        String, nullable=False, comment="Repo name or 'global'", index=True
    )
    version_number = Column(Integer, default=1, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    use_count = Column(Integer, default=0, nullable=False)
    last_used_at = Column(DateTime, nullable=True)
    ttl_days = Column(Integer, nullable=True)
    expires_at = Column(DateTime, nullable=True)
    created_at = Column(
        DateTime, default=lambda: datetime.now(timezone.utc), nullable=False
    )
    last_modified_at = Column(
        DateTime, default=lambda: datetime.now(timezone.utc), nullable=False
    )

    # Relationship: ordered changelog entries for this skill.
    changelog = relationship(
        "SkillChangelog",
        back_populates="skill",
        order_by="SkillChangelog.created_at",
    )

    def get_tags(self) -> list[str]:
        """Parse and return the tags list from the JSON-encoded ``tags`` column.

        Returns an empty list if the stored value is malformed or not a valid
        JSON array.

        Returns:
            A list of tag strings.
        """
        try:
            parsed = json.loads(self.tags)
            if isinstance(parsed, list):
                return [str(t) for t in parsed]
        except (json.JSONDecodeError, TypeError):
            pass
        return []

    def set_tags(self, tags: list[str]) -> None:
        """Serialize and persist *tags* as a JSON array in the ``tags`` column.

        Args:
            tags: The list of tag strings to store.
        """
        self.tags = json.dumps(tags, ensure_ascii=False)

    def compute_expires_at(self) -> None:
        """Set ``expires_at`` based on ``ttl_days`` and ``created_at``.

        If ``ttl_days`` is set and ``created_at`` is available, the expiry
        timestamp is computed as ``created_at + ttl_days``.  Otherwise
        ``expires_at`` is reset to ``None``.
        """
        if self.ttl_days and self.created_at:
            self.expires_at = self.created_at + timedelta(days=self.ttl_days)
        else:
            self.expires_at = None

    def is_expired(self) -> bool:
        """Check whether the skill has passed its expiration time.

        Returns:
            ``True`` if the skill has an ``expires_at`` timestamp and it is
            in the past; ``False`` otherwise.  Naïve datetimes (legacy data
            without timezone info) are handled gracefully.
        """
        if self.expires_at is None:
            return False
        now = datetime.now(timezone.utc)
        # Handle naive datetimes (old data without timezone info).
        if self.expires_at.tzinfo is None:
            return now.replace(tzinfo=None) > self.expires_at
        return now > self.expires_at

    def to_dict(self) -> dict:
        """Serialize the skill to a plain dictionary suitable for JSON output.

        Returns:
            A dictionary containing all skill fields with timestamps
            converted to ISO-8601 strings and an additional ``is_expired``
            boolean flag.
        """
        return {
            "id": self.id,
            "display_name": self.display_name,
            "description": self.description,
            "skill_type": self.skill_type,
            "repo_name": self.repo_name,
            "tags": self.get_tags(),
            "version_number": self.version_number,
            "is_active": self.is_active,
            "use_count": self.use_count,
            "last_used_at": (
                self.last_used_at.isoformat() if self.last_used_at else None
            ),
            "ttl_days": self.ttl_days,
            "expires_at": (
                self.expires_at.isoformat() if self.expires_at else None
            ),
            "is_expired": self.is_expired(),
            "created_at": (
                self.created_at.isoformat() if self.created_at else None
            ),
            "last_modified_at": (
                self.last_modified_at.isoformat() if self.last_modified_at else None
            ),
        }

    @staticmethod
    def validate_name(name: str) -> bool:
        """Validate that *name* is a well-formed kebab-case identifier.

        A valid name must be 2–64 characters long and match the pattern
        ``^[a-z0-9]([a-z0-9-]*[a-z0-9])?$`` (lowercase alphanumeric
        characters separated by hyphens).

        Args:
            name: The candidate skill name.

        Returns:
            ``True`` if *name* is valid, ``False`` otherwise.
        """
        if not name or len(name) < 2 or len(name) > 64:
            return False
        return bool(KEBAB_CASE_RE.match(name))

    @staticmethod
    def to_kebab_case(text: str) -> str:
        """Convert an arbitrary string to kebab-case.

        The input is lowercased, non-alphanumeric characters are replaced
        with hyphens, consecutive hyphens are collapsed, and the result is
        truncated to 64 characters.

        Args:
            text: The free-form text to convert.

        Returns:
            A normalized kebab-case string suitable for use as a skill name.
        """
        text = text.lower().strip()
        text = re.sub(r"[^a-z0-9]+", "-", text)
        text = re.sub(r"-+", "-", text).strip("-")
        return text[:64].rstrip("-") if len(text) > 64 else text


class SkillChangelog(Base):
    """ORM model for the **skill_changelog** table.

    Records the history of version transitions for each skill.  One skill
    name corresponds to exactly one row in the ``skills`` table but may
    have many rows in this changelog table — one per version transition.

    Attributes:
        id:              Auto-incrementing primary key.
        skill_id:        FK reference to ``skills.id``.
        from_version:    The version number before the transition.
        to_version:      The version number after the transition.
        trigger:         The ``EvolutionTrigger`` that caused the change.
        reason:          Human-readable explanation of why the change occurred.
        source_skill_id: For DERIVE/MERGE triggers — the originating skill name.
        created_at:      Timestamp when the changelog entry was created (UTC).
    """

    __tablename__ = "skill_changelog"

    id = Column(Integer, primary_key=True, autoincrement=True)
    skill_id = Column(
        String, ForeignKey("skills.id"), nullable=False, index=True
    )
    from_version = Column(Integer, nullable=False, comment="Version before transition")
    to_version = Column(Integer, nullable=False, comment="Version after transition")
    trigger = Column(
        String, nullable=False, comment="ARCHIVE, FIX, DERIVE, or MERGE"
    )
    reason = Column(Text, nullable=False)
    source_skill_id = Column(
        String, nullable=True, comment="For DERIVE/MERGE: source skill name"
    )
    created_at = Column(
        DateTime, default=lambda: datetime.now(timezone.utc), nullable=False
    )

    skill = relationship("Skill", back_populates="changelog")

    def to_dict(self) -> dict:
        """Serialize the changelog entry to a plain dictionary.

        Returns:
            A dictionary containing all changelog fields with timestamps
            converted to ISO-8601 strings.
        """
        return {
            "id": self.id,
            "skill_id": self.skill_id,
            "from_version": self.from_version,
            "to_version": self.to_version,
            "trigger": self.trigger,
            "reason": self.reason,
            "source_skill_id": self.source_skill_id,
            "created_at": (
                self.created_at.isoformat() if self.created_at else None
            ),
        }


# Backward-compatibility alias — existing code that references SkillLineage
# will continue to work without modification.
SkillLineage = SkillChangelog

# ---------------------------------------------------------------------------
# Module-level database engine / session management (singleton per process)
# ---------------------------------------------------------------------------

_engine = None
_session_factory = None
_scoped_factory = None
_db_lock = threading.Lock()


def init_db(workspace_path: str) -> None:
    """Initialize the SQLite database engine and session factories.

    Creates the workspace directory (if it does not exist), sets up a
    SQLite engine pointing at ``<workspace_path>/brain.db``, configures
    WAL journal mode for safe concurrent access, creates all tables, and
    prepares both a regular and a scoped session factory.

    This function is idempotent — calling it multiple times is safe thanks
    to the internal threading lock.

    Args:
        workspace_path: Absolute or relative path to the workspace directory
                        that will contain ``brain.db``.
    """
    global _engine, _session_factory, _scoped_factory
    with _db_lock:
        os.makedirs(workspace_path, exist_ok=True)
        db_path = os.path.join(workspace_path, "brain.db")
        db_url = f"sqlite:///{db_path}"
        _engine = create_engine(
            db_url,
            echo=False,
            connect_args={"timeout": 30},  # Wait up to 30s for write locks
        )

        from sqlalchemy import event

        @event.listens_for(_engine, "connect")
        def set_wal_mode(dbapi_connection, connection_record):
            """Enable WAL journal mode and set busy timeout on every new SQLite connection."""
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA synchronous=NORMAL")
            cursor.execute("PRAGMA busy_timeout=30000")  # 30 second busy timeout
            cursor.close()

        Base.metadata.create_all(_engine)

        from sqlalchemy.orm import sessionmaker

        _session_factory = sessionmaker(bind=_engine)
        _scoped_factory = scoped_session(_session_factory)


def get_session() -> Session:
    """Return a thread-safe SQLAlchemy session.

    Uses the scoped session factory when available (multi-worker scenarios),
    falling back to the regular session factory otherwise.

    Returns:
        A ``Session`` object.  The caller **must** call ``session.close()``
        when done to release the connection back to the pool.

    Raises:
        RuntimeError: If the database has not been initialized yet
                      (``init_db()`` was never called).
    """
    if _session_factory is None:
        raise RuntimeError(
            "Database not initialized. Call init_db() first."
        )
    # Prefer scoped_session for thread safety in multi-worker scenarios.
    if _scoped_factory is not None:
        return _scoped_factory()
    return _session_factory()
