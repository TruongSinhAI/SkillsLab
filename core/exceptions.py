"""
Skills Lab — Custom Exception Classes

Structured exception hierarchy for consistent error handling across
all modules (MCP server, dashboard, CLI, evolver, retriever).
"""


class SkillsLabError(Exception):
    """Base exception for all Skills Lab errors."""

    def __init__(self, message: str = "", details: str = ""):
        self.message = message
        self.details = details
        full = message
        if details:
            full = f"{message}: {details}"
        super().__init__(full)


# --- SKILL.md Parsing Errors ---

class SKILLParseError(SkillsLabError):
    """Raised when a SKILL.md file cannot be parsed (malformed YAML, missing delimiters, etc.)."""
    pass


class SKILLValidationError(SkillsLabError):
    """Raised when SKILL.md frontmatter fails validation (missing keys, unknown keys, etc.)."""
    pass


# --- Skill Lifecycle Errors ---

class SkillNotFoundError(SkillsLabError):
    """Raised when a requested skill does not exist in the database or filesystem."""
    pass


class SkillAlreadyExistsError(SkillsLabError):
    """Raised when attempting to create a skill that already exists."""
    pass


class SkillInactiveError(SkillsLabError):
    """Raised when attempting to modify an inactive (deprecated) skill."""
    pass


# --- Evolution Engine Errors ---

class EvolutionError(SkillsLabError):
    """Base exception for evolution engine errors."""
    pass


class InvalidActionError(EvolutionError):
    """Raised when an invalid evolution action is specified."""
    pass


class InvalidNameError(EvolutionError):
    """Raised when a skill name fails validation (not kebab-case, too short/long, etc.)."""
    pass


class MissingRequiredFieldError(EvolutionError):
    """Raised when a required field is missing for an evolution action."""
    pass


class MergeError(EvolutionError):
    """Raised when a merge operation fails (no sources, source not found, etc.)."""
    pass


# --- Search / Retriever Errors ---

class SearchError(SkillsLabError):
    """Base exception for search-related errors."""
    pass


class EmbeddingError(SearchError):
    """Raised when embedding model loading or inference fails."""
    pass


# --- Database Errors ---

class DatabaseError(SkillsLabError):
    """Raised when a database operation fails."""
    pass


class DatabaseNotInitializedError(DatabaseError):
    """Raised when the database has not been initialized (init_db() not called)."""
    pass


# --- Dashboard / API Errors ---

class APIError(SkillsLabError):
    """Base exception for API/dashboard errors."""
    pass


class NotFoundError(APIError):
    """Raised when a requested resource is not found (HTTP 404)."""
    pass


class ConflictError(APIError):
    """Raised when a resource conflicts with existing data (HTTP 409)."""
    pass


class BadRequestError(APIError):
    """Raised when the request is malformed or invalid (HTTP 400)."""
    pass
