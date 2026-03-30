"""
Skills Lab — SKILL.md File Manager

Manages SKILL.md files following industry standards:
  - YAML frontmatter: name, description, metadata (tags, skill-type, version, ...)
  - Markdown body: knowledge content (When to Use, Solution, Lessons Learned, ...)

Each Skill is stored in a directory: workspace/skills/{skill_name}/SKILL.md
"""

import difflib
import json
import os
import re
import shutil
from datetime import datetime, timezone
from typing import Any

import yaml

from core.exceptions import SKILLParseError


# ---------------------------------------------------------------------------
# SKILL.md Constants
# ---------------------------------------------------------------------------

FRONTMATTER_DELIMITER = "---"
LESSONS_HEADING = "## Lessons Learned"
REFERENCES_HEADING = "## References"
REFERENCE_PATTERN = re.compile(r"@([a-z0-9][a-z0-9-]*[a-z0-9])")

# Required frontmatter keys
REQUIRED_KEYS = {"name", "description"}

# Allowed top-level frontmatter keys
ALLOWED_KEYS = {"name", "description", "license", "metadata"}

# Allowed metadata sub-keys
ALLOWED_METADATA_KEYS = {
    "skill-type", "repo", "version", "tags", "created",
    "last-modified", "ttl-days", "author", "source", "references",
}


def _safe_int(value: str) -> int:
    """Convert a string to int for sorting, falling back to 0 for non-numeric values."""
    try:
        return int(value)
    except (ValueError, TypeError):
        return 0


class SKILLManager:
    """
    File I/O manager for SKILL.md files.

    Each Skill is represented by a single directory containing one SKILL.md file.
    Optionally, a ``references/`` subdirectory may hold supplementary documents.

    Attributes:
        workspace_path: Root workspace directory path.
        skills_dir: Path to the ``skills/`` directory within the workspace.
    """

    def __init__(self, workspace_path: str) -> None:
        """
        Initialize the SKILLManager.

        Args:
            workspace_path: Absolute or relative path to the workspace root directory.
                            Skills will be stored under ``{workspace_path}/skills/``.
        """
        self.workspace_path: str = workspace_path
        self.skills_dir: str = os.path.join(workspace_path, "skills")

    def _skill_dir(self, skill_name: str) -> str:
        """
        Return the directory path for a given skill.

        Args:
            skill_name: Kebab-case skill identifier.

        Returns:
            Absolute or relative path to the skill's directory.
        """
        return os.path.join(self.skills_dir, skill_name)

    def _skill_path(self, skill_name: str) -> str:
        """
        Return the file path to a skill's SKILL.md file.

        Args:
            skill_name: Kebab-case skill identifier.

        Returns:
            Path to ``{skills_dir}/{skill_name}/SKILL.md``.
        """
        return os.path.join(self._skill_dir(skill_name), "SKILL.md")

    def _ensure_dir(self, skill_name: str) -> None:
        """
        Ensure the skill directory exists, creating it if necessary.

        Args:
            skill_name: Kebab-case skill identifier.
        """
        os.makedirs(self._skill_dir(skill_name), exist_ok=True)

    # -----------------------------------------------------------------------
    # SKILL.md Parsing
    # -----------------------------------------------------------------------

    @staticmethod
    def _parse_frontmatter(raw: str) -> tuple[dict[str, Any], str]:
        """
        Parse the raw content of a SKILL.md file into a frontmatter dictionary
        and a markdown body string.

        Expected format::

            ---
            name: skill-name
            description: ...
            metadata:
              key: value
            ---
            # Body content
            ...

        Args:
            raw: The full text content of a SKILL.md file.

        Returns:
            A tuple of ``(frontmatter_dict, body_markdown)``.

        Raises:
            SKILLParseError: If the file does not start with ``---``, the closing
                             ``---`` is missing, the frontmatter is empty, the
                             YAML is invalid, or the frontmatter is not a mapping.
        """
        # Must start with ---
        if not raw.startswith(FRONTMATTER_DELIMITER):
            raise SKILLParseError("SKILL.md must start with YAML frontmatter (---)")

        # Find closing ---
        rest = raw[len(FRONTMATTER_DELIMITER):]
        end_idx = rest.find(FRONTMATTER_DELIMITER)
        if end_idx == -1:
            raise SKILLParseError("SKILL.md frontmatter not closed (missing ---)")

        yaml_str = rest[:end_idx].strip()
        body = rest[end_idx + len(FRONTMATTER_DELIMITER):].strip()

        if not yaml_str:
            raise SKILLParseError("SKILL.md frontmatter is empty")

        try:
            frontmatter = yaml.safe_load(yaml_str)
        except yaml.YAMLError as e:
            raise SKILLParseError(f"Invalid YAML frontmatter: {e}")

        if not isinstance(frontmatter, dict):
            raise SKILLParseError("SKILL.md frontmatter must be a YAML mapping")

        return frontmatter, body

    @staticmethod
    def validate_frontmatter(frontmatter: dict[str, Any]) -> list[str]:
        """
        Validate a parsed frontmatter dictionary against SKILL.md specification rules.

        Checks performed:
          - All required keys (``name``, ``description``) are present and non-empty.
          - No unknown top-level keys are used.
          - No unknown metadata sub-keys are used.
          - The ``name`` value is a non-empty kebab-case string.
          - The ``tags`` value (if present) is a list.

        Args:
            frontmatter: The parsed YAML frontmatter dictionary.

        Returns:
            A list of error message strings. An empty list indicates the frontmatter
            is valid.
        """
        errors: list[str] = []

        # Required keys
        for key in REQUIRED_KEYS:
            if key not in frontmatter:
                errors.append(f"Missing required key: '{key}'")
            elif not frontmatter[key]:
                errors.append(f"Required key '{key}' is empty")

        # Disallowed top-level keys
        for key in frontmatter:
            if key not in ALLOWED_KEYS:
                errors.append(f"Unknown top-level key: '{key}'. Allowed: {sorted(ALLOWED_KEYS)}")

        # Validate metadata sub-keys
        if "metadata" in frontmatter and isinstance(frontmatter["metadata"], dict):
            for key in frontmatter["metadata"]:
                if key not in ALLOWED_METADATA_KEYS:
                    errors.append(f"Unknown metadata key: '{key}'. Allowed: {sorted(ALLOWED_METADATA_KEYS)}")

        # Validate name format
        if "name" in frontmatter:
            if not frontmatter["name"] or not isinstance(frontmatter["name"], str):
                errors.append("'name' must be a non-empty string")
            else:
                import re
                if not re.match(r"^[a-z0-9]([a-z0-9-]*[a-z0-9])?$", frontmatter["name"]):
                    errors.append(f"'name' must be kebab-case, got: '{frontmatter['name']}'")

        # Validate tags is a list
        if "metadata" in frontmatter and isinstance(frontmatter["metadata"], dict):
            tags = frontmatter["metadata"].get("tags")
            if tags is not None and not isinstance(tags, list):
                errors.append("'tags' must be a JSON array")

        return errors

    @staticmethod
    def _build_frontmatter(
        name: str,
        description: str,
        metadata: dict[str, Any] | None = None,
        license: str | None = None,
    ) -> str:
        """
        Build a YAML frontmatter string from the given parameters.

        Only metadata keys present in ``ALLOWED_METADATA_KEYS`` are included;
        unknown keys are silently dropped.

        Args:
            name: Kebab-case skill identifier.
            description: Human-readable description of the skill.
            metadata: Optional dictionary of metadata key-value pairs.
            license: Optional license identifier.

        Returns:
            A string of the form ``"---\\n{yaml}\\n---"``.
        """
        fm: dict[str, Any] = {"name": name, "description": description}
        if license:
            fm["license"] = license
        if metadata:
            # Only include known metadata keys
            clean_meta = {k: v for k, v in metadata.items() if k in ALLOWED_METADATA_KEYS}
            fm["metadata"] = clean_meta

        yaml_str = yaml.dump(fm, default_flow_style=False, allow_unicode=True, sort_keys=False)
        return f"{FRONTMATTER_DELIMITER}\n{yaml_str}{FRONTMATTER_DELIMITER}"

    # -----------------------------------------------------------------------
    # Public interface
    # -----------------------------------------------------------------------

    def _history_dir(self, skill_name: str) -> str:
        """
        Return the path to the version history directory for a skill.

        Args:
            skill_name: Kebab-case skill identifier.

        Returns:
            Path to ``{skill_dir}/.history/``.
        """
        return os.path.join(self._skill_dir(skill_name), ".history")

    def _archive_current_version(self, skill_name: str) -> None:
        """
        Archive the current SKILL.md to ``.history/v{version}.md`` before overwriting.

        Reads the version number from the existing frontmatter. If no SKILL.md
        exists or it cannot be read/parsed, this method is a no-op.

        Args:
            skill_name: Kebab-case skill identifier.
        """
        skill_path = self._skill_path(skill_name)
        if not os.path.exists(skill_path):
            return

        try:
            fm = self.read_frontmatter(skill_name)
            meta = fm.get("metadata", {}) or {}
            version = meta.get("version", "0")

            history_dir = self._history_dir(skill_name)
            os.makedirs(history_dir, exist_ok=True)

            archive_path = os.path.join(history_dir, f"v{version}.md")
            shutil.copy2(skill_path, archive_path)
        except Exception:
            # Silently skip archiving if anything fails — don't block writes
            pass

    def write_skill(
        self,
        skill_name: str,
        description: str,
        body: str,
        display_name: str = "",
        skill_type: str = "IMPLEMENTATION",
        repo: str = "global",
        version: int = 1,
        tags: list[str] | None = None,
        ttl_days: int | None = None,
        author: str | None = None,
        source: str | None = None,
        references: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Write a new SKILL.md file (overwrites if it already exists).

        Creates the skill directory if it does not exist, then writes the
        SKILL.md file with the supplied frontmatter fields and markdown body.

        If a SKILL.md file already exists, the previous version is archived
        to ``.history/v{old_version}.md`` before overwriting, enabling
        version diff functionality.

        Args:
            skill_name: Kebab-case skill identifier (also used as the directory name).
            description: Short description used for agent matching.
            body: Markdown body content (When to Use, Solution, Lessons Learned, ...).
            display_name: Optional human-readable name (currently unused in output).
            skill_type: Skill type classifier (e.g. ``"IMPLEMENTATION"``).
            repo: Repository name or ``"global"``.
            version: Version number.
            tags: Optional list of tag strings for categorization.
            ttl_days: Optional time-to-live in days.
            author: Optional author name.
            source: Optional reference source URL or identifier.
            references: Optional list of skill names that this skill references.

        Returns:
            The metadata dictionary that was written to the frontmatter.

        Raises:
            ValueError: If ``description`` or ``body`` is empty or whitespace-only.
        """
        if not description or not description.strip():
            raise ValueError("description is required and cannot be empty")
        if not body or not body.strip():
            raise ValueError("body is required and cannot be empty")

        self._ensure_dir(skill_name)

        # Archive the current version before overwriting
        self._archive_current_version(skill_name)

        now = datetime.now(timezone.utc)
        metadata: dict[str, Any] = {
            "skill-type": skill_type,
            "repo": repo,
            "version": str(version),
            "tags": tags or [],
            "created": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "last-modified": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        if ttl_days is not None:
            metadata["ttl-days"] = ttl_days
        if author:
            metadata["author"] = author
        if source:
            metadata["source"] = source
        if references:
            metadata["references"] = references

        frontmatter = self._build_frontmatter(
            name=skill_name,
            description=description,
            metadata=metadata,
        )

        content = f"{frontmatter}\n\n{body}\n"

        with open(self._skill_path(skill_name), "w", encoding="utf-8") as f:
            f.write(content)

        return metadata

    def read_skill(self, skill_name: str) -> dict[str, Any]:
        """
        Read and parse a SKILL.md file.

        Args:
            skill_name: Kebab-case skill identifier.

        Returns:
            A dictionary with the following keys:

            - ``"frontmatter"`` (dict): The parsed YAML frontmatter.
            - ``"body"`` (str): The markdown body content.
            - ``"raw"`` (str): The full raw file content.

        Raises:
            FileNotFoundError: If the SKILL.md file does not exist.
            SKILLParseError: If the file format is invalid.
        """
        path = self._skill_path(skill_name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"SKILL.md not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            raw = f.read()

        frontmatter, body = self._parse_frontmatter(raw)
        return {"frontmatter": frontmatter, "body": body, "raw": raw}

    def read_body(self, skill_name: str) -> str:
        """
        Read only the markdown body of a SKILL.md file (Tier 2 content).

        Args:
            skill_name: Kebab-case skill identifier.

        Returns:
            The markdown body string.

        Raises:
            FileNotFoundError: If the SKILL.md file does not exist.
            SKILLParseError: If the file format is invalid.
        """
        return self.read_skill(skill_name)["body"]

    def read_frontmatter(self, skill_name: str) -> dict[str, Any]:
        """
        Read only the YAML frontmatter of a SKILL.md file (Tier 1 metadata).

        Args:
            skill_name: Kebab-case skill identifier.

        Returns:
            The parsed frontmatter dictionary.

        Raises:
            FileNotFoundError: If the SKILL.md file does not exist.
            SKILLParseError: If the file format is invalid.
        """
        return self.read_skill(skill_name)["frontmatter"]

    def read_raw(self, skill_name: str) -> str:
        """
        Read the entire raw content of a SKILL.md file without parsing.

        Args:
            skill_name: Kebab-case skill identifier.

        Returns:
            The raw file content as a string.

        Raises:
            FileNotFoundError: If the SKILL.md file does not exist.
        """
        path = self._skill_path(skill_name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"SKILL.md not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    # -----------------------------------------------------------------------
    # Lessons Learned — Append-only
    # -----------------------------------------------------------------------

    def append_lesson(
        self,
        skill_name: str,
        version: int,
        lesson: str,
    ) -> None:
        """
        Append a new lesson entry to the ``## Lessons Learned`` section.

        If the section does not yet exist, it is created at the end of the body.

        Appended format::

            - **V{version}** ({YYYY-MM-DD}): {lesson}

        Args:
            skill_name: Kebab-case skill identifier.
            version: The version number associated with this lesson.
            lesson: The lesson content text.

        Raises:
            FileNotFoundError: If the SKILL.md file does not exist.
            SKILLParseError: If the file format is invalid.
        """
        path = self._skill_path(skill_name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"SKILL.md not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            raw = f.read()

        frontmatter, body = self._parse_frontmatter(raw)

        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        new_entry = f"- **V{version}** ({today}): {lesson}"

        if LESSONS_HEADING in body:
            # Append to the existing section
            body = body.replace(
                LESSONS_HEADING,
                f"{LESSONS_HEADING}\n{new_entry}",
                1,
            )
        else:
            # Create a new section at the end of the body
            body = f"{body}\n\n{LESSONS_HEADING}\n{new_entry}"

        # Rebuild file
        yaml_str = yaml.dump(frontmatter, default_flow_style=False, allow_unicode=True, sort_keys=False)
        new_raw = f"{FRONTMATTER_DELIMITER}\n{yaml_str}{FRONTMATTER_DELIMITER}\n\n{body}\n"

        with open(path, "w", encoding="utf-8") as f:
            f.write(new_raw)

    def update_frontmatter_version(
        self,
        skill_name: str,
        version: int,
    ) -> None:
        """
        Update the version number and last-modified timestamp in the frontmatter metadata.

        Args:
            skill_name: Kebab-case skill identifier.
            version: The new version number to set.

        Raises:
            FileNotFoundError: If the SKILL.md file does not exist.
            SKILLParseError: If the file format is invalid.
        """
        path = self._skill_path(skill_name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"SKILL.md not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            raw = f.read()

        frontmatter, body = self._parse_frontmatter(raw)

        if "metadata" in frontmatter and isinstance(frontmatter["metadata"], dict):
            frontmatter["metadata"]["version"] = str(version)
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        frontmatter["metadata"]["last-modified"] = now

        yaml_str = yaml.dump(frontmatter, default_flow_style=False, allow_unicode=True, sort_keys=False)
        new_raw = f"{FRONTMATTER_DELIMITER}\n{yaml_str}{FRONTMATTER_DELIMITER}\n\n{body}\n"

        with open(path, "w", encoding="utf-8") as f:
            f.write(new_raw)

    # -----------------------------------------------------------------------
    # Skill lifecycle
    # -----------------------------------------------------------------------

    def delete_skill_dir(self, skill_name: str) -> None:
        """
        Delete the entire physical directory of a skill.

        If the skill directory does not exist, this method is a no-op.

        Args:
            skill_name: Kebab-case skill identifier.
        """
        dir_path = self._skill_dir(skill_name)
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)

    def skill_dir_exists(self, skill_name: str) -> bool:
        """
        Check whether a skill directory exists on disk.

        Args:
            skill_name: Kebab-case skill identifier.

        Returns:
            ``True`` if the skill directory exists, ``False`` otherwise.
        """
        return os.path.isdir(self._skill_dir(skill_name))

    def list_skills(self) -> list[str]:
        """
        List all skill names whose directories contain a SKILL.md file.

        Returns:
            A sorted list of skill name strings (directory names that have a
            ``SKILL.md`` file). Returns an empty list if the skills directory
            does not exist.
        """
        if not os.path.isdir(self.skills_dir):
            return []
        result: list[str] = []
        for name in os.listdir(self.skills_dir):
            skill_path = os.path.join(self.skills_dir, name, "SKILL.md")
            if os.path.isfile(skill_path):
                result.append(name)
        return sorted(result)

    def get_description_for_search(self, skill_name: str) -> str:
        """
        Retrieve the description and tags concatenated for search/embedding purposes.

        Falls back gracefully if parsing fails by extracting the first few
        meaningful lines from the raw file.

        Args:
            skill_name: Kebab-case skill identifier.

        Returns:
            A single string combining the description and tags, suitable for
            use as embedding input text. Returns an empty string if all
            retrieval attempts fail.
        """
        try:
            fm = self.read_frontmatter(skill_name)
            desc = fm.get("description", "")
            tags: list[str] = []
            if "metadata" in fm and isinstance(fm["metadata"], dict):
                tags = fm["metadata"].get("tags", [])
                if isinstance(tags, list):
                    tags = [str(t) for t in tags]
            parts = [desc] + tags
            return " ".join(parts)
        except Exception:
            # Fallback: read the raw file and extract the first few meaningful lines
            try:
                raw = self.read_raw(skill_name)
                lines = raw.split("\n")
                meaningful = [l.strip() for l in lines if l.strip() and not l.startswith("---") and not l.startswith("#")]
                return " ".join(meaningful[:3])
            except Exception:
                return ""

    def get_bm25_text(self, skill_name: str) -> str:
        """
        Retrieve the full text for BM25 indexing: name + description + tags + repo.

        Falls back to returning just the ``skill_name`` if parsing fails.

        Args:
            skill_name: Kebab-case skill identifier.

        Returns:
            A single string concatenating the skill name, description, tags,
            and repo, suitable for BM25 tokenization.
        """
        try:
            fm = self.read_frontmatter(skill_name)
            name = fm.get("name", skill_name)
            desc = fm.get("description", "")
            repo: str = ""
            tags: list[str] = []
            if "metadata" in fm and isinstance(fm["metadata"], dict):
                meta = fm["metadata"]
                repo = meta.get("repo", "global")
                tags = meta.get("tags", [])
                if isinstance(tags, list):
                    tags = [str(t) for t in tags]
            parts = [name, desc, " ".join(tags), repo]
            return " ".join(parts)
        except Exception:
            return skill_name

    def get_search_text(self, skill_name: str) -> str:
        """
        Retrieve the full text for BM25 indexing including the body content.

        Returns: ``name + description + tags + repo + body`` (body truncated to
        2000 characters to avoid excessive tokenization).

        Falls back gracefully to metadata-only text if the body cannot be read.

        Args:
            skill_name: Kebab-case skill identifier.

        Returns:
            A single string suitable for BM25 tokenization, including the
            skill body text for full-text search coverage.
        """
        try:
            fm = self.read_frontmatter(skill_name)
            name = fm.get("name", skill_name)
            desc = fm.get("description", "")
            repo: str = ""
            tags: list[str] = []
            if "metadata" in fm and isinstance(fm["metadata"], dict):
                meta = fm["metadata"]
                repo = meta.get("repo", "global")
                tags = meta.get("tags", [])
                if isinstance(tags, list):
                    tags = [str(t) for t in tags]

            # Attempt to read the body for full-text indexing
            body = ""
            try:
                body = self.read_body(skill_name)
            except Exception:
                pass

            # Truncate body to 2000 chars to avoid excessive tokenization
            if len(body) > 2000:
                body = body[:2000]

            parts = [name, desc, " ".join(tags), repo, body]
            return " ".join(parts)
        except Exception:
            return skill_name

    # -----------------------------------------------------------------------
    # References support
    # -----------------------------------------------------------------------

    @staticmethod
    def _parse_references_from_body(body: str) -> list[str]:
        """
        Parse @skill-name references from the ``## References`` section of a body.

        Scans the body for a ``## References`` heading and extracts all
        kebab-case identifiers prefixed with ``@`` within that section.

        Args:
            body: The markdown body string.

        Returns:
            A deduplicated, sorted list of referenced skill names.
            Returns an empty list if the section does not exist or has no
            ``@`` references.
        """
        if REFERENCES_HEADING not in body:
            return []

        # Extract text after the References heading (up to the next ## or EOF)
        ref_section_start = body.index(REFERENCES_HEADING) + len(REFERENCES_HEADING)
        ref_section = body[ref_section_start:]

        # Stop at the next ## heading (if any)
        next_heading = re.search(r"\n## ", ref_section)
        if next_heading:
            ref_section = ref_section[: next_heading.start()]

        matches = REFERENCE_PATTERN.findall(ref_section)
        return sorted(set(matches))

    def get_references(self, skill_name: str) -> list[str]:
        """
        Get the list of skill names referenced by this skill.

        Reads references from two sources and merges them:

        1. **Frontmatter metadata**: ``metadata.references`` (a YAML list).
        2. **Body ``## References`` section**: any ``@skill-name`` entries.

        The result is a deduplicated, sorted list.

        Args:
            skill_name: Kebab-case skill identifier.

        Returns:
            List of referenced skill names (empty if none).

        Raises:
            FileNotFoundError: If the SKILL.md file does not exist.
            SKILLParseError: If the file format is invalid.
        """
        frontmatter = self.read_frontmatter(skill_name)
        body = self.read_body(skill_name)

        refs: set[str] = set()

        # From frontmatter metadata.references
        if "metadata" in frontmatter and isinstance(frontmatter["metadata"], dict):
            meta_refs = frontmatter["metadata"].get("references", [])
            if isinstance(meta_refs, list):
                refs.update(str(r) for r in meta_refs if r)

        # From body ## References section with @skill-name
        body_refs = self._parse_references_from_body(body)
        refs.update(body_refs)

        return sorted(refs)

    def set_references(self, skill_name: str, references: list[str]) -> None:
        """
        Update the references field in the SKILL.md frontmatter metadata.

        Reads the existing file, updates the ``metadata.references`` list,
        and writes the file back. Existing frontmatter keys and body content
        are preserved.

        Args:
            skill_name: Kebab-case skill identifier.
            references: List of skill names to reference.

        Raises:
            FileNotFoundError: If the SKILL.md file does not exist.
            SKILLParseError: If the file format is invalid.
        """
        path = self._skill_path(skill_name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"SKILL.md not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            raw = f.read()

        frontmatter, body = self._parse_frontmatter(raw)

        if "metadata" not in frontmatter or not isinstance(frontmatter["metadata"], dict):
            frontmatter["metadata"] = {}

        frontmatter["metadata"]["references"] = references

        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        frontmatter["metadata"]["last-modified"] = now

        yaml_str = yaml.dump(frontmatter, default_flow_style=False, allow_unicode=True, sort_keys=False)
        new_raw = f"{FRONTMATTER_DELIMITER}\n{yaml_str}{FRONTMATTER_DELIMITER}\n\n{body}\n"

        with open(path, "w", encoding="utf-8") as f:
            f.write(new_raw)

    def list_skill_versions(self, skill_name: str) -> list[dict[str, str]]:
        """
        List all archived version snapshots for a skill.

        Each entry includes the version number and the file modification time.

        Args:
            skill_name: Kebab-case skill identifier.

        Returns:
            A sorted list of dicts, each with ``"version"`` (str) and
            ``"path"`` (str) keys. Returns an empty list if the history
            directory does not exist or contains no snapshots.
        """
        history_dir = self._history_dir(skill_name)
        if not os.path.isdir(history_dir):
            return []

        versions: list[dict[str, str]] = []
        for filename in os.listdir(history_dir):
            if filename.startswith("v") and filename.endswith(".md"):
                version_num = filename[1:-3]  # Strip "v" prefix and ".md" suffix
                file_path = os.path.join(history_dir, filename)
                versions.append({
                    "version": version_num,
                    "path": file_path,
                })

        # Sort by version number numerically (handle non-numeric gracefully)
        versions.sort(key=lambda v: _safe_int(v["version"]))
        return versions

    def _read_version_snapshot(self, skill_name: str, version: str) -> str:
        """
        Read the content of a specific version snapshot.

        If ``version`` is ``"current"``, reads the current SKILL.md file.

        Args:
            skill_name: Kebab-case skill identifier.
            version: Version number as a string, or ``"current"`` for the
                live SKILL.md file.

        Returns:
            The raw file content as a string.

        Raises:
            FileNotFoundError: If the requested snapshot file does not exist.
        """
        if version == "current":
            path = self._skill_path(skill_name)
        else:
            path = os.path.join(self._history_dir(skill_name), f"v{version}.md")

        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Version snapshot not found: v{version} for skill '{skill_name}'"
            )

        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    def get_version_diff(
        self,
        skill_name: str,
        version_a: str,
        version_b: str = "current",
    ) -> dict[str, str]:
        """
        Compute a unified diff between two versions of a skill.

        Uses Python's ``difflib.unified_diff`` to produce a standard diff output.

        Args:
            skill_name: Kebab-case skill identifier.
            version_a: The first version to compare (e.g. ``"1"``).
            version_b: The second version to compare. Defaults to ``"current"``,
                which uses the live SKILL.md file.

        Returns:
            A dict with keys:

            - ``"skill_name"``: The skill identifier.
            - ``"v1"``: The first version string.
            - ``"v2"``: The second version string.
            - ``"diff"``: The unified diff as a string (empty if the versions
              are identical or both files are missing).

        Raises:
            FileNotFoundError: If either version snapshot does not exist.
        """
        text_a = self._read_version_snapshot(skill_name, version_a)
        text_b = self._read_version_snapshot(skill_name, version_b)

        lines_a = text_a.splitlines(keepends=True)
        lines_b = text_b.splitlines(keepends=True)

        diff_lines = difflib.unified_diff(
            lines_a,
            lines_b,
            fromfile=f"v{version_a}",
            tofile=f"v{version_b}",
            lineterm="",
        )

        diff_text = "\n".join(diff_lines)

        return {
            "skill_name": skill_name,
            "v1": version_a,
            "v2": version_b,
            "diff": diff_text,
        }

    def find_referencing_skills(self, skill_name: str) -> list[str]:
        """
        Find all skills that reference the given skill.

        Scans every SKILL.md file in the workspace for references to
        ``skill_name`` — checking both the frontmatter ``metadata.references``
        field and ``@skill-name`` entries in the ``## References`` body section.

        Args:
            skill_name: Kebab-case skill identifier to search for.

        Returns:
            A sorted list of skill names that reference this skill.
            Returns an empty list if no references are found.
        """
        referencing: list[str] = []

        for candidate in self.list_skills():
            if candidate == skill_name:
                continue
            try:
                refs = self.get_references(candidate)
                if skill_name in refs:
                    referencing.append(candidate)
            except (FileNotFoundError, SKILLParseError):
                continue

        return sorted(referencing)
