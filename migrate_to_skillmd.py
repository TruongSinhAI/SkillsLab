"""
Migration script: Old format (snippet.txt + metadata.json + usage.md) → SKILL.md
"""

import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.models import Skill, init_db, get_session
from core.manager import SKILLManager


def migrate_skills(workspace_path: str, dry_run: bool = False):
    """
    Migrate old skill format to SKILL.md.

    Old: workspace/skills/{uuid}/snippet.txt + metadata.json + usage.md
    New: workspace/skills/{kebab-name}/SKILL.md
    """
    skills_dir = os.path.join(workspace_path, "skills")
    if not os.path.isdir(skills_dir):
        print("No skills directory found. Nothing to migrate.")
        return

    manager = SKILLManager(workspace_path)

    migrated = 0
    skipped = 0

    for entry in sorted(os.listdir(skills_dir)):
        old_dir = os.path.join(skills_dir, entry)
        if not os.path.isdir(old_dir):
            continue

        snippet_path = os.path.join(old_dir, "snippet.txt")
        meta_path = os.path.join(old_dir, "metadata.json")
        usage_path = os.path.join(old_dir, "usage.md")

        # Skip if already SKILL.md format
        if os.path.exists(os.path.join(old_dir, "SKILL.md")):
            print(f"  ⏭️  {entry}: already SKILL.md format, skip")
            skipped += 1
            continue

        # Need at least metadata.json
        if not os.path.exists(meta_path):
            print(f"  ⚠️  {entry}: no metadata.json, skip")
            skipped += 1
            continue

        # Read old files
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        snippet = ""
        if os.path.exists(snippet_path):
            with open(snippet_path, "r", encoding="utf-8") as f:
                snippet = f.read()

        usage = ""
        if os.path.exists(usage_path):
            with open(usage_path, "r", encoding="utf-8") as f:
                usage = f.read()

        # Build new name
        old_name = metadata.get("name", entry)
        new_name = Skill.to_kebab_case(old_name)
        if not Skill.validate_name(new_name):
            new_name = f"skill-{entry[:8]}"

        # Build description
        description = metadata.get("description", old_name)

        # Build body
        body_parts = [f"# {old_name}"]

        if snippet.strip():
            body_parts.append("\n## Solution\n")
            body_parts.append("```")
            body_parts.append(snippet.strip())
            body_parts.append("```")

        if usage.strip():
            body_parts.append("\n## Lessons Learned\n")
            body_parts.append(usage.strip())

        body = "\n".join(body_parts)

        # Build tags from description keywords
        words = description.lower().split()
        tags = list(set(w for w in words if len(w) > 3))[:5]

        skill_type = metadata.get("type", "IMPLEMENTATION")
        repo_name = metadata.get("repo_name", "global")
        version = metadata.get("version_number", 1)

        print(f"  📝 {entry} → {new_name} (V{version}, {skill_type}, {repo_name})")

        if dry_run:
            print(f"     [DRY RUN] Would write SKILL.md with {len(body)} chars")
            migrated += 1
            continue

        # Write SKILL.md
        new_dir = os.path.join(skills_dir, new_name)
        os.makedirs(new_dir, exist_ok=True)

        manager.write_skill(
            skill_name=new_name,
            description=description,
            body=body,
            display_name=old_name,
            skill_type=skill_type,
            repo=repo_name,
            version=version,
            tags=tags,
        )

        # Update DB
        session = get_session()
        try:
            # Remove old skill from DB
            old_skill = session.query(Skill).filter_by(id=entry).first()
            if old_skill:
                session.delete(old_skill)

            # Create new skill in DB (or update if exists)
            existing = session.query(Skill).filter_by(id=new_name).first()
            if existing:
                existing.version_number = max(existing.version_number, version)
            else:
                now = __import__("datetime").datetime.now(__import__("datetime").timezone.utc)
                new_skill = Skill(
                    id=new_name,
                    display_name=old_name,
                    description=description,
                    skill_type=skill_type,
                    repo_name=repo_name,
                    version_number=version,
                    is_active=metadata.get("is_active", True),
                    use_count=metadata.get("use_count", 0),
                    created_at=now,
                    last_modified_at=now,
                )
                new_skill.set_tags(tags)
                new_skill.compute_expires_at()
                session.add(new_skill)

            session.commit()
        except Exception as e:
            session.rollback()
            print(f"     ⚠️  DB error: {e}")
        finally:
            session.close()

        # Backup old directory
        backup_dir = os.path.join(skills_dir, f"{entry}.bak")
        if not os.path.exists(backup_dir):
            os.rename(old_dir, backup_dir)
            print(f"     📦 Old dir → {entry}.bak")

        migrated += 1

    print(f"\n✅ Migration complete: {migrated} migrated, {skipped} skipped")


if __name__ == "__main__":
    workspace = os.environ.get("SKILLS_LAB_WORKSPACE", os.path.join(os.path.dirname(os.path.abspath(__file__)), "workspace"))
    dry_run = "--dry-run" in sys.argv

    if dry_run:
        print("🔍 DRY RUN — no files will be modified\n")
    else:
        print(f"🔧 Migrating skills in: {workspace}\n")

    init_db(workspace)
    migrate_skills(workspace, dry_run=dry_run)
