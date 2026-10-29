"""Load skills: methodology guides in skills/<name>/SKILL.md with YAML frontmatter.

A broken skill is skipped with a warning; it never stops the app.
"""

import logging
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from agent.config import SKILLS_DIR

PROJECT_ROOT = Path(__file__).resolve().parent.parent
logger = logging.getLogger(__name__)

FRONTMATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*\n?(.*)\Z", re.DOTALL)
SKILL_NAME = re.compile(r"^[a-z0-9][a-z0-9-]*$")


@dataclass(frozen=True)
class Skill:
    name: str
    # One sentence: when to use the skill. Shown in the system prompt.
    description: str
    body: str
    path: Path


class SkillError(Exception):
    """A SKILL.md file that cannot be used."""


def load_skills(root: Path = PROJECT_ROOT / SKILLS_DIR) -> tuple[dict[str, Skill], list[str]]:
    """All valid skills by name, and warnings about the skipped ones."""
    skills: dict[str, Skill] = {}
    warnings: list[str] = []
    for path in sorted(root.glob("*/SKILL.md")):
        where = path.parent.name
        try:
            skill = parse_skill(path)
        except SkillError as exc:
            warnings.append(f"Skipped skill '{where}': {exc}")
            continue
        if skill.name in skills:
            warnings.append(
                f"Skipped skill '{where}': duplicate name '{skill.name}' "
                f"(already loaded from '{skills[skill.name].path.parent.name}')"
            )
            continue
        skills[skill.name] = skill
    for warning in warnings:
        logger.warning(warning)
    return skills, warnings


def parse_skill(path: Path) -> Skill:
    """Read one SKILL.md; SkillError if it has no usable frontmatter or body."""
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise SkillError(f"cannot read the file ({type(exc).__name__})") from exc
    match = FRONTMATTER.match(text.replace("\r\n", "\n"))
    if match is None:
        raise SkillError("no YAML frontmatter between '---' lines at the top")
    try:
        meta = yaml.safe_load(match.group(1))
    except yaml.YAMLError as exc:
        raise SkillError(f"invalid YAML frontmatter ({exc.__class__.__name__})") from exc
    if not isinstance(meta, dict):
        raise SkillError("the frontmatter is not a set of 'key: value' lines")
    name, description = meta.get("name"), meta.get("description")
    if not isinstance(name, str) or not SKILL_NAME.match(name.strip()):
        raise SkillError("missing or invalid 'name' (lowercase letters, digits and dashes)")
    if not isinstance(description, str) or not description.strip():
        raise SkillError("missing 'description'")
    body = match.group(2).strip()
    if not body:
        raise SkillError("empty body")
    return Skill(name=name.strip(), description=" ".join(description.split()), body=body, path=path)


def skills_list(skills: dict[str, Skill]) -> str:
    """'- name: description' lines for the system prompt."""
    return "\n".join(f"- {skill.name}: {skill.description}" for skill in skills.values())
