"""Bulk onboarding CLI for ANTS Platform.

Adds ``ants-platform[crewai]`` as a dependency to multiple CrewAI project
repositories. Modifies pyproject.toml files in-place — does not commit or push.

Usage::

    ants-onboard /path/to/repos/
    ants-onboard /path/to/repos/ --dry-run
"""
import argparse
import re
import sys
from pathlib import Path


_DEPENDENCY_PEP621 = '"ants-platform[crewai]>=3.5.1"'
_DEPENDENCY_POETRY = 'ants-platform = {version = ">=3.5.1", extras = ["crewai"]}'
_MARKER = "ants-platform"


def _already_has_dependency(content: str) -> bool:
    """Check if ants-platform is already in the file."""
    return _MARKER in content


def _find_section_block(content: str, header: str) -> tuple[int, int]:
    """Return (start, end) of a TOML section's body.

    ``start`` is the index after the header line's newline.
    ``end`` is the index of the next section header (``[...``) or EOF.
    """
    idx = content.find(header)
    if idx == -1:
        return (-1, -1)

    # Find end of header line
    nl = content.find("\n", idx)
    if nl == -1:
        return (len(content), len(content))
    body_start = nl + 1

    # Find next section header
    next_section = re.search(r"^\[", content[body_start:], re.MULTILINE)
    body_end = body_start + next_section.start() if next_section else len(content)

    return (body_start, body_end)


def _add_to_pep621(content: str) -> str | None:
    """Add dependency to [project] dependencies list."""
    body_start, body_end = _find_section_block(content, "[project]")
    if body_start == -1:
        return None

    section = content[body_start:body_end]

    # Find dependencies = [ ... ] within the [project] section only
    dep_match = re.search(r"(dependencies\s*=\s*\[)(.*?)(\n\s*\])", section, re.DOTALL)
    if not dep_match:
        return None

    abs_start = body_start + dep_match.start()
    abs_end = body_start + dep_match.end()

    before = dep_match.group(1)
    deps = dep_match.group(2)
    closing = dep_match.group(3)

    new_dep = f"\n    {_DEPENDENCY_PEP621},"
    return content[:abs_start] + before + deps.rstrip() + new_dep + closing + content[abs_end:]


def _add_to_poetry(content: str) -> str | None:
    """Add dependency to [tool.poetry.dependencies] section."""
    body_start, body_end = _find_section_block(content, "[tool.poetry.dependencies]")
    if body_start == -1:
        return None

    insertion = f"{_DEPENDENCY_POETRY}\n"
    return content[:body_start] + insertion + content[body_start:]


def _detect_format(content: str) -> str | None:
    """Detect whether pyproject.toml uses PEP 621 or Poetry for dependencies."""
    # Check for PEP 621: [project] section with a dependencies key
    body_start, body_end = _find_section_block(content, "[project]")
    if body_start != -1:
        section = content[body_start:body_end]
        if "dependencies" in section:
            return "pep621"

    if "[tool.poetry.dependencies]" in content:
        return "poetry"

    return None


def _process_file(pyproject_path: Path, dry_run: bool) -> bool:
    """Process a single pyproject.toml. Returns True if modified."""
    content = pyproject_path.read_text()

    if _already_has_dependency(content):
        print(f"  SKIP {pyproject_path} (already has ants-platform)")
        return False

    fmt = _detect_format(content)
    new_content = None

    if fmt == "pep621":
        new_content = _add_to_pep621(content)
    elif fmt == "poetry":
        new_content = _add_to_poetry(content)

    if new_content is None:
        print(f"  WARN {pyproject_path} (could not find dependencies section)")
        return False

    if dry_run:
        print(f"  DRY-RUN {pyproject_path} (would add ants-platform[crewai])")
    else:
        pyproject_path.write_text(new_content)
        print(f"  ADDED {pyproject_path}")

    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="ants-onboard",
        description=(
            "Add ants-platform[crewai] dependency to multiple CrewAI project repos. "
            "Scans a directory for subdirectories containing pyproject.toml and adds "
            "the dependency. Does not commit or push — you handle that."
        ),
    )
    parser.add_argument(
        "directory",
        type=Path,
        help="Directory containing CrewAI project subdirectories",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be changed without modifying files",
    )
    args = parser.parse_args()

    if not args.directory.is_dir():
        print(f"Error: {args.directory} is not a directory", file=sys.stderr)
        sys.exit(1)

    # Find pyproject.toml files in immediate subdirectories
    projects = sorted(args.directory.glob("*/pyproject.toml"))

    if not projects:
        # Also check if the directory itself is a project
        root_pyproject = args.directory / "pyproject.toml"
        if root_pyproject.exists():
            projects = [root_pyproject]
        else:
            print(f"No pyproject.toml files found in {args.directory}")
            sys.exit(1)

    print(f"Found {len(projects)} project(s) in {args.directory}\n")

    modified = 0
    for pyproject in projects:
        if _process_file(pyproject, args.dry_run):
            modified += 1

    action = "Would modify" if args.dry_run else "Modified"
    print(f"\n{action} {modified}/{len(projects)} project(s)")

    if modified > 0 and not args.dry_run:
        print("\nNext steps:")
        print("  1. Review the changes: git diff")
        print("  2. Commit: git add -A && git commit -m 'Add ANTS Platform observability'")
        print("  3. Push: git push")
        print("  4. Set env vars in each deployment:")
        print("     ANTS_AUTO_INSTRUMENT=1")
        print("     ANTS_PLATFORM_PUBLIC_KEY=pk-ap-...")
        print("     ANTS_PLATFORM_SECRET_KEY=sk-ap-...")
        print("     ANTS_PLATFORM_HOST=https://api.agenticants.ai")


if __name__ == "__main__":
    main()
