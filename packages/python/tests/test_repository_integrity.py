from __future__ import annotations

import re
import tomllib
from pathlib import Path
from urllib.parse import unquote, urlsplit

import pytest

from fixture_data import Fixture, load_fixture


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
CORE_DOCUMENTATION = (
    Path("docs/README.md"),
    Path("docs/ARCHITECTURE.md"),
    Path("docs/QUALITY.md"),
    Path("docs/LEGIBILITY_AUDIT.md"),
    Path("docs/decisions/README.md"),
    Path("spec/BEHAVIOR.md"),
)
MARKDOWN_SOURCES = (
    Path("AGENTS.md"),
    Path("README.md"),
    Path("packages/python/README.md"),
    Path("packages/typescript/README.md"),
    *sorted(
        path.relative_to(REPOSITORY_ROOT)
        for path in (REPOSITORY_ROOT / "docs").rglob("*.md")
    ),
    *sorted(
        path.relative_to(REPOSITORY_ROOT)
        for path in (REPOSITORY_ROOT / "spec").rglob("*.md")
    ),
)
INLINE_LINK_DESTINATION = re.compile(
    r"\]\(\s*(?:<(?P<bracketed>[^>\n]+)>|(?P<bare>[^\s)\n]+))"
)


def markdown_link_destinations(markdown: str) -> list[str]:
    """Return inline link destinations, ignoring fenced code blocks."""
    destinations: list[str] = []
    fence: str | None = None

    for line in markdown.splitlines():
        stripped = line.lstrip()
        marker = stripped[:3]
        if marker in {"```", "~~~"}:
            if fence is None:
                fence = marker
            elif fence == marker:
                fence = None
            continue
        if fence is not None:
            continue

        for match in INLINE_LINK_DESTINATION.finditer(line):
            destinations.append(match.group("bracketed") or match.group("bare"))

    return destinations


def local_link_target(source: Path, destination: str) -> Path | None:
    parsed = urlsplit(destination)
    if parsed.scheme or parsed.netloc or not parsed.path:
        return None

    path = Path(unquote(parsed.path))
    if path.is_absolute():
        return REPOSITORY_ROOT / path.relative_to("/")
    return source.parent / path


def test_core_documentation_files_exist() -> None:
    for relative_path in CORE_DOCUMENTATION:
        assert (REPOSITORY_ROOT / relative_path).is_file(), (
            f"Missing {relative_path}"
        )


@pytest.mark.parametrize("relative_path", MARKDOWN_SOURCES, ids=str)
def test_local_markdown_links_resolve(relative_path: Path) -> None:
    source = REPOSITORY_ROOT / relative_path
    markdown = source.read_text(encoding="utf-8")

    for destination in markdown_link_destinations(markdown):
        target = local_link_target(source, destination)
        if target is not None:
            resolved_target = target.resolve()
            assert resolved_target.is_relative_to(REPOSITORY_ROOT.resolve()), (
                f"Local link escapes the repository in {relative_path}: {destination}"
            )
            assert resolved_target.exists(), (
                f"Broken local link in {relative_path}: {destination} -> {target}"
            )


def test_package_metadata_matches_repository_policy() -> None:
    with (PACKAGE_ROOT / "pyproject.toml").open("rb") as pyproject_file:
        metadata: object = tomllib.load(pyproject_file)
    project = Fixture(metadata).mapping("project")

    assert project.value("requires-python") == ">=3.11"
    assert project.value("dependencies") == []
    assert (PACKAGE_ROOT / project.string("readme")).is_file()
    assert (PACKAGE_ROOT / project.mapping("license").string("file")).is_file()
    assert (PACKAGE_ROOT / "LICENSE").read_text(encoding="utf-8") == (
        REPOSITORY_ROOT / "LICENSE"
    ).read_text(encoding="utf-8")


def test_typescript_metadata_matches_repository_policy() -> None:
    typescript_root = REPOSITORY_ROOT / "packages" / "typescript"
    package = load_fixture(typescript_root / "package.json")
    package_lock = load_fixture(typescript_root / "package-lock.json")

    assert package.string("name") == "personalitygen"
    assert package.string("version") == package_lock.string("version")
    locked_package = package_lock.mapping("packages").mapping("")
    assert package.string("version") == locked_package.string("version")
    assert "dependencies" not in package.keys()
    assert package.mapping("devDependencies").string("typescript").startswith(">=7.")
    assert (typescript_root / "README.md").is_file()
    assert (typescript_root / "LICENSE").read_text(encoding="utf-8") == (
        REPOSITORY_ROOT / "LICENSE"
    ).read_text(encoding="utf-8")
