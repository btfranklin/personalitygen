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


def test_draft_release_requires_both_complete_ci_workflows() -> None:
    workflows = REPOSITORY_ROOT / ".github" / "workflows"
    draft = (workflows / "create-draft-release.yml").read_text(encoding="utf-8")
    assert 'tags:\n            - "v*.*.*"' in draft
    for language in ("python", "typescript"):
        assert f"uses: ./.github/workflows/{language}-package.yml" in draft
        ci = (workflows / f"{language}-package.yml").read_text(encoding="utf-8")
        assert "    workflow_call:" in ci
    release_job = draft.split("    draft-release:\n", 1)[1]
    assert "needs: [python-validation, typescript-validation]" in release_job
    assert "if:" not in release_job  # Default success gating must stay in force.
    assert "uses: btfranklin/release-notes-scribe@v0" in release_job
    assert 'create_release: "true"' in release_job
    assert 'draft: "true"' in release_job


def test_publication_only_builds_and_publishes_the_release_tag() -> None:
    workflow = (
        REPOSITORY_ROOT / ".github" / "workflows" / "publish-packages.yml"
    ).read_text(encoding="utf-8")
    assert "types: [published]" in workflow
    assert "ref: ${{ github.event.release.tag_name }}" in workflow
    assert "fetch-depth: 0" in workflow
    assert "pdm build --dest release-dist" in workflow
    assert "npm ci --ignore-scripts" in workflow
    assert "npm run build" in workflow
    assert "npm pack --ignore-scripts --pack-destination release-dist" in workflow
    assert 'npm publish "./$package" --ignore-scripts --provenance' in workflow
    assert "pypa/gh-action-pypi-publish@release/v1" in workflow
    for publisher in ("publish-python", "publish-typescript"):
        job = workflow.split(f"    {publisher}:\n", 1)[1].split("\n    publish-", 1)[0]
        assert "needs: build" in job
        assert "environment: release" in job
        assert "id-token: write" in job
    for forbidden in (
        "workflow_dispatch:", "artifact_run_id", "run-id:", "preflight",
        "workflow_call:", "uses: ./.github/workflows/", "setup-perfect-doc",
        "pdm install", "pdm run", "npm run check", "npm run test",
        "npm run lint", "npm run typecheck", "npm run verify:package",
    ):
        assert forbidden not in workflow
