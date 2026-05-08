from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_text(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


def test_agents_file_is_a_short_repo_specific_map() -> None:
    agents = read_text("AGENTS.md")

    assert "docs/README.md" in agents
    assert "docs/ARCHITECTURE.md" in agents
    assert "docs/QUALITY.md" in agents
    assert "Django" not in agents
    assert "Python 3.14+" not in agents


def test_docs_index_routes_to_core_sources_of_truth() -> None:
    docs_index = read_text("docs/README.md")

    for doc_path in (
        "docs/ARCHITECTURE.md",
        "docs/QUALITY.md",
        "docs/LEGIBILITY_AUDIT.md",
        "docs/decisions/README.md",
    ):
        assert (ROOT / doc_path).exists()

    assert "[Architecture](ARCHITECTURE.md)" in docs_index
    assert "[Quality](QUALITY.md)" in docs_index
    assert "[Legibility Audit](LEGIBILITY_AUDIT.md)" in docs_index
