from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_ROOT = ROOT.parents[1] / "examples" / "python"
EXAMPLES = (
    EXAMPLES_ROOT / "generate_npc.py",
    EXAMPLES_ROOT / "project_big_five_to_abbf.py",
    EXAMPLES_ROOT / "compare_characters.py",
    EXAMPLES_ROOT / "select_npcs_by_pole.py",
    EXAMPLES_ROOT / "rpg_party.py",
    EXAMPLES_ROOT / "mystery_casting.py",
    EXAMPLES_ROOT / "repeatable_population.py",
)


@pytest.mark.parametrize("script", EXAMPLES, ids=lambda path: path.name)
def test_example_script_runs(script: Path) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")

    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        check=True,
        text=True,
    )

    assert result.stdout.strip()
    assert result.stderr == ""
