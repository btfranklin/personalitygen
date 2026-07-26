from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = (
    ROOT / "examples" / "generate_npc.py",
    ROOT / "examples" / "project_big_five_to_abbf.py",
    ROOT / "examples" / "compare_characters.py",
    ROOT / "examples" / "select_npcs_by_pole.py",
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
