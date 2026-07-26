from __future__ import annotations

import argparse
import email
import json
import subprocess
import tarfile
import tempfile
import venv
import zipfile
from pathlib import Path


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect and install the built personalitygen package."
    )
    parser.add_argument(
        "--dist",
        type=Path,
        default=Path("dist"),
        help="Directory containing exactly one wheel and one source archive.",
    )
    parser.add_argument("--expected-version")
    return parser.parse_args()


def only_matching(directory: Path, pattern: str) -> Path:
    matches = sorted(directory.glob(pattern))
    if len(matches) != 1:
        raise AssertionError(
            f"Expected exactly one {pattern} in {directory}, found {matches}"
        )
    return matches[0]


def wheel_version(wheel: Path) -> str:
    with zipfile.ZipFile(wheel) as archive:
        members = set(archive.namelist())
        metadata_path = next(
            path for path in members if path.endswith(".dist-info/METADATA")
        )
        metadata = email.message_from_bytes(archive.read(metadata_path))

        required_suffixes = {
            "personalitygen/__init__.py",
            "personalitygen/py.typed",
            ".dist-info/licenses/LICENSE",
        }
        for suffix in required_suffixes:
            if not any(path.endswith(suffix) for path in members):
                raise AssertionError(f"Wheel is missing {suffix}")
        if any("/tests/" in f"/{path}" or "/spec/" in f"/{path}" for path in members):
            raise AssertionError("Wheel contains test or specification files")

    version = metadata["Version"]
    if version is None:
        raise AssertionError("Wheel metadata is missing Version")
    return version


def verify_sdist(sdist: Path, version: str) -> None:
    with tarfile.open(sdist, "r:gz") as archive:
        members = {member.name for member in archive.getmembers()}

    root = f"personalitygen-{version}"
    for required in {
        f"{root}/LICENSE",
        f"{root}/README.md",
        f"{root}/pyproject.toml",
        f"{root}/src/personalitygen/__init__.py",
        f"{root}/src/personalitygen/py.typed",
    }:
        if required not in members:
            raise AssertionError(f"Source distribution is missing {required}")
    if any("/tests/" in f"/{path}" or "/spec/" in f"/{path}" for path in members):
        raise AssertionError("Source distribution contains test or specification files")


def verify_isolated_install(wheel: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="personalitygen-wheel-") as directory:
        environment = Path(directory)
        venv.EnvBuilder(with_pip=True).create(environment)
        python = environment / "bin" / "python"
        subprocess.run(
            [
                python,
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--no-deps",
                wheel.resolve(),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        smoke_test = """
from personalitygen import (
    AdaptiveBifurcatedProfile,
    BigFiveConflictResolutionConfiguration,
    BigFiveConflictResolutionStyle,
    PriorityLevel,
)

conflict = BigFiveConflictResolutionConfiguration(
    BigFiveConflictResolutionStyle.AVOIDING
)
assert conflict.concern_for_self is PriorityLevel.LOW
assert len(AdaptiveBifurcatedProfile.random().vector) == 5
"""
        subprocess.run(
            [python, "-c", smoke_test],
            check=True,
            capture_output=True,
            text=True,
        )


def main() -> None:
    arguments = parse_arguments()
    wheel = only_matching(arguments.dist, "*.whl")
    sdist = only_matching(arguments.dist, "*.tar.gz")
    version = wheel_version(wheel)
    if arguments.expected_version is not None:
        if version != arguments.expected_version:
            raise AssertionError(
                f"Built version {version} does not match "
                f"{arguments.expected_version}"
            )
    verify_sdist(sdist, version)
    verify_isolated_install(wheel)
    print(
        json.dumps(
            {
                "version": version,
                "wheel": str(wheel),
                "sdist": str(sdist),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
