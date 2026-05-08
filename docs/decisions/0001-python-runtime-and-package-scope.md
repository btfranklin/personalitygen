# 0001: Python Runtime And Package Scope

## Status

Accepted

## Decision

`personalitygen` targets Python 3.11 and newer. The package should stay dependency-free at runtime unless a future feature clearly requires a dependency.

## Rationale

The current codebase runs cleanly on Python 3.11+ and uses only the standard library. Supporting 3.11+ keeps the package broadly usable while still allowing modern typing and packaging defaults.

## Consequences

- Do not use Python 3.14-only syntax without intentionally raising `requires-python`, README badges, docs, and CI together.
- Keep GitHub Actions testing aligned with the advertised runtime range.
- Use PDM for dependency management.
- Prefer deterministic tests around the domain model so the package remains safe to change without adding dependencies.
