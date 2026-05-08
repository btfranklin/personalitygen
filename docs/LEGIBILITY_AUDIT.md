# Legibility Audit

## Current Strengths

- The package has a small `src/` layout and a single public import surface.
- Runtime behavior is deterministic when a caller supplies a seeded random source.
- There are no runtime dependencies, so the package is easy to install and inspect.
- Tests and linting run quickly through PDM.

## Completed Cleanup

- `AGENTS.md` has been reduced to a short map and no longer contains unrelated Django, async-task, or Python 3.14-only guidance.
- Runtime policy is documented as Python 3.11+ to match `pyproject.toml`, README badges, and the CI matrix.
- Architecture and quality guidance now live under `docs/`.
- Trait aggregate scoring has a shared helper so every trait class uses the same validation and average calculation rule.
- Tests cover more of the package contract: trait validation, aggregate scoring, stage sampling, deterministic generation, conflict-style mapping, public exports, and documentation/runtime alignment.

## Remaining Pressure Points

- The life-stage sampling means are still embedded directly in `traits.py`. This is acceptable while the model is small, but a larger model may want a table-shaped specification that tests and docs can reference directly.
- Conflict-style weights are documented only by code comments and tests. If this mapping becomes user-visible or research-sensitive, promote it into a domain note with rationale and expected tradeoffs.
- The release workflow should be checked whenever Python support changes so package metadata, CI, and build jobs move together.

## Entropy Controls

- Keep `AGENTS.md` as a map, not an instruction dump.
- Add new domain rules to docs before or alongside code.
- Add parameterized tests when adding traits, components, life stages, or conflict styles.
- Prefer small shared helpers for repeated domain invariants, but avoid generic abstractions that make the simple model harder to inspect.
