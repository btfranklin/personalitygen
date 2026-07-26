# Legibility Audit

## Current Strengths

- The package has a small `src/` layout and a single public import surface.
- Runtime behavior is deterministic when a caller supplies a seeded random source.
- There are no runtime dependencies, so the package is easy to install and inspect.
- Tests and linting run quickly through PDM.

## Completed Cleanup

- `AGENTS.md` has been reduced to a short map and no longer contains unrelated Django, async-task, or Python 3.14-only guidance.
- Runtime policy is documented as Python 3.11+ to match
  `packages/python/pyproject.toml`, package badges, and the CI matrix.
- The Python package, tests, examples, manifest, and lockfile live together
  under `packages/python/`, leaving the repository root language-neutral.
- Shared model parameters and machine-readable behavior examples live under
  `spec/`, and the Python suite proves that it consumes the complete fixture
  set.
- Architecture and quality guidance now live under `docs/`.
- Score validation and vector math now live in a small internal helper shared by Big Five and ABBF models.
- Tests cover more of the package contract: trait validation, aggregate scoring, stage sampling, deterministic generation, conflict-style mapping, ABBF vectors, runnable examples, public exports, and documentation/runtime alignment.

## Remaining Pressure Points

- Runtime model tables remain explicit inside each implementation, while
  `spec/model.json` gives tests and other language implementations one
  canonical table-shaped reference.
- The release workflow should be checked whenever Python support changes so package metadata, CI, and build jobs move together.

## Entropy Controls

- Keep `AGENTS.md` as a map, not an instruction dump.
- Add new domain rules to docs before or alongside code.
- Add parameterized tests when adding traits, components, life stages, or conflict styles.
- Prefer small shared helpers for repeated domain invariants, but avoid generic abstractions that make the simple model harder to inspect.
