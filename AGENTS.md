# Repository Guidelines

## Start Here
- [docs/README.md](docs/README.md) is the documentation index and system of record.
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) explains the package boundaries, domain model, and extension rules.
- [docs/QUALITY.md](docs/QUALITY.md) defines validation commands, testing expectations, dependency policy, and release/runtime constraints.
- [docs/LEGIBILITY_AUDIT.md](docs/LEGIBILITY_AUDIT.md) tracks repo-legibility gaps and next cleanup work.

## Project Shape
- `personalitygen` is a language-neutral project for simulated Big Five and
  ABBF personality profiles.
- The Python 3.11+ package lives in `packages/python/`, with source under
  `packages/python/src/personalitygen/` and tests under
  `packages/python/tests/`.
- Keep `AGENTS.md` short. Put durable architecture, quality, and planning detail in `docs/`.

## Working Commands
- Install: `pdm install -p packages/python --group dev`
- Tests: `pdm run -p packages/python test`
- Lint: `pdm run -p packages/python lint`

## Change Rules
- Use PDM for dependency and environment management.
- Keep Python runtime support aligned across
  `packages/python/pyproject.toml`, package badges, docs, and GitHub Actions.
- Do not introduce Python 3.14-only syntax unless the package metadata and CI matrix are intentionally raised from Python 3.11+.
- Keep runtime dependencies empty unless the package genuinely needs one; dev dependencies should use lower bounds such as `>=`.
- Tests should encode the behavioral contract: unit-range validation, aggregate scoring, life-stage sampling, deterministic randomness, and conflict-style derivation.
