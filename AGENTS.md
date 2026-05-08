# Repository Guidelines

## Start Here
- [docs/README.md](docs/README.md) is the documentation index and system of record.
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) explains the package boundaries, domain model, and extension rules.
- [docs/QUALITY.md](docs/QUALITY.md) defines validation commands, testing expectations, dependency policy, and release/runtime constraints.
- [docs/LEGIBILITY_AUDIT.md](docs/LEGIBILITY_AUDIT.md) tracks repo-legibility gaps and next cleanup work.

## Project Shape
- `personalitygen` is a dependency-free Python 3.11+ library for simulated Big Five personality profiles.
- Source lives in `src/personalitygen/`; tests live in top-level `tests/`.
- Keep `AGENTS.md` short. Put durable architecture, quality, and planning detail in `docs/`.

## Working Commands
- Install: `pdm install --group dev`
- Tests: `pdm run test`
- Lint: `pdm run lint`

## Change Rules
- Use PDM for dependency and environment management.
- Keep runtime support aligned across `pyproject.toml`, README badges, docs, and GitHub Actions.
- Do not introduce Python 3.14-only syntax unless the package metadata and CI matrix are intentionally raised from Python 3.11+.
- Keep runtime dependencies empty unless the package genuinely needs one; dev dependencies should use lower bounds such as `>=`.
- Tests should encode the behavioral contract: unit-range validation, aggregate scoring, life-stage sampling, deterministic randomness, and conflict-style derivation.
