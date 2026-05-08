# personalitygen Docs

This directory is the source of truth for repository structure, domain rules, validation, and maintenance notes. `AGENTS.md` should stay as a short map back to these documents.

## Index

- [Architecture](ARCHITECTURE.md): package layout, domain model, dependency direction, and extension rules.
- [Quality](QUALITY.md): install commands, validation commands, test expectations, dependency policy, and release/runtime constraints.
- [Legibility Audit](LEGIBILITY_AUDIT.md): current repo-legibility strengths, completed cleanup, remaining pressure points, and entropy controls.
- [Decisions](decisions/README.md): durable technical decisions that should not be rediscovered from chat or commit history.

## Fast Orientation

`personalitygen` is a small, dependency-free Python 3.11+ package. Its public API exposes immutable value objects for Big Five traits, a trait configuration, and a conflict-resolution configuration. Random generation accepts an optional seeded random source so callers can reproduce simulations.

Before changing behavior, read the architecture and quality docs, then run `pdm run test` and `pdm run lint`.
