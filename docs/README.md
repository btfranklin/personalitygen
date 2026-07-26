# personalitygen Docs

This directory is the source of truth for repository structure, domain rules, validation, and maintenance notes. `AGENTS.md` should stay as a short map back to these documents.

## Index

- [Architecture](ARCHITECTURE.md): package layout, domain model, dependency direction, and extension rules.
- [Usage](USAGE.md): simulation-oriented recipes and links to runnable examples.
- [Quality](QUALITY.md): install commands, validation commands, test expectations, dependency policy, and release/runtime constraints.
- [Legibility Audit](LEGIBILITY_AUDIT.md): current repo-legibility strengths, completed cleanup, remaining pressure points, and entropy controls.
- [Decisions](decisions/README.md): durable technical decisions that should not be rediscovered from chat or commit history.
- [Behavioral Contract](../spec/BEHAVIOR.md): language-neutral model semantics
  and conformance-fixture policy.

## Fast Orientation

`personalitygen` is organized as a language-neutral repository with equal
Python and TypeScript implementations under `packages/`. Both are
dependency-free and expose immutable value objects for Big Five traits,
conflict-resolution behavior, and ABBF signed-vector profiles. Random
generation accepts an optional caller-owned random source so games, stories,
and simulations can control reproducibility.

The language-neutral model and conformance fixtures live under `spec/`. A
behavior change must update that contract together with every implementation.

Before changing behavior, read the architecture and quality docs, then run both
language suites described in [Quality](QUALITY.md).
