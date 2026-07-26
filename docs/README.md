# personalitygen Docs

This directory is the source of truth for repository structure, domain rules, validation, and maintenance notes. `AGENTS.md` should stay as a short map back to these documents.

## Index

- [Architecture](ARCHITECTURE.md): package layout, domain model, dependency direction, and extension rules.
- [Usage](USAGE.md): simulation-oriented recipes and links to runnable examples.
- [Quality](QUALITY.md): install commands, validation commands, test expectations, dependency policy, and release/runtime constraints.
- [Legibility Audit](LEGIBILITY_AUDIT.md): current repo-legibility strengths, completed cleanup, remaining pressure points, and entropy controls.
- [Decisions](decisions/README.md): durable technical decisions that should not be rediscovered from chat or commit history.

## Fast Orientation

`personalitygen` is organized as a language-neutral repository with
first-class implementations under `packages/`. The current Python 3.11+
package is dependency-free and exposes immutable value objects for Big Five
traits, conflict-resolution configuration, and ABBF signed-vector profiles.
Random generation accepts an optional seeded random source so callers can
reproduce game, story, and simulation characters.

Before changing Python behavior, read the architecture and quality docs, then
run `pdm run -p packages/python test` and
`pdm run -p packages/python lint`.
