# Quality

## Runtime And Dependency Policy

The package targets Python 3.11 and newer. Keep this aligned in:

- `pyproject.toml` `requires-python`
- README badges and wording
- GitHub Actions test matrix
- release/build workflow Python version
- `AGENTS.md` and these docs

Runtime dependencies should remain empty unless a feature genuinely requires one. Development dependencies should be managed by PDM and use lower-bound constraints such as `>=`.

## Setup

```bash
pdm install --group dev
```

## Validation

Run both commands before committing behavior changes:

```bash
pdm run test
pdm run lint
```

`pdm run test` runs the offline pytest suite. `pdm run lint` runs Ruff over source and tests.

## Test Contracts

Tests should protect the package's public behavior, not only current implementation details. Important contracts include:

- every trait component must stay in the unit range
- every trait aggregate score must equal the average of its components
- every supported life stage must be sampleable for every Big Five trait
- life-stage bias should keep its intended direction unless a domain decision changes it
- seeded generation should be deterministic
- conflict-resolution styles should remain reachable and mapped to concern priorities
- ABBF scores should stay in the signed range
- ABBF vector order, pole metadata, dominant-pole selection, and Big Five projection should stay stable
- ABBF dot product and cosine similarity should handle zero vectors predictably
- runnable examples should execute successfully under the test suite
- the public package export surface should stay explicit
- docs and metadata should agree about the supported Python runtime

Prefer deterministic fake random sources or seeded `random.Random` instances. Keep the suite offline and fast.

## Release Checks

Before release, verify:

1. The test matrix covers every advertised Python version.
2. `pdm build` succeeds from a clean checkout.
3. The generated package includes `py.typed`.
4. README usage examples still work against the public import surface.
