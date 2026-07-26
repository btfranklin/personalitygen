# Quality

## Runtime And Dependency Policy

The package targets Python 3.11 and newer. Keep this aligned in:

- `packages/python/pyproject.toml` `requires-python`
- README badges and wording
- GitHub Actions test matrix
- release/build workflow Python version
- `AGENTS.md` and these docs

Runtime dependencies should remain empty unless a feature genuinely requires one. Development dependencies should be managed by PDM and use lower-bound constraints such as `>=`.

## Cross-Language Contract

The model under `spec/` is the semantic source of truth. Every implementation
test suite must consume all files under `spec/conformance/`; published packages
must not load those files at runtime. Numeric comparisons use absolute
tolerance `1e-12` for ordinary arithmetic and `1e-6` for truncated-Gaussian
sampling.

## Setup

```bash
pdm install -p packages/python --group dev
```

## Validation

Run both commands before committing behavior changes:

```bash
pdm run -p packages/python test
pdm run -p packages/python lint
```

The PDM commands above run the offline pytest suite and Ruff over Python source,
tests, and examples.

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
- the Python implementation should satisfy every shared conformance fixture
- the fixture set should be explicit so new files cannot be silently ignored

Prefer deterministic fake random sources or seeded `random.Random` instances. Keep the suite offline and fast.

## Release Checks

Before release, verify:

1. The test matrix covers every advertised Python version.
2. `pdm build -p packages/python` succeeds from a clean checkout.
3. The generated package includes `py.typed`.
4. README usage examples still work against the public import surface.
