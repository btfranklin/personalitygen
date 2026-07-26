# Quality

## Runtime And Dependency Policy

The Python package targets Python 3.11–3.14. Keep this aligned in:

- `packages/python/pyproject.toml` `requires-python`
- README badges and wording
- GitHub Actions test matrix
- release/build workflow Python version
- `AGENTS.md` and these docs

Runtime dependencies should remain empty unless a feature genuinely requires one. Development dependencies should be managed by PDM and use lower-bound constraints such as `>=`.

The TypeScript package publishes ESM targeting ES2022 without a Node engine
restriction. TypeScript 7 is the build compiler, while the emitted declarations
must compile under TypeScript 6+. TypeScript is a development dependency, not a
consumer runtime dependency. Manage TypeScript tooling with npm, commit
`package-lock.json`, and use lower-bound development constraints.

Python and TypeScript public versions move in lockstep.

## Cross-Language Contract

The model under `spec/` is the semantic source of truth. Every implementation
test suite must consume all files under `spec/conformance/`; published packages
must not load those files at runtime. Numeric comparisons use absolute
tolerance `1e-12` for ordinary arithmetic and `1e-6` for truncated-Gaussian
sampling.

## Setup

```bash
pdm install -p packages/python --group dev
cd packages/typescript && npm ci
```

## Validation

Run both language checks before committing behavior changes:

```bash
pdm run -p packages/python test
pdm run -p packages/python lint
npm run --prefix packages/typescript check
```

The TypeScript check runs Biome, TypeScript 7 compilation, Node's built-in test
runner, all conformance fixtures, TypeScript 6 declaration consumption, package
content inspection, and an isolated tarball installation.

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
- the TypeScript package should expose only ESM JavaScript and declarations
- TypeScript value objects and categorical objects should stay frozen
- TypeScript invalid numeric and categorical inputs should retain their
  `RangeError` and `TypeError` split

Prefer deterministic fake random sources or seeded `random.Random` instances. Keep the suite offline and fast.

## Release Checks

Before release, verify:

1. Python CI passes on 3.11–3.14.
2. TypeScript CI passes on Node 22, 24, and 26.
3. `pdm build -p packages/python` succeeds and includes `py.typed`.
4. `npm run --prefix packages/typescript check` succeeds and the tarball
   contains only package metadata, license, README, ESM, declarations, and
   source maps.
5. Both package versions match the release tag.
6. Both OIDC publishing workflows build and test from the release commit.
