# 0003: First-Class Language Package Layout

## Status

Accepted

## Decision

`personalitygen` is a language-neutral repository. Each implementation lives
as an independently buildable distribution under `packages/<language>/`.
Python lives under `packages/python/`; future language implementations belong
beside it rather than under it.

Repository-wide architecture, behavioral decisions, and quality guidance stay
under top-level `docs/`. Package manifests, source, tests, examples, lockfiles,
and registry-facing README and license files stay inside their owning package.

## Rationale

Python and TypeScript are intended to be equal implementations, and their
long-term relative value is not yet known. Symmetric package boundaries make
that intent visible while allowing each ecosystem to use its native tooling
and release artifact.

## Consequences

- Pass `-p packages/python` to PDM commands from the repository root.
- Keep package builds self-contained; do not reach outside a package directory
  for files required in its published artifact.
- Do not make one language implementation a runtime dependency of another.
- Put cross-language behavioral truth in a repository-level specification.
