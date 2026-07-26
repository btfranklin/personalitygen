# 0005: TypeScript As An Equal Package

## Status

Accepted.

## Decision

Publish `packages/typescript/` as the unscoped npm package `personalitygen`.
Python and TypeScript are equal implementations, share public versions, and
conform independently to `spec/`.

The TypeScript package:

- publishes dependency-free ESM and declarations from `dist/`
- targets ES2022 without imposing a Node engine restriction
- uses frozen classes with object-parameter constructors and camelCase APIs
- represents categorical values with frozen const objects and string unions
- accepts a structural `RandomSource` rather than owning a seeded generator
- builds with TypeScript 7 and verifies declaration consumption with
  TypeScript 6
- uses Node's built-in test runner and Biome

## Consequences

Neither package depends on the other at build time or runtime. Shared semantics
change through the specification and conformance fixtures first, while each
implementation remains idiomatic. Releases are incomplete unless both language
jobs pass and both registry versions match.
