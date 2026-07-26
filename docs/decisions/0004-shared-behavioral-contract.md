# 0004: Shared Behavioral Contract

## Status

Accepted

## Decision

Cross-language personality semantics live under top-level `spec/`.
`model.json` records canonical parameters and names; machine-readable
conformance fixtures record representative inputs and expected outputs. Every
implementation consumes the fixtures in tests but retains its own explicit
runtime code.

The project guarantees semantic parity, not identical results from the same
integer seed across runtimes. Random conformance cases inject explicit uniform
fractions. Arithmetic comparisons use absolute tolerance `1e-12`;
truncated-Gaussian comparisons use `1e-6`.

## Rationale

Generating one implementation from another would make one language
authoritative and encourage non-idiomatic APIs. Loading shared JSON at runtime
would make otherwise dependency-free packages depend on repository layout and
packaged resources. A test-time contract preserves equal status and catches
drift without coupling runtime artifacts.

## Consequences

- Update the specification, fixtures, implementations, tests, and docs
  together when behavior changes.
- Do not add a cross-language seeded-output promise without first specifying a
  shared PRNG and numeric sampling algorithm.
- Fail tests when a conformance fixture exists but an implementation does not
  consume it.
- Keep language-specific formatting and ergonomics outside the shared contract.
