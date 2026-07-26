# 0006: Public Personality Vocabulary

## Status

Accepted.

## Decision

Use domain names for generated personality data rather than configuration
names:

- `BigFiveTraits` is the complete collection of five Big Five trait objects.
- `BigFiveConflictResolution` is a conflict style with its derived concern
  priorities.
- `BigFivePersonality` exposes these values as `traits` and
  `conflict_resolution` in Python, and `traits` and `conflictResolution` in
  TypeScript.
- Conflict-resolution objects expose their authored style as `style`.

TypeScript categorical const objects and their string-union types share the
same public name, such as `LifeStage`. The public API does not add `*Value`
aliases.

## Consequences

The API reads as character data rather than generator settings:
`personality.traits.openness` and `personality.conflict_resolution.style`.
Python and TypeScript retain idiomatic property casing while keeping the same
domain vocabulary.

This is an intentional breaking change for the 0.4.0 release line. The former
configuration names and TypeScript `*Value` aliases are removed directly,
without deprecated aliases or compatibility shims.
