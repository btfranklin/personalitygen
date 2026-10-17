# Runnable Examples

These examples are repository files. They are not included in the Python wheel,
Python source distribution, or npm package. They use the public personality API.
The application rules in this directory are examples that a developer can change.

| Example | Main library features |
| --- | --- |
| [RPG party disagreement](#rpg-party-disagreement) | Generated traits and conflict styles guide responses. |
| [Mystery casting](#mystery-casting) | ABBF projection, dominant poles, and vector comparison select a cast. |
| [Repeatable test population](#repeatable-test-population) | Seeded generation, life stages, and component scores supply test cases. |

## RPG Party Disagreement

Run these commands from the repository root. Install the development tools as
described in [Quality](../docs/QUALITY.md#setup).

Python:

```bash
pdm run -p packages/python python examples/python/rpg_party.py
```

TypeScript:

```bash
npm run --prefix packages/typescript example:rpg
```

Both scripts generate three adult characters. Asha proposes entering ruins.
Bram is wounded and prefers camp. His generated conflict style leads him to
seek a joint plan. The party agrees to scout the entrance, and trust in the
leader increases from 2 to 3 for each follower.

The script prints each member's goal, curiosity, anxiety, conflict style,
preferred plan, response, and change in trust. Generation is repeatable within
the current package version. Python and TypeScript use different random
sources, so their trait values and styles can differ.

| Responsibility | Python | TypeScript |
| --- | --- | --- |
| Generate characters and print the scene | [rpg_party.py](python/rpg_party.py) | [rpg-party.ts](typescript/rpg-party.ts) |
| Choose plans, responses, decisions, and trust changes | [rpg_rules.py](python/rpg_rules.py) | [rpg-rules.ts](typescript/rpg-rules.ts) |

The package supplies immutable traits and a conflict style selected with
trait-based probabilities. The game owns goals, injury state, and relationships.
The example combines curiosity, anxiety, a goal bias of `0.3` or `-0.3`, and an
injury penalty of `0.4` to choose between entering the ruins and returning to
camp. These weights are game design choices, not model parameters.

Members who agree with the proposal accept it. Members who disagree respond
through their saved conflict style:

| Style | Response in this game |
| --- | --- |
| Avoiding | Stand aside |
| Obliging | Yield |
| Integrating | Seek a joint plan |
| Dominating | Refuse |
| Compromising | Offer a limited visit |

The leader returns to camp if any member refuses. Otherwise, a request for a
joint plan or limited visit changes the decision to scouting. If neither occurs,
the original proposal stands. Scouting adds one trust point. Entering against
a member's preference removes one trust point from that member. Other outcomes
leave trust unchanged. The script calculates one update from an initial trust
value of 2; a full game would store the result in its relationship state.

The example does not use personality similarity as a conflict probability.
Disagreement comes from incompatible preferred actions in this scene. No rule
changes a generated personality or adds game behavior to the installed library.

## Mystery Casting

Run from the repository root:

```bash
pdm run -p packages/python python examples/python/mystery_casting.py
npm run --prefix packages/typescript example:mystery
```

The example generates 40 adult personalities with seed 42 and projects their
Big Five traits into ABBF. Separate casting rules select three distinct people:

1. Filter for a dominant proficiency pole above `0.1`, then select the highest
   competition-axis score as the protagonist.
2. Select the remaining profile with the highest cosine similarity to the
   protagonist as the ally.
3. Select the remaining profile with the lowest cosine similarity to the
   protagonist as the rival.

Equal scores keep the input order. A pool with fewer than three candidates,
duplicate names or IDs, or no eligible protagonist fails with a clear error.
The example does not retry generation or silently lower the threshold.

The output lists each selected character's dominant poles, the selection
reasons, and all three pairwise similarity scores. The rival has the least
similar remaining profile; the score can still be positive. A story role is
assigned by the application. Personality differences do not make a character
a villain, and similarity is not a conflict probability.

| Responsibility | Python | TypeScript |
| --- | --- | --- |
| Generate the pool and print the cast | [mystery_casting.py](python/mystery_casting.py) | [mystery-casting.ts](typescript/mystery-casting.ts) |
| Filter and select story roles | [mystery_rules.py](python/mystery_rules.py) | [mystery-rules.ts](typescript/mystery-rules.ts) |

## Repeatable Test Population

Run from the repository root:

```bash
pdm run -p packages/python python examples/python/repeatable_population.py
npm run --prefix packages/typescript example:population
```

The example uses seed 42 to generate 30 profiles for each supported life stage:
child, young adult, and adult. It generates the population again with a fresh
random source and the same seed, then compares all profiles. A mismatch fails
the run. It prints the successful repeat check, all five trait means for the
whole population and each life stage, and counts for all five conflict styles,
including any zero counts.

Separate selection rules find profiles with anxiety at least `0.7` and profiles
with sociability at most `0.3`. These are two independent filters on component
scores, not aggregate neuroticism or extraversion. The report shows each match
count and up to three profiles with IDs and component scores. An empty result
stays empty; generation does not retry to fill a test category.

The sample reflects this generator's settings. It is not a demographic forecast.
The counts and filter thresholds are example choices, not personality model
rules. Repeated output is stable within the current package version and random
source; Python and TypeScript do not promise matching output from the same
seed. Save selected profile values as fixtures for durable tests, since a later
model or sampling change can change seeded output.

| Responsibility | Python | TypeScript |
| --- | --- | --- |
| Verify the repeat run and print the report | [repeatable_population.py](python/repeatable_population.py) | [test-population.ts](typescript/test-population.ts) |
| Generate groups, report scores, and select cases | [population_rules.py](python/population_rules.py) | [population-rules.ts](typescript/population-rules.ts) |

## Other Examples

The two language directories also contain equivalent examples for:

- generating one character and projecting it into ABBF
- projecting authored Big Five traits into ABBF
- comparing authored ABBF vectors
- selecting characters by dominant ABBF poles

Run all Python examples through `pdm run -p packages/python test`. Run all
TypeScript examples through `npm run --prefix packages/typescript examples`.
Both package aggregate checks also run all examples and their application-rule
tests. The Python example tests run through pytest. The TypeScript example
tests run through Node's test runner as part of the `examples` command.
