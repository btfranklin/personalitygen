# Runnable Examples

These examples are repository files. They are not included in the Python wheel,
Python source distribution, or npm package. They use the public personality API.
The game rules in this directory are examples that a game developer can change.

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

## Other Examples

The two language directories also contain equivalent examples for:

- generating one character and projecting it into ABBF
- projecting authored Big Five traits into ABBF
- comparing authored ABBF vectors
- selecting characters by dominant ABBF poles

Run all Python examples through `pdm run -p packages/python test`. Run all
TypeScript examples through `npm run --prefix packages/typescript examples`.
The aggregate checks also run the example game-rule tests.
