"""Generate repeatable test data and inspect the selected component scores."""

from personalitygen import LifeStage

from population_rules import (
    generate_population,
    select_cases,
    style_counts,
    trait_means,
)


def main() -> None:
    population = generate_population()
    repeated = generate_population()
    if population != repeated:
        raise AssertionError("Repeated generation changed a profile")
    print(f"Test population: {len(population)} profiles; seed=42; 30 per life stage")
    print("Repeat check: every generated component and conflict style matches.")
    for label, group in [
        ("all", population),
        *[
            (stage.value, [p for p in population if p.life_stage == stage])
            for stage in LifeStage
        ],
    ]:
        means = "; ".join(
            f"{name}={score:.3f}" for name, score in trait_means(group).items()
        )
        print(f"{label}: count={len(group)}; {means}")
    print("Conflict styles:")
    for style, count in style_counts(population).items():
        print(f"  {style}: {count}")
    for label, matches in select_cases(population).items():
        print(f"{label}: {len(matches)} matches; showing up to 3")
        for person in matches[:3]:
            traits = person.personality.traits
            print(
                f"  {person.identifier}: "
                f"anxiety={traits.neuroticism.anxiety_score:.3f}; "
                f"sociability={traits.extraversion.sociability_score:.3f}"
            )
    print("These are generator samples, not a demographic forecast.")
    print(
        "Save selected profiles as fixtures; fresh random draws can miss a test case."
    )


if __name__ == "__main__":
    main()
