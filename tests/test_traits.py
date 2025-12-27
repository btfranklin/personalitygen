import random

from personalitygen.enums import LifeStage
from personalitygen.personality import BigFiveTraitConfiguration


def test_trait_configuration_is_deterministic_for_seed() -> None:
    rng_a = random.Random(123)
    rng_b = random.Random(123)

    traits_a = BigFiveTraitConfiguration.random(LifeStage.ADULT, rng=rng_a)
    traits_b = BigFiveTraitConfiguration.random(LifeStage.ADULT, rng=rng_b)

    assert traits_a == traits_b
