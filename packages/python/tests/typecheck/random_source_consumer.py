from personalitygen import AdaptiveBifurcatedProfile


class UniformOnlyRandom:
    def uniform(self, minimum: float, maximum: float) -> float:
        return minimum + ((maximum - minimum) * 0.5)


profile = AdaptiveBifurcatedProfile.random(rng=UniformOnlyRandom())
assert len(profile.vector) == 5
