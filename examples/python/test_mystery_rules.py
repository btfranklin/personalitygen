"""Check casting choices with authored vectors and explicit role rules."""

import pytest
from personalitygen import AdaptiveBifurcatedProfile

from mystery_rules import Candidate, cast_mystery


def test_cast_selects_distinct_roles_by_filter_and_similarity() -> None:
    protagonist = Candidate("lead", AdaptiveBifurcatedProfile(0.3, 0.2, 0.2, 0.1, 0.8))
    ally = Candidate("ally", AdaptiveBifurcatedProfile(0.15, 0.1, 0.1, 0.05, 0.4))
    rival = Candidate("rival", AdaptiveBifurcatedProfile(-0.3, -0.2, -0.2, -0.1, -0.8))
    spare = Candidate("spare", AdaptiveBifurcatedProfile(0, 0, 0, 0, 0.2))
    cast = cast_mystery([spare, ally, rival, protagonist])
    assert cast.protagonist == protagonist
    assert cast.ally == ally
    assert cast.rival == rival


def test_equal_scores_keep_pool_order() -> None:
    profile = AdaptiveBifurcatedProfile(0, 0, 0, 0, 0.5)
    candidates = [Candidate(name, profile) for name in ("first", "second", "third")]
    cast = cast_mystery(candidates)
    assert [cast.protagonist.name, cast.ally.name, cast.rival.name] == [
        "first",
        "second",
        "third",
    ]


def test_cast_rejects_incomplete_or_ineligible_pools() -> None:
    with pytest.raises(ValueError, match="at least three"):
        cast_mystery([])
    profile = AdaptiveBifurcatedProfile(0, 0, 0, 0, 0.1)
    candidates = [Candidate(name, profile) for name in ("one", "two", "three")]
    with pytest.raises(ValueError, match="above 0.1"):
        cast_mystery(candidates)
    with pytest.raises(ValueError, match="unique"):
        cast_mystery([candidates[0]] * 3)
