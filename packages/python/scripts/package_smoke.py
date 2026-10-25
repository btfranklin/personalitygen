from personalitygen import (
    AdaptiveBifurcatedProfile,
    BigFiveConflictResolution,
    BigFiveConflictResolutionStyle,
    PriorityLevel,
)

conflict = BigFiveConflictResolution(
    style=BigFiveConflictResolutionStyle.AVOIDING
)
assert conflict.concern_for_self is PriorityLevel.LOW
assert len(AdaptiveBifurcatedProfile.random().vector) == 5
