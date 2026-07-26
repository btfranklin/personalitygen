"""Compare authored ABBF character vectors."""

from __future__ import annotations

from itertools import combinations

from personalitygen import AdaptiveBifurcatedProfile


CAST = {
    "diplomat": AdaptiveBifurcatedProfile(0.4, 0.6, 0.9, 0.8, 0.2),
    "scout": AdaptiveBifurcatedProfile(-0.2, -0.1, -0.4, 0.1, 0.5),
    "warlord": AdaptiveBifurcatedProfile(0.2, -0.7, -0.8, -0.6, -0.9),
}


def main() -> None:
    print("Character similarity:")
    for left_name, right_name in combinations(CAST, 2):
        left = CAST[left_name]
        right = CAST[right_name]
        print(
            f"{left_name} / {right_name}: "
            f"dot={left.dot_product(right):.2f}, "
            f"cosine={left.cosine_similarity(right):.2f}"
        )


if __name__ == "__main__":
    main()
