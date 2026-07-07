"""
Elo Rating System for Hypothesis Tournament.

Implements pairwise comparison and rating updates inspired by
Google Co-Scientist's tournament-based hypothesis ranking.
"""

import math
from typing import Tuple


def expected_score(rating_a: int, rating_b: int) -> float:
    """Calculate expected score for player A against player B."""
    return 1.0 / (1.0 + math.pow(10, (rating_b - rating_a) / 400.0))


def update_ratings(
    rating_a: int,
    rating_b: int,
    winner: str,  # "a" | "b" | "draw"
    k_factor: int = 32,
) -> Tuple[int, int]:
    """
    Update Elo ratings after a pairwise comparison.

    Args:
        rating_a: Current Elo of hypothesis A
        rating_b: Current Elo of hypothesis B
        winner: "a" if A wins, "b" if B wins, "draw" for tie
        k_factor: Sensitivity of rating changes

    Returns:
        Tuple of (new_rating_a, new_rating_b)
    """
    ea = expected_score(rating_a, rating_b)
    eb = expected_score(rating_b, rating_a)

    if winner == "a":
        sa, sb = 1.0, 0.0
    elif winner == "b":
        sa, sb = 0.0, 1.0
    else:  # draw
        sa, sb = 0.5, 0.5

    new_a = round(rating_a + k_factor * (sa - ea))
    new_b = round(rating_b + k_factor * (sb - eb))

    return new_a, new_b


def rank_hypotheses(hypotheses: list) -> list:
    """Sort hypotheses by Elo rating (descending)."""
    return sorted(hypotheses, key=lambda h: h.elo_rating, reverse=True)
