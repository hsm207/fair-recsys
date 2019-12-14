from functools import partial
from typing import List, NamedTuple

import numpy as np

from src.custom_types import Movie


def compute_reclist_category_ratio(s: List[Movie], category: str) -> float:
    if not s:
        return 0

    is_category = [movie.category == category for movie in s]

    return np.mean(is_category).item()


def compute_movie_score(
    movie: Movie,
    current_reclist: List[Movie],
    longtail_pref: float,
    longtail_weight: float,
) -> float:
    base_score = movie.base_score
    movie_category = movie.category
    shorthead_pref = 1 - longtail_pref
    recsize = len(current_reclist)

    if movie_category not in ["shorthead", "longtail"]:
        return base_score

    category_ratio = compute_reclist_category_ratio(current_reclist, movie_category)
    category_pref = shorthead_pref if movie_category in "shorthead" else longtail_pref

    weighted_base_score = (1 - longtail_weight) * base_score
    weighted_category_booster = (
        longtail_weight * category_pref * (1 - category_ratio) ** recsize
    )

    return weighted_base_score + weighted_category_booster


def construct_reclist(
    candidate_set: List[Movie], size: int, longtail_pref: float, longtail_weight: float
) -> List[Movie]:

    if not candidate_set:
        return []
    elif size >= len(candidate_set):
        return candidate_set

    reclist = []

    movie_sort_fn = partial(
        compute_movie_score,
        current_reclist=reclist,
        longtail_pref=longtail_pref,
        longtail_weight=longtail_weight,
    )

    while len(reclist) < size:
        candidate_set = sorted(candidate_set, key=movie_sort_fn)
        best_movie = candidate_set.pop()
        reclist.append(best_movie)

    return reclist
