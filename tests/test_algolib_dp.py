"""SSPO dynamic programs (COUNTS / TABLE / EXPECT / DIFF / UHAT) against full-subset enumeration.

Ported from SSPO v1 (tests/test_01_dp_correctness.py, tests/test_02_theorem2_properties.py: counterexamples).
"""
from __future__ import annotations

import itertools
import math
import random
from typing import List, Tuple

import numpy as np
import pytest

from selectrl.algolib.combin import capped_draw_table
from selectrl.algolib.selector import classify_group, j_minus, w_star_mask
from selectrl.algolib.sspo_dp import (
    TABLE,
    UHAT,
    brute_force_selection_advantages,
    class_spec,
    dp_selection_advantages,
)

Arrays = Tuple[List[int], List[int], List[int], List[int]]
Config = Tuple[int, int, int, int, int, List[int]]  # (n, k, n_C, n_T, n_W_star, wrong cluster sizes)


def build_group(n_C: int, n_T: int, n_W_star: int, w_clusters: List[int], rng: random.Random) -> Arrays:
    """Arrays (g, chk, sig_id, sig_known_correct) for the given class counts, in random order.

    C uses signature 0; W* reuses signature 0 or a reference-only known-correct signature 50; W_m uses 1 + m; T
    samples have Chk = 0 (some with g = 1, i.e. false rejections) and no or unique signatures.
    """
    c_known = rng.random() < 0.7
    w_star_sig = 0 if rng.random() < 0.5 else 50
    rows = [(1, 1, 0, int(c_known))] * n_C
    rows += [(1 if j % 3 == 0 else 0, 0, -1 if j % 2 else 100 + j, 0) for j in range(n_T)]
    rows += [(0, 1, w_star_sig, 1)] * n_W_star
    for m, size in enumerate(w_clusters):
        rows += [(0, 1, 1 + m, 0)] * size
    rng.shuffle(rows)
    g, chk, sig, known = (list(col) for col in zip(*rows))
    return g, chk, sig, known


def random_counts(n: int, rng: random.Random) -> Tuple[int, int, int, List[int]]:
    rem = n
    n_C = rng.randint(0, rem)
    rem -= n_C
    n_T = rng.randint(0, rem)
    rem -= n_T
    n_W_star = rng.randint(0, min(rem, 4)) if rem > 0 and rng.random() < 0.4 else 0
    rem -= n_W_star
    clusters: List[int] = []
    while rem > 0:
        size = clusters[-1] if clusters and rem >= clusters[-1] and rng.random() < 0.45 else rng.randint(1, rem)
        clusters.append(size)
        rem -= size
    return n_C, n_T, n_W_star, sorted(clusters, reverse=True)


def generate_configurations(seed: int = 20261015) -> List[Config]:
    rng = random.Random(seed)
    configs: List[Config] = [
        (16, 8, 0, 16, 0, []),  # only T
        (16, 8, 16, 0, 0, []),  # all C
        (16, 8, 0, 0, 16, []),  # all W*
        (16, 8, 0, 0, 0, [16]),  # one giant wrong cluster
        (16, 8, 3, 4, 0, [9]),  # centering counterexample 1
        (16, 8, 3, 4, 0, [8, 1]),  # centering counterexample 2
        (16, 8, 4, 2, 2, [4, 4]),  # C tied with two wrong clusters, W* present
        (16, 8, 2, 2, 1, [3, 3, 3, 2]),  # three-way tie of wrong clusters
        (16, 8, 1, 0, 0, [1] * 15),  # singleton wrong clusters
        (16, 8, 5, 3, 3, [2, 2, 1]),  # heavy W*
        (16, 16, 4, 4, 2, [3, 3]),  # k = n
        (16, 1, 4, 4, 2, [3, 3]),  # k = 1
        (4, 2, 2, 0, 0, [2]),  # exact tie at small n
        (6, 3, 2, 1, 1, [2]),
        (8, 4, 2, 2, 0, [2, 2]),
        (8, 8, 0, 8, 0, []),
    ]
    for n in (4, 6, 8):
        for k in range(1, n + 1):
            for _ in range(6):
                configs.append((n, k, *random_counts(n, rng)))
    for k in range(1, 17):
        for _ in range(5):
            configs.append((16, k, *random_counts(16, rng)))
    return configs


def test_dp_matches_brute_force_on_200_plus_configurations():
    configs = generate_configurations()
    assert len(configs) >= 200
    rng = random.Random(7)
    max_err = 0.0
    seen_w_star = seen_ties = 0
    for idx, (n, k, n_C, n_T, n_W_star, w_clusters) in enumerate(configs):
        g, chk, sig, known = build_group(n_C, n_T, n_W_star, w_clusters, rng)
        classes = classify_group(g, chk, sig, known)
        assert not classes.impure, f"config {idx} must be cluster-pure"
        assert (classes.n_C, classes.n_T, classes.n_W_star) == (n_C, n_T, n_W_star)
        assert classes.n_W_clusters == w_clusters
        seen_w_star += n_W_star > 0
        seen_ties += len(set(w_clusters)) < len(w_clusters) or n_C in w_clusters

        adv_dp = dp_selection_advantages(classes, k)
        adv_bf, uhat_km1_bf, uhat_k_bf = brute_force_selection_advantages(g, chk, sig, known, k)
        max_err = max(max_err, float(np.max(np.abs(adv_dp - adv_bf))))

        for i in range(n):
            spec = class_spec(classes, i)
            kwargs = dict(n=n, n_C=n_C, n_T=n_T, n_W_clusters=w_clusters)
            max_err = max(max_err, abs(UHAT(spec, k - 1, **kwargs) - uhat_km1_bf[i]))
            if k < n:
                max_err = max(max_err, abs(UHAT(spec, k, **kwargs) - uhat_k_bf[i]))
    assert max_err <= 1e-9, f"max abs error {max_err:.3e}"
    assert seen_w_star >= 20 and seen_ties >= 20


def test_table_matches_enumeration():
    rng = random.Random(3)
    for _ in range(60):
        sizes = [rng.randint(0, 4) for _ in range(rng.randint(0, 4))]
        w = rng.randint(0, 4)
        max_s = rng.randint(0, 8)
        table = TABLE(w, sizes, max_s)
        expected = [[0] * (len(sizes) + 1) for _ in range(max_s + 1)]
        for draw in itertools.product(*[range(min(w, m) + 1) for m in sizes]):
            s = sum(draw)
            t = sum(1 for j in draw if j == w)
            if s > max_s or (w >= 1 and t == 0) or (w == 0 and s > 0):
                continue
            t = 0 if w == 0 else t
            expected[s][t] += math.prod(math.comb(m, j) for m, j in zip(sizes, draw))
        assert table == expected, (w, sizes, max_s)
        if w >= 1:
            raw = capped_draw_table(w, sizes, max_s)
            assert [row[1:] for row in raw] == [row[1:] for row in table]


def single_candidate_bf(i: int, g: List[int], chk: List[int], sig: List[int], known: List[int], k: int) -> float:
    """``A_i^(S,U)`` by explicit enumeration of all (k-1)-subsets of the other samples."""
    w_star = w_star_mask(g, chk, known)
    others = [j for j in range(len(g)) if j != i]
    subsets = list(itertools.combinations(others, k - 1))
    total = sum(j_minus((*s, i), g, chk, sig, w_star) - j_minus(s, g, chk, sig, w_star) for s in subsets)
    return total / len(subsets)


@pytest.mark.parametrize("case", ["c_two_signatures", "wrong_shares_c_signature", "impure_with_w_star"])
def test_impure_fallback(case: str):
    g = [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    chk = [1] * 12 + [0] * 4
    known = [0] * 16
    if case == "c_two_signatures":
        sig = [10, 10, 11, 1, 1, 1, 1, 1, 2, 2, 3, 4, -1, -1, -1, -1]
    elif case == "wrong_shares_c_signature":
        sig = [10, 10, 10, 10, 1, 1, 1, 1, 1, 2, 2, 2, -1, -1, -1, -1]
    else:
        sig = [10, 10, 11, 10, 1, 1, 1, 1, 2, 2, 3, 3, -1, -1, -1, -1]
        known = [1, 1, 0, 1] + [0] * 12  # sample 3 (g = 0, signature 10) is W*
    classes = classify_group(g, chk, sig, known)
    assert classes.impure
    with pytest.raises(ValueError):
        dp_selection_advantages(classes, 8)
    adv, _, _ = brute_force_selection_advantages(g, chk, sig, known, 8)
    for i in range(16):
        assert abs(adv[i] - single_candidate_bf(i, g, chk, sig, known, 8)) <= 1e-12
    if case != "wrong_shares_c_signature":  # every passed signature cluster is gold-consistent once W* is excluded
        assert all(adv[i] <= 1e-12 for i in range(16) if g[i] == 0)
        assert all(adv[i] >= -1e-12 for i in classes.c_idx)


def test_centering_counterexamples():
    """Leave-one-out centering of the selection term would give positive advantages to wrong rollouts."""
    # 3 C, 9 W_1, 4 T at n = 16, k = 8: b^S of a T rollout is -0.0210 while A^S = 0.
    b_t = UHAT("T", 8, n=16, n_C=3, n_T=4, n_W_clusters=[9]) - UHAT("T", 7, n=16, n_C=3, n_T=4, n_W_clusters=[9])
    assert abs(b_t - (-0.0210)) < 1e-4
    # 3 C, 8 W_1, 1 W_2, 4 T: the singleton W_2 rollout has A^S = 0 and b^S = -0.0272.
    g = [1] * 3 + [0] * 13
    chk = [1] * 12 + [0] * 4
    sig = [0] * 3 + [1] * 8 + [2] + [-1] * 4
    classes = classify_group(g, chk, sig, [0] * 16)
    adv = dp_selection_advantages(classes, 8)
    assert abs(adv[11]) < 1e-12
    kw = dict(n=16, n_C=3, n_T=4, n_W_clusters=[8, 1])
    b_w2 = UHAT(("W", 1), 8, **kw) - UHAT(("W", 1), 7, **kw)
    assert abs(b_w2 - (-0.0272)) < 1e-4
