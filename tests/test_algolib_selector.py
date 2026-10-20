"""Selector, exact select@k / cov@k and pass@k.

Ported from SSPO v1 (tests/test_10_eval_estimators.py: pass@k and select@k parts); impure-cluster cases are new.
"""
from __future__ import annotations

import itertools
import math
import random
from typing import List, Tuple

import numpy as np
import pytest

from selectrl.algolib.combin import comb, pass_at_k
from selectrl.algolib.selector import (
    classify_group,
    exact_cov_at_k,
    exact_select_at_k,
    j_cov,
    selector_success,
)
from selectrl.algolib.sspo_dp import UHAT

Pool = Tuple[List[int], List[int], List[int]]


def random_pool(n: int, rng: random.Random, n_sigs: int = 4, pure: bool = False) -> Pool:
    """Random (g, chk, sig_id). Impure by default: g is drawn independently of the signature."""
    sig_gold = [rng.randint(0, 1) for _ in range(n_sigs)]
    g, chk, sig = [], [], []
    for _ in range(n):
        s = rng.randrange(n_sigs)
        c = 1 if rng.random() < 0.7 else 0
        sig.append(s if c else -1)
        chk.append(c)
        g.append(sig_gold[s] if (pure and c) else rng.randint(0, 1))
    return g, chk, sig


def enumerate_select(g: List[int], chk: List[int], sig: List[int], k: int) -> float:
    subsets = list(itertools.combinations(range(len(g)), k))
    return sum(selector_success(s, g, chk, sig) for s in subsets) / len(subsets)


def test_selector_success_examples():
    g = [1, 0, 0, 1, 0, 1]
    chk = [1, 1, 1, 1, 0, 1]
    sig = [0, 1, 1, 0, -1, 2]
    assert selector_success([0, 1], g, chk, sig) == 0.5  # tie of two singletons
    assert selector_success([0, 1, 2], g, chk, sig) == 0.0  # wrong cluster is the largest
    assert selector_success([0, 3, 1], g, chk, sig) == 1.0
    assert selector_success([4], g, chk, sig) == 0.0  # nothing passes Chk
    assert selector_success([0, 1, 5], g, chk, sig) == pytest.approx(2.0 / 3.0)
    impure_sig = [0, 0, 1, 1, -1, 2]  # cluster 0 holds one correct and one wrong program
    assert selector_success([0, 1, 5], g, chk, impure_sig) == 0.5
    assert selector_success([0, 1, 2, 3], g, chk, impure_sig) == 0.5  # tie of {0, 1} (1/2) and {2, 3} (1/2)


def test_exact_select_at_k_matches_enumeration_with_impure_clusters():
    rng = random.Random(11)
    n_impure_pools = 0
    for _ in range(120):
        n = rng.randint(1, 10)
        g, chk, sig = random_pool(n, rng, n_sigs=rng.randint(1, 4))
        n_impure_pools += classify_group(g, chk, sig, [0] * n).impure
        for k in range(1, n + 1):
            assert abs(exact_select_at_k(g, chk, sig, k) - enumerate_select(g, chk, sig, k)) <= 1e-12
    assert n_impure_pools >= 30


def test_exact_select_at_k_large_pool_against_monte_carlo():
    rng = random.Random(5)
    g, chk, sig = random_pool(64, rng, n_sigs=10)
    sampler = np.random.default_rng(0)
    for k in (8, 16):
        exact = exact_select_at_k(g, chk, sig, k)
        draws = [selector_success(sampler.choice(64, size=k, replace=False).tolist(), g, chk, sig)
                 for _ in range(4000)]
        assert abs(exact - float(np.mean(draws))) < 4.0 * float(np.std(draws)) / math.sqrt(4000) + 1e-3
    assert 0.0 <= exact_select_at_k(g, chk, sig, 64) <= 1.0


def test_exact_select_at_k_matches_pure_dynamic_program():
    """Under purity select@k equals UHAT(T, k) on the pool plus one dummy T sample (SSPO v1 eval route)."""
    rng = random.Random(2)
    for _ in range(40):
        n = rng.randint(2, 16)
        g, chk, sig = random_pool(n, rng, n_sigs=5, pure=True)
        # make C a single signature: relabel every correct signature to the first one
        correct_sigs = sorted({s for gi, ci, s in zip(g, chk, sig) if ci and gi})
        sig = [correct_sigs[0] if (c and s in correct_sigs) else s for c, s in zip(chk, sig)]
        classes = classify_group(g, chk, sig, [0] * n)
        assert not classes.impure
        for k in range(1, n + 1):
            via_dp = UHAT("T", k, n=n + 1, n_C=classes.n_C, n_T=classes.n_T + 1, n_W_clusters=classes.n_W_clusters)
            assert abs(exact_select_at_k(g, chk, sig, k) - via_dp) <= 1e-9


def test_exact_cov_at_k_and_pass_at_k():
    assert abs(pass_at_k(16, 4, 8) - (1 - math.comb(12, 8) / math.comb(16, 8))) < 1e-15
    assert pass_at_k(16, 0, 8) == 0.0 and pass_at_k(16, 16, 1) == 1.0
    with pytest.raises(ValueError):
        pass_at_k(4, 1, 5)
    # unbiased: E[pass_at_k(n, c, k)] over c ~ Binomial(n, p) equals 1 - (1 - p)^k
    n = 16
    for p in (0.1, 0.25, 0.5, 0.7):
        for k in (1, 2, 4, 8, 16):
            mean = sum(comb(n, c) * p ** c * (1 - p) ** (n - c) * pass_at_k(n, c, k) for c in range(n + 1))
            assert abs(mean - (1 - (1 - p) ** k)) < 1e-14
    rng = random.Random(4)
    for _ in range(40):
        n = rng.randint(1, 10)
        g, chk, sig = random_pool(n, rng)
        n_c = sum(1 for a, b in zip(g, chk) if a and b)
        for k in range(1, n + 1):
            subsets = list(itertools.combinations(range(n), k))
            enum = sum(j_cov(s, g, chk) for s in subsets) / len(subsets)
            assert abs(exact_cov_at_k(g, chk, k) - enum) <= 1e-12
            assert exact_cov_at_k(g, chk, k) == pass_at_k(n, n_c, k)


def test_monotonicity_sanity():
    rng = random.Random(9)
    for _ in range(60):
        n = rng.randint(2, 12)
        g, chk, sig = random_pool(n, rng, n_sigs=4, pure=True)
        n_c = sum(1 for a, b in zip(g, chk) if a and b)
        assert abs(exact_select_at_k(g, chk, sig, 1) - n_c / n) < 1e-12  # select@1 = pass@1 on C
        for k in range(1, n + 1):  # J_S(K) <= J_cov(K) on every subset
            assert exact_select_at_k(g, chk, sig, k) <= exact_cov_at_k(g, chk, k) + 1e-12
        rejected = [i for i in range(n) if chk[i] == 0]
        passed = [i for i in range(n) if chk[i] == 1]
        if not rejected or not passed:
            continue
        i, j = rng.choice(rejected), rng.choice(passed)
        # Under purity, turning a rejected rollout into a copy of rollout j (same signature and gold label) cannot
        # lower select@k when j is correct and cannot raise it when j is wrong (SSPO lemma, per subset).
        g2, chk2, sig2 = list(g), list(chk), list(sig)
        g2[i], chk2[i], sig2[i] = g[j], 1, sig[j]
        for k in range(1, n + 1):
            before, after = exact_select_at_k(g, chk, sig, k), exact_select_at_k(g2, chk2, sig2, k)
            if g[j] == 1:
                assert after >= before - 1e-12
            else:
                assert after <= before + 1e-12
