"""Exact combinatorics shared by the selector, the SSPO dynamic programs and the reference algorithms.

Ported from SSPO v1 (sspo/dp.py: comb_exact; sspo/eval.py: estimate_pass_at_k; sspo/advantage.py:
coverage_advantages; the capped-draw table is the core of sspo/dp.py TABLE).

* ``comb(n, k)``: exact integer binomial, 0 outside ``0 <= k <= n``.
* ``pass_at_k(n, c, k) = 1 - C(n - c, k) / C(n, k)``: the unbiased estimator of ``1 - (1 - p)^k`` from ``c``
  successes among ``n`` i.i.d. samples (Chen et al., 2021), computed exactly.
* ``capped_draw_table(cap, sizes, max_s)``: ``D[s][t]`` = number of ways to draw ``s`` items in total from disjoint
  clusters of the given sizes such that every cluster contributes at most ``cap`` items and exactly ``t`` clusters
  contribute exactly ``cap`` items.
* ``coverage_loo_advantages(flags, k)``: the full-subset leave-one-out advantage of the coverage score
  ``J_cov(K) = 1[K contains a flagged sample]`` and its leave-one-out baseline (SSPO Theorem 2 (c), (e)).
"""
from __future__ import annotations

import math
from fractions import Fraction
from typing import List, Sequence, Tuple

import numpy as np


def comb(n: int, k: int) -> int:
    """Exact binomial coefficient ``C(n, k)``; 0 when ``k < 0``, ``n < 0`` or ``k > n``."""
    if n < 0 or k < 0 or k > n:
        return 0
    return math.comb(n, k)


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimate ``1 - C(n - c, k) / C(n, k)`` for ``c`` successes among ``n`` samples.

    Its expectation over ``c ~ Binomial(n, p)`` equals ``1 - (1 - p)^k`` exactly. Raises ``ValueError`` unless
    ``0 <= c <= n`` and ``1 <= k <= n``.
    """
    if not (0 <= c <= n) or not (1 <= k <= n):
        raise ValueError(f"pass_at_k needs 0 <= c <= n and 1 <= k <= n, got n={n}, c={c}, k={k}")
    return float(1 - Fraction(comb(n - c, k), comb(n, k)))


def capped_draw_table(cap: int, sizes: Sequence[int], max_s: int) -> List[List[int]]:
    """Count draws from disjoint clusters with a per-cluster cap, tracking how many clusters hit the cap.

    Returns ``D`` with ``D[s][t]`` for ``0 <= s <= max_s`` and ``0 <= t <= len(sizes)``:
    ``D[s][t] = sum over (j_1..j_M) with sum j_m = s, j_m <= min(cap, sizes[m]), #{m : j_m = cap} = t`` of
    ``prod_m C(sizes[m], j_m)``. Computed by the recursion
    ``D_m(s, t) = sum_j C(n_m, j) D_{m-1}(s - j, t - 1[j = cap])`` with ``D_0(0, 0) = 1``.
    ``cap`` must be >= 1 (a zero cap is the trivial table ``D[0][0] = 1`` and is handled by callers).
    """
    if cap < 1:
        raise ValueError(f"capped_draw_table needs cap >= 1, got {cap}")
    m_total = len(sizes)
    table = [[0] * (m_total + 1) for _ in range(max_s + 1)]
    table[0][0] = 1
    s_upper = 0  # largest s with a non-zero entry so far
    t_upper = 0  # largest t with a non-zero entry so far
    for n_m in sizes:
        if n_m <= 0:
            continue
        max_j = min(cap, n_m, max_s)
        new = [[0] * (m_total + 1) for _ in range(max_s + 1)]
        for j in range(max_j + 1):
            ways = comb(n_m, j)
            dt = 1 if j == cap else 0
            for s_prev in range(min(s_upper, max_s - j) + 1):
                row_prev = table[s_prev]
                row_new = new[s_prev + j]
                for t_prev in range(t_upper + 1):
                    val = row_prev[t_prev]
                    if val:
                        row_new[t_prev + dt] += ways * val
        table = new
        s_upper = min(max_s, s_upper + max_j)
        t_upper = min(m_total, t_upper + (1 if max_j == cap else 0))
    return table


def coverage_loo_advantages(flags: Sequence[int], k: int) -> Tuple[np.ndarray, np.ndarray]:
    """Leave-one-out coverage advantage ``A^cov`` and its leave-one-out baseline ``b^cov`` (both shape ``(n,)``).

    With ``c = sum(flags)`` and ``c'_i = c - f_i`` flagged samples among the other ``n - 1``:

    * ``A_i^cov = f_i * C(n - c, k - 1) / C(n - 1, k - 1)``: the mean over all ``(k-1)``-subsets ``K'`` of the other
      samples of ``J_cov(K' + {i}) - J_cov(K')`` (the difference is 1 exactly when ``i`` is flagged and ``K'`` is not
      covered). ``(k / n) * A_i^cov`` equals the PKPO binary closed form ``f_i * C(n - c, k - 1) / C(n, k)``.
    * ``b_i^cov = V_i(k) - V_i(k - 1)`` with ``V_i(j) = 1 - C(n - 1 - c'_i, j) / C(n - 1, j)`` the U-statistic of
      pass@j on the other ``n - 1`` samples; it does not depend on sample ``i`` and is >= 0.

    Requires ``1 <= k <= n - 1`` (the baseline needs k-subsets of the other ``n - 1`` samples).
    """
    f = [1 if int(x) == 1 else 0 for x in flags]
    n = len(f)
    if not (1 <= k <= n - 1):
        raise ValueError(f"coverage_loo_advantages needs 1 <= k <= n - 1, got n={n}, k={k}")
    c = sum(f)
    den_km1 = comb(n - 1, k - 1)
    den_k = comb(n - 1, k)
    adv_flagged = float(Fraction(comb(n - c, k - 1), den_km1))

    def baseline(c_other: int) -> float:
        n_miss = n - 1 - c_other
        v_k = 1 - Fraction(comb(n_miss, k), den_k)
        v_km1 = 1 - Fraction(comb(n_miss, k - 1), den_km1)
        return float(v_k - v_km1)

    b_flagged = baseline(c - 1) if c > 0 else 0.0
    b_other = baseline(c) if c < n else 0.0
    adv = np.array([adv_flagged if fi else 0.0 for fi in f], dtype=np.float64)
    base = np.array([b_flagged if fi else b_other for fi in f], dtype=np.float64)
    return adv, base
