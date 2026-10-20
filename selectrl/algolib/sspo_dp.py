"""SSPO selection leave-one-out advantage: exact dynamic programs and the full-enumeration fallback.

Ported from SSPO v1 (sspo/dp.py). Function names follow the method description (SSPO section 4).

For a sample i of class X the uncentered selection advantage is (SSPO Theorem 2 (c), Proposition D)

    A_i^(S,U) = C(n-1, k-1)^-1 * sum_{K' subset of the other n-1 samples, |K'| = k-1} [J^-(K' + {i}) - J^-(K')].

Under cluster purity (C forms a single signature and no wrong cluster shares it) it depends only on X and the
class counts of the other samples:

* ``COUNTS(X)``: counts ``(n'_C, n'_T, n'_d, n'_{W,1..M'})`` of the other samples after removing one member of
  X; ``n'_d`` is the rest of the designated cluster when X = W_{m*} (0 otherwise). W* samples never take part in a
  draw that scores (J^- = 0 as soon as a W* sample is drawn) but stay in the denominator ``C(n-1, kk)``.
* ``TABLE(w)``: ``N_w(s, t)`` = ways to draw s items from the non-designated wrong clusters with every cluster at
  most w and exactly t clusters at w (t >= 1 for w >= 1; ``N_0(0, 0) = 1``).
* ``EXPECT(X, kk, phi)``: ``sum_{w,t,c,a,s} C(n'_C, c) C(n'_d, a) N_w(s, t) C(n'_T, kk-c-a-s) / C(n-1, kk) *
  phi(c, a, w, t)``, the multivariate hypergeometric expectation over kk-subsets of the other samples.
* ``J(c, a, w, t)``: selector success with c correct, a designated-cluster and (w, t) other-cluster counts:
  with ``V* = max(a, w)`` and ``R = t * 1[w = V* >= 1] + 1[a = V* >= 1]``, it is 0 if ``c = 0`` or ``c < V*``,
  1 if ``c > V*``, and ``1 / (1 + R)`` if ``c = V* >= 1``.
* ``DIFF(C) = EXPECT(C, k-1, J(c+1,a,w,t) - J(c,a,w,t))``, ``DIFF(W_m*) = EXPECT(W_m*, k-1, J(c,a+1,w,t) -
  J(c,a,w,t))``, ``DIFF(T) = 0``, ``DIFF(W*) = -UHAT(W*, k-1)``.
* ``UHAT(X, kk) = EXPECT(X, kk, J)``: the U-statistic of ``J^-`` over kk-subsets of the other samples.

When clusters are impure, ``brute_force_selection_advantages`` enumerates all k- and (k-1)-subsets instead.
"""
from __future__ import annotations

import itertools
from typing import Callable, Dict, List, Sequence, Tuple, Union

import numpy as np

from selectrl.algolib.combin import capped_draw_table, comb
from selectrl.algolib.selector import (
    LABEL_C,
    LABEL_T,
    LABEL_W,
    LABEL_W_STAR,
    GroupClasses,
    j_minus,
    w_star_mask,
)

ClassSpec = Union[str, Tuple[str, int]]  # "C", "T", "W*" or ("W", m) for the wrong cluster W_m
Phi = Callable[[int, int, int, int], float]


def J(c: int, a: int, w: int, t: int) -> float:
    """Selector success under cluster purity with c correct, a designated-cluster and (w, t) other-cluster counts."""
    v_star = max(a, w)
    if c == 0 or c < v_star:
        return 0.0
    if c > v_star:
        return 1.0
    r = (t if w == v_star else 0) + (1 if a == v_star else 0)
    return 1.0 / (1.0 + r)


def COUNTS(X: ClassSpec, n_C: int, n_T: int, n_W_clusters: Sequence[int]) -> Tuple[int, int, int, List[int]]:
    """Counts ``(n'_C, n'_T, n'_d, n'_{W,1..M'})`` of the other samples after removing one member of class X."""
    w_list = [int(x) for x in n_W_clusters]
    if X == LABEL_C:
        return n_C - 1, n_T, 0, w_list
    if X == LABEL_T:
        return n_C, n_T - 1, 0, w_list
    if X == LABEL_W_STAR:
        return n_C, n_T, 0, w_list
    if isinstance(X, tuple) and len(X) == 2 and X[0] == LABEL_W:
        m_star = int(X[1])
        others = [w_list[m] for m in range(len(w_list)) if m != m_star]
        return n_C, n_T, w_list[m_star] - 1, others
    raise ValueError(f"unknown class specification: {X!r}")


def TABLE(w: int, n_prime_W: Sequence[int], max_s: int) -> List[List[int]]:
    """``N_w(s, t)`` for ``0 <= s <= max_s``, ``0 <= t <= M'`` (exact integers)."""
    if w == 0:
        table = [[0] * (len(n_prime_W) + 1) for _ in range(max_s + 1)]
        table[0][0] = 1
        return table
    table = capped_draw_table(w, n_prime_W, max_s)
    for row in table:
        row[0] = 0  # with w >= 1 the maximum w must be attained by at least one cluster
    return table


def EXPECT(X: ClassSpec, kk: int, phi: Phi, n: int, n_C: int, n_T: int, n_W_clusters: Sequence[int]) -> float:
    """Multivariate hypergeometric expectation of ``phi(c, a, w, t)`` over kk-subsets of the other n-1 samples.

    Draws that contain a W* sample contribute 0 (they are excluded from the sum but counted in ``C(n-1, kk)``).
    """
    if kk < 0 or kk > n - 1:
        return 0.0
    denom = comb(n - 1, kk)
    nc, nt, nd, nw = COUNTS(X, n_C, n_T, n_W_clusters)
    if nc < 0 or nt < 0 or nd < 0 or any(x < 0 for x in nw):
        return 0.0
    m_prime = len(nw)
    max_c = min(kk, nc)
    max_a = min(kk, nd)
    comb_c = [comb(nc, c) for c in range(max_c + 1)]
    comb_d = [comb(nd, a) for a in range(max_a + 1)]
    comb_t = [comb(nt, r) for r in range(kk + 1)]

    total = 0.0
    for w in range(min(kk, max(nw, default=0)) + 1):
        max_s = min(kk, sum(nw), w * m_prime)
        table = TABLE(w, nw, max_s)
        t_range = range(0, 1) if w == 0 else range(1, m_prime + 1)
        for s in range(max_s + 1):
            for t in t_range:
                n_wt = table[s][t]
                if n_wt == 0:
                    continue
                for c in range(min(max_c, kk - s) + 1):
                    rest = kk - s - c
                    for a in range(max(0, rest - nt), min(max_a, rest) + 1):
                        val = phi(c, a, w, t)
                        if val != 0.0:
                            total += float(n_wt * comb_c[c] * comb_d[a] * comb_t[rest - a]) * val
    return total / float(denom)


def UHAT(X: ClassSpec, kk: int, n: int, n_C: int, n_T: int, n_W_clusters: Sequence[int]) -> float:
    """Leave-one-out U-statistic ``UHAT(X, kk) = EXPECT(X, kk, J)`` of ``J^-`` on the other n-1 samples."""
    return EXPECT(X, kk, J, n=n, n_C=n_C, n_T=n_T, n_W_clusters=n_W_clusters)


def DIFF(X: ClassSpec, n: int, k: int, n_C: int, n_T: int, n_W_clusters: Sequence[int]) -> float:
    """Uncentered selection advantage of one member of class X (see the module docstring)."""
    if X == LABEL_C:
        return EXPECT(X, k - 1, lambda c, a, w, t: J(c + 1, a, w, t) - J(c, a, w, t),
                      n=n, n_C=n_C, n_T=n_T, n_W_clusters=n_W_clusters)
    if isinstance(X, tuple) and len(X) == 2 and X[0] == LABEL_W:
        return EXPECT(X, k - 1, lambda c, a, w, t: J(c, a + 1, w, t) - J(c, a, w, t),
                      n=n, n_C=n_C, n_T=n_T, n_W_clusters=n_W_clusters)
    if X == LABEL_T:
        return 0.0
    if X == LABEL_W_STAR:
        return -UHAT(X, k - 1, n=n, n_C=n_C, n_T=n_T, n_W_clusters=n_W_clusters)
    raise ValueError(f"unknown class specification: {X!r}")


def class_spec(classes: GroupClasses, i: int) -> ClassSpec:
    """The ``ClassSpec`` of sample i ("C", "T", "W*" or ("W", m))."""
    lab = classes.label[i]
    return (LABEL_W, classes.cluster[i]) if lab == LABEL_W else lab


def dp_selection_advantages(classes: GroupClasses, k: int) -> np.ndarray:
    """Exact ``A_i^(S,U)`` for every sample of a cluster-pure group (shape ``(n,)``), with ``1 <= k <= n``.

    One DIFF per class; wrong clusters of equal size share the value (DIFF(W_m) depends only on |W_m| and the
    multiset of the other cluster sizes).
    """
    n = classes.n
    if classes.impure:
        raise ValueError("dp_selection_advantages requires a cluster-pure group; use the brute-force fallback")
    if not (1 <= k <= n):
        raise ValueError(f"dp_selection_advantages needs 1 <= k <= n, got n={n}, k={k}")
    sizes = classes.n_W_clusters
    cache: Dict[Tuple[str, int], float] = {}
    out = np.zeros(n, dtype=np.float64)
    for i in range(n):
        spec = class_spec(classes, i)
        key = (LABEL_W, sizes[spec[1]]) if isinstance(spec, tuple) else (spec, 0)
        if key not in cache:
            cache[key] = DIFF(spec, n=n, k=k, n_C=classes.n_C, n_T=classes.n_T, n_W_clusters=sizes)
        out[i] = cache[key]
    return out


def brute_force_selection_advantages(
    g: Sequence[int],
    chk: Sequence[int],
    sig_id: Sequence[int],
    sig_known_correct: Sequence[int],
    k: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Selection advantage of ``J^-`` by enumerating all k- and (k-1)-subsets (any cluster structure).

    Returns ``(adv, uhat_km1, uhat_k)``, each of shape ``(n,)``:
      adv[i]      = C(n-1, k-1)^-1 * sum_{|K'| = k-1, i not in K'} [J^-(K' + {i}) - J^-(K')]
      uhat_km1[i] = mean of J^-(K') over (k-1)-subsets of the other samples   (UHAT(X_i, k-1))
      uhat_k[i]   = mean of J^-(K)  over k-subsets of the other samples       (UHAT(X_i, k); 0 when k = n)
    The sums over subsets that avoid i are obtained as the total minus the sum over subsets containing i, so the
    cost is C(n, k) + C(n, k-1) score evaluations (24,310 for n = 16, k = 8).
    """
    g_l = [int(x) for x in np.asarray(g).reshape(-1).tolist()]
    chk_l = [int(x) for x in np.asarray(chk).reshape(-1).tolist()]
    sig_l = [int(x) for x in np.asarray(sig_id).reshape(-1).tolist()]
    n = len(g_l)
    if not (1 <= k <= n):
        raise ValueError(f"brute_force_selection_advantages needs 1 <= k <= n, got n={n}, k={k}")
    w_star = w_star_mask(g_l, chk_l, sig_known_correct)

    def subset_sums(size: int) -> Tuple[float, np.ndarray]:
        total = 0.0
        with_i = np.zeros(n, dtype=np.float64)
        for subset in itertools.combinations(range(n), size):
            val = j_minus(subset, g_l, chk_l, sig_l, w_star)
            if val != 0.0:
                total += val
                with_i[list(subset)] += val
        return total, with_i

    total_k, with_i_k = subset_sums(k)
    total_km1, with_i_km1 = subset_sums(k - 1)
    without_i_km1 = total_km1 - with_i_km1
    den_km1 = float(comb(n - 1, k - 1))
    adv = (with_i_k - without_i_km1) / den_km1
    uhat_km1 = without_i_km1 / den_km1
    den_k = comb(n - 1, k)
    uhat_k = (total_k - with_i_k) / float(den_k) if den_k > 0 else np.zeros(n, dtype=np.float64)
    return adv, uhat_km1, uhat_k
