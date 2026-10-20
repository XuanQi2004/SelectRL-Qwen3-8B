"""The deployment selector S, the SSPO class partition of a group, and exact select@k / cov@k.

Ported from SSPO v1 (sspo/selector.py); ``exact_select_at_k`` is new (it replaces the purity-only exact value of
sspo/eval.py and handles impure clusters).

Classes of a group of n samples (SSPO section 3):
  C   = {g = 1, Chk = 1}
  T   = {Chk = 0}
  W*  = {g = 0, Chk = 1, signature already known to be correct}
  W_m = the remaining {g = 0, Chk = 1} samples, partitioned by signature (sorted by size, largest first).

Selector S on a candidate set K: drop Chk = 0 candidates; group the rest by signature; pick one of the largest
clusters uniformly at random; submit a uniformly random member of it. Set scores:
  J_S(K)   = probability that S submits a g = 1 program on K (0 when no candidate passes Chk);
  J^-(K)   = J_S(K) * 1[K contains no W* sample]   (conservative score, SSPO Proposition D);
  J_cov(K) = 1[K contains a C sample].
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Dict, FrozenSet, List, Sequence, Tuple

import numpy as np

from selectrl.algolib.combin import capped_draw_table, comb, pass_at_k

LABEL_C = "C"
LABEL_T = "T"
LABEL_W_STAR = "W*"
LABEL_W = "W"


@dataclass(frozen=True)
class GroupClasses:
    """Partition of the n samples of one group into C, T, W* and the wrong clusters W_1..W_M.

    ``label[i]`` is one of "C", "T", "W*", "W"; ``cluster[i]`` is the index m of W_m for "W" samples and -1
    otherwise. ``impure`` is True when C spans more than one signature or some W_m shares a signature with C; the
    dynamic program of ``sspo_dp`` then does not apply and the full-enumeration fallback is used.
    """

    n: int
    c_idx: Tuple[int, ...]
    t_idx: Tuple[int, ...]
    w_star_idx: Tuple[int, ...]
    w_clusters: Tuple[Tuple[int, ...], ...]
    w_cluster_sig: Tuple[int, ...]
    label: Tuple[str, ...]
    cluster: Tuple[int, ...]
    c_sigs: FrozenSet[int]
    impure: bool
    n_false_reject: int  # samples with Chk = 0 but g = 1 (checker false rejections)

    @property
    def n_C(self) -> int:
        return len(self.c_idx)

    @property
    def n_T(self) -> int:
        return len(self.t_idx)

    @property
    def n_W_star(self) -> int:
        return len(self.w_star_idx)

    @property
    def n_W_clusters(self) -> List[int]:
        return [len(c) for c in self.w_clusters]


def _ints(x: Sequence[int]) -> List[int]:
    return [int(v) for v in np.asarray(x).reshape(-1).tolist()]


def w_star_mask(g: Sequence[int], chk: Sequence[int], sig_known_correct: Sequence[int]) -> List[bool]:
    """``W*`` membership: a wrong program (g = 0) that passes Chk with a signature known to be correct."""
    return [gi == 0 and ci == 1 and ki == 1 for gi, ci, ki in zip(_ints(g), _ints(chk), _ints(sig_known_correct))]


def classify_group(
    g: Sequence[int],
    chk: Sequence[int],
    sig_id: Sequence[int],
    sig_known_correct: Sequence[int],
) -> GroupClasses:
    """Partition a group into C, T, W* and W_1..W_M (wrong clusters sorted by size descending, then signature id)."""
    g_l, chk_l, sig_l = _ints(g), _ints(chk), _ints(sig_id)
    n = len(g_l)
    if not (len(chk_l) == len(sig_l) == n and len(_ints(sig_known_correct)) == n):
        raise ValueError("classify_group: all arrays must have the same length")
    w_star = w_star_mask(g_l, chk_l, sig_known_correct)

    c_idx: List[int] = []
    t_idx: List[int] = []
    w_star_idx: List[int] = []
    w_by_sig: Dict[int, List[int]] = {}
    n_false_reject = 0
    for i in range(n):
        if chk_l[i] != 1:
            t_idx.append(i)
            n_false_reject += 1 if g_l[i] == 1 else 0
        elif g_l[i] == 1:
            c_idx.append(i)
        elif w_star[i]:
            w_star_idx.append(i)
        else:
            w_by_sig.setdefault(sig_l[i], []).append(i)

    ordered = sorted(w_by_sig.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    w_cluster_sig = tuple(sig for sig, _ in ordered)
    w_clusters = tuple(tuple(members) for _, members in ordered)

    label = [LABEL_T] * n
    cluster = [-1] * n
    for i in c_idx:
        label[i] = LABEL_C
    for i in w_star_idx:
        label[i] = LABEL_W_STAR
    for m, members in enumerate(w_clusters):
        for i in members:
            label[i] = LABEL_W
            cluster[i] = m

    c_sigs = frozenset(sig_l[i] for i in c_idx)
    impure = len(c_sigs) > 1 or any(sig in c_sigs for sig in w_cluster_sig)
    return GroupClasses(
        n=n,
        c_idx=tuple(c_idx),
        t_idx=tuple(t_idx),
        w_star_idx=tuple(w_star_idx),
        w_clusters=w_clusters,
        w_cluster_sig=w_cluster_sig,
        label=tuple(label),
        cluster=tuple(cluster),
        c_sigs=c_sigs,
        impure=impure,
        n_false_reject=n_false_reject,
    )


def selector_success(
    subset_indices: Sequence[int],
    g: Sequence[int],
    chk: Sequence[int],
    sig_id: Sequence[int],
) -> float:
    """``J_S(K)``: probability that the deployed selector submits a g = 1 program from the samples ``K``.

    All largest Chk = 1 clusters have the same size v, so with ``a_j`` correct members in winning cluster j the
    probability is ``sum_j a_j / (v * #winners)``. Impure clusters (mixed g within a signature) are allowed.
    """
    clusters: Dict[int, List[int]] = {}
    for i in subset_indices:
        if int(chk[i]) == 1:
            rec = clusters.setdefault(int(sig_id[i]), [0, 0])
            rec[0] += 1
            rec[1] += 1 if int(g[i]) == 1 else 0
    if not clusters:
        return 0.0
    v = max(size for size, _ in clusters.values())
    winners = [correct for size, correct in clusters.values() if size == v]
    return sum(winners) / (v * len(winners))


def j_minus(
    subset_indices: Sequence[int],
    g: Sequence[int],
    chk: Sequence[int],
    sig_id: Sequence[int],
    w_star: Sequence[bool],
) -> float:
    """Conservative score ``J^-(K) = J_S(K) * 1[K contains no W* sample]``."""
    for i in subset_indices:
        if w_star[i]:
            return 0.0
    return selector_success(subset_indices, g, chk, sig_id)


def j_cov(subset_indices: Sequence[int], g: Sequence[int], chk: Sequence[int]) -> float:
    """Checked coverage ``J_cov(K) = 1[K contains a sample with g = 1 and Chk = 1]``."""
    return 1.0 if any(int(g[i]) == 1 and int(chk[i]) == 1 for i in subset_indices) else 0.0


def exact_select_at_k(g: Sequence[int], chk: Sequence[int], sig_id: Sequence[int], k: int) -> float:
    """Exact select@k: the mean of ``J_S(K)`` over all ``C(n, k)`` k-subsets K of the n samples.

    Let the Chk = 1 samples form signature clusters j of size ``n_j`` with ``a_j`` correct members. Given that j is
    the chosen cluster with ``x_j`` members in K, the submitted member is uniform over a uniformly random
    ``x_j``-subset of cluster j, so it is correct with probability ``a_j / n_j``. Hence

        select@k = sum_j (a_j / n_j) * P(cluster j is chosen),

    and P(j chosen) = C(n, k)^-1 * sum_{v >= 1} C(n_j, v) * sum_{s, t} D_v[s][t] * C(f_v, k - v - s) / (1 + t),

    where ``D_v`` counts draws of s items from the other clusters with size >= v that keep every one of them at
    <= v items with exactly t of them tied at v (``capped_draw_table``), and the free pool of size ``f_v`` holds
    the Chk = 0 samples and the other clusters smaller than v (they can never reach v). P(j chosen) depends on j
    only through ``n_j``, so it is computed once per distinct cluster size. All counts are exact integers; the
    result is converted to float at the end. Polynomial in n and k; handles impure clusters.
    """
    g_l, chk_l, sig_l = _ints(g), _ints(chk), _ints(sig_id)
    n = len(g_l)
    if not (1 <= k <= n):
        raise ValueError(f"exact_select_at_k needs 1 <= k <= n, got n={n}, k={k}")
    sizes: Dict[int, int] = {}
    correct: Dict[int, int] = {}
    n_fail = 0
    for gi, ci, si in zip(g_l, chk_l, sig_l):
        if ci == 1:
            sizes[si] = sizes.get(si, 0) + 1
            correct[si] = correct.get(si, 0) + (1 if gi == 1 else 0)
        else:
            n_fail += 1

    # Sum of a_j / n_j over clusters of each size (clusters without correct members contribute nothing).
    ratio_by_size: Dict[int, Fraction] = {}
    for sig, size in sizes.items():
        if correct[sig] > 0:
            ratio_by_size[size] = ratio_by_size.get(size, Fraction(0)) + Fraction(correct[sig], size)

    all_sizes = list(sizes.values())
    total = Fraction(0)
    for size, ratio in ratio_by_size.items():
        others = list(all_sizes)
        others.remove(size)  # drop one cluster of this size: the designated cluster j
        num_by_t: Dict[int, int] = {}
        for v in range(1, min(size, k) + 1):
            capped = [m for m in others if m >= v]
            free = n_fail + sum(m for m in others if m < v)
            max_s = k - v
            table = capped_draw_table(v, capped, max_s)
            base = comb(size, v)
            for s in range(max_s + 1):
                fill = comb(free, max_s - s)
                if fill == 0:
                    continue
                for t, ways in enumerate(table[s]):
                    if ways:
                        num_by_t[t] = num_by_t.get(t, 0) + base * ways * fill
        p_chosen = sum((Fraction(num, 1 + t) for t, num in num_by_t.items()), Fraction(0))
        total += ratio * p_chosen
    return float(total / comb(n, k))


def exact_cov_at_k(g: Sequence[int], chk: Sequence[int], k: int) -> float:
    """Exact cov@k: the mean of ``J_cov(K)`` over all k-subsets, i.e. pass@k on the class C = {g = 1, Chk = 1}."""
    g_l, chk_l = _ints(g), _ints(chk)
    n_c = sum(1 for gi, ci in zip(g_l, chk_l) if gi == 1 and ci == 1)
    return pass_at_k(len(g_l), n_c, k)
