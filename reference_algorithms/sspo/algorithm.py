"""Reference method: SSPO (selection-aware set policy optimization) for the deployed execution selector.

Ported from SSPO v1 (sspo/advantage.py, ``compute_sspo_advantages`` with ``center_select = "none"``).

Objective per problem (SSPO section 4): ``J = (1 - alpha) J_cov^(k) + alpha J^-^(k)`` with ``alpha = 0.7`` and
``k = ctx.k_deploy``, where ``J_cov(K) = 1[K contains a C = {g = 1, Chk = 1} rollout]`` and
``J^-(K) = J_S(K) * 1[K contains no W* rollout]`` is the deployed selector's success with the wrong programs that
carry a known-correct signature (W*, from ``sig_known_correct``) counted as failures.

Advantage of rollout i (Theorem 2):

    A_i = (k / n) * [ (1 - alpha) (A_i^cov - b_i^cov) + alpha A_i^(S,U) ]

* ``A_i^cov - b_i^cov``: full-subset leave-one-out coverage advantage on C minus its leave-one-out baseline.
* ``A_i^(S,U)``: uncentered full-subset leave-one-out advantage of ``J^-``, by the exact dynamic program when the
  group is cluster-pure, else by enumerating all subsets. It is 0 for Chk = 0 rollouts, <= 0 for every g = 0
  rollout and >= 0 for C rollouts under purity (Theorem 2 (d1)); it is left uncentered because a leave-one-out
  baseline would hand positive advantages to rollouts that fail the checker.
* ``k / n``: the per-problem factor of the unbiased estimator ``sum_i (k / n) A_i s_i`` of the objective's
  gradient, folded into the advantages because the harness applies no per-group loss scale.

The coverage baseline needs ``k <= n - 1``; when ``ctx.k_deploy >= n`` the budget ``k = n - 1`` is used. Loss: PPO
clipped surrogate aggregated as Dr. GRPO (token sum / (number of sequences * loss_norm_tokens), a protocol constant).
"""
from typing import Dict, List

import numpy as np

from selectrl.algo_api import Algorithm, Group, StepContext
from selectrl.algolib.combin import coverage_loo_advantages
from selectrl.algolib.selector import classify_group
from selectrl.algolib.sspo_dp import brute_force_selection_advantages, dp_selection_advantages


class SSPO(Algorithm):
    name = "sspo"
    loss_agg_mode = "seq_mean_token_sum_norm"
    needs_ref_logp = False
    clip_eps_low = 0.2
    clip_eps_high = 0.2
    kl_coef = 0.0
    alpha = 0.7
    last_metrics: Dict[str, float] = {}  # replaced (never mutated) by every compute_advantages call

    def compute_advantages(self, groups: List[Group], ctx: StepContext) -> List[np.ndarray]:
        out: List[np.ndarray] = []
        n_impure = 0
        sel_terms: List[np.ndarray] = []
        cov_terms: List[np.ndarray] = []
        for grp in groups:
            n = grp.n
            if n < 2:
                out.append(np.zeros(n, dtype=np.float64))
                continue
            k = min(ctx.k_deploy, n - 1)
            classes = classify_group(grp.g, grp.chk, grp.sig_id, grp.sig_known_correct)
            if classes.impure:
                n_impure += 1
                adv_sel, _, _ = brute_force_selection_advantages(
                    grp.g, grp.chk, grp.sig_id, grp.sig_known_correct, k)
            else:
                adv_sel = dp_selection_advantages(classes, k)
            in_c = np.zeros(n, dtype=np.int64)
            in_c[list(classes.c_idx)] = 1
            adv_cov, base_cov = coverage_loo_advantages(in_c, k)
            cov = adv_cov - base_cov
            out.append((k / n) * ((1.0 - self.alpha) * cov + self.alpha * adv_sel))
            sel_terms.append(adv_sel)
            cov_terms.append(cov)

        self.last_metrics = {
            "frac_impure_groups": n_impure / len(groups) if groups else 0.0,
            "mean_selection_term": float(np.mean(np.concatenate(sel_terms))) if sel_terms else 0.0,
            "mean_coverage_term": float(np.mean(np.concatenate(cov_terms))) if cov_terms else 0.0,
        }
        return out

    def metrics(self) -> Dict[str, float]:
        """Metrics of the last ``compute_advantages`` call: fraction of groups that used the brute-force fallback
        (impure clusters), and the per-rollout means of the uncentered selection term ``A^(S,U)`` and of the
        centered coverage term ``A^cov - b^cov`` (both before the alpha weights and the k / n scale)."""
        return dict(self.last_metrics)


ALGORITHM = SSPO()
