"""Reference baseline: pass@k training with the PKPO unbiased binary closed form and a leave-one-out baseline.

Ported from SSPO v1 (sspo/advantage.py, arm ``pkpo_pass8``).

Reward ``f_i = g_i``, budget ``k = ctx.k_deploy``. With c correct rollouts among n, the full-subset leave-one-out
advantage of the pass@k score is ``A_i^cov = f_i C(n - c, k - 1) / C(n - 1, k - 1)``; the per-problem scale
``k / n`` that makes ``sum_i (k / n) A_i^cov s_i`` an unbiased estimate of the gradient of pass@k is folded into the
returned advantage, giving the PKPO closed form ``f_i C(n - c, k - 1) / C(n, k)``. The leave-one-out baseline
``b_i = V_i(k) - V_i(k - 1)`` (pass@j U-statistics on the other n - 1 rollouts; independent of rollout i, so the
estimate stays unbiased) is subtracted with the same scale:

    A_i = (k / n) * (A_i^cov - b_i) = f_i C(n - c, k - 1) / C(n, k) - (k / n) b_i.

The baseline needs ``k <= n - 1``; when ``ctx.k_deploy >= n`` the budget ``k = n - 1`` is used. Loss: PPO clipped
surrogate aggregated as Dr. GRPO (token sum / (number of sequences * loss_norm_tokens), a protocol constant).
"""
from typing import List

import numpy as np

from selectrl.algo_api import Algorithm, Group, StepContext
from selectrl.algolib.combin import coverage_loo_advantages


class PassAtK(Algorithm):
    name = "passk"
    loss_agg_mode = "seq_mean_token_sum_norm"
    needs_ref_logp = False
    clip_eps_low = 0.2
    clip_eps_high = 0.2
    kl_coef = 0.0

    def compute_advantages(self, groups: List[Group], ctx: StepContext) -> List[np.ndarray]:
        out: List[np.ndarray] = []
        for grp in groups:
            n = grp.n
            if n < 2:
                out.append(np.zeros(n, dtype=np.float64))
                continue
            k = min(ctx.k_deploy, n - 1)
            adv, base = coverage_loo_advantages(grp.g, k)
            out.append((k / n) * (adv - base))
        return out


ALGORITHM = PassAtK()
