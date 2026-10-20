"""Reference baseline: Dr. GRPO with the gold reward (Liu et al., 2025, "Understanding R1-Zero-Like Training").

Reward ``r_i = g_i``. Advantage, within each group of n rollouts: ``A_i = r_i - mean(r)`` (no standard-deviation
normalization). Loss: PPO clipped surrogate (eps = 0.2), no KL term, aggregated as Dr. GRPO (token sum / (number of
sequences * loss_norm_tokens), a protocol constant) -- the same loss form, clipping and KL setting as
``reference_algorithms/sspo``, so the comparison with SSPO differs only in the credit-assignment rule.
"""
from typing import List

import numpy as np

from selectrl.algo_api import Algorithm, Group, StepContext


class DrGRPO(Algorithm):
    name = "drgrpo"
    loss_agg_mode = "seq_mean_token_sum_norm"
    needs_ref_logp = False
    clip_eps_low = 0.2
    clip_eps_high = 0.2
    kl_coef = 0.0

    def compute_advantages(self, groups: List[Group], ctx: StepContext) -> List[np.ndarray]:
        out: List[np.ndarray] = []
        for grp in groups:
            r = np.asarray(grp.g, dtype=np.float64).reshape(-1)
            out.append(r - r.mean() if r.size else np.zeros(0, dtype=np.float64))
        return out


ALGORITHM = DrGRPO()
