"""Reference baseline: GRPO with the gold reward.

Reward ``r_i = g_i`` (1 if the program passes every hidden test of the training problem). Advantage, within each
group of n rollouts: ``A_i = (r_i - mean(r)) / (std(r) + 1e-6)`` with the population standard deviation, so a
group whose rewards are all equal gets all-zero advantages. Loss: PPO clipped surrogate (eps = 0.2), no KL term,
aggregated as the mean over sequences of the per-sequence token mean (original GRPO).
"""
from typing import List

import numpy as np

from selectrl.algo_api import Algorithm, Group, StepContext


class GRPO(Algorithm):
    name = "grpo"
    loss_agg_mode = "seq_mean_token_mean"
    needs_ref_logp = False
    clip_eps_low = 0.2
    clip_eps_high = 0.2
    kl_coef = 0.0
    std_eps = 1.0e-6

    def compute_advantages(self, groups: List[Group], ctx: StepContext) -> List[np.ndarray]:
        out: List[np.ndarray] = []
        for grp in groups:
            r = np.asarray(grp.g, dtype=np.float64).reshape(-1)
            if r.size == 0:
                out.append(np.zeros(0, dtype=np.float64))
                continue
            out.append((r - r.mean()) / (r.std() + self.std_eps))
        return out


ALGORITHM = GRPO()
