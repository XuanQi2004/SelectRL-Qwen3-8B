"""Candidate algorithm: the file a benchmark agent edits.

Rules for the agent:
* This file, and new files you add to this directory, are the only things you may edit. Everything else in the
  repository (the harness under selectrl/, reference_algorithms/, configs/protocol/, scripts/, data and caches) is
  locked by a sha256 manifest; a run that changes any of it is not scored.
* The contract is selectrl/algo_api.py: subclass ``Algorithm``, implement ``compute_advantages(groups, ctx)``
  (CPU, NumPy; one float array of shape (n,) per group) and optionally ``policy_loss(inp)`` (GPU, per-token loss),
  set the class attributes (loss_agg_mode, clip_eps_low/high, kl_coef, needs_ref_logp), and export
  ``ALGORITHM = <YourAlgorithm>()`` at module level. Use ``self.state`` (JSON-serializable) for anything that must
  survive across steps and checkpoints, and ``np.random.default_rng(ctx.seed)`` for any randomness.
* Allowed imports: numpy, math, torch, pure standard-library modules (itertools, functools,
  collections, dataclasses, typing, fractions, heapq, bisect, statistics), selectrl.algo_api and
  selectrl.algolib.* (exact combinatorics, the deployment selector, select@k, the SSPO dynamic programs).
  No file, network, process or reflection access (no open, eval, exec, getattr/setattr, and no dunder attribute
  access other than defining __init__); bench/guard.py checks this statically.
* The algorithm only ever sees training problems. Sampling, the number of rollouts, the optimizer, learning rate,
  LoRA shape, number of steps, the checker, the deployment selector and the evaluation are fixed by the protocol.
  The score is the exact select@8 of the final checkpoint on the hidden test split.

The code below starts as an exact copy of reference_algorithms/grpo/algorithm.py (GRPO):

Reference baseline: GRPO with the gold reward.

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
