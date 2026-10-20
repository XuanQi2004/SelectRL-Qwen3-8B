"""The contract between the locked harness and an RL algorithm (the only thing a benchmark agent may change).

An algorithm is a subclass of :class:`Algorithm` exported as ``ALGORITHM`` from ``<algorithm_dir>/algorithm.py``.
The harness calls, once per training step:

1. ``compute_advantages(groups, ctx)`` on CPU with plain NumPy data (one :class:`Group` per training problem,
   ``n`` rollouts each). It returns one float array of shape ``(n,)`` per group: the sequence-level advantage that
   every response token of that rollout receives.
2. ``policy_loss(inp)`` on GPU for every micro-batch, with token-level tensors (:class:`LossInputs`). It returns a
   per-token loss tensor; the harness aggregates it with ``loss_agg_mode`` using global (all-rank) normalizers, so
   the result does not depend on the number of GPUs.

Everything else (sampling, the number of rollouts, optimizer, learning rate, LoRA shape, number of steps, the
checker, the deployment selector and the evaluation) is fixed by the protocol and cannot be changed by the algorithm.
The algorithm never sees evaluation problems: training groups come from the training split only.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

LOSS_AGG_MODES = (
    "token_mean",  # sum of token losses / total number of response tokens in the mini-batch (DAPO style)
    "seq_mean_token_mean",  # mean over sequences of the per-sequence token mean (original GRPO)
    "seq_mean_token_sum_norm",  # sum of token losses / (number of sequences * train.loss_norm_tokens) (Dr. GRPO:
    #                             a constant normalizer, no per-sequence length normalization)
)


@dataclass(frozen=True)
class Group:
    """Signals of the ``n`` rollouts sampled for one training problem in one step (all arrays have shape ``(n,)``).

    Correctness signals:
      g            1 if the program passes every hidden test of the (training) problem, else 0.
      chk          1 if the frozen execution checker accepts the program (passes all public examples, finishes every
                   stress input within the time limit, matches every confirmed brute-force reference), else 0.
      public_pass  1 if the program passes all public examples of the statement.
      public_frac  fraction of public examples passed.
      has_code     1 if a Python code block could be extracted from the response.
      truncated    1 if generation stopped at the token limit.
    Execution signature (what the deployment selector votes on):
      sig_id       integer id of the program's output signature on all check inputs; equal ids <=> identical
                   normalized outputs. Ids are local to this group (0..num_signatures-1); -1 when there is no code.
      sig_known_correct
                   1 if the signature equals that of a program already known to be correct for this problem
                   (reference solutions and correct programs from earlier steps; fixed before this step's sampling).
    Other:
      response_len number of generated tokens.
    """

    problem_key: str
    g: np.ndarray
    chk: np.ndarray
    public_pass: np.ndarray
    public_frac: np.ndarray
    has_code: np.ndarray
    truncated: np.ndarray
    sig_id: np.ndarray
    sig_known_correct: np.ndarray
    response_len: np.ndarray

    @property
    def n(self) -> int:
        return int(self.g.shape[0])


@dataclass(frozen=True)
class StepContext:
    """Read-only step information."""

    step: int  # 1-based
    total_steps: int
    n_rollouts: int  # rollouts per problem (16 in the standard protocol)
    k_deploy: int  # deployment budget k of the evaluation metric select@k (8 in the standard protocol)
    max_new_tokens: int
    seed: int  # use np.random.default_rng(ctx.seed) for any randomness so runs are reproducible


@dataclass
class LossInputs:
    """Token-level inputs for one micro-batch (T response tokens of S sequences, flattened).

    logp          (T,) log-probabilities of the sampled tokens under the current policy (requires grad).
    old_logp      (T,) log-probabilities under the policy that started this step (no grad). Equal to
                  ``logp.detach()`` for the first optimizer update of a step.
    ref_logp      (T,) log-probabilities under the frozen base model (no grad), or None when the algorithm sets
                  ``needs_ref_logp = False``.
    rollout_logp  (T,) log-probabilities reported by the inference engine at sampling time (no grad).
    advantages    (T,) the sequence advantage broadcast to each token of the sequence.
    seq_index     (T,) long tensor, index of the sequence (0..S-1) each token belongs to.
    token_pos     (T,) long tensor, position of the token inside its response (0-based).
    seq_len       (S,) long tensor, number of response tokens of each sequence.
    """

    logp: Any
    old_logp: Any
    ref_logp: Optional[Any]
    rollout_logp: Any
    advantages: Any
    seq_index: Any
    token_pos: Any
    seq_len: Any


class Algorithm:
    """Base class. Override ``compute_advantages``; optionally override ``policy_loss`` and the class attributes."""

    name: str = "base"
    loss_agg_mode: str = "seq_mean_token_sum_norm"
    needs_ref_logp: bool = False  # set True if policy_loss uses ref_logp (costs one extra forward pass)
    clip_eps_low: float = 0.2
    clip_eps_high: float = 0.2
    kl_coef: float = 0.0  # coefficient of the k3 KL estimator against the base model (requires needs_ref_logp)
    drop_zero_adv_groups: bool = True  # skip groups whose advantages are all zero (no gradient without a KL term)

    def __init__(self) -> None:
        # JSON-serializable state that survives across steps and is saved in checkpoints (e.g. running statistics).
        self.state: Dict[str, Any] = {}

    # ------------------------------------------------------------------ required
    def compute_advantages(self, groups: List[Group], ctx: StepContext) -> List[np.ndarray]:
        raise NotImplementedError

    # ------------------------------------------------------------------ optional
    def policy_loss(self, inp: LossInputs):
        """Default: PPO clipped surrogate on per-token importance ratios, plus optional k3 KL to the base model."""
        import torch

        ratio = torch.exp(inp.logp - inp.old_logp)
        adv = inp.advantages
        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1.0 - self.clip_eps_low, 1.0 + self.clip_eps_high) * adv
        loss = -torch.minimum(surr1, surr2)
        if self.kl_coef > 0.0 and inp.ref_logp is not None:
            log_r = inp.ref_logp - inp.logp
            loss = loss + self.kl_coef * (torch.exp(log_r) - log_r - 1.0)
        return loss

    def metrics(self) -> Dict[str, float]:
        """Optional scalar metrics logged once per step (read after compute_advantages)."""
        return {}


@dataclass
class AdvantageReport:
    """What the harness logs about the advantages an algorithm returned (filled by the harness)."""

    mean: float = 0.0
    std: float = 0.0
    abs_max: float = 0.0
    frac_zero_groups: float = 0.0
    extra: Dict[str, float] = field(default_factory=dict)


MAX_ABS_ADVANTAGE = 1.0e4


def validate_advantages(groups: List[Group], advs: List[np.ndarray]) -> List[np.ndarray]:
    """Check shape and finiteness of an algorithm's output; return float64 copies (raises ValueError)."""
    if not isinstance(advs, (list, tuple)) or len(advs) != len(groups):
        raise ValueError(f"compute_advantages must return a list of {len(groups)} arrays, got {type(advs).__name__}"
                         f" of length {len(advs) if hasattr(advs, '__len__') else '?'}")
    out = []
    for gi, (grp, a) in enumerate(zip(groups, advs)):
        arr = np.asarray(a, dtype=np.float64).reshape(-1).copy()
        if arr.shape[0] != grp.n:
            raise ValueError(f"group {gi}: expected {grp.n} advantages, got {arr.shape[0]}")
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"group {gi}: advantages must be finite")
        if np.max(np.abs(arr), initial=0.0) > MAX_ABS_ADVANTAGE:
            raise ValueError(f"group {gi}: |advantage| exceeds {MAX_ABS_ADVANTAGE}")
        out.append(arr)
    return out
