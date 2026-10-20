# Task: design an RL algorithm that makes deployment-time selection succeed more often

You are given a fixed reinforcement-learning harness that fine-tunes `Qwen/Qwen3-8B` (LoRA, thinking off, 40,960-token context) on
competitive-programming problems. Your job is to write the **RL algorithm** — how the rollouts of one step are
turned into a training signal — so that the trained model scores higher on the benchmark metric.

## What is measured

At deployment the model is used as a *system*: for each problem, `k = 8` programs are sampled; a frozen
**execution checker** runs each program on the public examples, on stress inputs and on small inputs with
brute-force reference outputs; programs that fail (`Chk = 0`) are discarded; the survivors are grouped by their
**output signature** (identical normalized outputs on all check inputs); one program of the largest group is
submitted (ties broken uniformly at random). The submission is judged on the hidden tests.

**Score = select@8**: the expected success of this procedure over all 8-subsets of 32 samples per problem,
averaged over the hidden test problems, for the final checkpoint after the fixed number of training steps.
Reported alongside: pass@1, pass@8 (unbiased), select@k for other k, checker pass rate.

Plain pass@k training is not optimal for this metric: the selector can be fooled by a large cluster of wrong
programs that agree with each other and pass the checker, and it can never pick a correct program the checker
rejects. See `reference_algorithms/sspo/` for one algorithm that targets select@k directly (the reference to beat).

## What you may change

Only files in `algorithms/candidate/` (edit `algorithm.py`, add more `.py` files there if you like).
`algorithm.py` must export `ALGORITHM = <subclass of selectrl.algo_api.Algorithm>()`.

The contract (read `selectrl/algo_api.py`, it is short):

* `compute_advantages(groups, ctx) -> list[np.ndarray]` — called once per step on CPU. One `Group` per training
  problem (32 per step), each with `n = 16` rollouts and these per-rollout signals: `g` (passes all hidden tests of
  this *training* problem), `chk` (accepted by the checker), `sig_id` (output-signature cluster id within the
  group), `sig_known_correct` (signature equals a known-correct program's), `public_pass`, `public_frac`,
  `has_code`, `truncated`, `response_len`. Return one advantage per rollout; every token of the rollout gets it.
* `policy_loss(inp) -> per-token loss` — optional, called on GPU for every micro-batch with `logp` (current
  policy, differentiable), `old_logp` (policy at the start of the step), `ref_logp` (base model, only if you set
  `needs_ref_logp = True`), `rollout_logp` (inference engine at sampling time), the broadcast `advantages`,
  `seq_index`, `token_pos`, `seq_len`. The default is the PPO clipped surrogate (+ optional k3 KL).
* Class attributes: `loss_agg_mode` (`token_mean` | `seq_mean_token_mean` | `seq_mean_token_sum_norm`),
  `clip_eps_low/high`, `kl_coef`, `needs_ref_logp`, `drop_zero_adv_groups` (default True: groups whose
  advantages are all zero are skipped).
* `self.state` (JSON-serializable dict) persists across steps and checkpoints; use `np.random.default_rng(ctx.seed)`
  for randomness. `selectrl.algolib` offers exact combinatorics, the deployment selector, exact select@k and the
  SSPO dynamic programs.

Allowed imports: `numpy`, `math`, `torch` (tensor ops), pure standard-library modules (`itertools`,
`functools`, `collections`, `dataclasses`, `typing`, `fractions`, `heapq`, `bisect`, `statistics`, ...),
`selectrl.algo_api`, `selectrl.algolib.*`. No file, process, network or reflection access; `bench/guard.py`
rejects such code (`python -m selectrl.bench.guard algorithms/candidate`).

## What is fixed (do not try to change it; it is hash-locked)

Model, LoRA shape, optimizer (AdamW, lr 1e-5, grad-norm clip 1.0), 100 steps x 32 problems x 16 rollouts,
2 PPO mini-batches per step, sampling (temperature 1, top-p 1, no separate generation cap within the 40,960-token context, thinking off), the prompt, the
checker and its cache, the selector, the data splits and the evaluation (`configs/protocol/qwen3_8b_40960.yaml`,
`selectrl/`, `scripts/`, `data/`, `cache/`). `bash scripts/score.sh` refuses to score a modified harness.

## How to work

```bash
python -m selectrl.bench.guard algorithms/candidate          # static check, seconds
python -m pytest -q tests/                                    # CPU unit tests of the harness and references
bash scripts/smoke.sh                                         # ~20 min real run of a reference (GPU)
bash scripts/train.sh algorithms/candidate --seed 0           # full run (4 x H200, 2 serve + 2 train: ~9.5 h of training), evaluates on val
bash scripts/eval.sh runs/candidate_seed0 --split val --ckpt step_0050   # optional intermediate checkpoints
python -m selectrl.eval.report --runs "runs/*"                # table incl. paired comparisons
```

Baselines on the validation split are in `results/baselines/`. The hidden test split is not in your workspace;
the final score is computed by the maintainers with `bash scripts/score.sh <your algorithm dir>`.

## Rules

1. Do not modify, copy-patch or monkey-patch anything outside `algorithms/candidate/`.
2. Do not try to obtain the hidden test problems or their outputs, and do not hard-code problem-specific
   information; algorithms must work from the per-step signals only.
3. Report what you tried and why; a submission is the directory `algorithms/candidate/` at the end.
