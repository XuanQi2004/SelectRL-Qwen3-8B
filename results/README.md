# Reference results

Protocol `configs/protocol/qwen3_8b_40960.yaml` (hash `534c4dc6f841a404`). Seed 0, step-100 adapter, 32 samples
per problem. Layout: 1 node with 4 × H200 (2 serving, 2 training).

| run | val select@8 | test select@8 | test select@1 | test pass@1 | test pass@8 | test cov@8 | test checker pass rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| `frozen_base` (Qwen3-8B, no training) | 23.55 | 25.86 | 15.77 | 19.99 | 36.81 | 27.54 | 18.88 |
| `sspo_seed0` | 32.09 | 32.12 | 20.08 | 25.94 | 44.34 | 34.21 | 24.45 |
| `drgrpo_seed0` | running | | | | | | |
| `passk_seed0` | running | | | | | | |
| `grpo_seed0` | running | | | | | | |

- Numbers are in percent.
- Validation has 266 problems and test has 422.
- select@k is the exact expectation over all k-subsets of the 32 samples of each problem.
- cov@k is the probability that a k-subset contains at least one program that is correct and accepted by the checker.

## Files

- **`summaries/<run>/{val,test}_final.summary.json`**: aggregate metrics, copied from the run's evaluation summaries.
- **`summaries/sspo_seed0/metrics.jsonl`**: one line per training step.
  - correctness `g_rate`, checker pass rate `chk_rate`;
  - advantage statistics and the SSPO selection and coverage terms;
  - loss and gradient norm;
  - `mismatch_abs_lp`, the log-prob gap between the inference servers and the trainer;
  - timing.
- **`summaries/sspo_seed0/training_curves.png`**: plot of the per-step metrics.
- **`baselines/<run>/`**: installed by `bash scripts/fetch_reference.sh` and not tracked in git. Each run has its adapter, `manifest.json`, `metrics.jsonl` and per-problem evaluation rows.

`python -m selectrl.eval.report --runs "runs/*" "results/baselines/*"` builds the comparison table. It includes
paired, contest-clustered bootstrap intervals.
