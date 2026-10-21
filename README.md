# SelectRL-Qwen3-8B

**RL algorithm discovery for deployment-time execution selection on code, with Qwen3-8B.**

This is a fixed, reproducible reinforcement-learning harness in which the *RL algorithm* is the only free
variable. `Qwen/Qwen3-8B` (thinking off, 40,960-token context) is fine-tuned with LoRA on CodeContests problems.
The benchmark scores the **deployed system**, not single samples:
1. sample 8 programs;
2. filter them with a frozen execution checker;
3. group the survivors by output signature;
4. submit one program from the largest group.

The score is how often that submission passes the hidden tests (**select@8**).

An algorithm is one Python file that implements `compute_advantages` (and optionally `policy_loss`) against a
small contract (`selectrl/algo_api.py`). Everything else is fixed and hash-locked: model, data, sampling,
optimizer, compute budget, checker, selector and evaluation.

Chinese run guide: [README_zh.md](README_zh.md). Task statement for research agents: [AGENT_TASK.md](AGENT_TASK.md).
Design: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). Validation log: [docs/VALIDATION.md](docs/VALIDATION.md).

## Reference algorithms

| name | idea |
|---|---|
| `sspo` | **SSPO**, selection-aware set policy optimization. Exact leave-one-out credit for the deployed selector's success (J⁻ with a known-correct-signature penalty), plus a coverage term, computed by a dynamic program |
| `drgrpo` | Dr. GRPO on the hidden-test reward (group-mean baseline, no std normalization). Same loss form, clipping and KL as `sspo`, so the two differ only in credit assignment |
| `grpo` | standard GRPO on the hidden-test reward (group mean/std normalization, per-sequence token-mean loss) |
| `passk` | pass@k training with the PKPO unbiased closed form and a leave-one-out baseline |

## Results (seed 0, step 100, 32 samples per problem)

| run | val select@8 | test select@8 | test pass@1 | test pass@8 | test cov@8 |
|---|---:|---:|---:|---:|---:|
| frozen `Qwen/Qwen3-8B` (no training) | 23.55 | 25.86 | 19.99 | 36.81 | 27.54 |
| `sspo` | **32.09** | **32.12** | 25.94 | 44.34 | 34.21 |
| `drgrpo`, `passk`, `grpo` | running under the same protocol and layout | | | | |

- Numbers are in percent.
- Validation has 266 problems and test has 422.
- select@8 is the exact expectation over all 8-subsets of the 32 samples of each problem.
- cov@8 is the probability that an 8-subset contains at least one program that is correct and accepted by the checker.

Per-run summaries are in [results/](results/README.md).

## Quick start (one node, H200 recommended)

```bash
bash scripts/setup_env.sh            # inside a vLLM 0.29 environment (or: --new-venv)
bash scripts/download_model.sh       # Qwen/Qwen3-8B @ pinned revision -> models/ (16 GB)
bash scripts/fetch_assets.sh         # released problems, splits and frozen checker cache (sha256-verified)
bash scripts/smoke.sh                # ~20-30 min: 2 real training steps + resume + tiny eval -> "SMOKE PASS"
bash scripts/train.sh reference_algorithms/sspo --seed 0
bash scripts/collect_results.sh      # results_bundle.tar.gz (small files only)
```

Every script skips finished work. After an interruption, re-run the same command: training resumes from the last
finished step and evaluation from the last finished problem.

**Reference layout.** This is the layout the released runs used: 1 node, 4 × H200, 2 GPUs serving and 2 training.

```bash
SELECTRL_PLACEMENT=split SELECTRL_MEM_UTIL=0.88 SELECTRL_MAX_NUM_SEQS=128 SELECTRL_MICRO_TOKENS=8192 \
  bash scripts/train.sh reference_algorithms/sspo --seed 0 --gpus 0,1,2,3
```

On that layout one run took 12 h 42 min end to end:
- 100 training steps: 9.5 h;
- final evaluation of the trained adapter on val + test: about 3 h;
- startup: the rest.

## Protocol ([configs/protocol/qwen3_8b_40960.yaml](configs/protocol/qwen3_8b_40960.yaml), locked, hash `534c4dc6f841a404`)

| item | value |
|---|---|
| model | `Qwen/Qwen3-8B` @ `b968826`, bf16, `max_model_len` 40,960, thinking off |
| sampling | temperature 1.0, top-p 1.0, no separate generation cap (a response may use the rest of the 40,960-token window); identical for training rollouts and evaluation |
| training | 100 steps × 32 problems × 16 rollouts, 2 PPO mini-batches per step, LoRA r = 64 / alpha = 128 on all attention and MLP projections, AdamW lr 1e-5, grad-norm clip 1.0, data seed 1234 |
| data | CodeContests, contest-disjoint splits: train 2,121 / validation 266 / test 422 problems. The training pool keeps the candidates that the parent harness's base model (`Qwen/Qwen3.5-9B`) solved in 1-7 of 8 samples (`data/active/base_pass_summary.json`). It is a frozen asset |
| checker | per problem: 4 input generators, 4 brute-force solvers and 2 validators, written once by the parent harness's base model (`Qwen/Qwen3.5-9B`, thinking off) and frozen.<br>Chk = public examples pass ∧ stress inputs finish in 6 s CPU ∧ agreement with confirmed brute-force outputs.<br>Signature = hash of normalized outputs on all check inputs |
| metric | exact select@8 over all 8-subsets of 32 samples per problem, for the step-100 adapter. Also reported: pass@k (unbiased), select@k and cov@k for k ∈ {1, 2, 4, 8, 16}, checker pass rate |

## Released assets

The GitHub release [`v1.0.0`](https://github.com/XuanQi2004/SelectRL-Qwen3-8B/releases/tag/v1.0.0) contains:
- `selectrl-data-v1.tar.gz`: `data/problems/{train,val,test}.jsonl.gz`, `data/splits/`, `data/filter_stats.json`, `data/active/base_pass_summary.json`.
- `selectrl-checkers-v1.tar.gz.part000` … `.part007`: the frozen checker cache `cache/checkers/` (2,809 problems, with `MANIFEST.json`), split into 400 MB parts. `fetch_assets.sh` concatenates them before extraction.
- Archive sha256 are in `bench/ASSETS.sha256`, and per-file hashes of the installed files are in `bench/LOCK.json`.
- `reference-<run>.tar.gz`, one archive per reference run, unpacked to `results/baselines/<run>/`:
  - the step-100 LoRA adapter (`ckpt/step_0100/adapter/`, vLLM-loadable);
  - `manifest.json` and `metrics.jsonl`;
  - per-problem evaluation rows and summaries (`eval/{val,test}_final.{jsonl,summary.json}`), with no completions.

  The frozen base archive contains evaluation rows only. The sha256 of each archive is in `bench/REFERENCE.sha256`.

`bash scripts/fetch_assets.sh` installs the data and checker cache. `bash scripts/fetch_reference.sh` installs
the reference runs.

## Layout

```
selectrl/              locked harness (algo_api, algolib, sandbox, data, checker, rollout, train, eval, bench)
reference_algorithms/  sspo, drgrpo, grpo, passk
algorithms/candidate/  the file a research agent edits (starts as GRPO)
configs/protocol/      qwen3_8b_40960.yaml (benchmark), smoke.yaml (installation check)
scripts/               bash entry points
data/, cache/          problems, splits, checker cache (released as an asset)
results/               reference results
docs/                  architecture, validation log, OpenRSI task proposal draft
```

## License and data

Code: Apache-2.0. CodeContests (DeepMind) is CC BY 4.0, and the problem statements and tests in the released
assets are derived from it. Qwen3-8B weights are downloaded from Hugging Face under their own license
(Apache-2.0). See [THIRD_PARTY.md](THIRD_PARTY.md).
