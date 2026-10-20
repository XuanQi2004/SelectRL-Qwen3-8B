# SelectRL-Qwen3-8B architecture

This file is the internal contract between modules. Paths are relative to the repository root.

## 1. Task

Train `Qwen/Qwen3-8B` (thinking off, context window 40,960 tokens) with RL on competitive
programming problems (CodeContests) so that the *deployed system* — sample `k = 8` programs, run the frozen
execution checker, keep programs with `Chk = 1`, vote by output signature, submit one program of the largest
cluster — solves more hidden-test problems. The benchmark metric is **select@8** on a hidden test split.
Algorithms differ only in `algorithms/<name>/algorithm.py` (see `selectrl/algo_api.py`).

## 2. Layout

```
selectrl/                 LOCKED harness (hash-pinned by bench/integrity.py)
  algo_api.py             Algorithm / Group / StepContext / LossInputs contract
  algolib/                pure math helpers algorithms may import (combinatorics, selector, DP, set scores)
    combin.py             exact binomials, pass@k unbiased estimator
    selector.py           classify_group(), selector_success() on a subset, exact_select_at_k()
    sspo_dp.py            COUNTS/TABLE/EXPECT/DIFF/UHAT dynamic programs + brute-force fallback (from sspo v1)
  sandbox/                exec_worker.py (persistent forked sandbox), pool.py (process pool), compare.py
  data/                   codecontests.py (download+convert), filters.py, splits.py
  checker/                gen.py (helper programs via vLLM servers), build.py (cases), chk.py (Chk/sig/g),
                          cache.py (exec cache keyed by (pid, code_hash, checker_version))
  rollout/                servers.py (launch/health/LoRA hot-load of one vLLM server per GPU), client.py
                          (async HTTP sampling with token ids + logprobs), prompts.py (chat prompt, code extraction)
  train/                  main.py (torchrun entry), model.py (causal LM + LoRA, chunked token log-probs),
                          step.py (one RL step), ckpt.py (atomic checkpoints, resume)
  eval/                   run_eval.py (sample n_eval per problem, label), metrics.py (pass@k, exact select@k,
                          bootstrap), report.py (tables, leaderboard)
  bench/                  guard.py (AST policy for algorithm code), integrity.py (sha256 manifest),
                          score.py (official scoring), workspace.py (agent workspace without hidden data)
  config.py               protocol + hardware preset loading
  utils.py                atomic io, jsonl, logging, hashing
reference_algorithms/     LOCKED baselines: grpo/, passk/, sspo/   (each has algorithm.py exporting ALGORITHM)
algorithms/candidate/     the ONLY place an agent edits (starts as a copy of reference grpo)
configs/protocol/         qwen3_8b_40960.yaml (locked benchmark protocol), smoke.yaml (tiny real run)
scripts/                  bash entry points (setup, data, checkers, train, eval, score, bundle)
data/                     problems/{train,val,test}.jsonl.gz, splits/*.txt (+ sha256), filter_stats.json
cache/                    checkers/{artifacts,cases}/<pid>.json.gz, exec/<checker_version>/<pid>.json
runs/<run_name>/          manifest.json, metrics.jsonl, ckpt/, adapters/, eval/, summary.json, done
```

## 3. Data records

Problem (`data/problems/<split>.jsonl.gz`, one JSON per line):

```json
{"pid": "cc-1549_C", "source": "code_contests", "orig_split": "train", "name": "1549_C. Web of Lies",
 "contest_id": "cf_1549", "difficulty": "1600", "time_limit_s": 6.0, "question": "...",
 "public": [["input", "output"], ...], "hidden": [["input", "output"], ...], "py_refs": ["code", ...]}
```

`hidden` holds at most 40 tests (private tests first, then generated tests). `py_refs` holds at most 3 Python 3
reference solutions (used only to seed the set of known-correct signatures; never shown to the policy).

Checker cases (`cache/checkers/cases/<pid>.json.gz`): as in sspo v1 `ec_build.build()`:
`{"pid", "inputs": [{"id", "mode": "max|small|edge", "gen", "seed", "inp", "valid", "ref", "ref_key",
"ref_agree", "n_bf_ok"}], "n_val_kept", "n_bf_kept", ...}`.

Execution record of one program (`selectrl.checker.chk.evaluate_program`):

```json
{"key": "<code_hash>", "has_code": 1, "public_pass": 1, "public_frac": 1.0, "chk": 1, "sig": "<16 hex>",
 "g": 0, "g_frac": 0.35, "checker_version": "ccv1"}
```

## 4. Algorithm contract

See `selectrl/algo_api.py`. `compute_advantages(groups, ctx) -> list[np.ndarray]` (CPU, NumPy) and optional
`policy_loss(LossInputs) -> per-token loss` (GPU). The harness:

* builds `Group` objects only from training problems (no evaluation data ever reaches the algorithm),
* validates outputs (`validate_advantages`), drops groups whose advantages are all zero from the loss but keeps
  the normalizers fixed,
* aggregates per-token losses with `loss_agg_mode` using global normalizers broadcast from rank 0,
* applies the locked optimizer (AdamW, lr, grad-norm clip 1.0) and LoRA shape.

`algorithms/<name>/algorithm.py` must define `ALGORITHM = <AlgorithmSubclass>()`. Imports are restricted by
`bench/guard.py` (numpy, math, torch, scipy.special, standard pure-Python modules, `selectrl.algo_api`,
`selectrl.algolib.*`); no file, network, process or reflection access.

## 5. Process layout of a training run (single node, G GPUs)

```
scripts/train.sh
  ├─ python -m selectrl.rollout.servers launch   (one `vllm serve` per GPU, CUDA_VISIBLE_DEVICES=i,
  │                                               --gpu-memory-utilization from the hardware preset,
  │                                               --enable-lora, --max-model-len 40960, --language-model-only)
  └─ torchrun --nproc_per_node G -m selectrl.train.main --run-dir runs/<name> --algo <dir> --protocol <yaml>
       rank 0: sample problems -> HTTP sampling on all servers -> extract code -> sandbox (Chk, sig, g)
               -> Group list -> algorithm.compute_advantages -> step batch file (atomic)
       all ranks: load their token-balanced shard -> ref/old log-probs (no grad) -> PPO mini-batch updates
               (DDP over LoRA parameters) -> rank 0 saves adapter -> servers hot-load it -> checkpoint
```

`placement: colocate` (default for >= 80 GB GPUs) puts a server and a training rank on every GPU with a static
memory split; `placement: split` dedicates the first half of the GPUs to servers (used for 48 GB validation GPUs).

## 6. Evaluation

For each problem of a split: `n_eval` samples (fixed seeds) from the policy under the protocol's sampling
parameters, execution with the same checker, then per problem: pass@k (unbiased) and **exact** select@k (expected
selector success over all k-subsets of the n_eval samples, computed by `algolib.selector.exact_select_at_k`).
The benchmark score is the mean exact select@8 on the hidden test split of the final checkpoint.

## 7. Locking

`bench/integrity.py` hashes every file under `selectrl/`, `reference_algorithms/`, `configs/protocol/`,
`scripts/`, `data/splits/` and the checker cache index into `bench/LOCK.json`. `bench/score.py` refuses to score
when any locked file differs, when the algorithm fails the guard, or when the run manifest does not match the
protocol hash. Evaluation runs in a fresh process that never imports algorithm code.
