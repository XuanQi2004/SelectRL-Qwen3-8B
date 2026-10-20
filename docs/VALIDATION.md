# Validation log

## 1. Qwen3-8B reference run (protocol `534c4dc6f841a404`)

Run: SSPO, seed 0, one node with 4 × H200 (2 vLLM servers, 2 training ranks), vLLM 0.29.0, torch 2.13.0,
transformers 5.12.1, peft 0.19.1. These checks were recorded during the released reference run.

| check | result |
|---|---|
| full-context capacity, before training: real model, 40,960-token forward/backward | passed; 16.1 s, peak allocated 42.3 GiB |
| 100 training steps | all finite (loss, gradient norm); 0 CUDA OOM, 0 host OOM |
| inference/training log-prob gap `mismatch_abs_lp`, every step (abort threshold 0.08) | at most 0.015 |
| native resume: step-12 checkpoint restored on CPU and on GPU | all 504 adapter tensors and 1,512 optimizer tensors identical; one further full-context update finite |
| LoRA inference at the full 40,960-token context on a 48 GB L40S | passed (capacity check only, not used for training) |
| checker replay: 256 saved validation samples, 16 / 32 / 64 sandbox workers | `g` and `Chk` identical in every trial; the only signature differences are on `Chk = 0` samples, which the selector discards, so pass/select/cov are unchanged |
| frozen-base evaluation on a second host | problem ids, sample indices, protocol hash and SHA256 of all artifacts verified before import |

## 2. Earlier harness checks (same code, `Qwen/Qwen3.5-9B` protocol of the parent harness)

Environment of every run below: vLLM 0.29.0, torch 2.13.0+cu130, transformers 5.12.1, peft 0.19.1,
flash-linear-attention 0.5.2, Python 3.12, NVIDIA L40S / A40 (46 GB), `Qwen/Qwen3.5-9B` @ `c2022362`.
These are correctness checks of the harness on 48 GB GPUs (split placement: one GPU serves, one trains); they
are not benchmark results. The H200 colocated placement uses the same code path with a different memory split.

### 2.1 Inference server and LoRA serving

| check | result |
|---|---|
| vLLM server, `--max-model-len 262144 --language-model-only --enable-lora --max-lora-rank 64` | starts (2.5-6 min incl. compilation and CUDA graphs); KV cache 18 GiB on a 46 GB GPU = 581,615 tokens, 2.2x concurrency at 262,144 tokens |
| `/v1/completions` with token-id prompts, `return_token_ids`, `logprobs: 0` | returns sampled token ids and their log-probs; 60k-token prompt served |
| HF `Qwen3_5ForCausalLM` (training) vs vLLM (serving), base model, 102-200 sampled tokens | mean abs log-prob difference 0.003-0.009 |
| PEFT adapter saved from the text-only model, loaded with `/v1/load_lora_adapter` | **loaded "successfully" but not applied** (vLLM log-probs equal the base model's) |
| same adapter with keys rewritten to `model.language_model.layers.*` (what `export_for_vllm` does) | applied: vLLM-with-LoRA vs HF-with-LoRA 0.0094, equal to the base-model difference |
| LoRA also on Gated DeltaNet projections (`in_proj_qkv`, `in_proj_z`, `out_proj`) | applied with a larger mismatch (0.016), therefore excluded from the protocol |

### 2.2 The v1 checker-building failure

The v1 helper generator (`LLM(max_model_len=16384, gpu_memory_utilization=0.85, enable_prefix_caching=True,
max_num_seqs=128)`) on vLLM 0.29.0:

* on an empty GPU: starts in 437 s and generates;
* with another process holding 10 GB of the same GPU: `ValueError: Free memory on device cuda:0 (33.55/44.39 GiB)
  on startup is less than desired GPU memory utilization (0.85, 37.73 GiB)` followed by
  `RuntimeError: Engine core initialization failed. See root cause above.`

### 2.3 End-to-end run on 2 x 46 GB GPUs with real CodeContests problems

8 training problems + 8 validation problems, `configs/protocol/smoke.yaml` (4 problems x 4 rollouts per step,
2 steps, 2,048 new tokens), split placement.

| stage | result |
|---|---|
| checker helpers through the vLLM server (16 problems) | 64 generators, 63 brute-force solvers, 32 validators extracted, 4 min |
| checker cases on CPU | 16/16 problems built, 0 failures |
| `scripts/smoke.sh` (GRPO): step 1, forced exit, restart, `resumed after step 1`, step 2, eval | `SMOKE PASS`; step 2: 8 sequences trained, 2 optimizer updates, grad norm 0.47, inference/training log-prob gap 0.0085, clip fraction 0.0008 |
| SSPO and Pass@k, 2 steps each | finish; rollouts identical to the GRPO run (same seed, same base model: sampling is reproducible) |
| base model evaluation (8 val problems x 4 samples) | select@2 0.479, pass@1 0.406 |

### 2.4 LoRA effect during training

Not finished before this release: in the end-to-end run above the first step had no trainable group (all
rollouts of each problem got the same reward), so step 2 sampled from an unchanged adapter. The key-layout fix
itself is verified in section 1, and the trainer enforces it at run time: at step 2 (and the two steps after a
resume) it aborts if the mean |log-prob| gap between the inference servers and the trained policy exceeds 0.08,
and `metrics.jsonl` records `mismatch_abs_lp` (and `shift_vs_base_abs_lp` when reference log-probs are on)
every step.
