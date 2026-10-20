# Third-party components

| component | use | license |
|---|---|---|
| CodeContests (Li et al., 2022, DeepMind) | problem statements, public/hidden tests and splits in the released assets are derived from it | CC BY 4.0 |
| `Qwen/Qwen3-8B` | policy model, downloaded from Hugging Face at the pinned revision; not redistributed | Apache-2.0 |
| vLLM 0.29.0, PyTorch 2.13.0, Transformers 5.12.1, PEFT 0.19.1 | inference servers and training | Apache-2.0 / BSD-3-Clause |
| PKPO (Walder & Karkhanis, 2025) | closed form used by `reference_algorithms/passk` | algorithm, reimplemented |
| GRPO (Shao et al., 2024), Dr. GRPO (Liu et al., 2025) | `reference_algorithms/grpo`, `reference_algorithms/drgrpo` | algorithms, reimplemented |

The released LoRA adapters are derivatives of `Qwen/Qwen3-8B` and are distributed under Apache-2.0.
