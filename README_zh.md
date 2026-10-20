# SelectRL-Qwen3-8B 中文运行说明

这份说明写给在 GPU 机器上跑实验的人，照着做即可。

**实验在做什么。** 用 LoRA 在 CodeContests 上对 `Qwen/Qwen3-8B` 做强化学习。整个训练框架是固定的，唯一能改的是 RL 算法本身。

**评分方式。** 部署时按下面的流程挑出一个程序提交，这个程序通过隐藏测试的概率就是分数，叫 **select@8**：
1. 对每道题采 8 个程序；
2. 用冻结的执行检查器筛掉不合格的；
3. 剩下的按输出签名分组；
4. 从最大的组里选一个提交。

**机器要求。** 单节点，推荐 H200。正式结果用的布局是 4 张 H200：2 张跑推理，2 张跑训练。

## 1. 环境（约 5–15 分钟）

已有 vLLM 0.29 环境时（例如官方镜像 `vllm/vllm-openai:v0.29.0`），在该环境里执行：

```bash
bash scripts/setup_env.sh
```

这个脚本只补装少量包，并用约束文件锁住已装的 torch、vLLM、transformers 版本。没有现成环境时用：

```bash
bash scripts/setup_env.sh --new-venv
```

## 2. 模型（约 16 GB）

```bash
bash scripts/download_model.sh
```

模型下载到 `models/Qwen3-8B`，固定版本 `b968826`。模型是公开的，`HF_TOKEN` 设不设都行。

## 3. 数据和检查器缓存

```bash
bash scripts/fetch_assets.sh
```

脚本从 GitHub Release `v1.0.0` 下载题目、划分和冻结的检查器缓存，并按 `bench/ASSETS.sha256` 和 `bench/LOCK.json` 校验。
- 划分：训练 2,121 题 / 验证 266 题 / 测试 422 题，按比赛划分，互不重叠。
- 这些文件是协议的一部分，**不要重新生成**。`scripts/build_data.sh` 只用来说明它们是怎么建出来的，重新生成会得到不同的文件。

参考结果（adapter 和逐题评测行）用下面的命令下载到 `results/baselines/`：

```bash
bash scripts/fetch_reference.sh
```

## 4. 冒烟测试（约 20–30 分钟，先跑这个）

```bash
bash scripts/smoke.sh
```

跑完会打印 `SMOKE PASS`。

## 5. 正式训练

按参考布局训练一个算法，训练结束后会自动在验证集和测试集上评测：

```bash
SELECTRL_PLACEMENT=split SELECTRL_MEM_UTIL=0.88 SELECTRL_MAX_NUM_SEQS=128 SELECTRL_MICRO_TOKENS=8192 \
  bash scripts/train.sh reference_algorithms/sspo --seed 0 --gpus 0,1,2,3
```

- `reference_algorithms/` 下有四个参考算法：`sspo`、`drgrpo`、`grpo`、`passk`。
- 实测一次运行（参考布局）：100 步训练约 9.5 小时，最终评测验证集和测试集约 3 小时，端到端 12 小时 42 分。
- 冻结基座的评测：`bash scripts/eval.sh base --name frozen_base`。

## 6. 输出在哪里

每个运行的结果都在 `runs/<名字>/` 下：

| 文件 | 内容 |
|---|---|
| `manifest.json` | 协议名与哈希、算法名与 sha256、种子、卡数 |
| `metrics.jsonl` | 每步一行：正确率、检查器通过率、优势统计、损失、梯度范数、推理端与训练端的 log-prob 差、各阶段耗时 |
| `eval/{val,test}_final.jsonl` | 每题一行：pass@k、select@k、cov@k（k=1,2,4,8,16） |
| `eval/{val,test}_final.summary.json` | 汇总，主指标是 `select@8` |
| `ckpt/step_XXXX/adapter/` | 每 10 步和最后一步的 LoRA，vLLM 可以直接加载 |

汇总表：`python -m selectrl.eval.report --runs "runs/*" "results/baselines/*"`，会附上按比赛聚类的配对 bootstrap 区间。

## 7. 中断与续跑

直接重新执行原来的命令：训练从最后完成的一步继续，评测跳过已经完成的题目。

## 8. 常见问题

- **vLLM 起不来**：看 `runs/<名字>/logs/vllm_gpu*.log` 末尾。最常见的原因是同一张卡上有别的进程占着显存。
- **训练端显存不够**：调小 `SELECTRL_MICRO_TOKENS`。它只决定一次前向放几条序列，**不会缩短上下文**，上下文始终是 40,960。
- **`mismatch_abs_lp` 报错退出**：说明 vLLM 没有用上训练出的 LoRA。请把 `logs/` 发回。
