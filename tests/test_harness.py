"""CPU tests of the harness pieces that do not need a GPU: guard, batching, logprob indexing, presets, client parsing."""
from __future__ import annotations

import os
import textwrap
from pathlib import Path

import numpy as np
import pytest
import torch

from selectrl.algo_api import Group, LossInputs, StepContext, validate_advantages
from selectrl.bench.guard import GuardError, check_algorithm_dir, load_algorithm
from selectrl.config import hardware_preset, load_protocol, protocol_hash
from selectrl.rollout.client import parse_choice
from selectrl.train.model import MicroBatch, balance_across_ranks, chunked_token_logprobs, pack_micro_batches

ROOT = Path(__file__).resolve().parent.parent


def _algo_dir(tmp_path: Path, body: str) -> Path:
    d = tmp_path / "algo"
    d.mkdir()
    (d / "algorithm.py").write_text(textwrap.dedent(body))
    return d


GOOD = """
import numpy as np
from selectrl.algo_api import Algorithm

class A(Algorithm):
    name = "t"
    def compute_advantages(self, groups, ctx):
        return [np.asarray(g.g, dtype=float) - np.mean(g.g) for g in groups]

ALGORITHM = A()
"""


def test_guard_accepts_reference_algorithms():
    for d in ["reference_algorithms/grpo", "reference_algorithms/passk", "reference_algorithms/sspo",
              "algorithms/candidate"]:
        errs, digest = check_algorithm_dir(ROOT / d)
        assert errs == [], (d, errs)
        assert len(digest) == 64


@pytest.mark.parametrize("snippet", [
    "import os", "import subprocess", "from pathlib import Path", "x = open('f')", "eval('1')",
    "import numpy as np\nnp.load('x')", "import torch\ntorch.save(1, 'x')", "y = (1).__class__",
    "getattr(object, 'x')", "import socket", "from numpy import load", "import importlib",
    "import torch\ndef f(t):\n    return t.grad_fn.next_functions",
    "import dataclasses\nx = dataclasses.sys.modules", "import operator\nf = operator.attrgetter('load')",
    "from operator import attrgetter", "import torch\ng = torch.compile", "import typing\nos_ = typing.sys",
])
def test_guard_rejects(tmp_path, snippet):
    d = _algo_dir(tmp_path, GOOD + "\n" + snippet + "\n")
    errs, _ = check_algorithm_dir(d)
    assert errs, snippet
    with pytest.raises(GuardError):
        load_algorithm(d)


def test_guard_rejects_data_files(tmp_path):
    d = _algo_dir(tmp_path, GOOD)
    (d / "weights.npy").write_bytes(b"0")
    errs, _ = check_algorithm_dir(d)
    assert any("non-Python" in e for e in errs)


def test_load_algorithm_and_validate(tmp_path):
    algo, info = load_algorithm(_algo_dir(tmp_path, GOOD))
    grp = Group(problem_key="p", g=np.array([1, 0, 0, 1]), chk=np.ones(4, int), public_pass=np.ones(4, int),
                public_frac=np.ones(4), has_code=np.ones(4, int), truncated=np.zeros(4, int),
                sig_id=np.array([0, 1, 1, 0]), sig_known_correct=np.zeros(4, int), response_len=np.full(4, 10))
    ctx = StepContext(step=1, total_steps=2, n_rollouts=4, k_deploy=2, max_new_tokens=16, seed=0)
    advs = validate_advantages([grp], algo.compute_advantages([grp], ctx))
    assert np.allclose(advs[0], [0.5, -0.5, -0.5, 0.5])
    with pytest.raises(ValueError):
        validate_advantages([grp], [np.array([np.nan, 0, 0, 0])])
    with pytest.raises(ValueError):
        validate_advantages([grp], [np.zeros(3)])


def test_default_policy_loss_is_ppo_clip():
    from selectrl.algo_api import Algorithm

    a = Algorithm()
    logp = torch.tensor([-1.0, -2.0, -0.5], requires_grad=True)
    old = torch.tensor([-1.0, -1.0, -1.0])
    adv = torch.tensor([1.0, 1.0, -1.0])
    inp = LossInputs(logp=logp, old_logp=old, ref_logp=None, rollout_logp=old, advantages=adv,
                     seq_index=torch.zeros(3, dtype=torch.long), token_pos=torch.arange(3),
                     seq_len=torch.tensor([3]))
    loss = a.policy_loss(inp)
    ratio = torch.exp(logp.detach() - old)
    expect = -torch.minimum(ratio * adv, torch.clamp(ratio, 0.8, 1.2) * adv)
    assert torch.allclose(loss.detach(), expect)


def test_pack_and_balance():
    lens = [(100, 50), (10, 5), (100, 900), (20, 20), (300, 10)]
    mbs = pack_micro_batches(lens, budget=1100)
    assert sorted(i for mb in mbs for i in mb) == list(range(5))
    for mb in mbs:
        L = max(sum(lens[i]) for i in mb)
        assert len(mb) == 1 or L * len(mb) <= 1100
    shards = balance_across_ranks([5, 1, 9, 3, 7], world=2)
    assert sorted(shards[0] + shards[1]) == list(range(5))
    assert abs(sum([5, 1, 9, 3, 7][i] for i in shards[0]) - sum([5, 1, 9, 3, 7][i] for i in shards[1])) <= 3


def test_microbatch_indices_select_response_positions():
    seqs = [([1, 2, 3], [4, 5]), ([7], [8, 9, 10])]
    b = MicroBatch(seqs, pad_id=0, device=torch.device("cpu"))
    L = b.input_ids.shape[1]
    assert L == 5 and b.num_tokens == 5
    flat = b.input_ids.reshape(-1)
    # the hidden state at pred_index predicts the next token, which must be the target
    assert torch.equal(flat[b.pred_index + 1], b.targets)
    assert b.seq_len.tolist() == [2, 3] and b.token_pos.tolist() == [0, 1, 0, 1, 2]


def test_chunked_logprobs_match_full_softmax():
    torch.manual_seed(0)
    h = torch.randn(37, 8, requires_grad=True)
    w = torch.randn(50, 8)
    y = torch.randint(0, 50, (37,))
    lp = chunked_token_logprobs(h, y, w, grad=True, chunk=10)
    ref = torch.log_softmax(h @ w.t(), -1).gather(1, y[:, None])[:, 0]
    assert torch.allclose(lp, ref, atol=1e-5)
    lp.sum().backward()
    h2 = h.detach().clone().requires_grad_(True)
    torch.log_softmax(h2 @ w.t(), -1).gather(1, y[:, None])[:, 0].sum().backward()
    assert torch.allclose(h.grad, h2.grad, atol=1e-5)


def test_protocol_loads_and_hash_is_stable():
    std = load_protocol("configs/protocol/qwen3_8b_40960.yaml")
    smoke = load_protocol("configs/protocol/smoke.yaml")
    assert std["model"]["hf_id"] == "Qwen/Qwen3-8B" and std["model"]["max_model_len"] == 40960
    assert protocol_hash(std) == "534c4dc6f841a404"  # hash recorded by the released reference runs
    assert smoke["model"] == std["model"] and smoke["train"]["steps"] == 2 and smoke["train"]["lr"] == std["train"]["lr"]
    assert protocol_hash(std) == protocol_hash(load_protocol("configs/protocol/qwen3_8b_40960.yaml"))
    assert protocol_hash(std) != protocol_hash(smoke)


@pytest.mark.parametrize("mem,n,placement", [(141, 8, "colocate"), (80, 4, "colocate"), (141, 1, "colocate"),
                                             (48, 2, "split"), (48, 4, "split")])
def test_hardware_presets(mem, n, placement):
    p = hardware_preset([str(i) for i in range(n)], mem_gb=mem)
    assert p["placement"] == placement
    assert set(p["server_gpus"]) | set(p["train_gpus"]) == {str(i) for i in range(n)}
    if placement == "split":
        assert not set(p["server_gpus"]) & set(p["train_gpus"])
    assert 0.3 <= p["mem_util"] <= 0.9 and p["micro_tokens"] >= 4096


def test_hardware_preset_rejects_single_small_gpu():
    with pytest.raises(RuntimeError):
        hardware_preset(["0"], mem_gb=48)


def test_parse_choice_token_ids_and_logprobs():
    c = {"text": "hi", "token_ids": [5, 6], "logprobs": {"token_logprobs": [-0.1, None]}, "finish_reason": "stop"}
    out = parse_choice(c)
    assert out["token_ids"] == [5, 6] and out["logprobs"] == [-0.1, 0.0] and out["finish_reason"] == "stop"
    c2 = {"text": "x", "logprobs": {"tokens": ["token_id:7"], "token_logprobs": [-1.0]}}
    assert parse_choice(c2)["token_ids"] == [7]
    with pytest.raises(RuntimeError):
        parse_choice({"text": "x", "logprobs": {}})
