"""Reference algorithms (GRPO, pass@k, SSPO) and the candidate starting point.

Ported in part from SSPO v1 (tests/test_02_theorem2_properties.py, tests/test_03_unbiased_mc.py).
"""
from __future__ import annotations

import ast
import importlib.util
import itertools
import math
import random
from pathlib import Path
from types import ModuleType
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
import pytest

from selectrl.algo_api import LOSS_AGG_MODES, Group, StepContext, validate_advantages
from selectrl.algolib.selector import classify_group, j_cov
from selectrl.algolib.sspo_dp import dp_selection_advantages

ROOT = Path(__file__).resolve().parents[1]
ALGO_FILES = {
    "grpo": ROOT / "reference_algorithms" / "grpo" / "algorithm.py",
    "passk": ROOT / "reference_algorithms" / "passk" / "algorithm.py",
    "sspo": ROOT / "reference_algorithms" / "sspo" / "algorithm.py",
    "candidate": ROOT / "algorithms" / "candidate" / "algorithm.py",
}


def load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(f"algo_under_test_{name}", ALGO_FILES[name])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_group(g: Sequence[int], chk: Sequence[int], sig: Sequence[int], known: Sequence[int],
               key: str = "p") -> Group:
    n = len(g)
    as_int = lambda x: np.asarray(x, dtype=np.int64)  # noqa: E731
    return Group(problem_key=key, g=as_int(g), chk=as_int(chk), public_pass=as_int(chk),
                 public_frac=np.asarray(chk, dtype=np.float64), has_code=np.ones(n, dtype=np.int64),
                 truncated=np.zeros(n, dtype=np.int64), sig_id=as_int(sig), sig_known_correct=as_int(known),
                 response_len=np.full(n, 100, dtype=np.int64))


def ctx(n: int = 16, k: int = 8) -> StepContext:
    return StepContext(step=1, total_steps=10, n_rollouts=n, k_deploy=k, max_new_tokens=1024, seed=0)


def random_group(n: int, rng: random.Random, pure: bool = True, key: str = "p") -> Group:
    """Random group; when ``pure`` the correct rollouts share signature 0 and wrong clusters use other ids."""
    rows = []
    for _ in range(n):
        u = rng.random()
        if u < 0.3:
            rows.append((1, 1, 0 if pure else rng.randint(0, 2), int(rng.random() < 0.5)))
        elif u < 0.4:
            rows.append((0, 1, 0, 1))  # W*: wrong program with the known-correct signature
        elif u < 0.75:
            rows.append((0, 1, rng.randint(1, 3), 0))
        else:
            rows.append((int(rng.random() < 0.2), 0, -1, 0))
    g, chk, sig, known = zip(*rows)
    return make_group(g, chk, sig, known, key)


@pytest.mark.parametrize("name", list(ALGO_FILES))
def test_shapes_finiteness_and_determinism(name: str):
    algo = load(name).ALGORITHM
    assert algo.loss_agg_mode in LOSS_AGG_MODES
    rng = random.Random(1)
    groups = [random_group(16, rng, pure=(j % 3 != 0), key=f"p{j}") for j in range(12)]
    groups += [make_group([0] * 16, [0] * 16, [-1] * 16, [0] * 16), make_group([1] * 16, [1] * 16, [0] * 16, [1] * 16)]
    first = validate_advantages(groups, algo.compute_advantages(groups, ctx()))
    second = validate_advantages(groups, algo.compute_advantages(groups, ctx()))
    for a, b in zip(first, second):
        assert a.shape == (16,) and np.all(np.isfinite(a))
        np.testing.assert_array_equal(a, b)
    for metric in algo.metrics().values():
        assert math.isfinite(metric)


@pytest.mark.parametrize("name", ["passk", "sspo"])
def test_budget_is_clamped_below_group_size(name: str):
    """With k_deploy >= n the leave-one-out baseline is undefined; the algorithms use k = n - 1."""
    algo = load(name).ALGORITHM
    grp = make_group([1, 0, 0, 0], [1, 1, 1, 0], [0, 1, 1, -1], [0] * 4)
    np.testing.assert_array_equal(algo.compute_advantages([grp], ctx(4, 8))[0],
                                  algo.compute_advantages([grp], ctx(4, 3))[0])


def test_grpo_formula_and_zero_variance_groups():
    algo = load("grpo").ALGORITHM
    assert (algo.loss_agg_mode, algo.clip_eps_low, algo.clip_eps_high, algo.kl_coef, algo.needs_ref_logp) == \
        ("seq_mean_token_mean", 0.2, 0.2, 0.0, False)
    g = [1, 0, 0, 1, 0, 0, 0, 0]
    groups = [make_group(g, [1] * 8, [0] * 8, [0] * 8),
              make_group([0] * 8, [1] * 8, [0] * 8, [0] * 8),
              make_group([1] * 8, [1] * 8, [0] * 8, [0] * 8)]
    adv = algo.compute_advantages(groups, ctx(8, 4))
    r = np.asarray(g, dtype=np.float64)
    np.testing.assert_allclose(adv[0], (r - r.mean()) / (r.std() + 1e-6), rtol=0, atol=1e-15)
    assert np.all(adv[1] == 0.0) and np.all(adv[2] == 0.0)


def test_candidate_is_a_copy_of_grpo():
    def body(path: Path) -> str:
        tree = ast.parse(path.read_text())
        tree.body = tree.body[1:]  # drop the module docstring
        return ast.dump(tree)

    assert body(ALGO_FILES["candidate"]) == body(ALGO_FILES["grpo"])


def test_passk_equals_pkpo_closed_form_minus_loo_baseline():
    algo = load("passk").ALGORITHM
    assert algo.loss_agg_mode == "seq_mean_token_sum_norm"
    rng = random.Random(3)
    for _ in range(50):
        n = rng.choice([4, 8, 16])
        k = rng.randint(1, n - 1)
        g = [int(rng.random() < rng.random()) for _ in range(n)]
        adv = algo.compute_advantages([make_group(g, [1] * n, [0] * n, [0] * n)], ctx(n, k))[0]
        c = sum(g)
        for i in range(n):
            closed_form = g[i] * math.comb(n - c, k - 1) / math.comb(n, k)
            c_other = c - g[i]
            v = lambda j: 1 - math.comb(n - 1 - c_other, j) / math.comb(n - 1, j)  # noqa: E731
            assert abs(adv[i] - (closed_form - (k / n) * (v(k) - v(k - 1)))) <= 1e-12
    # explicit leave-one-out enumeration on a small group
    n, k, g = 8, 3, [1, 0, 0, 1, 0, 0, 0, 0]
    adv = algo.compute_advantages([make_group(g, [1] * n, [0] * n, [0] * n)], ctx(n, k))[0]
    ones = [1] * n
    for i in range(n):
        others = [j for j in range(n) if j != i]
        sub_km1 = list(itertools.combinations(others, k - 1))
        a_i = np.mean([j_cov((*s, i), g, ones) - j_cov(s, g, ones) for s in sub_km1])
        b_i = np.mean([j_cov(s, g, ones) for s in itertools.combinations(others, k)]) - \
            np.mean([j_cov(s, g, ones) for s in sub_km1])
        assert abs(adv[i] - (k / n) * (a_i - b_i)) <= 1e-12


def test_sspo_theorem2_properties():
    module = load("sspo")
    algo = module.ALGORITHM
    assert (algo.alpha, algo.loss_agg_mode) == (0.7, "seq_mean_token_sum_norm")
    rng = random.Random(42)
    for trial in range(150):
        n = rng.choice([8, 16])
        k = rng.randint(1, n - 1)
        grp = random_group(n, rng, pure=True)
        classes = classify_group(grp.g, grp.chk, grp.sig_id, grp.sig_known_correct)
        assert not classes.impure
        adv = algo.compute_advantages([grp], ctx(n, k))[0]
        sel = dp_selection_advantages(classes, k)
        cov_only = load("passk").ALGORITHM.compute_advantages(
            [make_group([int(lab == "C") for lab in classes.label], [1] * n, [0] * n, [0] * n)], ctx(n, k))[0]
        # total = (1 - alpha) * (k/n)(A^cov - b^cov) + alpha * (k/n) A^S
        np.testing.assert_allclose(adv, (1 - algo.alpha) * cov_only + algo.alpha * (k / n) * sel, atol=1e-12)
        for i in range(n):
            if classes.label[i] == "T":
                assert sel[i] == 0.0
            if classes.label[i] == "C":
                assert sel[i] >= -1e-12 and adv[i] >= -1e-12
            if int(grp.g[i]) == 0:
                assert sel[i] <= 1e-12 and adv[i] <= 1e-12, (trial, i, classes.label[i])


def test_sspo_impure_groups_use_fallback_and_metrics():
    algo = load("sspo").ALGORITHM
    pure = make_group([1, 1, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 0, 0, 0], [0, 0, 1, 1, 2, -1, -1, -1], [0] * 8)
    impure = make_group([1, 1, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 0, 0, 0], [0, 5, 1, 1, 2, -1, -1, -1], [0] * 8)
    adv = algo.compute_advantages([pure, impure], ctx(8, 4))
    m = algo.metrics()
    assert m["frac_impure_groups"] == 0.5
    assert set(m) == {"frac_impure_groups", "mean_selection_term", "mean_coverage_term"}
    assert all(np.all(np.isfinite(a)) for a in adv)


# --------------------------------------------------------------------------- unbiasedness on a tabular policy
# Classes of the tabular policy: 0 = C (g=1, Chk=1, sig 0 known-correct), 1 = W* (g=0, Chk=1, sig 0 known-correct),
# 2 = W_1 (g=0, Chk=1, sig 1), 3 = T (g=0, Chk=0).
CLASS_ROWS = [(1, 1, 0, 1), (0, 1, 0, 1), (0, 1, 1, 0), (0, 0, -1, 0)]


def softmax(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - z.max())
    return e / e.sum()


def objective(logits: np.ndarray, k: int, alpha: float) -> float:
    """``(1 - alpha) J_cov^(k) + alpha J^-^(k)`` for k i.i.d. draws from the tabular policy."""
    p_c, _, p_w, p_t = softmax(logits)
    j_minus = 0.0
    for c in range(1, k + 1):
        for w in range(0, k - c + 1):
            prob = math.comb(k, c) * math.comb(k - c, w) * p_c ** c * p_w ** w * p_t ** (k - c - w)
            j_minus += prob * (1.0 if c > w else 0.5 if c == w else 0.0)
    return (1 - alpha) * (1 - (1 - p_c) ** k) + alpha * j_minus


def class_advantages(algo, counts: Tuple[int, ...], n: int, k: int) -> Dict[int, float]:
    rows = [CLASS_ROWS[x] for x, cnt in enumerate(counts) for _ in range(cnt)]
    g, chk, sig, known = zip(*rows)
    adv = algo.compute_advantages([make_group(g, chk, sig, known)], ctx(n, k))[0]
    out, start = {}, 0
    for x, cnt in enumerate(counts):
        if cnt:
            out[x] = float(adv[start])
            start += cnt
    return out


def gradient_estimate(adv_by_class: Dict[int, float], counts: Tuple[int, ...], probs: np.ndarray) -> np.ndarray:
    """``sum_i A_i s_i`` with the score ``s(x) = e_x - probs`` of the softmax policy."""
    scores = np.eye(len(probs)) - probs[None, :]
    return sum(cnt * adv_by_class[x] * scores[x] for x, cnt in enumerate(counts) if cnt)


def compositions(n: int, parts: int):
    for cut in itertools.combinations(range(n + parts - 1), parts - 1):
        bounds = (-1,) + cut + (n + parts - 1,)
        yield tuple(bounds[j + 1] - bounds[j] - 1 for j in range(parts))


def finite_difference(f: Callable[[np.ndarray], float], x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    grad = np.zeros_like(x)
    for d in range(len(x)):
        step = np.zeros_like(x)
        step[d] = eps
        grad[d] = (f(x + step) - f(x - step)) / (2 * eps)
    return grad


def test_sspo_gradient_estimator_is_unbiased():
    """E[sum_i A_i s_i] equals the gradient of (1 - alpha) J_cov^(k) + alpha J^-^(k) (SSPO Theorem 2 (a))."""
    algo = load("sspo").ALGORITHM
    logits = np.array([-0.2, -1.0, 0.4, -0.1])
    probs = softmax(logits)
    n, k = 8, 4
    target = finite_difference(lambda z: objective(z, k, algo.alpha), logits)

    cache = {counts: class_advantages(algo, counts, n, k) for counts in compositions(n, 4)}
    exact = np.zeros(4)
    for counts, adv in cache.items():
        prob = math.factorial(n) / math.prod(math.factorial(c) for c in counts) * math.prod(
            p ** c for p, c in zip(probs, counts))
        exact += prob * gradient_estimate(adv, counts, probs)
    np.testing.assert_allclose(exact, target, atol=1e-8)

    # Monte Carlo over independent groups (small), within sampling noise
    rng = np.random.default_rng(20261015)
    draws = rng.choice(4, size=(3000, n), p=probs)
    est = np.array([gradient_estimate(cache[c], c, probs)
                    for c in (tuple(np.bincount(row, minlength=4).tolist()) for row in draws)])
    se = est.std(axis=0, ddof=1) / np.sqrt(len(est))
    assert np.all(np.abs(est.mean(axis=0) - target) <= 4.0 * se + 1e-6)


def test_passk_gradient_estimator_is_unbiased():
    """E[sum_i A_i s_i] equals the gradient of pass@k = 1 - (1 - p)^k for a two-class policy."""
    algo = load("passk").ALGORITHM
    logits = np.array([-0.5, 0.3])
    probs = softmax(logits)
    n, k = 16, 8
    target = finite_difference(lambda z: 1 - (1 - softmax(z)[0]) ** k, logits)
    exact = np.zeros(2)
    scores = np.eye(2) - probs[None, :]
    for c in range(n + 1):
        g = [1] * c + [0] * (n - c)
        adv = algo.compute_advantages([make_group(g, [1] * n, [0] * n, [0] * n)], ctx(n, k))[0]
        grad = sum(adv[i] * scores[g[i] ^ 1] for i in range(n))
        exact += math.comb(n, c) * probs[0] ** c * probs[1] ** (n - c) * grad
    np.testing.assert_allclose(exact, target, atol=1e-8)
