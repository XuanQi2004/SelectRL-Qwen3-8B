"""Checker pipeline on three hand-written problems: mock sampler -> gen -> build -> evaluate_program / ExecCache."""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from selectrl.checker import CHECKER_VERSION, ExecCache, build_all, code_hash, last_code_block, load_cases
from selectrl.checker.chk import NOCODE_SIG, evaluate_chk_and_sig, evaluate_program, run_candidate_on_cases
from selectrl.checker.gen import KINDS, MockSampler, generate_artifacts, helper_seed, mock_render_chat
from selectrl.utils import atomic_write_json_gz, read_json_gz

GEN = (
    "import sys, random\n"
    "mode, seed = sys.stdin.read().split()\n"
    "seed = int(seed)\n"
    "random.seed(seed)\n"
    "if mode == 'max':\n"
    "    print(100 + seed, 200 + seed)\n"
    "elif mode == 'edge':\n"
    "    print(seed, seed)\n"
    "else:\n"
    "    print(seed + 2, seed + 5)\n"
)
GEN_BAD_EDGE = GEN.replace("print(seed, seed)", "print(seed, seed, seed) if seed == 2 else print(seed, seed)")
GEN_HUGE = "import sys\nmode, seed = sys.stdin.read().split()\nprint('1 ' * (5 << 20) if mode == 'max' else '3 4')\n"
VAL_GOOD = (
    "import sys\n"
    "toks = sys.stdin.read().split()\n"
    "print('VALID' if len(toks) == 2 and all(t.lstrip('-').isdigit() for t in toks) else 'INVALID')\n"
)
VAL_REJECTS_ALL = "print('INVALID')\n"
BF_BAD = "print(-999999)\n"

ADD = "a, b = map(int, input().split())\nprint(a + b)\n"
MUL = "a, b = map(int, input().split())\nprint(a * b)\n"
DIV = "a, b = map(int, input().split())\nprint(f'{(a + b) / 3.0:.8f}')\n"


def _problems() -> list[dict]:
    specs = [
        ("hw-add", "Compute a + b.", ADD, lambda a, b: f"{a + b}\n"),
        ("hw-mul", "Compute a * b.", MUL, lambda a, b: f"{a * b}\n"),
        ("hw-div", "Compute (a + b) / 3.", DIV, lambda a, b: f"{(a + b) / 3:.6f}\n"),
    ]
    probs = []
    for pid, q, _, f in specs:
        pub = [[f"{a} {b}\n", f(a, b)] for a, b in ((2, 3), (10, 20))]
        hidden = [[f"{a} {b}\n", f(a, b)] for a, b in ((1, 1), (7, 9), (40, 2), (5, 5), (8, 1))]
        probs.append({"pid": pid, "question": q, "time_limit_s": 1.0, "public": pub, "hidden": hidden})
    return probs


def _programs() -> dict:
    table = {}
    for (q, ref), gens in zip((("Compute a + b.", ADD), ("Compute a * b.", MUL), ("Compute (a + b) / 3.", DIV)),
                              ([GEN, GEN, GEN_BAD_EDGE, GEN_HUGE], [GEN] * 4, [GEN] * 4)):
        table[q] = {"gen": gens, "bf": [ref, ref, ref, BF_BAD], "val": [VAL_GOOD, VAL_REJECTS_ALL]}
    return table


@pytest.fixture(scope="module")
def built(tmp_path_factory) -> dict:
    root = tmp_path_factory.mktemp("cache")
    art_dir, cases_dir = root / "checkers" / "artifacts", root / "checkers" / "cases"
    probs = _problems()
    sampler = MockSampler(_programs())
    summary = asyncio.run(generate_artifacts(probs, art_dir, sampler, mock_render_chat, concurrency=2))
    assert summary["generated"] == 3 and not summary["failed"]
    assert len(sampler.calls) == 3 * len(KINDS)
    bsum = build_all(probs, art_dir, cases_dir)
    assert bsum["built"] == 3 and not bsum["failed"]
    return {"root": root, "art_dir": art_dir, "cases_dir": cases_dir, "probs": {p["pid"]: p for p in probs},
            "cases": {p["pid"]: load_cases(cases_dir, p["pid"]) for p in probs}, "sampler": sampler}


def test_generation_is_deterministic_and_resumable(built):
    calls = built["sampler"].calls
    seeds = {(c["question"], c["kind"]): c["seed"] for c in calls}
    assert seeds[("Compute a + b.", "gen")] == helper_seed("hw-add", "gen")
    assert {c["kind"]: c["n"] for c in calls} == KINDS
    assert all(c["max_tokens"] == 8000 for c in calls)
    art = read_json_gz(built["art_dir"] / "hw-add.json.gz")
    assert art["gen"][0] == GEN and len(art["bf"]) == 4 and len(art["val"]) == 2
    again = MockSampler(_programs())
    summary = asyncio.run(generate_artifacts(list(built["probs"].values()), built["art_dir"], again, mock_render_chat))
    assert summary["reused"] == 3 and summary["generated"] == 0 and again.calls == []
    assert build_all(list(built["probs"].values()), built["art_dir"], built["cases_dir"])["reused"] == 3
    stale = dict(built["cases"]["hw-mul"], checker_version="old")
    atomic_write_json_gz(built["cases_dir"] / "hw-mul.json.gz", stale)
    assert load_cases(built["cases_dir"], "hw-mul") is None
    rebuilt = build_all(list(built["probs"].values()), built["art_dir"], built["cases_dir"])
    assert (rebuilt["built"], rebuilt["reused"]) == (1, 2)
    assert load_cases(built["cases_dir"], "hw-mul") == built["cases"]["hw-mul"]


def test_cases_construction(built):
    for pid, c in built["cases"].items():
        assert c["checker_version"] == CHECKER_VERSION
        assert c["n_val_kept"] == 1 and c["val_info"] == ["kept", "rejected_examples"]
        assert c["n_bf_kept"] == 3 and c["bf_info"][-1] == "failed_examples"
        confirmed = [x for x in c["inputs"] if x["ref"] is not None and x["ref_agree"] >= 2]
        assert len(confirmed) >= 5 and all(x["ref_agree"] == 3 and x["n_bf_ok"] == 3 for x in confirmed)
        assert len({x["id"] for x in c["inputs"]}) == len(c["inputs"])  # de-duplicated
        assert {"id", "mode", "gen", "seed", "inp", "valid", "ref", "ref_key", "ref_agree", "n_bf_ok"} <= set(
            c["inputs"][0])
    add = built["cases"]["hw-add"]
    assert add["gen_info"][3] == "8/10"  # the two > 8 MiB max inputs were dropped
    bad = [x for x in add["inputs"] if x["valid"] is False]
    assert [x["inp"] for x in bad] == ["2 2 2\n"] and bad[0]["ref"] is None


def test_chk_conditions(built):
    p, c = built["probs"]["hw-add"], built["cases"]["hw-add"]

    good = evaluate_program(p, c, ADD)
    assert good == {"key": code_hash(ADD), "has_code": 1, "public_pass": 1, "public_frac": 1.0, "chk": 1,
                    "sig": good["sig"], "g": 1, "g_frac": 1.0, "checker_version": CHECKER_VERSION}

    # (1) fails a public example
    fail_pub = "a, b = map(int, input().split())\nprint(0 if (a, b) == (2, 3) else a + b)\n"
    r = evaluate_program(p, c, fail_pub)
    assert (r["public_pass"], r["public_frac"], r["chk"]) == (0, 0.5, 0)

    # (2) crashes or exceeds the CPU limit on a stress input
    crash_max = "import sys\na, b = map(int, input().split())\nif a >= 100:\n    sys.exit(1)\nprint(a + b)\n"
    slow_max = ("import time\na, b = map(int, input().split())\nt = time.process_time()\n"
                "while a >= 100 and time.process_time() - t < 1.3:\n    pass\nprint(a + b)\n")
    for code in (crash_max, slow_max):
        r = evaluate_program(p, c, code)
        assert r["public_pass"] == 1 and r["chk"] == 0 and r["g"] == 1
    rec = run_candidate_on_cases(slow_max, p, c)
    assert [m[1] for m in rec["max"]] == ["timeout"]  # stress inputs stop at the first failure

    # (3) disagrees with a confirmed reference on a small input
    fail_ref = "a, b = map(int, input().split())\nprint(a + b if (a, b) in ((2, 3), (10, 20)) or a >= 100 else a - b)\n"
    r = evaluate_program(p, c, fail_ref)
    assert r["public_pass"] == 1 and r["chk"] == 0

    # hidden label: passes the first two hidden tests, fails the third, stops there
    fail_hidden = "a, b = map(int, input().split())\nprint(a + b + (a == 40))\n"
    r = evaluate_program(p, c, fail_hidden)
    assert (r["chk"], r["g"], r["g_frac"]) == (1, 0, pytest.approx(2 / 3))

    none = evaluate_program(p, c, None)
    assert (none["key"], none["has_code"], none["chk"], none["g"], none["sig"]) == ("nocode", 0, 0, 0, NOCODE_SIG)


def test_signature_normalization(built):
    p, c = built["probs"]["hw-div"], built["cases"]["hw-div"]
    spaced = "a, b = map(int, input().split())\nprint(f'  {(a + b) / 3.0:.9f}   \\n')\n"
    tiny_diff = "a, b = map(int, input().split())\nprint(f'{(a + b) / 3.0 + 1e-9:.7f}')\n"
    off = "a, b = map(int, input().split())\nprint(f'{(a + b) / 3.0 + 1e-2:.6f}')\n"
    r1, r2, r3 = (evaluate_program(p, c, code) for code in (spaced, tiny_diff, off))
    assert r1["chk"] == r2["chk"] == 1 and r3["chk"] == 0
    assert r1["sig"] == r2["sig"] == evaluate_program(p, c, DIV)["sig"]
    assert r3["sig"] != r1["sig"]

    rec = run_candidate_on_cases(spaced, p, c)
    swapped = dict(rec, small=[[o[0], s[1], s[2]] for o, s in zip(rec["small"], reversed(rec["small"]))])
    assert evaluate_chk_and_sig(swapped, c)[1] != evaluate_chk_and_sig(rec, c)[1]  # order matters

    # different programs with identical behaviour share the signature; comments do not change the key
    other = "import sys\nx, y = map(int, sys.stdin.readline().split())\nsys.stdout.write('%.8f\\n' % ((x + y) / 3))\n"
    assert evaluate_program(p, c, other)["sig"] == r1["sig"]
    assert code_hash("# hi\n" + DIV + "\n\n# bye") == code_hash(DIV) != code_hash(MUL)

    # without check inputs the signature falls back to the public output keys
    empty = {"inputs": []}
    assert evaluate_program(p, empty, DIV)["sig"] == evaluate_program(p, empty, spaced)["sig"]


def test_exec_cache(built, tmp_path: Path):
    p, c = built["probs"]["hw-mul"], built["cases"]["hw-mul"]
    cache = ExecCache(tmp_path, CHECKER_VERSION)
    commented = "# a comment\n" + MUL
    items = [(p, c, MUL), (p, c, commented), (p, c, None), (p, c, ADD)]
    recs = cache.evaluate_many(items)
    assert [r["g"] for r in recs] == [1, 1, 0, 0] and recs[0] == recs[1]
    assert (cache.misses, cache.hits) == (2, 2)
    assert (tmp_path / "exec" / CHECKER_VERSION / "hw-mul.json").exists()

    fresh = ExecCache(tmp_path, CHECKER_VERSION)  # persisted: nothing runs again
    assert fresh.evaluate_many(items) == recs and (fresh.misses, fresh.hits) == (0, 4)
    assert fresh.hit_rate == 1.0
    with pytest.raises(ValueError):
        ExecCache(tmp_path, "other-version")


def test_code_extraction():
    resp = "<think>\n```python\nprint('draft')\n```\n</think>\nFirst:\n```python\nprint(1)\n```\nFinal:\n```py\nprint(2)\n```"
    assert last_code_block(resp) == "print(2)\n"
    assert last_code_block("```\nprint(3)\n```") == "print(3)\n"
    assert last_code_block("<think>\n```python\nprint('unfinished')\n```") is None
    assert last_code_block("no code here") is None and last_code_block(None) is None
    assert last_code_block("```python\n\n```") is None
