"""CodeContests conversion, unique-answer filter and contest-disjoint splits on synthetic rows (no network)."""
from __future__ import annotations

import random
from pathlib import Path

from selectrl.data.codecontests import MAX_HIDDEN, MAX_TEST_BYTES, convert_row, make_contest_id, make_pid, prepare
from selectrl.data.filters import filter_problems, filter_reason
from selectrl.data.splits import assign_splits, check_contest_disjoint, contest_rank
from selectrl.utils import read_json, read_jsonl_gz


def _tests(n: int, tag: str) -> dict:
    return {"input": [f"{tag} {i}\n" for i in range(n)], "output": [f"{tag}{i}\n" for i in range(n)]}


def row(name: str, cf_contest_id: int = 0, source: int = 2, description: str = "Compute the value.",
        n_public: int = 1, n_private: int = 3, n_generated: int = 5, **extra) -> dict:
    """A CodeContests-like row."""
    r = {
        "name": name, "description": description, "source": source, "difficulty": 7, "cf_contest_id": cf_contest_id,
        "cf_rating": 1600, "public_tests": _tests(n_public, "pub"), "private_tests": _tests(n_private, "priv"),
        "generated_tests": _tests(n_generated, "gen"),
        "solutions": {"language": [3, 2, 1, 3], "solution": ["print(1)", "int main(){}", "print 1", "print(2)"]},
    }
    r.update(extra)
    return r


def test_convert_row():
    big = "x" * (MAX_TEST_BYTES + 1)
    gen = _tests(30, "gen")
    gen["input"][0] = big
    r = row("1549_C. Web of Lies", 1549, n_private=30, generated_tests=gen,
            solutions={"language": [3] * 5 + [2], "solution": [f"print({i})" for i in range(6)]})
    p, skipped = convert_row(r, "train")
    assert skipped == 1
    assert p["pid"] == "cc-1549_C" and p["contest_id"] == "cf_1549" and p["orig_split"] == "train"
    assert p["time_limit_s"] == 6.0 and p["difficulty"] == "1600" and p["source"] == "code_contests"
    assert len(p["hidden"]) == MAX_HIDDEN
    assert p["hidden"][:30] == [[f"priv {i}\n", f"priv{i}\n"] for i in range(30)]  # private tests first
    assert p["hidden"][30] == ["gen 1\n", "gen1\n"]  # the oversized generated test was skipped
    assert p["py_refs"] == ["print(0)", "print(1)", "print(2)"]
    assert p["public"] == [["pub 0\n", "pub0\n"]]
    assert set(p) >= {"pid", "source", "orig_split", "name", "contest_id", "difficulty", "time_limit_s", "question",
                      "public", "hidden", "py_refs"}
    assert convert_row(row("1_A. X"), "train")[0]["py_refs"] == ["print(1)", "print(2)"]  # language 3 only


def test_ids():
    assert make_contest_id(row("1549_C. Web of Lies", 0)) == "cf_1549"  # Codeforces name fallback
    atc = row("p02543 AtCoder Library Practice Contest - Disjoint Set Union", source=5)
    assert make_contest_id(atc) == "atcoder_atcoder_library_practice_contest"
    assert make_contest_id(row("p02544 AtCoder Library Practice Contest - Fenwick Tree", source=5)) == \
        make_contest_id(atc)
    assert make_pid(atc["name"]) == "cc-p02543"
    slug = make_pid("chef and strings")
    assert slug.startswith("cc-chef_and_strings-") and slug != make_pid("chef and strings 2")
    assert make_contest_id(row("brcktsrm", source=1)) == "codechef_brcktsrm"


def test_filters():
    def prob(**kw) -> dict:
        p = {"pid": kw.pop("pid", "x"), "question": "Find the sum.", "public": [["1\n", "1\n"]],
             "hidden": [[f"{i}\n", f"{i}\n"] for i in range(2, 8)]}
        p.update(kw)
        return p

    assert filter_reason(prob()) == "kept"
    assert filter_reason(prob(question="If there are multiple answers, print any of them.")) == "multi_answer_statement"
    assert filter_reason(prob(question="This is an interactive problem.")) == "multi_answer_statement"
    assert filter_reason(prob(question="Print the elements in any order.")) == "multi_answer_statement"
    assert filter_reason(prob(question=" ")) == "empty_statement"
    assert filter_reason(prob(public=[])) == "insufficient_tests"
    assert filter_reason(prob(hidden=[["2\n", "2\n"]] * 4)) == "insufficient_tests"
    assert filter_reason(prob(hidden=[["2\n", "2\n"]] * 5 + [["3\n", " \n"]])) == "insufficient_tests"
    inconsistent = prob(hidden=[[f"{i}\n", f"{i}\n"] for i in range(2, 8)] + [["2 \n", "3\n"]])
    assert filter_reason(inconsistent) == "inconsistent_expected_outputs"
    consistent_float = prob(hidden=[[f"{i}\n", f"{i}\n"] for i in range(2, 8)] + [["1\n", "1.0000000\n"]])
    assert filter_reason(consistent_float) == "kept"  # same canonical output key
    kept, stats = filter_problems([prob(pid="a"), prob(pid="b", public=[]), prob(pid="c", question="print any")])
    assert [p["pid"] for p in kept] == ["a"]
    assert stats["total_raw"] == 3 and stats["kept"] == 1 and stats["removed_fraction"] == round(2 / 3, 4)
    assert stats["removed_pids"] == {"insufficient_tests": ["b"], "multi_answer_statement": ["c"]}


def _meta() -> list[dict]:
    metas = []
    for c in range(60):
        metas += [{"pid": f"t{c}_{i}", "contest_id": f"cf_{c}", "orig_split": "train"} for i in range(1 + c % 5)]
    metas += [{"pid": f"v{i}", "contest_id": "cf_900", "orig_split": "valid"} for i in range(4)]
    metas += [{"pid": f"e{i}", "contest_id": f"cf_{950 + i % 2}", "orig_split": "test"} for i in range(6)]
    metas += [{"pid": "t_in_test_contest", "contest_id": "cf_950", "orig_split": "train"}]
    metas += [{"pid": "v_in_test_contest", "contest_id": "cf_951", "orig_split": "valid"}]
    return metas


def test_assign_splits():
    metas = _meta()
    assignment, info = assign_splits(metas, n_test_heldout=20, n_val_heldout=12)
    check_contest_disjoint(metas, assignment)
    assert assignment["t_in_test_contest"] == "test" and assignment["v_in_test_contest"] == "test"
    assert all(assignment[f"e{i}"] == "test" for i in range(6)) and assignment["v0"] == "val"
    assert info["moved_with_contest"] == {"train->test": 1, "valid->test": 1}
    assert 20 <= info["n_heldout_test_problems"] < 25 and 12 <= info["n_heldout_val_problems"] < 17
    assert not set(info["heldout_test_contests"]) & set(info["heldout_val_contests"])
    by_rank = sorted({m["contest_id"] for m in metas if m["orig_split"] == "train"} - {"cf_950"}, key=contest_rank)
    n_test = len(info["heldout_test_contests"])
    assert sorted(by_rank[:n_test]) == info["heldout_test_contests"]  # test takes the first contests by hash
    assert info["counts"]["test"] == 6 + 2 + info["n_heldout_test_problems"]
    assert sum(info["counts"].values()) == len(metas)

    shuffled = metas[:]
    random.Random(0).shuffle(shuffled)
    assert assign_splits(shuffled, n_test_heldout=20, n_val_heldout=12)[0] == assignment  # row order is irrelevant


def test_prepare_end_to_end(tmp_path: Path):
    train = [row(f"{c}_{x}. P", c) for c in range(1, 41) for x in "AB"]
    train += [row("41_A. Any", 41, description="Print any of them."), row("42_A. Few", 42, n_private=1, n_generated=0)]
    train += [row("100_A. Dup", 100)]  # same pid as a test problem
    rows = {"train": train, "valid": [row(f"200_{x}. V", 200) for x in "AB"],
            "test": [row(f"100_{x}. T", 100) for x in "AB"]}
    stats = prepare(tmp_path, rows, n_test_heldout=6, n_val_heldout=4)

    probs = {s: read_jsonl_gz(tmp_path / "problems" / f"{s}.jsonl.gz") for s in ("train_candidates", "val", "test")}
    lists = {s: (tmp_path / "splits" / f"{s}.txt").read_text().split() for s in probs}
    for s in probs:
        assert lists[s] == sorted(p["pid"] for p in probs[s])
    assert {"cc-100_A", "cc-100_B"} <= set(lists["test"]) and {"cc-200_A", "cc-200_B"} <= set(lists["val"])
    assert len(lists["test"]) == 2 + stats["splits"]["n_heldout_test_problems"] >= 8
    assert len(lists["val"]) == 2 + stats["splits"]["n_heldout_val_problems"] >= 6
    n_out = stats["splits"]["n_heldout_test_problems"] + stats["splits"]["n_heldout_val_problems"]
    assert len(lists["train_candidates"]) == 80 - n_out
    contests = {s: {p["contest_id"] for p in probs[s]} for s in probs}
    assert not (contests["train_candidates"] & contests["val"] or contests["train_candidates"] & contests["test"]
                or contests["val"] & contests["test"])

    saved = read_json(tmp_path / "filter_stats.json")
    assert saved == stats
    tr = stats["per_orig_split"]["train"]
    assert tr["removed_by_reason"] == {"duplicate_pid": 1, "insufficient_tests": 1, "multi_answer_statement": 1}
    assert tr["kept"] == 80 and stats["conversion"]["train"]["rows"] == 83
    assert not (tmp_path / "problems" / ".staging").exists()
