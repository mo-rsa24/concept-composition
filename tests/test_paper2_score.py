"""Paper 2's judge lists, setting choice and scorer (species judge steps 0.17 to 0.21), on a synthetic split labelled
set. Laptop only: no model is loaded.

    python3 -m pytest tests/test_paper2_score.py -q
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "2-classifier-guidance/scripts/paper2"))

import paper2_judge_lists as jl  # noqa: E402
import paper2_score as sc  # noqa: E402
from paper2_bars import pair_type  # noqa: E402
from paper2_kinds import KINDS  # noqa: E402

TEST_PAIRS = [("lion", "tiger"), ("wolf", "fox"), ("cheetah", "cougar"),
              ("zebra", "flamingo"), ("eagle", "fox"), ("boar", "pheasant")]
CHOOSING_PAIRS = [("cat", "dog"), ("duck", "goose")]
COLS = ["picture", "renamed", "renaming", "a", "b", "pair", "pair_type", "seed", "source", "kind", "rules_version",
        "part", "unselected_sample", "batch", "name", "call_1", "call_2", "agreed", "replaces", "notes"]


def slug(a, b):
    art = lambda s: ("an_" if s[0] in "aeiou" else "a_") + s
    return f"{art(a)}__x__{art(b)}"


def labelled_rows(tmp: Path) -> list[dict]:
    rows, n = [], 0

    def add(a, b, kind, part, **extra):
        nonlocal n
        n += 1
        r = {c: "" for c in COLS} | {"picture": str(tmp / "pics" / f"p{n:04d}.png"), "renamed": "False", "a": a, "b": b,
                                      "pair": slug(a, b), "pair_type": pair_type(a, b), "seed": str(n), "kind": kind,
                                      "part": part, "unselected_sample": "False", "call_1": kind, "call_2": kind}
        r.update(extra)
        rows.append(r)
        return r
    for kind in KINDS:
        for i in range(40):
            a, b = TEST_PAIRS[i % 6]
            r = add(a, b, kind, "read")
            if i == 0:
                r["call_2"] = "hybrid" if kind != "hybrid" else "two_clear"     # reviewers disagree once a kind
        for i in range(10):
            a, b = TEST_PAIRS[i % 6]
            add(a, b, "", "confirmation", call_1="", call_2="")
        for i in range(16):
            a, b = CHOOSING_PAIRS[i % 2]
            add(a, b, kind, "choosing")
    for i in range(8):
        a, b = TEST_PAIRS[i % 6]
        add(a, b, "two_clear" if i % 2 else "hybrid", "spare", unselected_sample="True")
    for i in range(3):
        add("crow", "raven", "two_of_one_kind", "apart")
    ws_read = [r for r in rows if r["kind"] == "wrong_species" and r["part"] == "read"][:6]
    for r in ws_read:                                   # six of the read wrong species are renamed copies
        r.update(renamed="True", renaming="cougar as leopard", b="leopard")
    return rows


def write_set(tmp: Path, rows: list[dict]) -> tuple[Path, Path]:
    tmp.mkdir(parents=True, exist_ok=True)
    ls = tmp / "labelled-set.csv"
    with open(ls, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)
    split = tmp / "split.json"
    split.write_text(json.dumps({"seed": 0, "attempt": 1, "choosing_pairs": [slug(*p) for p in CHOOSING_PAIRS],
                                 "test_pairs": [slug(*p) for p in TEST_PAIRS]}))
    return ls, split


def judge_calls(rows, kind_of_row, folder: Path, name="clip", settings=None, written="2026-11-27T10:00:00",
                chosen_written="2026-11-27T09:00:00"):
    settings = settings or {"margin": 0.15, "flag_share": 0.25}
    pics = []
    for r in rows:
        k = kind_of_row(r)
        pics.append({"picture": r["picture"], "a": r["a"], "b": r["b"], "covered": True, "n_bodies": 2,
                     "bodies": [{"box": [0, 0, 500, 500], "call": "A", "flag": False},
                                {"box": [400, 0, 900, 500] if int(r["seed"]) % 3 == 0 else [600, 0, 900, 500],
                                 "call": "B", "flag": False}],
                     "kind": k, "two_clear_score": 0.9 if k == "two_clear" else 0.1})
    (folder / "test").mkdir(parents=True, exist_ok=True)
    (folder / "test/calls.json").write_text(json.dumps({"judge": name, "settings": settings, "pictures": pics}))
    (folder / "test/settings.json").write_text(json.dumps({"judge": name, "settings": settings, "written": written}))
    if name != "old":
        (folder / "settings-chosen.json").write_text(json.dumps({"judge": name, "chosen": settings,
                                                                 "written": chosen_written}))
    return folder


def called_rows(rows):
    return [r for r in rows if r["part"] in ("read", "spare", "apart")]


def owner_files(tmp: Path, rows, right_per_kind: dict | int):
    read = [r for r in rows if r["part"] == "read"]
    key, calls = {}, []
    for kind in KINDS:
        mine = [r for r in read if r["kind"] == kind][:10]
        good = right_per_kind if isinstance(right_per_kind, int) else right_per_kind.get(kind, 10)
        for i, r in enumerate(mine):
            name = f"k{KINDS.index(kind)}n{i:02d}"
            key[name] = {"picture_path": r["picture"], "a": r["a"], "b": r["b"]}
            calls.append({"name": name, "kind": kind if i < good else ("other" if kind != "other" else "hybrid"),
                          "note": ""})
    (tmp / "owner").mkdir(exist_ok=True)
    (tmp / "owner/key.json").write_text(json.dumps({"pictures": key}))
    with open(tmp / "owner/owner-calls.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["name", "kind", "note"])
        w.writeheader()
        w.writerows(calls)
    return tmp / "owner/key.json", tmp / "owner/owner-calls.csv"


def score(tmp, ls, split, judges, owner=None, extra=()):
    args = ["--labelled-set", str(ls), "--split", str(split), "--out", str(tmp / "score"), "--no-sheet"]
    for name, folder in judges.items():
        args += ["--judge", f"{name}={folder}"]
    if owner:
        args += ["--owner-key", str(owner[0]), "--owner-calls", str(owner[1])]
    assert sc.main(args + list(extra)) == 0
    return json.loads((tmp / "score/bars.json").read_text())


# ---- the lists

def test_the_lists_hold_choosing_and_test_pairs_and_never_a_confirmation_picture(tmp_path):
    rows = labelled_rows(tmp_path)
    ls, split = write_set(tmp_path, rows)
    assert jl.main(["lists", "--labelled-set", str(ls), "--split", str(split), "--out", str(tmp_path / "lists")]) == 0
    ch = jl.read_csv(tmp_path / "lists/choosing.csv")
    te = jl.read_csv(tmp_path / "lists/test.csv")
    assert len(ch) == 96 and {r["part"] for r in ch} == {"choosing"}
    assert len(te) == 240 + 8 + 3 and {r["part"] for r in te} == {"read", "spare", "apart"}
    assert sum(r["unselected_sample"] == "True" for r in te) == 8
    pics = jl.read_csv(tmp_path / "lists/pictures.csv")
    assert len(pics) == len({r["picture"] for r in ch + te})


def test_confirm_json_is_refused_everywhere(tmp_path):
    rows = labelled_rows(tmp_path)
    ls, split = write_set(tmp_path, rows)
    (tmp_path / "confirm.json").write_text("{}")
    with pytest.raises(SystemExit, match="sealed"):
        jl.main(["lists", "--labelled-set", str(tmp_path / "confirm.json"), "--split", str(split), "--out", str(tmp_path)])
    with pytest.raises(SystemExit, match="sealed"):
        sc.main(["--labelled-set", str(ls), "--split", str(split), "--judge", "clip=x", "--out", str(tmp_path / "s"),
                 "--candidates", str(tmp_path / "confirm.json")])
    src = (REPO / "2-classifier-guidance/scripts/paper2/paper2_score.py").read_text()
    assert src.count('"confirm.json"') == 0          # the scorer never names the file it refuses


def test_a_judge_that_called_the_confirmation_slice_is_not_scored(tmp_path):
    rows = labelled_rows(tmp_path)
    ls, split = write_set(tmp_path, rows)
    called = called_rows(rows) + [r for r in rows if r["part"] == "confirmation"][:1]
    j = judge_calls(called, lambda r: r["kind"] or "two_clear", tmp_path / "clip")
    with pytest.raises(SystemExit, match="called 1 confirmation pictures"):
        score(tmp_path, ls, split, {"clip": j})


# ---- choosing a setting

def test_choose_keeps_the_grid_point_with_the_highest_agreement_on_choosing_pairs(tmp_path):
    rows = labelled_rows(tmp_path)
    ls, split = write_set(tmp_path, rows)
    jl.main(["lists", "--labelled-set", str(ls), "--split", str(split), "--out", str(tmp_path / "lists")])
    choosing = jl.read_csv(tmp_path / "lists/choosing.csv")
    # body 0 leans 0.62 to A (a gap of 0.24), body 1 is a clear B: mixed (so hybrid) only at margin 0.30, else two clear
    pics = [{"picture": r["picture"], "a": r["a"], "b": r["b"], "covered": True, "n_bodies": 2,
             "bodies": [{"box": [0, 0, 1, 1], "raw": {"p_a": 0.62, "p_b": 0.38, "p_neither_best": 0.0}, "call": "A", "flag": False},
                        {"box": [2, 2, 3, 3], "raw": {"p_a": 0.02, "p_b": 0.98, "p_neither_best": 0.0}, "call": "B", "flag": False}],
             "kind": "", "two_clear_score": 0.0} for r in choosing]
    for r in choosing:
        r["kind"] = "hybrid"
    jl.write_csv(tmp_path / "lists/choosing.csv", choosing)
    calls = tmp_path / "choosing-calls.json"
    calls.write_text(json.dumps({"judge": "clip", "settings": {"margin": 0.15, "flag_share": 0.25}, "pictures": pics}))
    out = tmp_path / "settings-chosen.json"
    assert jl.main(["choose", "--judge", "clip", "--calls", str(calls), "--lists", str(tmp_path / "lists"),
                    "--out", str(out)]) == 0
    got = json.loads(out.read_text())
    assert got["chosen"] == {"margin": 0.30, "flag_share": 0.15} and got["agreement"] == 1.0
    assert len(got["grid"]) == 15 and "written" in got
    with pytest.raises(SystemExit, match="out of steps"):
        jl.grid_of("dac")


# ---- the scorer

def test_a_perfect_judge_meets_every_read_bar_once_the_owner_reaches_57(tmp_path):
    rows = labelled_rows(tmp_path)
    ls, split = write_set(tmp_path, rows)
    j = judge_calls(called_rows(rows), lambda r: r["kind"], tmp_path / "clip")
    res = score(tmp_path, ls, split, {"clip": j}, owner_files(tmp_path, rows, 10))
    c = res["judges"]["clip"]
    assert res["owner"]["state"] == "absolute" and res["owner"]["right"] == 60
    assert c["verdict"] == "meets every read bar", c.get("missed")
    assert set(c["read_bars"]) >= {"agree_all", "agree_kind", "lookalike", "within_pair_auc", "drop_one_pair"}
    assert c["bars"]["agree_all"] == {"right": 240, "of": 240, "need": 216, "pass": True}
    assert c["bars"]["within_pair_auc"]["auc"] == 1.0 and len(c["bars"]["within_pair_auc"]["pairs"]) == 6
    assert c["wrong_species_apart"]["renamed"] == {"right": 6, "of": 6}
    assert c["wrong_species_apart"]["by_renaming"] == {"cougar as leopard": {"right": 6, "of": 6}}
    assert c["crow_and_raven"]["pictures"] == 3
    assert c["underpowered_lookalike"] == {"extra_subject": "look-alike minimum 8, underpowered on look-alike pairs"}
    assert c["two_clear_precision_unselected"]["pictures"] == 8
    assert (tmp_path / "score/confusion-clip.csv").exists()


def test_the_owner_check_decides_which_bars_are_read(tmp_path):
    rows = labelled_rows(tmp_path)
    ls, split = write_set(tmp_path, rows)
    j = judge_calls(called_rows(rows), lambda r: r["kind"], tmp_path / "clip")
    pending = score(tmp_path, ls, split, {"clip": j})
    assert pending["owner"]["state"] == "pending"
    assert "agree_all" not in pending["judges"]["clip"]["read_bars"]
    assert "wait for the owner's check" in pending["judges"]["clip"]["verdict"]
    full = score(tmp_path, ls, split, {"clip": j}, owner_files(tmp_path, rows, {"two_clear": 8, "hybrid": 9}))
    assert full["owner"]["right"] == 57 and full["owner"]["state"] == "absolute"
    lim = score(tmp_path, ls, split, {"clip": j}, owner_files(tmp_path, rows, {"two_clear": 8, "hybrid": 8, "borrowed_part": 9}))
    assert lim["owner"]["right"] == 55 and lim["owner"]["state"] == "limited"
    assert "agree_all" not in lim["judges"]["clip"]["read_bars"]
    assert lim["judges"]["clip"]["verdict"] == "meets every read bar (limited by labels)"
    bad = score(tmp_path, ls, split, {"clip": j}, owner_files(tmp_path, rows, {"hybrid": 7}))
    assert bad["owner"]["state"] == "untrusted"
    assert bad["judges"]["clip"]["verdict"] == "nothing read: the labels cannot be trusted"


def test_drop_one_pair_names_the_pair_that_carries_a_bar(tmp_path):
    rows = labelled_rows(tmp_path)
    ls, split = write_set(tmp_path, rows)
    # two clear only on four pairs: drop any of them and fewer than AUC_PAIRS_MIN pairs remain
    keep = {slug(*p) for p in TEST_PAIRS[:4]}
    rows = [r for r in rows if not (r["kind"] == "two_clear" and r["part"] == "read" and r["pair"] not in keep)]
    ls, split = write_set(tmp_path, rows)
    j = judge_calls(called_rows(rows), lambda r: r["kind"], tmp_path / "clip")
    c = score(tmp_path, ls, split, {"clip": j}, owner_files(tmp_path, rows, 10))["judges"]["clip"]
    assert c["bars"]["within_pair_auc"]["pass"]
    assert sorted(c["drop_one_pair"]["flips"]["within_pair_auc"]) == sorted(keep)
    assert "drop_one_pair" in c["missed"]


def test_a_judge_right_on_unlike_pairs_only_misses_the_lookalike_reading(tmp_path):
    rows = labelled_rows(tmp_path)
    ls, split = write_set(tmp_path, rows)
    j = judge_calls(called_rows(rows), lambda r: r["kind"] if r["pair_type"] == "unlike" else "other", tmp_path / "clip")
    c = score(tmp_path, ls, split, {"clip": j}, owner_files(tmp_path, rows, 10))["judges"]["clip"]
    assert c["bars"]["lookalike"]["pass"] is False and "lookalike" in c["missed"]
    assert c["bars"]["lookalike"]["unlike"]["rate"] == 1.0


def test_settings_chosen_after_the_test_call_fail_the_check(tmp_path):
    rows = labelled_rows(tmp_path)
    ls, split = write_set(tmp_path, rows)
    j = judge_calls(called_rows(rows), lambda r: r["kind"], tmp_path / "clip", chosen_written="2026-11-27T11:00:00")
    c = score(tmp_path, ls, split, {"clip": j}, owner_files(tmp_path, rows, 10))["judges"]["clip"]
    assert c["settings_check"]["chosen_first"] is False
    assert "settings chosen before the test call" in c["missed"]


def test_the_old_judge_is_the_floor_row_on_its_three_kinds(tmp_path):
    rows = labelled_rows(tmp_path)
    ls, split = write_set(tmp_path, rows)
    j = judge_calls(called_rows(rows), lambda r: r["kind"] if r["kind"] in ("two_clear", "hybrid", "two_of_one_kind")
                    else "other", tmp_path / "old", name="old", settings={"hybrid_margin": 0.1})
    c = score(tmp_path, ls, split, {"old": j})["judges"]["old"]
    assert c["verdict"].startswith("the floor row") and set(c["bars"]["agree_kind"]["kinds"]) == {
        "two_clear", "hybrid", "two_of_one_kind"}
    assert c["bars"]["agree_kind"]["pass"]


def test_a_passing_judge_gets_its_disagreements_drawn_at_full_size(tmp_path):
    from PIL import Image
    rows = labelled_rows(tmp_path)
    ls, split = write_set(tmp_path, rows)
    (tmp_path / "pics").mkdir()
    for r in called_rows(rows):
        Image.new("RGB", (1024, 1024), (90, 120, 150)).save(r["picture"])
    first = {}
    for r in rows:
        if r["part"] == "read":
            first.setdefault(r["kind"], r["picture"])
    wrong = set(first.values())                      # one wrong call a kind: 39 of 40 still passes
    j = judge_calls(called_rows(rows), lambda r: "other" if r["picture"] in wrong else r["kind"], tmp_path / "clip")
    ok, kc = owner_files(tmp_path, rows, 10)
    assert sc.main(["--labelled-set", str(ls), "--split", str(split), "--judge", f"clip={j}", "--owner-key", str(ok),
                    "--owner-calls", str(kc), "--out", str(tmp_path / "score")]) == 0
    res = json.loads((tmp_path / "score/bars.json").read_text())["judges"]["clip"]
    assert res["verdict"] == "meets every read bar" and res["disagreement_pictures"] == 6
    drawn = sorted((tmp_path / "score/disagreements/clip").glob("*.png"))
    assert len(drawn) == 6 and Image.open(drawn[0]).size == (1024, 1024)
    assert len(jl.read_csv(tmp_path / "score/disagreements.csv")) == 6
