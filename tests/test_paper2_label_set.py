"""Paper 2's labelled set (species judge step 0.8): settling, recalls, the rubric-version rule, the split and the seal.

    python3 -m pytest tests/test_paper2_label_set.py -q

Synthetic batches in the sheet builder's own format (key.json and calls files); no pictures are opened.
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "2-classifier-guidance/scripts/paper2"))

import paper2_label_set as ls  # noqa: E402
from paper2_bars import (LOOKALIKE_MIN_BY_KIND, PER_KIND_CHOOSING_MIN, PER_KIND_CONFIRM,  # noqa: E402
                         PER_KIND_ONE_PAIR_MAX, PER_KIND_TEST_PAIRS_MIN, PER_KIND_TEST_READ)
from paper2_kinds import KINDS  # noqa: E402

LOOK = [("cheetah", "cougar"), ("duck", "goose"), ("cat", "dog"), ("tiger", "dog"), ("cat", "fox"), ("lion", "tiger")]
UNLIKE = [("boar", "pheasant"), ("bee", "owl"), ("lobster", "whale"), ("zebra", "flamingo"), ("seal", "gull"),
          ("eagle", "fox")]


def slug(a, b):
    art = lambda x: ("an_" if x[0] in "aeiou" else "a_") + x
    return f"{art(a)}__x__{art(b)}"


def write_batch(folder: Path, entries: list[dict], calls: list[list[str]], version=None) -> dict:
    """entries: picture, a, b, unselected; calls: one list of kinds per reviewer (None = not called)."""
    folder.mkdir(parents=True)
    key = {"seed": 0, "pictures": {}}
    if version is not None:
        key["rules_version"] = version
    names = [f"n{i:05d}" for i in range(len(entries))]
    for name, e in zip(names, entries):
        key["pictures"][name] = {"a": e["a"], "b": e["b"], "pair": e.get("pair") or slug(e["a"], e["b"]), "seed": e["seed"],
                                 "source": "box", "unselected": str(e.get("unselected", False)), "picture_path": e["picture"],
                                 "renamed": str(e.get("renamed", False)), "renaming": e.get("renaming", "")}
    (folder / "key.json").write_text(json.dumps(key))
    files = []
    for i, kinds in enumerate(calls, 1):
        f = folder / f"calls-{i}.csv"
        with open(f, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["name", "kind", "note"])
            for name, k in zip(names, kinds):
                if k:
                    w.writerow([name, k.replace("_", " ") if i == 1 else k, f"note {i}"])
        files.append(str(f))
    return {"name": folder.name, "key": str(folder / "key.json"), "calls": files}


def rich_entries(per_pair_kind=12):
    out = []
    for a, b in LOOK + UNLIKE:
        for kind in KINDS:
            for j in range(per_pair_kind):
                out.append({"picture": f"/d/{slug(a, b)}/{kind}/{j}.png", "a": a, "b": b, "seed": 6000 + j,
                            "kind": kind, "unselected": j == 0})
    return out


def run(tmp_path, manifest):
    (tmp_path / "batches.json").write_text(json.dumps(manifest))
    return ls.main(["--batches", str(tmp_path / "batches.json"), "--out", str(tmp_path / "out")])


def read_out(tmp_path):
    with open(tmp_path / "out/labelled-set.csv", newline="") as f:
        return list(csv.DictReader(f))


def test_settling_agree_third_and_unsettled(tmp_path):
    e = [{"picture": f"/d/p{i}.png", "a": "cat", "b": "dog", "seed": i} for i in range(4)]
    b = write_batch(tmp_path / "b1", e, [["two_clear", "hybrid", "hybrid", "other"],
                                         ["two_clear", "two_of_one_kind", "hybrid", "hybrid"],
                                         [None, "two_of_one_kind", None, None]], version=2)
    s = ls.settle_batch(b)
    assert {p: r["kind"] for p, r in s["settled"].items()} == {"/d/p0.png": "two_clear", "/d/p1.png": "two_of_one_kind",
                                                              "/d/p2.png": "hybrid"}
    assert s["unsettled"] == ["n00003"]                   # split, and no third call
    assert s["settled"]["/d/p1.png"]["agreed"] is False and s["version"] == 2


def test_a_v1_flag_kind_cannot_mix_with_v2_but_other_kinds_stand(tmp_path):
    e = [{"picture": f"/d/p{i}.png", "a": "cat", "b": "dog", "seed": i} for i in range(3)]
    b1 = write_batch(tmp_path / "b1", e, [["borrowed_part", "two_clear", "hybrid"]] * 2)          # no version: v1
    recall = write_batch(tmp_path / "rc", e[:1], [["two_clear"]] * 2, version=2)
    labels, refused = ls.merge([ls.settle_batch(b1), ls.settle_batch(recall)])
    assert labels["/d/p0.png"]["kind"] == "two_clear" and "replaces" in labels["/d/p0.png"]   # the recall wins
    assert len(refused) == 1 and refused[0].startswith("b1/n00001: two_clear under rubric v1, where rubric v2")
    assert labels["/d/p2.png"]["kind"] == "hybrid"                                        # hybrid stands from v1
    (tmp_path / "batches.json").write_text(json.dumps([b1, recall]))
    with pytest.raises(SystemExit, match="refused: 1 pictures"):
        ls.main(["--batches", str(tmp_path / "batches.json"), "--out", str(tmp_path / "out")])


def test_a_manifest_version_must_match_the_key(tmp_path):
    e = [{"picture": "/d/p0.png", "a": "cat", "b": "dog", "seed": 0}]
    b = write_batch(tmp_path / "b1", e, [["hybrid"]] * 2, version=2)
    b["rules_version"] = 1
    with pytest.raises(SystemExit, match="manifest says rubric version 1"):
        ls.settle_batch(b)


def test_the_split_meets_the_size_table_and_seals_the_confirmation_slice(tmp_path):
    entries = rich_entries()
    kinds = [e["kind"] for e in entries]
    b = write_batch(tmp_path / "b1", entries, [kinds, kinds], version=2)
    assert run(tmp_path, [b]) == 0
    rows = read_out(tmp_path)
    split = json.loads((tmp_path / "out/split.json").read_text())
    assert not set(split["choosing_pairs"]) & set(split["test_pairs"])
    confirm = json.loads((tmp_path / "out/confirm.json").read_text())["pictures"]
    truth = {e["picture"]: e["kind"] for e in entries}
    for kind in KINDS:
        assert sum(r["kind"] == kind and r["part"] == "choosing" for r in rows) >= PER_KIND_CHOOSING_MIN
        read = [r for r in rows if r["kind"] == kind and r["part"] == "read"]
        assert len(read) == PER_KIND_TEST_READ
        assert sum(r["pair_type"] == "look-alike" for r in read) >= LOOKALIKE_MIN_BY_KIND[kind]
        assert len({r["pair"] for r in read}) >= PER_KIND_TEST_PAIRS_MIN
        assert max(Counter(r["pair"] for r in read).values()) <= PER_KIND_ONE_PAIR_MAX
        assert sum(truth[p] == kind for p in confirm) == PER_KIND_CONFIRM
    sealed = [r for r in rows if r["part"] == "confirmation"]
    assert len(sealed) == PER_KIND_CONFIRM * len(KINDS)
    assert all(r["kind"] == "" and r["call_1"] == "" and r["notes"] == "" for r in sealed)   # kinds only in confirm.json
    assert {r["picture"] for r in sealed} == set(confirm)
    assert all(r["pair"] in split["test_pairs"] for r in sealed)
    owner = list(csv.DictReader(open(tmp_path / "out/owner-list.csv")))
    assert len(owner) == 60 and set(owner[0]) == {"picture", "a", "b"}
    assert all(not (r["unselected_sample"] == "True" and r["part"] == "confirmation") for r in rows)


def test_the_split_is_deterministic(tmp_path):
    entries = rich_entries()
    kinds = [e["kind"] for e in entries]
    b = write_batch(tmp_path / "b1", entries, [kinds, kinds], version=2)
    out = []
    for i in range(2):
        (tmp_path / "batches.json").write_text(json.dumps([b]))
        ls.main(["--batches", str(tmp_path / "batches.json"), "--out", str(tmp_path / f"out{i}")])
        out.append((tmp_path / f"out{i}/labelled-set.csv").read_text() + (tmp_path / f"out{i}/confirm.json").read_text())
    assert out[0] == out[1]


def test_thin_data_names_the_shortfall_and_the_next_candidates(tmp_path):
    entries = rich_entries(per_pair_kind=2)
    kinds = [e["kind"] for e in entries]
    b = write_batch(tmp_path / "b1", entries, [kinds, kinds], version=2)
    cands = tmp_path / "candidates.csv"
    with open(cands, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["picture", "a", "b", "pair_type", "earlier_call"])
        w.writerow(["/d/new-hybrid.png", "cat", "dog", "look-alike", "hybrid"])
        w.writerow(["/d/new-plain.png", "cat", "dog", "look-alike", "two_bodies"])
        w.writerow([entries[0]["picture"], "x", "y", "unlike", "hybrid"])        # already called: never listed
    (tmp_path / "batches.json").write_text(json.dumps([b]))
    assert ls.main(["--batches", str(tmp_path / "batches.json"), "--candidates", str(cands),
                    "--out", str(tmp_path / "out")]) == 1
    counts = (tmp_path / "out/counts.txt").read_text()
    assert "NOT SPLIT" in counts and "short:" in counts
    nxt = list(csv.DictReader(open(tmp_path / "out/next-candidates.csv")))
    assert {r["picture"] for r in nxt} == {"/d/new-hybrid.png", "/d/new-plain.png"}     # the called picture is never listed
    assert {r["picture"]: r["may_be"] for r in nxt}["/d/new-hybrid.png"] == "hybrid"
    assert not (tmp_path / "out/confirm.json").exists()


def test_a_short_unselected_sample_writes_the_top_up_from_test_pairs(tmp_path):
    import paper2_collect
    entries = rich_entries()                                  # one unselected picture a pair and kind: 72 in all
    kinds = [e["kind"] for e in entries]
    b = write_batch(tmp_path / "b1", entries, [kinds, kinds], version=2)
    cands = tmp_path / "candidates.csv"
    cols = ["picture", "a", "b", "pair", "pair_type", "seed", "source", "made", "earlier_call", "unselected"]
    with open(cands, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for a, b_ in LOOK + UNLIKE:
            for j in range(5):                                # uncalled CO3-alone pictures on every pair
                w.writerow({"picture": f"/d/new/{slug(a, b_)}/{j}.png", "a": a, "b": b_, "pair": slug(a, b_),
                            "pair_type": "x", "seed": 7000 + j, "source": "m1", "made": paper2_collect.CO3_ALONE,
                            "earlier_call": "two_bodies", "unselected": "False"})
            w.writerow({"picture": f"/d/new/{slug(a, b_)}/pushed.png", "a": a, "b": b_, "pair": slug(a, b_),
                        "pair_type": "x", "seed": 7100, "source": "box", "made": "CO3 with a push",
                        "earlier_call": "hybrid", "unselected": "False"})
        w.writerow({**{c: "" for c in cols}, "picture": entries[0]["picture"], "pair": slug(*LOOK[0]),
                    "made": paper2_collect.CO3_ALONE})                 # already called: never drawn
    (tmp_path / "batches.json").write_text(json.dumps([b]))
    assert ls.main(["--batches", str(tmp_path / "batches.json"), "--candidates", str(cands),
                    "--out", str(tmp_path / "out")]) == 0
    rows = read_out(tmp_path)
    have = sum(r["unselected_sample"] == "True" for r in rows)
    assert have < 100
    test_pairs = set(json.loads((tmp_path / "out/split.json").read_text())["test_pairs"])
    top = list(csv.DictReader(open(tmp_path / "out/unselected-next.csv")))
    assert len(top) == min(100 - have, 5 * len(test_pairs))
    assert all(r["pair"] in test_pairs and r["made"] == paper2_collect.CO3_ALONE for r in top)
    assert entries[0]["picture"] not in {r["picture"] for r in top}
    assert list(top[0]) == cols                                        # the collector's columns, for paper2_sheets.py
    again = ls.unselected_next(cands, {r["picture"] for r in rows}, test_pairs, 100 - have)
    assert [r["picture"] for r in again] == [r["picture"] for r in top]   # drawn by seed 2, the same each time


def test_with_no_split_the_top_up_waits(tmp_path):
    entries = rich_entries(per_pair_kind=2)
    kinds = [e["kind"] for e in entries]
    b = write_batch(tmp_path / "b1", entries, [kinds, kinds], version=2)
    (tmp_path / "batches.json").write_text(json.dumps([b]))
    ls.main(["--batches", str(tmp_path / "batches.json"), "--out", str(tmp_path / "out")])
    assert "top-up to 100 waits for the split" in (tmp_path / "out/counts.txt").read_text()
    assert not (tmp_path / "out/unselected-next.csv").exists()


def test_a_v2_hybrid_is_refused_once_a_v3_batch_exists_and_a_v3_recall_replaces_it(tmp_path):
    e = [{"picture": f"/d/p{i}.png", "a": "cheetah", "b": "cougar", "seed": i} for i in range(4)]
    b2 = write_batch(tmp_path / "b2", e, [["hybrid", "hybrid", "two_clear", "borrowed_part"]] * 2, version=2)
    r3 = write_batch(tmp_path / "r3", e[:1], [["borrowed_part"]] * 2, version=3)       # recalls p0 only
    labels, refused = ls.merge([ls.settle_batch(b2), ls.settle_batch(r3)])
    assert labels["/d/p0.png"]["kind"] == "borrowed_part" and labels["/d/p0.png"]["rules_version"] == 3
    assert "b2 (hybrid, v2)" in labels["/d/p0.png"]["replaces"]
    assert len(refused) == 1 and refused[0].startswith("b2/n00001: hybrid under rubric v2, where rubric v3")
    assert labels["/d/p2.png"]["kind"] == "two_clear" and labels["/d/p3.png"]["kind"] == "borrowed_part"   # v2 stands
    (tmp_path / "batches.json").write_text(json.dumps([b2, r3]))
    with pytest.raises(SystemExit, match="b2/n00001: hybrid under rubric v2"):
        ls.main(["--batches", str(tmp_path / "batches.json"), "--out", str(tmp_path / "out")])
    r3all = write_batch(tmp_path / "r3all", e[:2], [["borrowed_part", "hybrid"]] * 2, version=3)
    labels, refused = ls.merge([ls.settle_batch(b2), ls.settle_batch(r3all)])
    assert refused == [] and labels["/d/p1.png"]["rules_version"] == 3


def test_without_a_v3_batch_v2_hybrids_stand():
    labels = {"/d/p.png": {"batch": "b2", "name": "n0", "kind": "hybrid", "rules_version": 2}}
    _, refused = ls.merge([{"name": "b2", "version": 2, "settled": labels}])
    assert refused == []


# ---- rubric v4, renamed pictures, and the split diagnostic

def test_a_v3_wrong_species_is_refused_once_a_v4_batch_exists():
    lab = {"/d/p.png": {"batch": "b3", "name": "n0", "kind": "wrong_species", "rules_version": 3}}
    _, refused = ls.merge([{"name": "b3", "version": 3, "settled": lab},
                           {"name": "r4", "version": 4, "settled": {}}])
    assert len(refused) == 1 and "wrong_species under rubric v3, where rubric v4" in refused[0]
    _, refused = ls.merge([{"name": "b3", "version": 3, "settled": lab}])
    assert refused == []


def test_a_renamed_copy_stands_beside_its_original(tmp_path):
    e = [{"picture": "/d/p0.png", "a": "cheetah", "b": "cougar", "seed": 1}]
    b1 = write_batch(tmp_path / "b1", e, [["two_clear"]] * 2, version=4)
    ren = [{"picture": "/d/p0.png", "a": "cheetah", "b": "leopard", "seed": 1, "pair": "a_cheetah__x__a_cougar",
            "renamed": True, "renaming": "cougar as leopard"}]
    b2 = write_batch(tmp_path / "b2", ren, [["wrong_species"]] * 2, version=4)
    labels, refused = ls.merge([ls.settle_batch(b1), ls.settle_batch(b2)])
    assert refused == [] and len(labels) == 2
    assert labels["/d/p0.png"]["kind"] == "two_clear"
    assert labels["/d/p0.png|renamed cougar as leopard"]["kind"] == "wrong_species"
    assert labels["/d/p0.png|renamed cougar as leopard"]["pair"] == "a_cheetah__x__a_cougar"


def test_the_read_holds_at_most_20_renamed_and_reports_why_it_falls_short():
    def r(i, renamed, pair):
        return {"item": f"i{i}", "pair": pair, "pair_type": "look-alike", "renamed": renamed}
    pics = [r(i, True, f"p{i % 6}") for i in range(30)] + [r(100 + i, False, f"p{i % 6}") for i in range(30)]
    read, fails = ls.read_selection(pics, ls.random.Random(0))
    assert len(read) == 40 and sum(x["renamed"] for x in read) <= 20 and fails == []
    read, fails = ls.read_selection(pics[:30] + pics[30:45], ls.random.Random(0))       # 15 natural only
    assert len(read) == 35 and sum(x["renamed"] for x in read) == 20
    assert "read under 40: the 20-renamed cap" in fails
    few_pairs = [r(i, False, "p0" if i < 20 else "p1") for i in range(60)]
    read, fails = ls.read_selection(few_pairs, ls.random.Random(0))
    assert "read under 40: the 12-a-pair cap" in fails and "pairs under 4" in fails


def test_a_mostly_renamed_wrong_species_read_is_marked(tmp_path):
    entries = [e for e in rich_entries() if e["kind"] != "wrong_species"]
    natural_ws = [e for e in rich_entries() if e["kind"] == "wrong_species"][::4]          # 36 natural
    renamed_ws = [{**e, "picture": e["picture"] + "-r", "renamed": True, "renaming": "x as y"}
                  for e in rich_entries() if e["kind"] == "wrong_species"]                # 144 renamed
    allp = entries + natural_ws + renamed_ws
    b = write_batch(tmp_path / "b1", allp, [[e["kind"] for e in allp]] * 2, version=4)
    assert run(tmp_path, [b]) == 0
    counts = (tmp_path / "out/counts.txt").read_text()
    line = next(l for l in counts.splitlines() if l.strip().startswith("wrong_species"))
    assert "renamed 20" in line and "natural 20" in line or "MOSTLY RENAMED" in line
    rows = read_out(tmp_path)
    ws_read = [r for r in rows if r["kind"] == "wrong_species" and r["part"] == "read"]
    assert len(ws_read) == 40 and sum(r["renamed"] == "True" for r in ws_read) <= 20
    assert "settled natural and renamed apart: wrong_species 36 natural, 144 renamed" in counts


def test_the_split_diagnostic_names_what_blocks_it(tmp_path):
    entries = rich_entries(per_pair_kind=2)
    kinds = [e["kind"] for e in entries]
    b = write_batch(tmp_path / "b1", entries, [kinds, kinds], version=4)
    assert run(tmp_path, [b]) == 1
    counts = (tmp_path / "out/counts.txt").read_text()
    assert "split diagnostic over 20000 draws" in counts
    diag = json.loads((tmp_path / "out/split-diagnostic.json").read_text())
    assert diag["draws"] == 20000
    assert diag["fails"]["hybrid"]["test under 50 (confirmation 10 and read 40)"] > 0
    assert set(diag["fails"]["hybrid"]) <= set(ls.REASONS)


def test_a_v3_extra_subject_is_refused_once_v4_exists_and_a_v4_recall_replaces_it(tmp_path):
    e = [{"picture": f"/d/p{i}.png", "a": "boar", "b": "pheasant", "seed": i} for i in range(2)]
    b3 = write_batch(tmp_path / "b3", e, [["extra_subject", "extra_subject"]] * 2, version=3)
    r4 = write_batch(tmp_path / "r4", e[:1], [["two_clear"]] * 2, version=4)          # a blurred third bird no longer counts
    labels, refused = ls.merge([ls.settle_batch(b3), ls.settle_batch(r4)])
    assert labels["/d/p0.png"]["kind"] == "two_clear" and labels["/d/p0.png"]["rules_version"] == 4
    assert "b3 (extra_subject, v3)" in labels["/d/p0.png"]["replaces"]
    assert len(refused) == 1 and refused[0].startswith("b3/n00001: extra_subject under rubric v3, where rubric v4")
    assert ls.VERSION_KINDS[4] == {"wrong_species", "extra_subject"}


def short_lookalike_extra_subject():
    """Rich for every kind, except extra subject: 8 pictures on look-alike pairs (4 on each of two), many unlike."""
    entries = [e for e in rich_entries() if e["kind"] != "extra_subject"]
    for a, b in LOOK[:2]:
        for j in range(4):
            entries.append({"picture": f"/d/{slug(a, b)}/extra/{j}.png", "a": a, "b": b, "seed": 6100 + j,
                            "kind": "extra_subject"})
    for a, b in UNLIKE:
        for j in range(16):
            entries.append({"picture": f"/d/{slug(a, b)}/extra/{j}.png", "a": a, "b": b, "seed": 6100 + j,
                            "kind": "extra_subject"})
    return entries


def test_a_set_short_only_on_one_kinds_lookalike_count_splits_under_its_lowered_minimum(tmp_path, monkeypatch):
    entries = short_lookalike_extra_subject()
    kinds = [e["kind"] for e in entries]
    b = write_batch(tmp_path / "b1", entries, [kinds, kinds], version=4)
    assert run(tmp_path, [b]) == 0                                          # extra subject's minimum is 8
    split = json.loads((tmp_path / "out/split.json").read_text())
    assert split["lookalike_min_by_kind"]["extra_subject"] == 8
    counts = (tmp_path / "out/counts.txt").read_text()
    assert "look-alike minimum per kind" in counts and "extra_subject 8" in counts
    assert "look-alike minimum 8, underpowered on look-alike pairs" in counts
    rows = read_out(tmp_path)
    es_read = [r for r in rows if r["kind"] == "extra_subject" and r["part"] == "read"]
    assert len(es_read) == 40 and sum(r["pair_type"] == "look-alike" for r in es_read) >= 8

    monkeypatch.setattr(ls, "LOOKALIKE_MIN_BY_KIND", {k: 20 for k in KINDS})
    (tmp_path / "batches.json").write_text(json.dumps([b]))
    assert ls.main(["--batches", str(tmp_path / "batches.json"), "--out", str(tmp_path / "out20")]) == 1
    diag = json.loads((tmp_path / "out20/split-diagnostic.json").read_text())
    es = diag["fails"]["extra_subject"]
    # every draw fails on extra subject: on its look-alike minimum, or before the read is drawn, on its test count
    assert es["look-alike under its minimum"] + es["test under 50 (confirmation 10 and read 40)"] >= diag["draws"]
    assert "'look-alike pictures': 12" in (tmp_path / "out20/counts.txt").read_text()
