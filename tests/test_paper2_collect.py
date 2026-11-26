"""Paper 2's candidate collector (species judge step 0.4): every source at the plan's count, from committed records only.

    python3 -m pytest tests/test_paper2_collect.py -q

Runs on the laptop and on the node: it reads only committed records (and their copies in scripts/paper2/records/),
never plans/ and never /datasets. Whether the files open is the collector's --check, on the node.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "2-classifier-guidance/scripts/paper2"))

import paper2_collect as collect  # noqa: E402

ROWS, REPORT = collect.collect()
BY_PICTURE = {r["picture"]: r for r in ROWS}


def test_every_source_resolves_to_the_plans_count():
    for name, s in REPORT["sources"].items():
        assert s["resolved"] == collect.PLAN_COUNTS[name], name
    assert REPORT["sources"]["m1"]["animal"] == 88


def test_left_out_and_merged_are_counted():
    assert dict(REPORT["left_out"]) == {"not animals (lighthouse and sailboat)": 64,
                                        "its finding records it failed the subject check": 1}
    assert REPORT["merged"] == 112          # round 03 and 07 share 48 CO3 files; binding's plain and lora are the dial's
    assert len(ROWS) == 1335
    assert not any("lighthouse" in r["pair"] for r in ROWS)


def test_the_unselected_hundred_are_co3_alone_and_fixed_by_seed_2():
    picked = sorted(r["picture"] for r in ROWS if r["unselected"])
    assert len(picked) == 100
    assert all(BY_PICTURE[p]["made"] in (collect.CO3_ALONE, "CO3, no push") for p in picked)
    again, _ = collect.collect()
    assert sorted(r["picture"] for r in again if r["unselected"]) == picked


def test_worked_examples_from_the_records():
    p = "/datasets/mmolefe/poe_repair_min/loop-12/z2/full/a_bobcat__x__a_quail/seed-6003.png"
    r = BY_PICTURE[p]                       # round 03's id ysmvpyya27ex: calls-1 two_clear, the third two_bodies
    assert (r["source"], r["round"], r["row"], r["earlier_call"]) == ("box", "03", "co3", "two_bodies")
    assert "07/co3" in r["also_called_in"]  # the same file, called again in round 07
    p = "/datasets/mmolefe/poe_repair_min/loop-12/z8/cases/poe/a_puma__x__a_condor/seed-6001/random-2.png"
    assert BY_PICTURE[p]["earlier_call"] == "hybrid" and BY_PICTURE[p]["push_checker"] == "none"
    p = "/datasets/mmolefe/poe_repair_min/paper2/paper2-upload-joint/8f113ecec769.png"
    assert (BY_PICTURE[p]["pair"], BY_PICTURE[p]["seed"]) == ("a_duck__x__a_goose", 6009)


def test_no_push_read_clip_or_imagenet_and_clip_only_named_boxes():
    checkers = {r["push_checker"] for r in ROWS}
    assert checkers == {"none", collect.READOUT}
    assert {(r["round"].split("-")[0], r["row"]) for r in ROWS if r["clip_named_boxes"]} == {("12", "box"), ("13", "own")}


def test_rows_read_as_a_picture_list():
    for r in ROWS:
        assert r["a"] and r["b"] and r["pair_type"] in ("look-alike", "unlike") and r["picture"].startswith("/datasets/")
    assert sum(r["reported_apart"] for r in ROWS) == 64        # crow and raven


def test_a_blind_folder_holding_a_stray_file_is_read(tmp_path, monkeypatch):
    """The dial's blind folder holds bar.json beside its batch folders; on Python 3.10 glob("*/") returned it too."""
    import json
    blind = tmp_path / "10-co-occurrence-dial/blind"
    (blind / "batch-1").mkdir(parents=True)
    (blind / "bar.json").write_text("{}")
    (blind / "stray.txt").write_text("not a batch")
    (blind / "batch-1/settled.json").write_text(json.dumps({"settled": {
        "abc": {"pair": "a_lobster__x__a_whale", "seed": 6001, "weight": 0.5, "rung": "one"}}}))
    (blind / "batch-1/calls-1.csv").write_text("id,rung,broken,note\nabc,one,1,lobster alone\n")
    monkeypatch.setattr(collect, "LEAKAGE", tmp_path)
    rows = collect.dial()
    assert len(rows) == 1 and rows[0]["picture"].endswith("a_lobster__x__a_whale/seed-6001-w0.50.png")
    assert rows[0]["notes"] == "lobster alone"
