"""Paper 2's look-alike renders (species judge steps 0.9 and 0.10): the job list, paths, guard, contact sheet, record,
and the collector's view of them. Laptop only: nothing here draws a picture (the drawing needs SDXL on the node).

    python3 -m pytest tests/test_paper2_render_lookalike.py -q
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "2-classifier-guidance/scripts/paper2"))

import paper2_collect as collect  # noqa: E402
import paper2_render_lookalike as render  # noqa: E402
from paper2_bars import NEW_LOOKALIKE_PAIRS, NEW_LOOKALIKE_SEEDS, pair_type  # noqa: E402


def test_the_job_list_is_eight_pairs_by_eight_seeds_all_lookalike_and_allowed():
    jobs = render.jobs()
    assert len(jobs) == 64 and len(set(jobs)) == 64
    assert {s for _, _, s in jobs} == set(range(8001, 8009))
    assert all(pair_type(a, b) == "look-alike" for a, b, _ in jobs)
    render.guard_subjects(NEW_LOOKALIKE_PAIRS)            # raises if any pair is refused


def test_paths_and_slugs():
    out = Path("/x")
    assert render.tile_path(out, "otter", "weasel", 8003) == out / "renders/an_otter__x__a_weasel/seed-8003.png"
    assert render.joint_prompt("lion", "tiger") == "a lion and a tiger"
    assert render.RUN_NAMES == {"plain": "paper2-render-plain-lookalike", "co3": "paper2-render-co3-lookalike"}


def fill(out, n_missing=0):
    jobs = render.jobs()
    for a, b, s in jobs[n_missing:]:
        p = render.tile_path(out, a, b, s)
        p.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (1024, 1024), (s % 255, 60, 90)).save(p)


def test_sheet_only_redraws_the_contact_sheet_and_counts_what_is_on_disk(tmp_path):
    fill(tmp_path, n_missing=3)
    assert render.main(["plain", "--out", str(tmp_path), "--sheet-only"]) == 1        # 61 of 64 on disk
    sheet = Image.open(tmp_path / "contact-sheet.png")
    assert sheet.size == (render.TILE * 8 + 170, render.TILE * 8)
    fill(tmp_path)
    assert render.main(["plain", "--out", str(tmp_path), "--sheet-only"]) == 0


def test_the_record_keeps_earlier_pictures_on_a_resumed_run(tmp_path):
    render.write_record(tmp_path, "co3", [{"pair": "a_lion__x__a_tiger", "seed": 8001, "picture": "p1", "seconds": 1}],
                        [], {"card": "x"})
    render.write_record(tmp_path, "co3", [{"pair": "a_lion__x__a_tiger", "seed": 8002, "picture": "p2", "seconds": 1}],
                        ["a_rabbit__x__a_hare: CO3's parser found 1 concepts"], {"card": "x"})
    rec = json.loads((tmp_path / "record.json").read_text())
    assert [p["seed"] for p in rec["pictures"]] == [8001, 8002] and rec["skipped_pairs"]


def test_the_collector_reads_the_128_new_pictures_as_uncalled_lookalike_candidates():
    rows, report = collect.collect(("new-plain", "new-co3"))
    assert len(rows) == 128 and {r["source"] for r in rows} == {"new-plain", "new-co3"}
    assert all(r["pair_type"] == "look-alike" and r["earlier_call"] == "" and not r["unselected"] for r in rows)
    plain = [r for r in rows if r["source"] == "new-plain"]
    a, b = NEW_LOOKALIKE_PAIRS[0]
    assert str(render.tile_path(collect.PAPER2_DATASETS / render.RUN_NAMES["plain"], a, b, 8001)) in {r["picture"] for r in plain}
    assert {r["made"] for r in rows if r["source"] == "new-co3"} == {collect.CO3_ALONE}


def test_the_default_collection_is_unchanged():
    rows, report = collect.collect()
    assert len(rows) == 1335 and sum(r["unselected"] for r in rows) == 100
    assert "new-plain" not in report["sources"]


# ---- the borrowed-part top-up (step 0.8)

from paper2_bars import BORROWED_TOPUP_PAIRS, BORROWED_TOPUP_SEEDS  # noqa: E402


def test_the_topup_is_five_pairs_by_64_seeds_and_every_pair_is_allowed():
    s = render.SETS["borrowed"]
    jobs = render.jobs(s["pairs"], s["seeds"])
    assert len(jobs) == 320 and len(set(jobs)) == 320
    assert {render.slug(a, b) for a, b in s["pairs"]} == {"a_zebra__x__a_flamingo", "a_lion__x__a_tiger",
                                                          "a_bobcat__x__a_quail", "a_puma__x__a_condor", "a_tiger__x__a_dog"}
    assert s["seeds"] == list(range(8101, 8165))
    assert s["runs"]["co3"] == "paper2-render-co3-borrowed"
    render.guard_subjects(s["pairs"])


def test_no_earlier_source_used_a_topup_seed_on_a_topup_pair():
    rows, _ = collect.collect(tuple(n for n in collect.READERS if n != "new-borrowed"))
    pairs = {render.slug(a, b) for a, b in BORROWED_TOPUP_PAIRS}
    used = {(r["pair"], r["seed"]) for r in rows if r["pair"] in pairs}
    assert used, "the top-up pairs have earlier pictures"
    assert not any(seed in BORROWED_TOPUP_SEEDS for _, seed in used)


def test_pairs_and_seeds_flags():
    assert render.parse_seeds("8101-8104") == [8101, 8102, 8103, 8104]
    assert render.parse_seeds("8101,8110") == [8101, 8110]
    assert render.pairs_from_slugs(["a_tiger__x__a_dog", "an_otter__x__a_weasel"]) == [("tiger", "dog"), ("otter", "weasel")]
    import pytest
    with pytest.raises(SystemExit, match="not a known pair slug"):
        render.pairs_from_slugs(["a_cat__x__a_unicorn"])


def test_the_borrowed_set_writes_its_own_folder_and_wraps_the_sheet(tmp_path):
    s = render.SETS["borrowed"]
    for a, b in s["pairs"][:2]:
        for seed in s["seeds"]:
            p = render.tile_path(tmp_path, a, b, seed)
            p.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (64, 64), (seed % 255, 10, 10)).save(p)
    assert render.main(["co3", "--set", "borrowed", "--out", str(tmp_path), "--sheet-only",
                        "--pairs", "a_zebra__x__a_flamingo", "a_lion__x__a_tiger"]) == 0
    assert Image.open(tmp_path / "contact-sheet.png").size == (render.TILE * 16 + 170, render.TILE * 4 * 2)
    assert render.main(["co3", "--set", "borrowed", "--out", str(tmp_path), "--sheet-only"]) == 1   # 3 pairs missing
    render.write_record(tmp_path, "co3", [{"pair": "a_tiger__x__a_dog", "seed": 8101, "picture": "p", "seconds": 9}],
                        [], {"set": "borrowed"}, s["pairs"], s["seeds"])
    rec = json.loads((tmp_path / "record.json").read_text())
    assert rec["seeds"][0] == 8101 and len(rec["pairs"]) == 5 and rec["set"] == "borrowed"


def test_the_collector_reads_the_320_topup_pictures_as_uncalled_co3_candidates():
    rows, report = collect.collect(("new-borrowed",))
    assert len(rows) == 320 and report["sources"]["new-borrowed"]["resolved"] == 320
    assert all(r["made"] == collect.CO3_ALONE and r["earlier_call"] == "" and not r["unselected"] for r in rows)
    assert all("/paper2/paper2-render-co3-borrowed/renders/" in r["picture"] for r in rows)
    types = {r["pair"]: r["pair_type"] for r in rows}
    assert types["a_lion__x__a_tiger"] == "look-alike" and types["a_zebra__x__a_flamingo"] == "unlike"
