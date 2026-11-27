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
