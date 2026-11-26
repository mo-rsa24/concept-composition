"""Paper 2's step 0.15 reader: the per-picture count the bar reads, the per-crop share and the bodies histogram.

    python3 -m pytest tests/test_paper2_crop_check.py -q
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "2-classifier-guidance/scripts/paper2"))

import paper2_crop_check as check  # noqa: E402


def calls_file(tmp_path, name, bodies_per_picture):
    pictures = [{"picture": f"seed_{i:02d}.png", "kind": "x",
                 "bodies": [{"call": c, "flag": f} for c, f in bodies]} for i, bodies in enumerate(bodies_per_picture)]
    p = tmp_path / f"{name}.json"
    p.write_text(json.dumps({"pictures": pictures, "settings": {}}))
    return p


def test_a_split_cat_counts_as_named_and_shows_in_the_histogram(tmp_path):
    cats = [[("A", False)]] * 61 + [[("A", False), ("A", False)]] + [[("B", False)]] + [[]]
    r = check.read_judge(json.loads(calls_file(tmp_path, "j", cats).read_text()))
    assert r["named_cat"] == 62 and r["passes"]                     # the split cat counts; dog and no-body miss
    assert r["bodies_per_picture"] == {"0": 1, "1": 62, "2": 1}
    assert r["crops"] == 64 and r["crops_called_cat"] == 63
    assert [m["picture"] for m in r["misses"]] == ["seed_62.png", "seed_63.png"]


def test_a_flag_or_a_mixed_call_is_a_miss(tmp_path):
    cats = [[("A", False)]] * 61 + [[("A", True)], [("mixed", False)], [("other", False)]]
    r = check.read_judge(json.loads(calls_file(tmp_path, "j", cats).read_text()))
    assert r["named_cat"] == 61 and not r["passes"]


def test_main_writes_cats_json_and_exits_1_when_a_judge_misses(tmp_path):
    good = calls_file(tmp_path, "good", [[("A", False)]] * 64)
    bad = calls_file(tmp_path, "bad", [[("A", False)]] * 60 + [[("B", False)]] * 4)
    out = tmp_path / "cats.json"
    assert check.main(["--calls", f"clip={good}", "--out", str(out)]) == 0
    assert check.main(["--calls", f"clip={good}", f"vlm={bad}", "--out", str(out)]) == 1
    saved = json.loads(out.read_text())
    assert saved["judges"]["clip"]["passes"] and not saved["judges"]["vlm"]["passes"]
    assert saved["judges"]["vlm"]["named_cat"] == 60
