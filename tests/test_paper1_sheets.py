"""Paper 1's blind sheets (2-classifier-guidance/scripts/paper1/paper1_sheets.py). Needs numpy and Pillow only, so it runs on the laptop.

  python3 -m pytest tests/test_paper1_sheets.py
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, PngImagePlugin

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "2-classifier-guidance" / "scripts" / "paper1"))
import paper1_sheets as sh  # noqa: E402


def _sources(tmp_path):
    """Two rows of 64-pixel pictures whose PNGs carry the seed in a text chunk, as a careless writer might leave it."""
    items = []
    for row in ("region", "ours"):
        for seed in (20017, 20042):
            folder = tmp_path / f"paper1-first-row-{row}" / "count-2"
            folder.mkdir(parents=True, exist_ok=True)
            info = PngImagePlugin.PngInfo()
            info.add_text("seed", str(seed))
            path = folder / f"{seed:06d}.png"
            Image.fromarray(np.full((64, 64, 3), 40 if row == "region" else 200, np.uint8)).save(path, pnginfo=info)
            items.append({"row": row, "seed": seed, "count": 2, "side": "touching", "path": str(path)})
    (tmp_path / "items.json").write_text(json.dumps({"items": items}))
    return items


def _run(argv):
    try:
        sh.main(argv)
    except SystemExit as e:
        return e.code or 0
    return 0


def test_build_hides_rows_and_check_passes(tmp_path):
    _sources(tmp_path)
    assert _run(["build", "--root", str(tmp_path), "--run", "sheet-run", "--items", "items.json", "--rubric", "without-shadows",
                 "--shuffle-seed", "1"]) == 0
    sheet = tmp_path / "sheet-run" / "sheet"
    pics = sorted(sheet.glob("*.png"))
    assert len(pics) == 4 and all(p.stem.isalpha() and len(p.stem) == sh.NAME_LENGTH for p in pics)
    assert all(Image.open(p).size == (256, 256) for p in pics)
    assert all("seed" not in Image.open(p).info for p in pics)                 # the source's text chunk did not travel
    assert not (sheet / "key.json").exists() and (tmp_path / "sheet-run" / "key.json").exists()
    assert "Rubric v1, without-shadows" in (sheet / "index.html").read_text()
    with open(sheet / "calls-a.csv") as f:
        assert next(csv.reader(f)) == ["picture", "red_count", "blue_count", "blended_band", "occlusion_correct"]
    assert _run(["check", "--root", str(tmp_path), "--run", "sheet-run"]) == 0


def test_check_catches_a_row_name_and_a_seed(tmp_path):
    _sources(tmp_path)
    _run(["build", "--root", str(tmp_path), "--run", "sheet-run", "--items", "items.json", "--rubric", "with-shadows", "--shuffle-seed", "2"])
    sheet = tmp_path / "sheet-run" / "sheet"
    (sheet / "notes.txt").write_text("this one is ours")
    assert _run(["check", "--root", str(tmp_path), "--run", "sheet-run"]) == 1
    (sheet / "notes.txt").unlink()
    info = PngImagePlugin.PngInfo()
    info.add_text("comment", "drawn at 20042")
    first = sorted(sheet.glob("*.png"))[0]
    Image.open(first).save(first, pnginfo=info)
    assert _run(["check", "--root", str(tmp_path), "--run", "sheet-run"]) == 1


def test_names_come_from_the_seed_alone():
    assert sh.neutral_names(10, 5) == sh.neutral_names(10, 5) and sh.neutral_names(10, 5) != sh.neutral_names(10, 6)
    assert all(set(n) <= set(sh.NAME_LETTERS) for n in sh.neutral_names(500, 3))


def test_settle_field_by_field():
    a = {"p.png": {"red_count": "2", "blue_count": "2", "blended_band": "no", "occlusion_correct": "yes"}}
    b = {"p.png": {"red_count": "2", "blue_count": "1", "blended_band": "no", "occlusion_correct": "no"}}
    c = {"p.png": {"red_count": "", "blue_count": "2", "blended_band": "", "occlusion_correct": ""}}
    settled, decided, unresolved = sh.settle(a, b, c)
    assert settled["p.png"]["red_count"] == "2" and settled["p.png"]["blue_count"] == "2"
    assert decided == {"agreed": 2, "third": 1}
    assert unresolved == [{"picture": "p.png", "field": "occlusion_correct", "first": "yes", "second": "no"}]


def test_select_takes_first_touching_and_fills_apart_from_two_objects(tmp_path):
    split = {"evaluation": {"count-1": {"touching": [20005, 20001], "apart": list(range(20100, 20140))},
                            "count-2": {"touching": [20003], "apart": [20200, 20201]},
                            "count-3": {"touching": [20002], "apart": [20300]}}}
    (tmp_path / "split.json").write_text(json.dumps(split))
    for row in ("ours", "paste"):
        for k, seeds in ((1, [20001, 20005] + list(range(20100, 20140))), (2, [20003, 20200, 20201]), (3, [20002, 20300])):
            d = tmp_path / row / f"count-{k}"
            d.mkdir(parents=True, exist_ok=True)
            for s in seeds:
                Image.fromarray(np.zeros((64, 64, 3), np.uint8)).save(d / f"{s:06d}.png")
    assert _run(["select", "--root", str(tmp_path), "--run", "sel", "--split", "split.json", "--rows", "ours=ours,paste=paste",
                 "--apart-rows", "ours", "--apart-total", "9"]) == 0
    out = json.loads((tmp_path / "sel" / "items.json").read_text())
    touching = [it["seed"] for it in out["items"] if it["row"] == "ours" and it["side"] == "touching" and it["count"] == 1]
    assert touching == [20001, 20005]                                          # seed order, not file order
    assert out["summary"]["apart_per_count"] == {"1": 6, "2": 2, "3": 1} or out["summary"]["apart_per_count"] == {1: 6, 2: 2, 3: 1}


def test_tuning_total_takes_eleven_eleven_ten(tmp_path):
    split = {"tuning": {f"count-{k}": {"touching": list(range(10000 + 100 * k, 10000 + 100 * k + 40)), "apart": []} for k in (1, 2, 3)}}
    (tmp_path / "split.json").write_text(json.dumps(split))
    for k in (1, 2, 3):
        d = tmp_path / "w1" / f"count-{k}"
        d.mkdir(parents=True)
        for s in range(10000 + 100 * k, 10000 + 100 * k + 40):
            Image.fromarray(np.zeros((64, 64, 3), np.uint8)).save(d / f"{s:06d}.png")
    assert _run(["select", "--root", str(tmp_path), "--run", "tune", "--split", "split.json", "--set", "tuning", "--rows", "w1=w1",
                 "--touch-total", "32"]) == 0
    per = json.loads((tmp_path / "tune" / "items.json").read_text())["summary"]["touching_per_count"]
    assert [per[str(k)] if str(k) in per else per[k] for k in (1, 2, 3)] == [11, 11, 10]
