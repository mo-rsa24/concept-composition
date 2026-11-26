"""Paper 2's blind-sheet builder: full-size pages, neutral names, and a check that finds any leak.

    python3 -m pytest tests/test_paper2_sheets.py -q
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest
from PIL import Image, PngImagePlugin

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "2-classifier-guidance/scripts/paper2"))

import paper2_sheets as sheets  # noqa: E402


def make_list(tmp_path, n=5, size=1024, text_chunk=None):
    src = tmp_path / "loop-x" / "row-joint" / "renders"
    src.mkdir(parents=True)
    rows = []
    for i in range(n):
        p = src / f"seed-{7100 + i}.png"
        info = None
        if text_chunk:
            info = PngImagePlugin.PngInfo()
            info.add_text("parameters", text_chunk)
        Image.new("RGB", (size, size), (40 * i, 90, 160)).save(p, pnginfo=info)
        rows.append({"picture": str(p), "a": "cheetah", "b": "cougar", "row": "dial-at-1", "seed": 7100 + i})
    lst = tmp_path / "list.csv"
    with open(lst, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["picture", "a", "b", "row", "seed"])
        w.writeheader()
        w.writerows(rows)
    return lst


def test_build_then_check_passes(tmp_path):
    out = tmp_path / "out"
    sheets.main(["build", "--pictures", str(make_list(tmp_path)), "--seed", "0", "--out", str(out)])
    assert sheets.main(["check", "--out", str(out)]) == 0
    pages = sorted((out / "sheet/pages").glob("*.png"))
    assert len(pages) == 5
    with Image.open(pages[0]) as im:
        assert im.size == (1024, 1024 + sheets.BAND)
    key = json.loads((out / "key/key.json").read_text())["pictures"]
    assert set(key) == {p.stem for p in pages}
    assert all(len(name) == sheets.NAME_HEX for name in key)
    assert not (out / "sheet" / "key.json").exists()


def test_a_text_chunk_in_the_source_does_not_reach_the_sheet(tmp_path):
    out = tmp_path / "out"
    lst = make_list(tmp_path, text_chunk="seed 7101 row dial-at-1")
    sheets.main(["build", "--pictures", str(lst), "--seed", "0", "--out", str(out)])
    assert sheets.main(["check", "--out", str(out)]) == 0


def test_a_planted_leak_fails_the_check(tmp_path):
    out = tmp_path / "out"
    sheets.main(["build", "--pictures", str(make_list(tmp_path)), "--seed", "0", "--out", str(out)])
    p = next((out / "sheet/images").glob("*.png"))
    info = PngImagePlugin.PngInfo()
    info.add_text("note", "seed-7102.png from dial-at-1")
    Image.open(p).save(p, pnginfo=info)
    assert sheets.main(["check", "--out", str(out)]) == 1


def test_same_seed_same_sheet_other_seed_other_order(tmp_path):
    lst = make_list(tmp_path)
    for name in ("a", "b", "c"):
        sheets.main(["build", "--pictures", str(lst), "--seed", "1" if name == "c" else "0",
                     "--out", str(tmp_path / name)])
    read = lambda n: (tmp_path / n / "sheet/sheet.csv").read_text()
    assert read("a") == read("b") and read("a") != read("c")


def test_a_picture_not_1024_stops_the_build(tmp_path):
    with pytest.raises(SystemExit, match="not 1024 by 1024"):
        sheets.main(["build", "--pictures", str(make_list(tmp_path, size=512)), "--seed", "0",
                     "--out", str(tmp_path / "out")])
