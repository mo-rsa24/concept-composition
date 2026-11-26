"""Paper 2's body finder: bodies.json in the shape the judges read, the crops and the box sheet.

    python3 -m pytest tests/test_paper2_bodies.py -q

Runs on the laptop: the dry run and a stand-in counter need no torch. The real counter runs on the held node.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
PAPER2 = REPO / "2-classifier-guidance/scripts/paper2"
sys.path.insert(0, str(PAPER2))

import paper2_bodies as bodies  # noqa: E402
from paper2_common import load_bodies, write_m14_list  # noqa: E402

M14 = REPO / "outputs/loop-6/m14/blind"


@pytest.mark.skipif(not (M14 / "key.json").exists(), reason="joint-prompt pictures not on this machine")
def test_dry_run_writes_what_the_judges_read(tmp_path):
    lst = tmp_path / "list.csv"
    rows = write_m14_list(lst, 3, seed=0)
    r = subprocess.run([sys.executable, str(PAPER2 / "paper2_bodies.py"), "--pictures", str(lst),
                        "--out", str(tmp_path / "out"), "--dry-run"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    read = load_bodies(tmp_path / "out/bodies.json")
    assert set(read) == {row["picture"] for row in rows}
    assert all(len(b) == 2 for b in read.values())
    assert len(list((tmp_path / "out/crops").glob("*.png"))) == 6
    assert Image.open(tmp_path / "out/sheet.png").size == (bodies.SHEET_COLUMNS * bodies.SHEET_TILE, bodies.SHEET_TILE)
    saved = json.loads((tmp_path / "out/bodies.json").read_text())
    assert saved["count_histogram"] == {"2": 3} and saved["seconds_per_picture"] >= 0


def test_a_counter_with_three_bodies_and_one_with_none(tmp_path, monkeypatch):
    pics = []
    for i in range(2):
        p = tmp_path / f"p{i}.png"
        Image.new("RGB", (1024, 1024), (100, 120 + i, 140)).save(p)
        pics.append(p)
    lst = tmp_path / "list.csv"
    lst.write_text("picture,a,b\n" + "".join(f"{p},cat,dog\n" for p in pics))
    found = {pics[0]: [[10, 10, 400, 400], [380, 50, 700, 600], [650, 500, 1000, 1000]], pics[1]: []}
    monkeypatch.setattr(bodies, "stand_in_counter",
                        lambda path: (len(found[path]), [{"box": b, "confidence": 0.5} for b in found[path]]))
    assert bodies.main(["--pictures", str(lst), "--out", str(tmp_path / "out"), "--dry-run"]) == 0
    read = load_bodies(tmp_path / "out/bodies.json")
    assert read[str(pics[0])] == found[pics[0]] and read[str(pics[1])] == []
    assert len(list((tmp_path / "out/crops").glob("*.png"))) == 3
    assert json.loads((tmp_path / "out/bodies.json").read_text())["count_histogram"] == {"0": 1, "3": 1}
