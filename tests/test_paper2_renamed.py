"""Paper 2's renamed wrong-species pictures: who is eligible, the renaming, and the trip through the sheet builder and
the label set. Laptop only.

    python3 -m pytest tests/test_paper2_renamed.py -q
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "2-classifier-guidance/scripts/paper2"))

import paper2_label_set as ls  # noqa: E402
import paper2_renamed as ren  # noqa: E402
import paper2_sheets as sheets  # noqa: E402

CAND_COLS = ["picture", "a", "b", "pair", "pair_type", "seed", "source", "round", "row", "made", "earlier_call",
             "earlier_scale", "notes", "push_checker", "clip_named_boxes", "reported_apart", "unselected", "also_called_in"]


def labelled(tmp_path):
    rows = [("p1", "cheetah", "cougar", "two_clear", "read", ""), ("p2", "duck", "goose", "two_clear", "choosing", ""),
            ("p3", "eagle", "fox", "two_clear", "spare", ""), ("p4", "cat", "dog", "two_clear", "read", ""),
            ("p5", "cheetah", "cougar", "hybrid", "read", ""), ("p6", "zebra", "flamingo", "", "confirmation", ""),
            ("p7", "cheetah", "leopard", "wrong_species", "read", "True")]
    lab, cands = [], {}
    for name, a, b, kind, part, renamed in rows:
        pic = str(tmp_path / "pics" / f"{name}.png")
        slug = f"a_{a}__x__a_{b}".replace("a_eagle", "an_eagle")
        lab.append({"picture": pic, "a": a, "b": b, "pair": slug, "seed": "1", "kind": kind, "part": part,
                    "renamed": renamed, "source": "box"})
        cands[pic] = {c: "" for c in CAND_COLS} | {"picture": pic, "a": a, "b": b, "pair": slug, "pair_type": "x",
                                                   "seed": "1", "source": "box", "row": "co3", "made": "CO3 alone"}
    return lab, cands


def test_only_settled_two_clear_on_a_renaming_pair_is_renamed(tmp_path):
    lab, cands = labelled(tmp_path)
    rows = ren.renamed_rows(lab, cands)
    got = {Path(r["picture"]).stem: (r["a"], r["b"], r["renaming"], r["pair_type"]) for r in rows}
    assert got == {"p1": ("cheetah", "leopard", "cougar as leopard", "look-alike"),
                   "p2": ("duck", "swan", "goose as swan", "look-alike"),
                   "p3": ("eagle", "coyote", "fox as coyote", "unlike")}      # p4 no renaming species; p5 hybrid;
    assert all(r["renamed"] is True and r["pair"].startswith(("a_cheetah", "a_duck", "an_eagle")) for r in rows)
    assert [r["picture"] for r in rows] == [r["picture"] for r in ren.renamed_rows(lab, cands)]   # seed 5 order


def test_a_renamed_batch_builds_checks_and_settles_apart(tmp_path):
    lab, cands = labelled(tmp_path)
    (tmp_path / "pics").mkdir()
    for r in lab:
        Image.new("RGB", (1024, 1024), (40, 80, 120)).save(r["picture"])
    lcsv, ccsv = tmp_path / "labelled.csv", tmp_path / "candidates.csv"
    with open(lcsv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(lab[0]))
        w.writeheader()
        w.writerows(lab)
    with open(ccsv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CAND_COLS)
        w.writeheader()
        w.writerows(cands.values())
    out_list = tmp_path / "renamed-list.csv"
    assert ren.main(["--labelled-set", str(lcsv), "--candidates", str(ccsv), "--out", str(out_list)]) == 0
    sheet = tmp_path / "batch"
    sheets.main(["build", "--pictures", str(out_list), "--seed", "9", "--out", str(sheet)])
    assert sheets.main(["check", "--out", str(sheet)]) == 0
    asked = (sheet / "sheet/sheet.csv").read_text()
    assert "a cheetah and a leopard" in asked and "cougar" not in asked
    key = json.loads((sheet / "key/key.json").read_text())
    names = list(key["pictures"])
    for i, who in enumerate(("calls-1.csv", "calls-2.csv")):
        with open(tmp_path / who, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["name", "kind", "note"])
            w.writerows((n, "wrong species", "") for n in names)
    s = ls.settle_batch({"name": "renamed-1", "key": str(sheet / "key/key.json"),
                         "calls": [str(tmp_path / "calls-1.csv"), str(tmp_path / "calls-2.csv")]})
    assert all(r["renamed"] and r["item"].endswith(r["renaming"]) for r in s["settled"].values())
    assert {r["kind"] for r in s["settled"].values()} == {"wrong_species"}
