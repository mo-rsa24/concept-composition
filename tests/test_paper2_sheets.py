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
    assert all(len(name) == sheets.NAME_LETTERS and set(name) <= set(sheets.NAME_ALPHABET) for name in key)
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


COLLECTOR_COLUMNS = ["picture", "a", "b", "pair", "pair_type", "seed", "source", "round", "row", "made", "earlier_call",
                     "earlier_scale", "notes", "push_checker", "clip_named_boxes", "reported_apart", "unselected",
                     "also_called_in"]


def collector_list(tmp_path, rows_and_calls):
    """A list in paper2_collect.py's columns, over stand-in pictures in a folder like the cluster's."""
    src = tmp_path / "loop-12" / "z13" / "own"
    src.mkdir(parents=True)
    rows = []
    for i, (row_, call) in enumerate(rows_and_calls):
        p = src / f"seed-{6025 + i}.png"
        Image.new("RGB", (1024, 1024), (30 * i, 90, 160)).save(p)
        rows.append({"picture": str(p), "a": "cheetah", "b": "cougar", "pair": "a_cheetah__x__a_cougar",
                     "pair_type": "look-alike", "seed": 6025 + i, "source": "box", "round": "13-rest", "row": row_,
                     "made": "CO3 with each animal's own prediction inside its detected box, steps 5 to 30",
                     "earlier_call": call, "earlier_scale": "four-step (one, hybrid, two_bodies, two_clear)",
                     "notes": "one cheetah | hybrid with cougar ears", "push_checker": "none",
                     "clip_named_boxes": True, "reported_apart": False, "unselected": False, "also_called_in": ""})
    lst = tmp_path / "list.csv"
    with open(lst, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLLECTOR_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    return lst


def test_names_are_consonants_only_and_never_hold_a_seed():
    for seed in range(50):
        names = sheets.neutral_names(200, seed)
        assert len(set(names)) == 200
        assert all(len(n) == 10 and set(n) <= set(sheets.NAME_ALPHABET) for n in names)
    assert not set(sheets.NAME_ALPHABET) & set("aeiouy0123456789")


def test_ordinary_call_words_and_short_row_words_do_not_fire(tmp_path):
    # earlier_call "one" and "split", rows "both" and "plain": the very words the rules use
    lst = collector_list(tmp_path, [("both", "one"), ("plain", "split"), ("own", "hybrid"), ("co3", "two_clear")])
    out = tmp_path / "out"
    sheets.main(["build", "--pictures", str(lst), "--seed", "2", "--out", str(out)])
    rules = (out / "sheet/rules.txt").read_text()
    assert "both at once" in rules and "one plain A" in rules
    assert sheets.main(["check", "--out", str(out)]) == 0


def test_a_planted_seed_in_a_page_and_a_planted_slug_in_sheet_csv_still_fail(tmp_path):
    lst = collector_list(tmp_path, [("own", "one"), ("co3", "two_clear")])
    for name, plant in (("seed", "page"), ("slug", "csv"), ("rules", "rules")):
        out = tmp_path / f"out-{name}"
        sheets.main(["build", "--pictures", str(lst), "--seed", "0", "--out", str(out)])
        if plant == "page":
            p = next((out / "sheet/pages").glob("*.png"))
            info = PngImagePlugin.PngInfo()
            info.add_text("Comment", "drawn at 6026")
            Image.open(p).save(p, pnginfo=info)
        elif plant == "csv":
            with open(out / "sheet/sheet.csv", "a") as f:
                f.write("zzzz,a_cheetah__x__a_cougar\n")
        else:
            (out / "sheet/rules.txt").write_text(sheets.rules_text() + "drawn by the own row\n")
        assert sheets.main(["check", "--out", str(out)]) == 1, name


def test_the_searched_columns_are_named_and_the_call_columns_are_not():
    assert sheets.IDENTIFYING_COLUMNS == ("pair", "seed", "source", "round", "row", "made", "also_called_in")
    key = {"pictures": {"x": {"source": "/d/loop-12/z13/own/seed-6025.png", "a": "cat", "b": "dog", "seed": "6025",
                              "row": "own", "earlier_call": "one", "notes": "a hybrid", "pair": "a_cat__x__a_dog"}}}
    found = sheets.unblinding_strings(key)
    assert {"6025", "own", "a_cat__x__a_dog", "seed-6025.png", "z13"} <= found
    assert not {"one", "a hybrid"} & found


# ---- the rubric's versions

V1_SHA256 = "3848cc6189d9ba206ae409355d89e970b604f023a672213c28435bc2853c077c"   # batch-01's rules.txt


def test_rubric_v1_is_unchanged():
    import hashlib
    assert hashlib.sha256(sheets.rules_text(1).encode()).hexdigest() == V1_SHA256
    assert not sheets.rules_text(1).startswith("Rules v")


def test_rubric_v2_names_the_parts_and_changes_nothing_else():
    v2 = sheets.rules_text(2)
    assert v2.startswith("Rules v2\n")
    for part in ("spots or stripes", "a mane", "wings", "feathers", "a beak", "horns", "hooves", "a tail",
                 "more\nthan a small patch", "When unsure, do not flag.", "When unsure, call it two clear."):
        assert part in v2, part
    back = v2[len("Rules v2\n"):].replace(sheets.V2_FLAG, sheets.V1_FLAG).replace(sheets.V2_BORROWED, "")
    assert back == sheets.rules_text(1)           # every other word is version 1's
    import paper2_kinds
    assert paper2_kinds.kind_of([("A", True), ("B", False)]) == ("borrowed_part", 3)   # the rule itself is unchanged


def test_a_v2_sheet_records_its_version_and_checks(tmp_path):
    out = tmp_path / "out"
    sheets.main(["build", "--pictures", str(make_list(tmp_path)), "--seed", "0", "--out", str(out)])
    assert json.loads((out / "key/key.json").read_text())["rules_version"] == 2
    assert (out / "sheet/rules.txt").read_text().splitlines()[0] == "Rules v2"
    assert sheets.main(["check", "--out", str(out)]) == 0
    (out / "sheet/rules.txt").write_text(sheets.rules_text(2).replace("When unsure, do not flag.", "Flag freely."))
    assert sheets.main(["check", "--out", str(out)]) == 1


def test_a_v1_sheet_still_checks_even_with_no_version_in_its_key(tmp_path):
    out = tmp_path / "out"
    sheets.main(["build", "--pictures", str(make_list(tmp_path)), "--seed", "0", "--out", str(out),
                 "--rules-version", "1"])
    key = json.loads((out / "key/key.json").read_text())
    key.pop("rules_version")                       # as batch-01's key was written
    (out / "key/key.json").write_text(json.dumps(key))
    assert sheets.main(["check", "--out", str(out)]) == 0
    (out / "sheet/rules.txt").write_text(sheets.rules_text(2))   # a v2 rubric on a v1 sheet is caught
    assert sheets.main(["check", "--out", str(out)]) == 1


def test_an_unknown_version_is_refused():
    with pytest.raises(SystemExit, match="no rubric version 3"):
        sheets.rules_text(3)
