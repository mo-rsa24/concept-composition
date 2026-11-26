"""Paper 2's seven rules, the crop, and each crop judge's dry run.

The five worked examples are the species judge plan's own ("What happens"), in its order; the
setup plan's task 2 passes when they return borrowed part, two of one kind, extra subject,
extra subject and wrong species. Runs on the laptop with no torch:

    python3 -m pytest tests/test_paper2_kinds.py -q
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

from paper2_common import crop_body, redecide, species_from_slug, write_m14_list  # noqa: E402
from paper2_kinds import (A, B, BORROWED_PART, EXTRA_SUBJECT, HYBRID, MIXED, OTHER,  # noqa: E402
                          OTHER_KIND, TWO_CLEAR, TWO_OF_ONE_KIND, WRONG_SPECIES, kind_of)

# (asked, bodies as (call, flag), expected kind, expected rule), in the plan's order.
WORKED = [
    ("a cheetah and a cougar: cheetah, cougar flagged with cheetah spots",
     [(A, False), (B, True)], BORROWED_PART, 3),
    ("a cheetah and a cougar: cheetah, cheetah",
     [(A, False), (A, False)], TWO_OF_ONE_KIND, 6),
    ("a boar and a pheasant: boar, pheasant, pheasant",
     [(A, False), (B, False), (B, False)], EXTRA_SUBJECT, 1),
    ("a boar and a pheasant: boar, pheasant, a hen (other)",
     [(A, False), (B, False), (OTHER, False)], EXTRA_SUBJECT, 1),
    ("a boar and a pheasant: boar, a hen (other)",
     [(A, False), (OTHER, False)], WRONG_SPECIES, 2),
]


def test_five_worked_examples_in_order():
    got = [kind_of(bodies)[0] for _, bodies, _, _ in WORKED]
    assert got == [BORROWED_PART, TWO_OF_ONE_KIND, EXTRA_SUBJECT, EXTRA_SUBJECT, WRONG_SPECIES]


@pytest.mark.parametrize("label,bodies,kind,rule", WORKED, ids=[w[0] for w in WORKED])
def test_worked_example_rule(label, bodies, kind, rule):
    assert kind_of(bodies) == (kind, rule)


@pytest.mark.parametrize("bodies,kind,rule", [
    ([(A, False), (B, False)], TWO_CLEAR, 4),
    ([(MIXED, False)], HYBRID, 5),
    ([(A, False), (MIXED, False)], HYBRID, 5),
    ([(B, True), (B, False), (B, False)], TWO_OF_ONE_KIND, 6),
    ([(A, False)], OTHER_KIND, 7),
    ([], OTHER_KIND, 7),
    ([(MIXED, False), (MIXED, False)], HYBRID, 5),
    ([(A, True), (B, True), (MIXED, False)], EXTRA_SUBJECT, 1),
])
def test_other_rules(bodies, kind, rule):
    assert kind_of(bodies) == (kind, rule)


def test_unknown_call_is_refused():
    with pytest.raises(ValueError):
        kind_of([("cat", False)])


def test_species_from_slug():
    assert species_from_slug("a_duck__x__a_goose") == ("duck", "goose")
    assert species_from_slug("an_eagle__x__a_fox") == ("eagle", "fox")
    assert species_from_slug("a_hippopotamus__x__a_swallow") == ("hippopotamus", "swallow")


def test_crop_margin_and_paint_out():
    # A white picture with a red left body and a blue right body overlapping it.
    im = Image.new("RGB", (1000, 1000), (255, 255, 255))
    for x in range(100, 500):
        for y in range(100, 600):
            im.putpixel((x, y), (255, 0, 0))
    boxes = [[100, 100, 500, 600], [400, 200, 900, 700]]
    crop, crop_box = crop_body(im, boxes, 0)
    # 10% of 400 wide and 500 high on each side.
    assert crop_box == [60, 50, 540, 650]
    assert crop.size == (480, 600)
    # Inside body 0's box but also inside body 1's: painted the picture's mean colour.
    mean = crop.getpixel((420 - 60, 300 - 50))
    assert mean != (255, 0, 0) and mean != (255, 255, 255)
    # Inside body 0's box and outside body 1's: untouched.
    assert crop.getpixel((200 - 60, 150 - 50)) == (255, 0, 0)


def test_body_wholly_inside_another_keeps_it():
    im = Image.new("RGB", (400, 400), (10, 200, 10))
    boxes = [[100, 100, 200, 200], [0, 0, 400, 400]]
    crop, _ = crop_body(im, boxes, 0)
    assert all(px == (10, 200, 10) for px in (crop.getpixel((x, y)) for x in range(crop.width) for y in range(crop.height)))


JUDGES = ["clip", "imagenet", "dac", "vlm"]
M14 = REPO / "outputs/loop-6/m14/blind"


@pytest.mark.skipif(not (M14 / "key.json").exists(), reason="joint-prompt pictures not on this machine")
@pytest.mark.parametrize("judge", JUDGES)
def test_judge_dry_run_on_two_pictures(judge, tmp_path):
    pics = tmp_path / "list.csv"
    write_m14_list(pics, 2, seed=0)
    out = tmp_path / judge
    subprocess.run([sys.executable, str(PAPER2 / f"paper2_judge_{judge}.py"), "--pictures", str(pics),
                    "--out", str(out), "--dry-run"], check=True)
    calls = json.loads((out / "calls.json").read_text())
    assert len(calls["pictures"]) == 2
    for p in calls["pictures"]:
        assert p["kind"] in {TWO_CLEAR, BORROWED_PART, HYBRID, TWO_OF_ONE_KIND, WRONG_SPECIES,
                             EXTRA_SUBJECT, OTHER_KIND}
        for body in p["bodies"]:
            assert (out / body["crop"]).exists()
    # The stand-in gives every picture the same fixed scores, so both pictures get one kind.
    assert calls["pictures"][0]["kind"] == calls["pictures"][1]["kind"]
    settings = json.loads((out / "settings.json").read_text())
    assert settings["dry_run"] is True
    assert (out / "settings.json").stat().st_mtime <= (out / "calls.json").stat().st_mtime

    # Re-deciding from the stored raw scores at the same settings changes nothing.
    mod = __import__(f"paper2_judge_{judge}")
    judge_cls = next(v for k, v in vars(mod).items() if k.endswith("Judge") and k != "Judge")
    again = redecide(judge_cls.__new__(judge_cls), calls["pictures"], calls["settings"])
    assert [p["kind"] for p in again] == [p["kind"] for p in calls["pictures"]]
    assert [p["two_clear_score"] for p in again] == [p["two_clear_score"] for p in calls["pictures"]]


# One body leaning to A, and a second body leaning to B, in each judge's raw-score shape.
BODY_A = {"clip": {"p_a": 0.9, "p_b": 0.1, "p_neither_best": 0.0},
          "imagenet": {"p_a": 0.9, "p_b": 0.1, "p_neither_best": 0.0},
          "dac": {"sim_a": 0.7, "sim_b": 0.2, "sim_neither_best": None},
          "vlm": {"yes_a": 0.9, "yes_b": 0.1, "yes_parts": 0.0, "parts_of": "B"}}
BODY_B = {"clip": {"p_a": 0.1, "p_b": 0.9, "p_neither_best": 0.0},
          "imagenet": {"p_a": 0.1, "p_b": 0.9, "p_neither_best": 0.0},
          "dac": {"sim_a": 0.2, "sim_b": 0.7, "sim_neither_best": None},
          "vlm": {"yes_a": 0.1, "yes_b": 0.9, "yes_parts": 0.0, "parts_of": "A"}}


@pytest.mark.parametrize("judge", JUDGES)
def test_fewer_than_two_bodies_scores_zero(judge):
    mod = __import__(f"paper2_judge_{judge}")
    judge_cls = next(v for k, v in vars(mod).items() if k.endswith("Judge") and k != "Judge")
    j = judge_cls.__new__(judge_cls)
    assert j.two_clear_score([{"raw": BODY_A[judge], "call": "A", "flag": False}], {}) == 0.0
    assert j.two_clear_score([], {}) == 0.0
    two = [{"raw": BODY_A[judge], "call": "A", "flag": False},
           {"raw": BODY_B[judge], "call": "B", "flag": False}]
    assert j.two_clear_score(two, {}) > 0.0
