"""Paper 1's first-result bars and reader (paper1_first_bars.py). Needs no torch, so it runs on the laptop.

  python3 -m pytest tests/test_paper1_first_bars.py
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "2-classifier-guidance" / "scripts" / "paper1"))
import paper1_first_bars as fb  # noqa: E402


def test_every_constant_matches_the_plans_table():
    table = {"FIRST_TOUCH_SEEDS": 96, "FIRST_APART_SEEDS": 96, "FIRST_GAIN_MIN": 15, "FIRST_P_MAX": 0.05, "FIRST_RANDOM_BAND": 5,
             "FIRST_NULL_GAIN_MAX": 5, "FIRST_REAL_RIGHT_MIN": 82, "FIRST_PUSH_MIN_GREY": 1, "FIRST_DIRECTION_COS_MIN": 0.10,
             "FIRST_W_ANCHOR_GREY": 5, "FIRST_CONTACT_DROP_MAX": 0.10, "FIRST_GPU_HOURS_STOP": 4, "FIRST_GPU_HOURS_STOP_ALIGNED": 7}
    assert {k: getattr(fb, k) for k in table} == table


def test_p_values_match_the_verdict_files_examples():
    assert fb.one_sided_binomial_p(20, 5) == pytest.approx(0.002, abs=0.0005)
    assert fb.one_sided_binomial_p(30, 15) == pytest.approx(0.018, abs=0.0005)
    assert fb.one_sided_binomial_p(40, 25) == pytest.approx(0.041, abs=0.0005)
    assert fb.one_sided_binomial_p(43, 28) == pytest.approx(0.048, abs=0.0005)
    assert fb.one_sided_binomial_p(10, 5) == pytest.approx(0.151, abs=0.0005)
    assert fb.one_sided_binomial_p(5, 0) == pytest.approx(0.031, abs=0.0005)


def test_the_rule():
    assert fb.answer(15, 0.002, 3) == "yes"
    assert fb.answer(15, 0.002, -5) == "yes"           # at the band's edge meets it
    assert fb.answer(15, 0.002, 6) == "inconclusive"   # the random push moved too
    assert fb.answer(15, 0.05, 0) == "inconclusive"    # p must be below 0.05
    assert fb.answer(10, 0.01, 0) == "inconclusive"
    assert fb.answer(5, 0.031, 0) == "no"              # at or below 5 is a no, whatever p
    assert fb.answer(-3, 1.0, 0) == "no"


def _call(right, k=2, side="touching"):
    if right:
        return {"red_count": str(k), "blue_count": str(k), "blended_band": "no", "occlusion_correct": "yes" if side == "touching" else ""}
    return {"red_count": str(k), "blue_count": str(k), "blended_band": "yes", "occlusion_correct": "no"}


def _worked_example(real_right=None):
    """The plan's worked example on 96 touching seeds (ours wins 20, loses 5, both right on 30; random right on 37), with
    real_right of 96 real touching scenes called right when given."""
    key, settled, n = [], {}, 0
    for i in range(96):
        k, seed = i % 3 + 1, 20000 + i
        both, ours_only, region_only = i < 30, 30 <= i < 50, 50 <= i < 55
        rows = [("ours", both or ours_only), ("region", both or region_only), ("randompush", i < 37)]
        if real_right is not None:
            rows.append(("real", i < real_right))
        for row, ok in rows:
            name = f"p{n}.png"; n += 1
            key.append({"picture": name, "row": row, "seed": seed, "count": k, "side": "touching", "path": "x"})
            settled[name] = _call(ok, k)
    return fb.read(settled, key)


def test_too_few_real_scenes_right_makes_the_answer_inconclusive():
    low = _worked_example(real_right=81)
    assert low["answer"] == "inconclusive" and low["reason"] == "could not measure it: the rubric at 64 pixels"
    assert low["rubric_readable"] is False
    ok = _worked_example(real_right=82)
    assert ok["answer"] == "yes" and ok["rubric_readable"] is True
    assert fb.answer_and_reason(5, 0.031, 0, real_right=81) == ("inconclusive", "could not measure it: the rubric at 64 pixels")
    assert fb.answer(5, 0.031, 0, real_right=82) == "no"


def test_reader_counts_pairs_and_answers():
    # 96 touching seeds: ours right on 50, region right on 35; ours wins 20, loses 5 (both right on 30); random right on 37
    key, settled, n = [], {}, 0
    for i in range(96):
        k, seed = i % 3 + 1, 20000 + i
        both, ours_only, region_only = i < 30, 30 <= i < 50, 50 <= i < 55
        for row, ok in (("ours", both or ours_only), ("region", both or region_only), ("randompush", i < 37)):
            name = f"p{n}.png"; n += 1
            key.append({"picture": name, "row": row, "seed": seed, "count": k, "side": "touching", "path": "x"})
            settled[name] = _call(ok, k)
    out = fb.read(settled, key)
    assert out["two_by_two"] == {"ours right, region right": 30, "ours right, region wrong": 20,
                                 "ours wrong, region right": 5, "ours wrong, region wrong": 41}
    assert out["gain"] == 15 and out["p_one_sided"] == pytest.approx(0.002, abs=0.0005)
    assert out["random_push_minus_region"] == 2 and out["answer"] == "yes"
    assert out["rows"]["ours"]["touching"]["right"] == 50


def test_occlusion_matters_only_on_touching_pictures():
    call = {"red_count": "1", "blue_count": "1", "blended_band": "no", "occlusion_correct": ""}
    assert fb.is_right(call, 1, "apart") and not fb.is_right(call, 1, "touching")
    assert not fb.is_right({"red_count": "x", "blue_count": "1", "blended_band": "no", "occlusion_correct": "yes"}, 1, "touching")


def test_choose_w_keeps_contact_then_most_right_then_smaller_w():
    key, calls, n = [], {}, 0
    rights = {"region": 10, "win15-22-w1": 20, "win15-22-w2": 24, "win15-22-w4": 24, "win10-22-w1": 24}
    for row, r in rights.items():
        for i in range(32):
            name = f"p{n}.png"; n += 1
            key.append({"picture": name, "row": row, "seed": 10000 + i, "count": 1, "side": "touching", "path": f"/x/{row}/count-1/{i}.png"})
            calls[name] = _call(i < r, 1)
    contact = {"region": 0.80, "win15-22-w1": 0.78, "win15-22-w2": 0.72, "win15-22-w4": 0.60, "win10-22-w1": 0.75}
    chosen, table = fb.choose_setting(calls, key, contact)
    assert chosen["row"] == "win10-22-w1" or chosen["row"] == "win15-22-w2"
    assert chosen["w"] == 1.0 and chosen["window"] == "10-22"      # 24 right at w = 1 beats 24 at w = 2; w = 4 dropped contact by 0.20
    assert not [t for t in table["settings"] if t["row"] == "win15-22-w4"][0]["allowed"]
    none, _ = fb.choose_setting(calls, key, {k: (0.80 if k == "region" else 0.5) for k in contact})
    assert none is None
