"""Paper 2's first-result bars and rule (paper2_first_result_bars.py). Needs no torch, so it runs on the laptop.

  python3 -m pytest tests/test_paper2_first_result_bars.py
"""
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "2-classifier-guidance" / "scripts" / "paper2"))
import paper2_first_result_bars as fb  # noqa: E402

PLAN = REPO / "plans/paper-2-dials/plans/03-identity-dial-first-result.md"
TABLE = {"FIRST_TWO_CLEAR_LEAD_MIN": 0.20, "FIRST_MCNEMAR_P_MAX": 0.05, "FIRST_BORROWED_MUST_FALL": True,
         "FIRST_RANDOM_DIAL_CLOSE": 0.05, "FIRST_NO_LEAD_MAX": 0.05, "FIRST_GAMING_GAP_MAX_RISE": 0.10,
         "FIRST_REDRAW_KIND_CHANGES_MAX": 2, "FIRST_GPU_HOURS_STOP": 24}


def test_every_constant_matches_the_plans_table():
    assert {k: getattr(fb, k) for k in TABLE} == TABLE


def test_the_plans_bars_table_is_what_the_code_holds():
    if not PLAN.exists():
        pytest.skip("plans/ is not in this checkout")
    text = PLAN.read_text().split("### The bars", 1)[1].split("### What each result means", 1)[0]
    found = dict(re.findall(r"^\| `(FIRST_[A-Z_]+) = ([^`]+)` \|", text, flags=re.M))
    assert set(found) == set(TABLE), "the plan's bars table and this test name different constants"
    for name, value in found.items():
        assert getattr(fb, name) == eval(value), name   # the plan writes Python literals: 0.20, True, 24


def test_p_values_match_the_review_files_examples():
    assert fb.one_sided_mcnemar_p(15, 5) == pytest.approx(0.021, abs=0.0005)    # "15 towards the dial and 5 away"
    assert fb.one_sided_mcnemar_p(12, 2) == pytest.approx(0.0065, abs=0.00005)  # "12 towards and 2 away"
    assert fb.one_sided_mcnemar_p(3, 0) == pytest.approx(0.125)                 # "3 towards the dial and none away"
    assert fb.one_sided_mcnemar_p(4, 0) >= 0.05 and fb.one_sided_mcnemar_p(5, 0) < 0.05   # under 5 pictures cannot pass
    assert fb.one_sided_mcnemar_p(0, 0) == 1.0


def test_the_rule():
    n = fb.MATCHED_PICTURES
    yes = dict(lead=10 / n, p=0.021, dial_borrowed=5 / n, co3_borrowed=12 / n, random_minus_co3=2 / n)
    assert fb.answer(**yes) == "yes"                                            # 10 more in 48 is 0.208
    assert fb.answer(**{**yes, "lead": 0.20}) == "yes"                          # at the bar meets it
    assert fb.answer(**{**yes, "random_minus_co3": -0.05}) == "yes"             # at the band's edge meets it
    assert fb.answer(**{**yes, "random_minus_co3": 3 / n}) == "inconclusive"    # the random dial moved
    assert fb.answer(**{**yes, "p": 0.05}) == "inconclusive"                    # p must be below 0.05
    assert fb.answer(**{**yes, "dial_borrowed": 12 / n}) == "inconclusive"      # borrowed parts must fall
    assert fb.answer(**{**yes, "lead": 9 / n}) == "inconclusive"                # 0.1875 is between the bars
    assert fb.answer(**{**yes, "lead": 2 / n}) == "no"                          # 0.042 is at most 0.05
    assert fb.answer(**{**yes, "lead": 0.05}) == "no"
    assert fb.answer(**{**yes, "lead": 3 / n}) == "inconclusive"                # 0.0625 is above 0.05
    assert fb.answer(**{**yes, "lead": -0.1}) == "no"
    verdict, reason = fb.answer_and_reason(**{**yes, "p": 0.08, "random_minus_co3": 0.1})
    assert verdict == "inconclusive" and "p 0.0800" in reason and "random dial moved" in reason


def test_gaming_alarm_reads_the_rise_from_co3():
    assert not fb.gaming_alarm(0.10, 0.20)          # a rise of exactly 0.10 is allowed
    assert fb.gaming_alarm(0.10, 0.21)
    assert fb.gaming_alarm(-0.05, 0.10)             # a gap already open counts from CO3
    assert not fb.gaming_alarm(0.30, 0.10)


def test_redraw_band_and_the_gpu_stop():
    assert fb.redraw_band_needed(2) is None
    assert fb.redraw_band_needed(3) == pytest.approx(3 / 32)
    assert not fb.projects_past_stop(17.4) and not fb.projects_past_stop(24) and fb.projects_past_stop(24.1)
