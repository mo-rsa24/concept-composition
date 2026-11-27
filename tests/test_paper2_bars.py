"""Paper 2's species judge bars: every constant at the plan's value, every pair at the plan's type, the ImageNet table checked.

The tests that compare the code with plans/paper-2-dials/plans/02-species-judge.md read the plan itself, so a bar or a
pair type that drifts from it fails; plans/ is not committed, so those run on the laptop and skip where the plan is
absent (the cluster). Every other test checks the committed code alone and runs everywhere.

    python3 -m pytest tests/test_paper2_bars.py -q
"""
from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "2-classifier-guidance/scripts/paper2"))
sys.path.insert(0, str(REPO / "scripts/shared"))

import paper2_bars as bars  # noqa: E402
from disallowed_subjects import is_allowed  # noqa: E402

PLAN_FILE = REPO / "plans/paper-2-dials/plans/02-species-judge.md"
PLAN = PLAN_FILE.read_text() if PLAN_FILE.exists() else None
needs_plan = pytest.mark.skipif(PLAN is None, reason="the plan is not on this machine (plans/ is not committed): laptop only")


def section(start: str, end: str) -> str:
    return PLAN[PLAN.index(start):PLAN.index(end)]


@needs_plan
def test_every_bar_in_the_plans_table_is_in_the_code_at_its_value():
    table = section("### The bars", "### What each result means")
    found = re.findall(r"`([A-Z_]+) = ([^`]+)`", table)
    assert len(found) == 17
    for name, value in found:
        assert getattr(bars, name) == eval(value), name       # values are Python literals: 0.80, 54, True


@needs_plan
def test_every_pair_in_the_plans_table_has_its_type():
    table = section("### The pairs", "**The look-alike supply is thin.**")
    plan_pairs = {}
    for row in table.splitlines():
        if not row.startswith("| ") or " and " not in row.split("|")[1]:
            continue
        kind = "look-alike" if row.split("|")[2].strip().startswith("look-alike") else "unlike"
        for a, b in re.findall(r"(\w+) and (\w+)", row.split("|")[1]):
            plan_pairs[(a, b)] = kind
    assert len(plan_pairs) == 23
    assert {(a, b) for a, b, _ in bars.PAIRS} == set(plan_pairs)
    for (a, b), kind in plan_pairs.items():
        assert bars.pair_type(a, b) == kind, (a, b)
    assert bars.REPORTED_APART == {("crow", "raven")}


def test_new_lookalike_pairs_are_lookalike_allowed_and_new():
    pooled = json.loads((REPO / "artifacts/notes/progress/leakage/10-co-occurrence-dial/pairs.json").read_text())["pairs"]
    taken = {frozenset((a, b)) for a, b, _ in bars.PAIRS}
    taken |= {frozenset(re.findall(r"(?:an?) (\w+)", p)) for p in pooled}
    assert len(bars.NEW_LOOKALIKE_PAIRS) == 8 and list(bars.NEW_LOOKALIKE_SEEDS) == list(range(8001, 8009))
    for a, b in bars.NEW_LOOKALIKE_PAIRS:
        assert bars.pair_type(a, b) == "look-alike", (a, b)
        assert is_allowed(f"a {a} and a {b}"), (a, b)
        assert frozenset((a, b)) not in taken, (a, b)


def test_grids_renaming_and_old_judge_match_the_plan():
    assert bars.MARGIN_GRID == (0.05, 0.10, 0.15, 0.20, 0.30)
    assert bars.FLAG_SHARE_GRID == (0.15, 0.25, 0.35)
    assert bars.BORROW_GAP_GRID == (0.1, 0.2, 0.3) and bars.N_REFS_GRID == (8, 16)
    assert bars.YES_THRESHOLD_GRID == (0.3, 0.5, 0.7)
    assert bars.CROP_CHECK_SETTINGS == {"margin": 0.15, "flag_share": 0.25, "yes_threshold": 0.5,
                                        "borrow_gap": 0.2, "n_refs": 8}
    assert bars.RENAMING == {"cougar": "leopard", "goose": "swan", "fox": "coyote", "flamingo": "spoonbill"}
    assert set(bars.OLD_JUDGE_KINDS.values()) <= set(bars.ALL_KINDS)
    assert bars.kind_of([("A", False), ("B", True)]) == ("borrowed_part", 3)   # the seven rules, one copy


def test_every_species_has_an_order_and_an_imagenet_answer():
    species = {s for a, b, _ in bars.PAIRS for s in (a, b)} | {s for p in bars.NEW_LOOKALIKE_PAIRS for s in p}
    species |= set(bars.RENAMING.values())
    for s in species:
        assert s in bars.ORDER, s
        assert (s in bars.IMAGENET_CLASSES) != (s in bars.IMAGENET_UNCOVERED), s
    for s, classes in bars.IMAGENET_CLASSES.items():
        assert classes and all(0 <= i < 1000 for i in classes), s
        for i in classes:
            assert i in bars.IMAGENET_CLASS_NAMES or i in bars.DOG_BREEDS, (s, i)


def test_coverage_counts():
    c = bars.coverage()
    plan = [r for r in c["pairs"] if r["where"] == "plan"]
    assert sum(r["covered"] for r in plan) == 13 and len(plan) == 23


@pytest.mark.skipif(importlib.util.find_spec("torchvision") is None, reason="needs torchvision's list of the 1,000 classes")
def test_imagenet_names_match_torchvisions_list():
    from torchvision.models._meta import _IMAGENET_CATEGORIES as names
    for i, name in bars.IMAGENET_CLASS_NAMES.items():
        assert names[i] == name, (i, names[i], name)
    assert names[151] == "Chihuahua" and names[268] == "Mexican hairless" and names[269] == "timber wolf"


@needs_plan
def test_the_crop_check_settings_are_the_plans():
    assert "m 0.15, p 0.25, t 0.5, b 0.2 with 8 references" in PLAN


def test_pair_types_hold_without_the_plan():
    """The same types as the plan's table, from the committed code alone, so the cluster checks them too."""
    lookalike = {("cheetah", "cougar"), ("duck", "goose"), ("crow", "raven"), ("cat", "dog"), ("tiger", "dog"),
                 ("cat", "fox")}
    assert len(bars.PAIRS) == 23
    for a, b, _ in bars.PAIRS:
        assert bars.pair_type(a, b) == ("look-alike" if (a, b) in lookalike else "unlike"), (a, b)


def test_the_lookalike_minimum_is_per_kind_and_lowered_only_for_extra_subject():
    assert set(bars.LOOKALIKE_MIN_BY_KIND) == set(bars.KINDS)
    assert bars.LOOKALIKE_MIN_BY_KIND["extra_subject"] == 8
    assert all(v == bars.LOOKALIKE_PER_KIND_MIN == 20 for k, v in bars.LOOKALIKE_MIN_BY_KIND.items() if k != "extra_subject")
