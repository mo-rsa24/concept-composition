"""Paper 2's adapter for the old judge: its label and verdict copied faithfully, its kinds mapped by the plan's table.

    python3 -m pytest tests/test_paper2_judge_old.py -q

Runs on the laptop; the real namer (CLIP through species_judge.Namer) runs on the node.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
PAPER2 = REPO / "2-classifier-guidance/scripts/paper2"
sys.path.insert(0, str(PAPER2))

import paper2_judge_old as old  # noqa: E402
from paper2_common import write_m14_list  # noqa: E402

SOURCE = (REPO / "3-understanding-compositionality/scripts/species_judge.py").read_text()


def test_the_copied_lines_still_read_the_same_in_species_judge():
    for line in ('if ph >= max(pa, pb) or abs(pa - pb) < HYBRID_MARGIN:',
                 'label = a if pa > pb else b',
                 'verdict = "hybrid" if labels == ["hybrid"] else "one_body"',
                 'elif "hybrid" in labels:',
                 'elif a in labels and b in labels:',
                 'verdict = "composed"',
                 'verdict = "two_of_one"',
                 'HYBRID_MARGIN = 0.10'):
        assert line in SOURCE, line


@pytest.mark.parametrize("n,labels,expected", [
    (1, ["a cat"], "one_body"), (1, ["hybrid"], "hybrid"), (0, [], "one_body"),
    (2, ["a cat", "a dog"], "composed"), (2, ["a cat", "a cat"], "two_of_one"),
    (3, ["a cat", "a dog", "hybrid"], "hybrid"), (3, ["a cat", "a dog", "a dog"], "composed"),
])
def test_verdict(n, labels, expected):
    assert old.verdict(n, labels, "a cat", "a dog") == expected


def test_body_label_and_mapping():
    assert old.body_label(0.9, 0.05, 0.05, "a cat", "a dog", 0.10) == "a cat"
    assert old.body_label(0.45, 0.40, 0.15, "a cat", "a dog", 0.10) == "hybrid"   # within the margin
    assert old.body_label(0.3, 0.2, 0.5, "a cat", "a dog", 0.10) == "hybrid"      # the half-and-half text wins
    from paper2_bars import OLD_JUDGE_KINDS
    assert OLD_JUDGE_KINDS == {"composed": "two_clear", "two_of_one": "two_of_one_kind", "hybrid": "hybrid",
                               "one_body": "other"}


def test_two_clear_score():
    mk = lambda pa, pb: {"raw": {"p_a": pa, "p_b": pb}}
    assert old.two_clear_score([mk(0.9, 0.1)]) == 0.0
    assert old.two_clear_score([mk(0.9, 0.1), mk(0.2, 0.7)]) == 0.7


@pytest.mark.skipif(not (REPO / "outputs/loop-6/m14/blind/key.json").exists(), reason="joint-prompt pictures not here")
def test_dry_run_writes_the_shared_calls_json(tmp_path):
    lst = tmp_path / "list.csv"
    write_m14_list(lst, 2, seed=0)
    r = subprocess.run([sys.executable, str(PAPER2 / "paper2_judge_old.py"), "--pictures", str(lst),
                        "--out", str(tmp_path / "out"), "--dry-run"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    calls = json.loads((tmp_path / "out/calls.json").read_text())
    assert [p["kind"] for p in calls["pictures"]] == ["two_clear", "two_clear"]
    assert [p["old_verdict"] for p in calls["pictures"]] == ["composed", "composed"]
    assert all((tmp_path / "out" / b["crop"]).exists() for p in calls["pictures"] for b in p["bodies"])
