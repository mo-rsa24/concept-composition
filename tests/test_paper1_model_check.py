"""Paper 1's model-file check (paper1_model_check.py) on stand-in files. Needs no torch, so it runs on the laptop.

  python3 -m pytest tests/test_paper1_model_check.py
"""
import hashlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "2-classifier-guidance" / "scripts" / "paper1"))
import paper1_model_check as mc  # noqa: E402


def _expert(folder, payload, record=None):
    folder.mkdir(parents=True)
    snap = folder / mc.EXPERT_SNAPSHOT
    snap.write_bytes(payload)
    digest = record or hashlib.sha256(payload).hexdigest()
    (folder / "pod.sha256").write_text(f"{digest}  models/{folder.name}/{mc.EXPERT_SNAPSHOT}\n")


def _setup(tmp_path, monkeypatch, blue_record=None, empty_payload=b"empty"):
    _expert(tmp_path / "r-full", b"red")
    _expert(tmp_path / "b-full", b"blue", blue_record)
    empty = tmp_path / "empty.pkl"
    empty.write_bytes(empty_payload)
    monkeypatch.setattr(mc, "EXPERTS", {"red": tmp_path / "r-full", "blue": tmp_path / "b-full"})
    monkeypatch.setattr(mc, "EMPTY_NET", empty)
    monkeypatch.setattr(mc, "EMPTY_BYTES", len(b"empty"))
    monkeypatch.setattr(mc, "EMPTY_SHA256", hashlib.sha256(b"empty").hexdigest())


def test_all_three_match(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    with pytest.raises(SystemExit) as e:
        mc.main(["--root", str(tmp_path), "--run", "check"])
    assert e.value.code == 0
    text = (tmp_path / "check" / "models.txt").read_text()
    assert "passes: True" in text and "no training-state-*.pt" in text


def test_a_mismatch_or_a_changed_empty_model_fails(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch, blue_record="0" * 64)
    with pytest.raises(SystemExit) as e:
        mc.main(["--root", str(tmp_path), "--run", "check"])
    assert e.value.code == 1 and "MISMATCH" in (tmp_path / "check" / "models.txt").read_text()


def test_recorded_values_are_the_ones_relayed():
    assert mc.EMPTY_BYTES == 248_538_254
    assert mc.EMPTY_SHA256 == "5e7905efd3b723d79507dde0af9d5d63d959738a2e08ab234646101ece8fb272"
    assert str(mc.EMPTY_NET) == "/datasets/mmolefe/poe_repair_min/loop-6/m7/full/models/empty/network-snapshot-0002000-0.100.pkl"
