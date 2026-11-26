"""Paper 2's attention fallback: CO3 builds without xformers on PyTorch's own attention, and says so.

    python3 -m pytest tests/test_paper2_co3_patch.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "2-classifier-guidance/scripts/paper2"))

import paper2_co3_patch as patch  # noqa: E402


def stand_in_pipelines():
    calls = []

    class Pipeline:
        def enable_xformers_memory_efficient_attention(self, *a, **k):
            calls.append("xformers switched on")

    class XLPipeline(Pipeline):
        pass

    return Pipeline, XLPipeline, calls


def test_without_xformers_the_switch_prints_one_line_and_does_nothing(capsys):
    Pipeline, XLPipeline, calls = stand_in_pipelines()
    got = patch.use_pytorch_attention_if_no_xformers([Pipeline], available=False)
    assert got == "pytorch-sdpa"
    XLPipeline().enable_xformers_memory_efficient_attention()     # what CO3's Co3.__init__ calls
    assert calls == []
    assert capsys.readouterr().out.strip() == patch.MESSAGE


def test_with_xformers_nothing_changes(capsys):
    Pipeline, XLPipeline, calls = stand_in_pipelines()
    before = Pipeline.enable_xformers_memory_efficient_attention
    got = patch.use_pytorch_attention_if_no_xformers([Pipeline], available=True)
    assert got == "xformers"
    assert Pipeline.enable_xformers_memory_efficient_attention is before
    XLPipeline().enable_xformers_memory_efficient_attention()
    assert calls == ["xformers switched on"]
    assert capsys.readouterr().out == ""


def test_a_subclass_with_its_own_switch_is_patched_when_named(capsys):
    Pipeline, _, calls = stand_in_pipelines()

    class OwnSwitch(Pipeline):
        def enable_xformers_memory_efficient_attention(self, *a, **k):
            calls.append("own switch")

    patch.use_pytorch_attention_if_no_xformers([Pipeline, OwnSwitch], available=False)
    OwnSwitch().enable_xformers_memory_efficient_attention()
    assert calls == [] and patch.MESSAGE in capsys.readouterr().out


def test_the_laptop_has_no_xformers_and_the_check_says_so():
    import importlib.util
    assert patch.xformers_importable() == (importlib.util.find_spec("xformers") is not None)


def test_every_co3_entry_point_calls_the_patch_before_building_co3():
    for name in ("paper2_co3_load.py", "paper2_identity_dial.py"):
        text = (REPO / "2-classifier-guidance/scripts/paper2" / name).read_text()
        call = text.index("use_pytorch_attention_if_no_xformers()")
        assert call < text.index("import paper2_rows as rows"), name
