"""Paper 1's pack (paper1_pack.py) on synthetic renders, and the renderer's joint-mode geometry (clevr_render.py) without Blender.

Needs numpy and Pillow only, so it runs on the laptop. scipy, which clevr_data.py imports for another stage, is stubbed if missing.

  python3 -m pytest tests/test_paper1_pack.py
"""
import ast
import json
import math
import sys
import types
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "2-classifier-guidance" / "scripts"
if "scipy" not in sys.modules:
    try:
        import scipy.ndimage  # noqa: F401
    except ImportError:
        scipy = types.ModuleType("scipy"); ndimage = types.ModuleType("scipy.ndimage"); ndimage.label = None
        scipy.ndimage = ndimage; sys.modules["scipy"] = scipy; sys.modules["scipy.ndimage"] = ndimage
sys.path.insert(0, str(SCRIPTS / "paper1"))
import paper1_pack as pk  # noqa: E402


def _render(renders, index, objects):
    """One synthetic joint render: objects are (colour, row0, row1, col0, col1, depth) boxes at 64 pixels."""
    name = renders.name
    img = np.full((240, 320, 3), 128, np.uint8)
    masks = []
    for colour, r0, r1, c0, c1, _ in objects:
        m = np.zeros((64, 64), np.uint8); m[r0:r1, c0:c1] = 255
        masks.append(m)
        img[int(r0 * 3.75):int(r1 * 3.75), c0 * 5:c1 * 5] = (200, 30, 30) if colour == "red" else (40, 70, 210)
    for sub in ("images", "scenes", "masks"):
        (renders / sub).mkdir(parents=True, exist_ok=True)
    stem = "%s_%06d" % (name, index)
    Image.fromarray(img).save(renders / "images" / f"{stem}.png")
    scene = {"objects": [{"color": col, "pixel_coords": [0, 0, d]} for col, *_, d in objects]}
    (renders / "scenes" / f"{stem}.json").write_text(json.dumps(scene))
    np.savez_compressed(renders / "masks" / f"{stem}.npz", masks=np.stack(masks))


def test_pack_writes_the_contract_and_counts(tmp_path):
    renders = tmp_path / "render-run" / "renders" / "joint"
    for i in range(6):
        k = i % 3 + 1
        objs = []
        for j in range(k):
            objs.append(("red", 40, 50, 5 + 10 * j, 10 + 10 * j, 5.0))                 # red low in the frame, nearer
            objs.append(("blue", 30, 41 if i % 2 == 0 else 35, 10 + 10 * j, 15 + 10 * j, 9.0))  # touching red on even scenes
        _render(renders, i, objs)
    pk.main(["--root", str(tmp_path), "--run", "pack", "--renders", "render-run/renders/joint", "--count", "6", "--k", "3"])   # no exit
    out = tmp_path / "pack"
    assert sorted(p.name for p in (out / "k3-s1").glob("count-*/*.png")) == ["000000.png", "000001.png", "000002.png"]
    assert (out / "heldout" / "count-1" / "000003.png").exists() and (out / "heldout" / "masks" / "count-1" / "000003-red.png").exists()
    assert Image.open(out / "k3-s1" / "count-2" / "000001.png").size == (64, 64)
    flags = json.loads((out / "k3-s1" / "touching.json").read_text())["files"]
    assert flags == {"count-1/000000.png": True, "count-2/000001.png": False, "count-3/000002.png": True}
    depth = json.loads((out / "depth-order.json").read_text())["per_count"]
    assert all(v["share"] == 1.0 for v in depth.values() if v["pairs"])          # the nearer red is the lower one
    assert "renders shared by SHA-256: 0" in (out / "leakage.txt").read_text()
    assert "touching share" in (out / "composition.txt").read_text()


def test_pack_exits_one_on_a_picture_shared_between_subset_and_held_out(tmp_path):
    renders = tmp_path / "r" / "joint"
    for i in range(2):
        _render(renders, i, [("red", 40, 50, 5, 10, 5.0), ("blue", 30, 41, 10, 15, 9.0)])   # identical scenes
    with pytest.raises(SystemExit) as e:
        pk.main(["--root", str(tmp_path), "--run", "pack", "--renders", "r/joint", "--count", "2", "--k", "1"])
    assert e.value.code == 1


def _renderer_functions():
    """The joint mode's pure functions from clevr_render.py, executed without Blender."""
    tree = ast.parse((SCRIPTS / "method_search" / "clevr_render.py").read_text())
    wanted = {"_corners", "_point_square_distance", "_squares_overlap", "footprints_overlap", "joint_counts"}
    ns = {"math": math}
    exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in wanted], type_ignores=[]),
                 "clevr_render_part", "exec"), ns)
    return ns


def test_renderer_has_no_syntax_blender_python_35_cannot_read():
    tree = ast.parse((SCRIPTS / "method_search" / "clevr_render.py").read_text())
    assert not [n for n in ast.walk(tree) if isinstance(n, ast.JoinedStr)]       # no f-strings
    assert not [n for n in ast.walk(tree) if isinstance(n, ast.NamedExpr)]       # no walrus


def test_footprints_touch_but_never_overlap():
    f = _renderer_functions()
    overlap = f["footprints_overlap"]
    assert not overlap(("Sphere", 0, 0, 0.35, 0), ("Sphere", 1.05, 0, 0.7, 0))    # gap exactly 0: touching allowed
    assert overlap(("Sphere", 0, 0, 0.35, 0), ("Sphere", 1.0, 0, 0.7, 0))
    half = 0.7 / math.sqrt(2)
    assert not overlap(("Cube", 0, 0, half, 0), ("Cube", 2 * half, 0, half, 0))   # faces touching
    assert overlap(("Cube", 0, 0, half, 45), ("Cube", 2 * half, 0, half, 0))      # a turned corner cuts in
    assert overlap(("Cube", 0, 0, half, 45), ("Sphere", half + 0.3, 0, 0.35, 0))  # corner reaches past half-side + radius
    assert not overlap(("Cube", 0, 0, half, 0), ("Cylinder", half + 0.35, 0, 0.35, 0))
    assert [f["joint_counts"]("cycle", i) for i in range(4)] == [(1, 1), (2, 2), (3, 3), (1, 1)]
    assert f["joint_counts"]("2,1", 7) == (2, 1)
