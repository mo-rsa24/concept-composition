"""Paper 2's follow-on judges, DreamSim against SDXL references and BioCLIP: their decide rules, a dry run through the
shared loop, the reference job list, and choose re-deciding them with no model. Laptop only.

    python3 -m pytest tests/test_paper2_followon_judges.py -q
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "2-classifier-guidance/scripts/paper2"))

import paper2_bars as bars  # noqa: E402
import paper2_judge_bioclip as bio  # noqa: E402
import paper2_judge_dreamsim as ds  # noqa: E402
import paper2_judge_lists as jl  # noqa: E402
import paper2_render_references as refs  # noqa: E402


def two_pictures(tmp: Path) -> Path:
    (tmp / "pics").mkdir(parents=True)
    rows = ["picture,a,b"]
    for i, (a, b) in enumerate([("cat", "dog"), ("lion", "tiger")]):
        p = tmp / "pics" / f"p{i}.png"
        Image.new("RGB", (1024, 1024), (100 + 40 * i, 90, 80)).save(p)
        rows.append(f"{p},{a},{b}")
    (tmp / "list.csv").write_text("\n".join(rows) + "\n")
    return tmp / "list.csv"


def test_dreamsim_decide_rule():
    d = ds.decide_by_similarity
    assert d({"sim_a": 0.30, "sim_b": 0.35}, 0.04, 0.10, 0.4) == ("other", False)       # under the floor
    assert d({"sim_a": 0.30, "sim_b": 0.35}, 0.04, 0.10, 0.0) == ("B", True)            # floor off: lead 0.05 < 0.10
    assert d({"sim_a": 0.62, "sim_b": 0.60}, 0.04, 0.10, 0.4) == ("mixed", False)
    assert d({"sim_a": 0.75, "sim_b": 0.60}, 0.04, 0.10, 0.4) == ("A", False)
    bodies = [{"raw": {"sim_a": 0.72, "sim_b": 0.55}}, {"raw": {"sim_a": 0.50, "sim_b": 0.70}}]
    assert ds.lead_two_clear(bodies) == pytest.approx(0.17)
    assert ds.lead_two_clear(bodies[:1]) == 0.0
    assert ds.DEFAULTS == {"margin": ds.MARGIN_GRID[2], "flag_gap": ds.FLAG_GAP_GRID[1], "floor": ds.FLOOR_GRID[1]}


@pytest.mark.parametrize("module,name", [(ds, "dreamsim"), (bio, "bioclip")])
def test_a_dry_run_writes_calls_with_raw_scores_and_settings(tmp_path, module, name):
    lst = two_pictures(tmp_path)
    module.main(["--pictures", str(lst), "--out", str(tmp_path / "out"), "--dry-run"])
    calls = json.loads((tmp_path / "out/calls.json").read_text())
    assert calls["judge"] == name and len(calls["pictures"]) == 2
    first = calls["pictures"][0]
    assert first["kind"] == "two_clear" and first["n_bodies"] == 2 and first["two_clear_score"] > 0
    assert all("raw" in b for b in first["bodies"])
    assert json.loads((tmp_path / "out/settings.json").read_text())["judge"] == name


def test_the_references_are_56_species_at_seeds_apart_from_the_census_cats(tmp_path, capsys):
    sp = bars.asked_species()
    assert len(sp) == 56 and {"cat", "dog", "leopard", "swan", "coyote", "spoonbill"} <= set(sp)
    assert list(bars.REFERENCE_SEEDS) == list(range(9001, 9009)) and not set(bars.REFERENCE_SEEDS) & set(range(64))
    assert refs.prompt("eagle") == "a photo of an eagle"
    assert refs.main(["--out", str(tmp_path), "--list-only"]) == 1
    assert "448 references asked (56 species, seeds 9001 to 9008), 448 to draw" in capsys.readouterr().out
    for s in bars.REFERENCE_SEEDS:
        p = refs.ref_path(tmp_path, "cat", s)
        p.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (8, 8)).save(p)
    assert refs.main(["--out", str(tmp_path), "--species", "cat"]) == 0           # all on disk: nothing to draw
    assert len(refs.references(tmp_path, "cat", 8)) == 8 and refs.references(tmp_path, "dog", 8) == []
    with pytest.raises(SystemExit, match="not an asked species"):
        refs.main(["--out", str(tmp_path), "--species", "unicorn", "--list-only"])


def choosing_lists(tmp: Path, n=12):
    rows = [{"picture": f"/x/p{i}.png", "a": "cat", "b": "dog", "kind": "hybrid", "part": "choosing"} for i in range(n)]
    jl.write_csv(tmp / "lists/choosing.csv", rows)
    return rows


def test_choose_redecides_dreamsim_over_its_45_points(tmp_path):
    rows = choosing_lists(tmp_path)
    # body 0 leads A by 0.03 (mixed only at margins 0.04 and up), body 1 is a clear B: hybrid needs margin >= 0.04
    pics = [{"picture": r["picture"], "a": "cat", "b": "dog", "covered": True, "n_bodies": 2,
             "bodies": [{"box": [0, 0, 1, 1], "raw": {"sim_a": 0.63, "sim_b": 0.60}, "call": "A", "flag": False},
                        {"box": [2, 2, 3, 3], "raw": {"sim_a": 0.45, "sim_b": 0.70}, "call": "B", "flag": False}],
             "kind": "", "two_clear_score": 0.0} for r in rows]
    (tmp_path / "calls.json").write_text(json.dumps({"judge": "dreamsim", "settings": dict(ds.DEFAULTS), "pictures": pics}))
    assert jl.main(["choose", "--judge", "dreamsim", "--calls", str(tmp_path / "calls.json"), "--lists",
                    str(tmp_path / "lists"), "--out", str(tmp_path / "chosen.json")]) == 0
    got = json.loads((tmp_path / "chosen.json").read_text())
    assert len(got["grid"]) == 45 and got["agreement"] == 1.0
    assert got["chosen"] == {"margin": 0.04, "flag_gap": 0.05, "floor": 0.0}             # the first point reaching 12/12


def test_choose_redecides_bioclip_on_the_clip_grids(tmp_path):
    rows = choosing_lists(tmp_path)
    pics = [{"picture": r["picture"], "a": "cat", "b": "dog", "covered": True, "n_bodies": 2,
             "bodies": [{"box": [0, 0, 1, 1], "raw": {"p_a": 0.62, "p_b": 0.38, "p_neither_best": 0.0}, "call": "A", "flag": False},
                        {"box": [2, 2, 3, 3], "raw": {"p_a": 0.02, "p_b": 0.98, "p_neither_best": 0.0}, "call": "B", "flag": False}],
             "kind": "", "two_clear_score": 0.0} for r in rows]
    (tmp_path / "calls.json").write_text(json.dumps({"judge": "bioclip", "settings": {"margin": 0.15, "flag_share": 0.25},
                                                     "pictures": pics}))
    assert jl.main(["choose", "--judge", "bioclip", "--calls", str(tmp_path / "calls.json"), "--lists",
                    str(tmp_path / "lists"), "--out", str(tmp_path / "chosen.json")]) == 0
    got = json.loads((tmp_path / "chosen.json").read_text())
    assert len(got["grid"]) == 15 and got["chosen"] == {"margin": 0.30, "flag_share": 0.15}


def test_bioclip_is_fetched_by_the_followon_set():
    import paper2_fetch_weights as fw
    assert fw.SETS["followon"] == fw.FOLLOWON_MODELS and fw.FOLLOWON_MODELS[0][0] == "imageomics/bioclip"
    assert fw.FOLLOWON_MODELS[0][2] == "open_clip_config.json"
