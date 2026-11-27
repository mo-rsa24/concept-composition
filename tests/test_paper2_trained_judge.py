"""Paper 2's trained species judge (paper2_trained_judge.py) on a synthetic split labelled set, with a stand-in backbone.

Needs torch and runs on the CPU; no CLIP weights are loaded:

    /home-mscluster/mmolefe/miniforge3/envs/co3_bw/bin/python -m pytest tests/test_paper2_trained_judge.py -q

What it checks: no read or confirmation picture, nor any picture sharing a file or a pair and seed with one, reaches the
training pool or the head's training; train never opens the test features; the score goes through paper2_score.py
unchanged and passes its settings check; the sealed read refuses a second run; the two-clear probability's gradient
reaches the input pixels; and the cached features equal the ones the dial trainer's path computes.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "2-classifier-guidance/scripts/paper2"))

import paper2_score as sc  # noqa: E402
import paper2_trained_judge as tj  # noqa: E402
from paper2_bars import pair_type  # noqa: E402
from paper2_kinds import KINDS  # noqa: E402

TEST_PAIRS = [("lion", "tiger"), ("wolf", "fox"), ("cheetah", "cougar"),
              ("zebra", "flamingo"), ("eagle", "fox"), ("boar", "pheasant")]
CHOOSING_PAIRS = [("cat", "dog"), ("duck", "goose")]
COLS = ["picture", "renamed", "renaming", "a", "b", "pair", "pair_type", "seed", "source", "kind", "rules_version",
        "part", "unselected_sample", "batch", "name", "call_1", "call_2", "agreed", "replaces", "notes"]
SIDE = 32


def slug(a, b):
    art = lambda s: ("an_" if s[0] in "aeiou" else "a_") + s
    return f"{art(a)}__x__{art(b)}"


def build(tmp: Path) -> dict:
    """A labelled set like the real one: read, confirmation, choosing, spare, unselected, apart and other rows, plus
    spare rows that share a file or a pair and seed with a read picture. Writes the pictures and bodies.json."""
    from PIL import Image
    rows, n = [], 0
    (tmp / "pics").mkdir(parents=True, exist_ok=True)

    def add(a, b, kind, part, picture=None, seed=None, **extra):
        nonlocal n
        n += 1
        r = {c: "" for c in COLS} | {"picture": picture or str(tmp / "pics" / f"p{n:04d}.png"), "renamed": "False",
                                      "a": a, "b": b, "pair": slug(a, b), "pair_type": pair_type(a, b),
                                      "seed": str(seed if seed is not None else n), "kind": kind, "part": part,
                                      "unselected_sample": "False", "call_1": kind, "call_2": kind}
        r.update(extra)
        rows.append(r)
        return r

    for kind in KINDS:
        for i in range(40):
            a, b = TEST_PAIRS[i % 6]
            add(a, b, kind, "read")
        for i in range(10):
            a, b = TEST_PAIRS[i % 6]
            add(a, b, "", "confirmation", call_1="", call_2="")
        for i in range(16):
            a, b = CHOOSING_PAIRS[i % 2]
            add(a, b, kind, "choosing")
        for i in range(6):
            a, b = TEST_PAIRS[i % 6]
            add(a, b, kind, "spare")
    read = [r for r in rows if r["part"] == "read"]
    conf = [r for r in rows if r["part"] == "confirmation"]
    same_file = add(read[0]["a"], "leopard", "wrong_species", "spare", picture=read[0]["picture"], seed=read[0]["seed"],
                    renamed="True", renaming="cougar as leopard", pair=read[0]["pair"], pair_type=read[0]["pair_type"])
    same_seed = add(conf[0]["a"], conf[0]["b"], "two_clear", "spare", seed=conf[0]["seed"])
    unselected = add("lion", "tiger", "two_clear", "spare", unselected_sample="True")
    for i in range(3):
        add("crow", "raven", "two_of_one_kind", "apart")
    add("lion", "tiger", "other", "other")

    for r in rows:
        p = Path(r["picture"])
        if not p.exists():
            k = KINDS.index(r["kind"]) if r["kind"] in KINDS else 6
            Image.new("RGB", (SIDE, SIDE), (30 * k, 200 - 25 * k, (int(r["seed"]) * 7) % 255)).save(p)
    bodies = {}
    for r in rows:
        three = r["kind"] == "extra_subject"
        boxes = [[1, 2, 15, 30], [16, 1, 31, 29]] + ([[8, 8, 24, 20]] if three else [])
        bodies[r["picture"]] = {"boxes": [{"box": b, "confidence": 0.5} for b in boxes]}
    (tmp / "bodies.json").write_text(json.dumps({"pictures": bodies}))
    ls = tmp / "labelled-set.csv"
    with open(ls, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)
    split = tmp / "split.json"
    split.write_text(json.dumps({"seed": 0, "attempt": 1, "choosing_pairs": [slug(*p) for p in CHOOSING_PAIRS],
                                 "test_pairs": [slug(*p) for p in TEST_PAIRS]}))
    return {"rows": rows, "ls": ls, "split": split, "bodies": tmp / "bodies.json", "same_file": same_file,
            "same_seed": same_seed, "unselected": unselected}


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("trained")
    w = build(tmp)
    w["run"] = tmp / "run"
    common = ["--labelled-set", str(w["ls"]), "--split", str(w["split"]), "--run", str(w["run"]), "--device", "cpu"]
    w["common"] = common
    assert tj.main(["features", *common, "--bodies", str(w["bodies"]), "--backbone", "stub"]) == 0
    test_pt = w["run"] / "features" / "test.pt"
    hidden = test_pt.with_suffix(".away")
    test_pt.rename(hidden)                        # train must not need, or open, the test features
    try:
        assert tj.main(["train", *common, "--epochs", "8", "--patience", "8", "--batch", "32"]) == 0
    finally:
        hidden.rename(test_pt)
    return w


def sealed(rows):
    return [r for r in rows if r["part"] in ("read", "confirmation")]


# ---------------------------------------------------------------- no sealed picture reaches training

@pytest.mark.parametrize("pool", tj.POOLS)
def test_pool_holds_no_sealed_picture(tmp_path, pool):
    w = build(tmp_path)
    rows = tj.read_labelled(w["ls"])
    got, dropped = tj.training_pool(rows, json.loads(w["split"].read_text()), pool)
    files = {r["picture"] for r in sealed(rows)}
    seeds = {(r["pair"], r["seed"]) for r in sealed(rows)}
    assert got and not [r for r in got if r["part"] in ("read", "confirmation", "apart", "other")]
    assert not [r for r in got if r["picture"] in files or (r["pair"], r["seed"]) in seeds]
    keys = {(r["picture"], r["a"], r["b"]) for r in got}
    for name in ("same_file", "same_seed", "unselected"):
        assert (w[name]["picture"], w[name]["a"], w[name]["b"]) not in keys
    if pool == "choosing":
        assert {r["part"] for r in got} == {"choosing"}
    else:
        assert {r["part"] for r in got} == {"choosing", "spare"}
        assert dropped["same file as a read or confirmation picture"] == 1
        assert dropped["same pair and seed as a read or confirmation picture"] == 1
        assert dropped["the unselected sample on a test pair"] == 1


def test_dev_split_keeps_pair_and_seed_together(tmp_path):
    w = build(tmp_path)
    rows = tj.read_labelled(w["ls"])
    pool, _ = tj.training_pool(rows, json.loads(w["split"].read_text()), "unsealed")
    train, dev = tj.dev_split(pool)
    assert len(train) + len(dev) == len(pool) and dev
    assert not {(r["pair"], r["seed"]) for r in train} & {(r["pair"], r["seed"]) for r in dev}
    assert {r["kind"] for r in dev} == set(KINDS)
    assert tj.dev_split(pool) == (train, dev)              # fixed


def test_features_and_training_never_hold_a_sealed_picture(world):
    files = {r["picture"] for r in sealed(world["rows"])}
    pool = torch.load(world["run"] / "features" / "pool.pt", weights_only=False)
    assert pool["pictures"] and not files & set(pool["pictures"])
    test = torch.load(world["run"] / "features" / "test.pt", weights_only=False)
    conf = {r["picture"] for r in world["rows"] if r["part"] == "confirmation"}
    assert not conf & set(test["pictures"])
    rows = tj.read_labelled(world["ls"])
    expected, _ = tj.training_pool(rows, json.loads(world["split"].read_text()), "unsealed")
    assert set(pool["pictures"]) == {r["picture"] for r in expected}
    counts = json.loads((world["run"] / "split-counts.json").read_text())
    assert sum(v["all"] for v in counts["train"].values()) + sum(v["all"] for v in counts["dev"].values()) == len(expected)


def test_train_refuses_a_tampered_pool(world, tmp_path):
    import shutil
    run = tmp_path / "run"
    shutil.copytree(world["run"] / "features", run / "features")
    pt = run / "features" / "pool.pt"
    f = torch.load(pt, weights_only=False)
    f["pictures"][0] = next(r["picture"] for r in world["rows"] if r["part"] == "read")
    torch.save(f, pt)
    with pytest.raises(SystemExit, match="read or confirmation"):
        tj.main(["train", "--labelled-set", str(world["ls"]), "--split", str(world["split"]), "--run", str(run),
                 "--device", "cpu", "--epochs", "1"])


def test_settings_chosen_names_the_head(world):
    chosen = json.loads((world["run"] / "settings-chosen.json").read_text())
    assert chosen["chosen"]["head_sha256"] == tj.sha256(world["run"] / "head.pt")
    log = json.loads((world["run"] / "train-log.json").read_text())
    assert log["seconds_per_epoch"] > 0 and all("dev_per_kind" in e for e in log["epochs"])


# ---------------------------------------------------------------- the score goes through the existing scorer, once

def test_score_is_read_by_the_existing_scorer_and_runs_once(world, tmp_path):
    assert tj.main(["score", *world["common"]]) == 0
    with pytest.raises(SystemExit, match="runs once"):
        tj.main(["score", *world["common"]])
    out = tmp_path / "score"
    assert sc.main(["--labelled-set", str(world["ls"]), "--split", str(world["split"]),
                    "--judge", f"trained={world['run']}", "--out", str(out), "--no-sheet"]) == 0
    res = json.loads((out / "bars.json").read_text())
    t = res["judges"]["trained"]
    assert res["read_pictures"] == 240 and t["bars"]["agree_all"]["of"] == 240 and t["bars"]["agree_all"]["need"] == 216
    assert t["settings_check"]["ok"] is True
    assert "settings chosen before the test call" not in t.get("missed", [])
    calls = json.loads((world["run"] / "test" / "calls.json").read_text())
    assert all(abs(sum(p["probs"].values()) - 1) < 1e-4 for p in calls["pictures"])
    assert all(p["two_clear_score"] == p["probs"]["two_clear"] for p in calls["pictures"])


# ---------------------------------------------------------------- the gradient reaches the pixels

def test_two_clear_gradient_reaches_the_pixels(world):
    boxes = [[1, 2, 15, 30], [16, 1, 31, 29]]
    reward = tj.TrainedJudgeReward(world["run"] / "head.pt", "cpu", backbone=tj.StubBackbone("cpu"),
                                   boxes_fn=lambda im: boxes)
    image = torch.rand(3, SIDE, SIDE, requires_grad=True)
    score, info = reward(image, "lion", "tiger")
    assert score.requires_grad and info["kind"] in KINDS and info["n_bodies"] == 2
    score.backward()
    assert image.grad is not None and torch.isfinite(image.grad).all() and image.grad.abs().sum() > 0


def test_names_in_either_order_give_the_same_probabilities(world):
    reward = tj.TrainedJudgeReward(world["run"] / "head.pt", "cpu", backbone=tj.StubBackbone("cpu"),
                                   boxes_fn=lambda im: [[1, 2, 15, 30], [16, 1, 31, 29]])
    image = torch.rand(3, SIDE, SIDE)
    boxes = reward.boxes(image)
    assert torch.allclose(reward.probs(image, "lion", "tiger", boxes), reward.probs(image, "tiger", "lion", boxes), atol=1e-6)


def test_cached_features_equal_the_reward_path(world):
    pool = torch.load(world["run"] / "features" / "pool.pt", weights_only=False)
    pic = pool["pictures"][0]
    image = tj.load_picture(pic, "cpu").requires_grad_(True)
    f = tj.picture_features(tj.StubBackbone("cpu"), image, pool["boxes"][0])
    assert torch.allclose(f["views"].detach(), pool["views"][0, 0].float(), atol=2e-3)
    assert torch.allclose(f["bodies"].detach(), pool["bodies"][0, 0].float(), atol=2e-3)
    assert torch.equal(f["mask"], pool["mask"][0, 0])
