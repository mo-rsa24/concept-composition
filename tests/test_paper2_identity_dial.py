"""Paper 2's identity dial: the adapter, the random dial, CO3's rollout, the guards and the dry runs.

Needs torch, so it skips on the laptop's own python and runs on the cluster in co3_bw (or co3):

    /home-mscluster/mmolefe/miniforge3/envs/co3_bw/bin/python -m pytest tests/test_paper2_identity_dial.py -q

It also runs on any CPU-only torch; nothing here loads SDXL, CO3 or a judge's model.
"""
from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[1]
PAPER2 = REPO / "2-classifier-guidance/scripts/paper2"
DIAL = PAPER2 / "paper2_identity_dial.py"
sys.path.insert(0, str(PAPER2))

import paper2_identity_dial as pid  # noqa: E402
from paper2_common import crop_body, share_two_clear  # noqa: E402


def stand_in_layer():
    """One attention block of width 8, named attn1 as SDXL's are."""
    torch.manual_seed(0)
    return pid.StandInUNet()


def randomise(dial, seed=1):
    dial.load_state(pid.random_state_like(dial.state(), 0.5, seed))


def test_strength_zero_is_the_frozen_layer_bit_for_bit():
    frozen = stand_in_layer()
    dialled = copy.deepcopy(frozen)
    dial = pid.attach_dial(dialled)
    assert len(dial.layers) == 4  # to_q, to_k, to_v, to_out.0
    randomise(dial)               # weights that would change the output if they acted
    x = torch.randn(2, 16, 8)
    dial.set_strength(0.0)
    with torch.no_grad():
        assert torch.equal(dialled(x), frozen(x))


def test_strength_one_changes_the_output():
    frozen = stand_in_layer()
    dialled = copy.deepcopy(frozen)
    dial = pid.attach_dial(dialled)
    randomise(dial)
    dial.set_strength(1.0)
    x = torch.randn(2, 16, 8)
    with torch.no_grad():
        assert not torch.allclose(dialled(x), frozen(x))


def test_rank_is_four_and_up_starts_at_zero():
    dial = pid.attach_dial(stand_in_layer())
    for m in dial.layers.values():
        assert m.lora_down.weight.shape[0] == 4 and m.lora_up.weight.shape[1] == 4
        assert torch.count_nonzero(m.lora_up.weight) == 0


@pytest.mark.parametrize("asked", [0.0123, 1.0, 37.5])
def test_random_dial_has_the_asked_norm_and_the_same_shapes(asked):
    dial = pid.attach_dial(stand_in_layer())
    randomise(dial, seed=3)
    state = dial.state()
    rnd = pid.random_state_like(state, asked, seed=7)
    assert {k: v.shape for k, v in rnd.items()} == {k: v.shape for k, v in state.items()}
    assert round(pid.state_norm(rnd), 4) == round(asked, 4)


def test_rollout_is_co3s_step_and_the_dial_switches_on_at_dial_from():
    sampler = pid.stand_in_sampler(0)
    dial = pid.attach_dial(sampler.unet)
    x = sampler.start(11001)
    sampler.rollout(x, 0, 7, dial, dial_from=4, strength=1.0)
    # every step went through the model's own denoise_step, in order, the dial off before step 4
    assert sampler.model.calls == [(0, 0.0), (1, 0.0), (2, 0.0), (3, 0.0), (4, 1.0), (5, 1.0), (6, 1.0)]
    # and the plain product's prediction is nowhere in the trainer
    assert "poe_eps" not in DIAL.read_text()


def test_load_dial_refuses_strength_zero(tmp_path):
    dial = pid.attach_dial(stand_in_layer())
    pid.save_dial(dial.state(), tmp_path / "dial.pt", {})
    with pytest.raises(SystemExit):
        pid.load_dial(stand_in_layer(), tmp_path / "dial.pt", 0.0)


def test_torch_crop_matches_the_judges_crop():
    from PIL import Image
    pic = REPO / "outputs/loop-6/m14/blind/pictures/95b7a187cfa1.png"
    if not pic.exists():
        pytest.skip("joint-prompt picture not on this machine")
    im = Image.open(pic).convert("RGB")
    boxes = [[60, 180, 560, 900], [500, 160, 980, 880]]
    image = torch.from_numpy(__import__("numpy").array(im)).permute(2, 0, 1).float() / 255
    for i, crop_t in enumerate(pid.torch_crops(image, boxes)):
        crop_p, _ = crop_body(im, boxes, i)
        ref = torch.from_numpy(__import__("numpy").array(crop_p)).permute(2, 0, 1).float() / 255
        assert crop_t.shape == ref.shape
        # equal outside the painted box; the painted mean colour differs only by PIL's rounding
        assert (crop_t - ref).abs().max() <= 1.5 / 255


@pytest.mark.parametrize("s_a,flags", [([0.9, 0.2], [False, False]), ([0.9, 0.3], [False, True]),
                                       ([0.6], [False]), ([0.8, 0.1, 0.7], [True, False, False])])
def test_torch_two_clear_matches_the_judges_formula(s_a, flags):
    bodies = [{"raw": {"p_a": s, "p_b": 1 - s}, "flag": f} for s, f in zip(s_a, flags)]
    got = float(pid.two_clear_from_shares(torch.tensor(s_a), flags))
    assert got == pytest.approx(share_two_clear(bodies), abs=1e-6)


def run(*args):
    return subprocess.run([sys.executable, str(DIAL), *args], capture_output=True, text=True)


@pytest.mark.parametrize("route", ["gradient", "kept"])
def test_dry_run_trains_two_steps_and_writes_a_checkpoint(route, tmp_path):
    r = run("train", "--dry-run", "--route", route, "--out", str(tmp_path))
    assert r.returncode == 0, r.stdout + r.stderr
    ckpt = torch.load(str(tmp_path / "ckpt-2" / "dial.pt"), weights_only=False)
    assert ckpt["step"] == 2 and ckpt["rank"] == 4 and ckpt["norm"] > 0
    log = [json.loads(x) for x in (tmp_path / "train-log.jsonl").read_text().splitlines()]
    assert [row["step"] for row in log] == [1, 2]
    assert log[1]["closeness_term"] > 0 and log[1]["closeness_grad_norm"] > 0
    assert (tmp_path / "closeness.json").exists()
    assert "reward per pair" in r.stdout


def test_zero_closeness_weight_exits_non_zero(tmp_path):
    r = run("train", "--dry-run", "--closeness-weight", "0", "--out", str(tmp_path))
    assert r.returncode != 0
    assert "closeness weight must be above 0" in r.stdout + r.stderr


def test_dial_before_the_window_is_refused(tmp_path):
    r = run("train", "--dry-run", "--window", "3", "5", "--dial-from-step", "2", "--out", str(tmp_path))
    assert r.returncode != 0 and "never trained on" in r.stdout + r.stderr


def test_window_inside_co3s_corrected_steps_is_refused(tmp_path):
    r = run("train", "--dry-run", "--window", "0", "3", "--out", str(tmp_path))
    assert r.returncode != 0 and "corrected steps" in r.stdout + r.stderr


def test_random_and_draw_use_the_same_switch(tmp_path):
    assert run("train", "--dry-run", "--out", str(tmp_path / "train")).returncode == 0
    r = run("random", "--like", str(tmp_path / "train/ckpt-2/dial.pt"), "--out", str(tmp_path / "random/dial.pt"))
    assert r.returncode == 0, r.stderr
    trained = torch.load(str(tmp_path / "train/ckpt-2/dial.pt"), weights_only=False)
    rnd = torch.load(str(tmp_path / "random/dial.pt"), weights_only=False)
    assert round(rnd["norm"], 4) == round(trained["norm"], 4)
    assert rnd["dial_from_step"] == trained["dial_from_step"] == 2
    for name in ("train/ckpt-2/dial.pt", "random/dial.pt"):
        out = tmp_path / ("draw-" + name.split("/")[0])
        r = run("draw", "--dry-run", "--ckpt", str(tmp_path / name), "--out", str(out))
        assert r.returncode == 0, r.stderr
        drawn = json.loads((out / "draw.json").read_text())
        assert drawn["dial_from_step"] == 2 and len(drawn["pictures"]) == 1
    assert run("draw", "--dry-run", "--ckpt", str(tmp_path / "random/dial.pt"), "--strength", "0",
               "--out", str(tmp_path / "off")).returncode != 0


class _Scripted:
    """A stand-in reward that plays a fixed script: (score, has gradient, bodies) per call."""

    def __init__(self, script):
        self.script, self.i = script, 0

    def __call__(self, image, a, b):
        score, grad, bodies = self.script[min(self.i, len(self.script) - 1)]
        self.i += 1
        value = image[0].mean() - image[2].mean() if grad else torch.tensor(float(score))
        return value, {"kind": "stand-in", "n_bodies": bodies}


def test_kept_route_survives_two_skipped_draws_with_two_bodies(tmp_path, monkeypatch):
    script = [(0.0, False, 2), (0.0, False, 2), (1.0, False, 2)]
    monkeypatch.setattr(pid, "StandInKeep", lambda: _Scripted(script))
    pid.main(["train", "--dry-run", "--route", "kept", "--steps", "5", "--out", str(tmp_path)])
    log = [json.loads(x) for x in (tmp_path / "train-log.jsonl").read_text().splitlines()]
    assert [row["reward"] for row in log[:3]] == [0.0, 0.0, 1.0]
    assert log[2]["task_grad_norm"] > 0                                   # the kept draw trained the dial
    assert log[3]["closeness_term"] > 0 and log[3]["closeness_grad_norm"] > 0   # checked after it moved


def test_gradient_route_survives_a_first_step_with_one_body(tmp_path, monkeypatch):
    script = [(0.0, False, 1), (0.0, True, 2)]
    monkeypatch.setattr(pid, "StandInReward", lambda: _Scripted(script))
    pid.main(["train", "--dry-run", "--steps", "4", "--out", str(tmp_path)])
    log = [json.loads(x) for x in (tmp_path / "train-log.jsonl").read_text().splitlines()]
    assert log[0]["task_grad_norm"] == 0 and log[1]["task_grad_norm"] > 0
    assert log[1]["closeness_term"] == 0 and log[2]["closeness_term"] > 0


def test_a_reward_that_never_reaches_the_dial_stops_the_run(tmp_path, monkeypatch):
    monkeypatch.setattr(pid, "StandInReward", lambda: _Scripted([(0.0, False, 1)]))
    with pytest.raises(SystemExit, match="gave the reward a gradient"):
        pid.main(["train", "--dry-run", "--steps", "12", "--out", str(tmp_path)])


# ---- setup plan task 3 as reworded: the closeness cache, weighted pairs, and the silent-term guards

def test_dial_leaves_the_output_unchanged_before_its_first_step_and_changes_it_after():
    plain = pid.stand_in_sampler(0)
    dialled = pid.stand_in_sampler(0)
    dial = pid.attach_dial(dialled.unet)
    randomise(dial)
    x = plain.start(11001)
    before_plain = plain.rollout(x, 0, 4, None, plain.n_steps, 0.0)
    before_dial = dialled.rollout(x, 0, 4, dial, dial_from=4, strength=1.0)
    assert torch.equal(before_plain, before_dial)          # steps 0 to 3: the dial is off
    after_plain = plain.rollout(x, 0, 7, None, plain.n_steps, 0.0)
    after_dial = dialled.rollout(x, 0, 7, dial, dial_from=4, strength=1.0)
    assert not torch.allclose(after_plain, after_dial)     # steps 4 to 6: it acts


def test_closeness_cache_is_made_with_no_dial_and_holds_the_frozen_prediction(tmp_path):
    out = tmp_path / "cache.pt"
    r = run("closeness-cache", "--dry-run", "--out", str(out))
    assert r.returncode == 0, r.stdout + r.stderr
    cache = torch.load(str(out), weights_only=False)
    assert cache["dial_attached"] is False and cache["kind"] == "stand-in"
    assert len(cache["entries"]) == len(pid.STAND_IN_CLOSENESS_PROMPTS) * 2
    untouched = pid.stand_in_sampler(0)                    # a fresh model, never given a dial
    for e in cache["entries"]:
        assert torch.equal(untouched.prompt_eps(e["x"], e["k"], e["prompt"]), e["eps"])
    dialled = pid.stand_in_sampler(0)
    pid.attach_dial(dialled.unet)
    with pytest.raises(SystemExit, match="no dial attached"):
        pid.build_closeness_cache(dialled, ["a cat"], [1], (2, 5), 0, tmp_path / "refused.pt")


def test_train_reads_the_cache_and_builds_no_second_model(tmp_path, monkeypatch):
    cache = tmp_path / "cache.pt"
    assert run("closeness-cache", "--dry-run", "--out", str(cache)).returncode == 0
    built = []
    real = pid.stand_in_sampler
    monkeypatch.setattr(pid, "stand_in_sampler", lambda seed=0: built.append(seed) or real(seed))
    pid.main(["train", "--dry-run", "--closeness-cache", str(cache), "--out", str(tmp_path / "run")])
    assert len(built) == 1                                 # the trainer's own sampler, and nothing else
    assert json.loads((tmp_path / "run/run.json").read_text())["closeness_cache"] == str(cache)
    assert not (tmp_path / "run/closeness-cache.pt").exists()


def test_a_cache_made_with_a_dial_is_refused(tmp_path):
    cache = tmp_path / "cache.pt"
    assert run("closeness-cache", "--dry-run", "--out", str(cache)).returncode == 0
    c = torch.load(str(cache), weights_only=False)
    c["dial_attached"] = True
    torch.save(c, cache)
    r = run("train", "--dry-run", "--closeness-cache", str(cache), "--out", str(tmp_path / "run"))
    assert r.returncode != 0 and "no dial attached" in r.stdout + r.stderr


def test_pairs_file_carries_weights(tmp_path):
    lines = tmp_path / "pairs.txt"
    lines.write_text("a_cheetah__x__a_cougar 3   # look-alike\nan_eagle__x__a_fox\n")
    assert pid.read_pairs(lines) == [("cheetah", "cougar", 3.0), ("eagle", "fox", 1.0)]
    js = tmp_path / "pairs.json"
    js.write_text(json.dumps([{"pair": "a_duck__x__a_goose", "weight": 2}]))
    assert pid.read_pairs(js) == [("duck", "goose", 2.0)]
    lines.write_text("a_duck__x__a_goose 0\n")
    with pytest.raises(SystemExit, match="above 0"):
        pid.read_pairs(lines)


def test_draws_follow_the_weights_and_each_pairs_reward_is_logged_on_its_own_key(tmp_path, monkeypatch):
    logged = []

    class Recorder:
        def __init__(self, *a, **k):
            pass

        def log(self, row, step):
            logged.append(row)

        def finish(self, summary):
            pass
    monkeypatch.setattr(pid, "Tracker", Recorder)
    pid.main(["train", "--dry-run", "--steps", "60", "--closeness-batch", "1", "--out", str(tmp_path)])
    rows = [json.loads(x) for x in (tmp_path / "train-log.jsonl").read_text().splitlines()]
    n_cheetah = sum(r["pair"] == "cheetah|cougar" for r in rows)
    assert 35 <= n_cheetah <= 55                           # weight 3 against 1: about 45 of 60
    keys = {k for row in logged for k in row if k.startswith("reward/pair/")}
    assert keys == {"reward/pair/a_cheetah__x__a_cougar", "reward/pair/a_duck__x__a_goose"}
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert set(summary["reward_per_pair"]) == {"cheetah|cougar", "duck|goose"}


def test_a_reward_of_exactly_zero_stops_the_dry_run_by_name(tmp_path, monkeypatch):
    class Zero:
        def __call__(self, image, a, b):
            return image.sum() * 0, {"kind": "stand-in", "n_bodies": 2}
    monkeypatch.setattr(pid, "StandInReward", Zero)
    with pytest.raises(SystemExit, match="reward term is exactly 0 at step 1"):
        pid.main(["train", "--dry-run", "--out", str(tmp_path)])


def test_closeness_prompts_are_single_animals_and_unlike_pairs_none_a_known_pair():
    import re
    sys.path.insert(0, str(REPO / "scripts/shared"))
    from disallowed_subjects import is_allowed
    import paper2_bars
    prompts = pid.read_prompts(PAPER2 / "paper2_closeness_prompts.txt")
    # The species judge plan's pairs as committed in paper2_bars.PAIRS (plans/ is not on the cluster; that PAIRS
    # matches the plan's table is checked on the laptop by tests/test_paper2_bars.py), and the pooled dial's pairs.
    known = {frozenset((a, b)) for a, b, _ in paper2_bars.PAIRS}
    pooled = json.loads((REPO / "artifacts/notes/progress/leakage/10-co-occurrence-dial/pairs.json").read_text())
    known |= {frozenset(re.findall(r"(?:an?) (\w+)", p)) for p in pooled["pairs"]}
    singles = [p for p in prompts if " and " not in p]
    pairs = [p for p in prompts if " and " in p]
    assert singles and pairs
    for p in prompts:
        assert is_allowed(p), p
    for p in pairs:
        assert frozenset(re.findall(r"(?:an?) (\w+)", p)) not in known, p


# ---- --reward trained: the trained species judge as the gradient route's reward (owner's choice of 2026-11-27)

def _stub_head(path, zero=False):
    """A trained-judge head on the stub backbone (16 numbers a view), saved as paper2_trained_judge.cmd_train saves one.
    zero=True zeroes every weight, so the head's logits are constant and its two-clear probability has no gradient."""
    import paper2_trained_judge as tj
    torch.manual_seed(0)
    head = tj.JudgeHead(tj.StubBackbone.dim)
    if zero:
        with torch.no_grad():
            for p in head.parameters():
                p.zero_()
    torch.save({"state": head.state_dict(), "config": head.config, "kinds": list(tj.KINDS),
                "bank": ["horse", "parrot", "owl"], "backbone": "stub", "logit_scale": tj.StubBackbone.logit_scale,
                "epoch": 1}, path)
    return path


class _TwoBoxes:
    """Stands in for the body counter: two fixed bodies, left and right halves of the stand-in's 8 by 8 picture."""

    def __init__(self, device):
        pass

    def __call__(self, image):
        return [[0, 0, 4, 8], [4, 0, 8, 8]]


@pytest.fixture
def trained_world(tmp_path, monkeypatch):
    """The non-dry-run path of `train` on CPU: the stand-in sampler in place of CO3, the stub body counter, a real
    closeness cache from another process, a weighted pairs file, and the route each reward took recorded."""
    cache = tmp_path / "cache.pt"
    assert run("closeness-cache", "--dry-run", "--out", str(cache)).returncode == 0
    pairs = tmp_path / "pairs.txt"
    pairs.write_text("a_cheetah__x__a_cougar 3\na_duck__x__a_goose 1\n")
    monkeypatch.setattr(pid, "co3_sampler", lambda device, out: pid.stand_in_sampler(0))
    monkeypatch.setattr(pid, "BodyBoxes", _TwoBoxes)
    built = []
    real = pid.trained_judge_reward
    monkeypatch.setattr(pid, "trained_judge_reward",
                        lambda head, device: built.append(("trained", head)) or real(head, device))

    def args(*extra, reward="trained", route="gradient", out="run"):
        return ["train", "--route", route, "--reward", reward, "--pairs-file", str(pairs), "--seeds", "11001", "11002",
                "--window", "2", "5", "--closeness-cache", str(cache), "--closeness-weight", "1", "--steps", "2",
                "--ckpt-steps", "2", "--device", "cpu", "--wandb-mode", "disabled", "--out", str(tmp_path / out),
                *extra]
    return {"tmp": tmp_path, "args": args, "built": built}


def test_reward_trained_routes_to_the_trained_judge_and_trains(trained_world):
    import paper2_trained_judge as tj
    w = trained_world
    head = _stub_head(w["tmp"] / "head.pt")
    seen = []
    real_call = tj.TrainedJudgeReward.__call__
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(tj.TrainedJudgeReward, "__call__", lambda self, *a: seen.append(type(self)) or real_call(self, *a))
        pid.main(w["args"]("--judge-head", str(head)))
    assert w["built"] == [("trained", head)]
    assert seen and all(t is tj.TrainedJudgeReward for t in seen)       # every step's reward was the trained judge
    run_json = json.loads((w["tmp"] / "run/run.json").read_text())
    assert run_json["reward"] == "trained" and run_json["judge_head"] == str(head)
    assert len(run_json["judge_head_sha256"]) == 64
    log = [json.loads(x) for x in (w["tmp"] / "run/train-log.jsonl").read_text().splitlines()]
    assert log[0]["task_grad_norm"] > 0                                  # the head's gradient reached the dial
    assert all(0 < row["reward"] < 1 for row in log)                     # a two-clear probability
    assert (w["tmp"] / "run/ckpt-2/dial.pt").exists()


def test_reward_trained_with_a_zero_gradient_head_stops_at_step_1(trained_world):
    w = trained_world
    head = _stub_head(w["tmp"] / "zero.pt", zero=True)
    # the value is the head's flat two-clear probability, 1/6, so the stop comes from the gradient norm being 0
    with pytest.raises(SystemExit, match=r"reward term is exactly 0 at step 1 \(value -0\.1666\d*, gradient norm 0\.0\)"):
        pid.main(w["args"]("--judge-head", str(head)))
    assert w["built"] == [("trained", head)]


def test_reward_trained_needs_its_head_and_the_gradient_route(trained_world):
    w = trained_world
    with pytest.raises(SystemExit, match="--judge-head is required"):
        pid.main(w["args"]())
    head = _stub_head(w["tmp"] / "head.pt")
    with pytest.raises(SystemExit, match="gradient-route reward"):
        pid.main(w["args"]("--judge-head", str(head), route="kept"))
    with pytest.raises(SystemExit, match="is not a file"):
        pid.main(w["args"]("--judge-head", str(w["tmp"] / "missing.pt")))


def test_crop_judge_routes_are_unchanged(trained_world, monkeypatch):
    """clip and imagenet still build CropShareReward from --judge-settings on the gradient route, any crop judge still
    builds JudgeKeep on the kept route, and neither touches the trained judge."""
    w = trained_world
    settings = w["tmp"] / "settings.json"
    settings.write_text(json.dumps({"settings": {"margin": 0.3, "flag_share": 0.25}}))
    made = []

    class Crop:
        def __init__(self, kind, device, s, class_table=None):
            made.append(("crop", kind, s))

        def __call__(self, image, a, b):
            return image[0].mean() - image[2].mean(), {"kind": "stand-in", "n_bodies": 2}

    class Keep:
        def __init__(self, judge, device, settings_file):
            made.append(("keep", judge, Path(settings_file)))

        def __call__(self, image, a, b):
            return torch.tensor(1.0), {"kind": "two_clear", "n_bodies": 2}
    monkeypatch.setattr(pid, "CropShareReward", Crop)
    monkeypatch.setattr(pid, "JudgeKeep", Keep)
    for kind in ("clip", "imagenet"):
        pid.main(w["args"]("--judge-settings", str(settings), reward=kind, out=f"run-{kind}"))
    pid.main(w["args"]("--judge-settings", str(settings), reward="vlm", route="kept", out="run-kept"))
    assert made == [("crop", "clip", {"margin": 0.3, "flag_share": 0.25}),
                    ("crop", "imagenet", {"margin": 0.3, "flag_share": 0.25}), ("keep", "vlm", settings)]
    assert w["built"] == []
    for out in ("run-clip", "run-imagenet", "run-kept"):
        assert "judge_head" not in json.loads((w["tmp"] / out / "run.json").read_text())
    with pytest.raises(SystemExit, match="--judge-settings is required"):
        pid.main(w["args"](reward="clip", out="run-none"))


# ---- the first result's additions: the CO3 row with no adapter, the probe, the logged tuning pictures, memory

def test_draw_co3_attaches_no_adapter_and_lists_its_pictures(tmp_path):
    (tmp_path / "pairs.txt").write_text("a_wolf__x__a_husky\na_crow__x__a_raven\n")
    r = run("draw-co3", "--dry-run", "--pairs-file", str(tmp_path / "pairs.txt"), "--seeds", "13001:13004",
            "--out", str(tmp_path / "co3"))
    assert r.returncode == 0, r.stdout + r.stderr
    import csv
    rows = list(csv.DictReader(open(tmp_path / "co3/pictures.csv")))
    assert len(rows) == 6 and set(rows[0]) == {"picture", "a", "b", "pair", "seed"}
    assert json.loads((tmp_path / "co3/draw.json").read_text())["ckpt"] is None


def test_co3_row_and_dial_row_at_one_seed_are_matched_pictures(tmp_path):
    """The same pair and seed through the same rollout: CO3 alone and an untrained dial (up at zero) give the same
    picture, so the rows differ only by what the dial learned."""
    (tmp_path / "pairs.txt").write_text("a_wolf__x__a_husky\n")
    assert run("draw-co3", "--dry-run", "--pairs-file", str(tmp_path / "pairs.txt"), "--seeds", "13001",
               "--out", str(tmp_path / "co3")).returncode == 0
    sampler = pid.stand_in_sampler(0)
    dial = pid.attach_dial(sampler.unet)
    pid.save_dial(dial.state(), tmp_path / "zero.pt", {"dial_from_step": 2})
    pid.main(["draw", "--dry-run", "--ckpt", str(tmp_path / "zero.pt"), "--out", str(tmp_path / "dial")])
    from PIL import Image
    import numpy as np
    a = np.asarray(Image.open(tmp_path / "co3/a_wolf__x__a_husky/seed-13001.png"))
    b = np.asarray(Image.open(tmp_path / "dial/a_cheetah__x__a_cougar/seed-13001.png"))
    assert a.shape == b.shape
    sampler2 = pid.stand_in_sampler(0)
    sampler2.set_pair("cheetah", "cougar")
    x = sampler2.rollout(sampler2.start(13001), 0, sampler2.n_steps, None, sampler2.n_steps, 0.0)
    c = np.asarray(pid.to_pil(sampler2.decode(x.float())))
    assert np.array_equal(b, c)


def test_probe_strength_0_equals_off_and_the_change_grows(tmp_path):
    assert run("train", "--dry-run", "--steps", "3", "--out", str(tmp_path / "t")).returncode == 0
    r = run("probe", "--dry-run", "--ckpt", str(tmp_path / "t/ckpt-3/dial.pt"), "--out", str(tmp_path / "p"))
    assert r.returncode == 0, r.stdout + r.stderr
    p = json.loads((tmp_path / "p/probe.json").read_text())
    assert p["strength_0_equals_off"] and p["change_grows_with_strength"]
    assert p["strengths"]["0.0"]["largest_grey"] == 0 and p["strengths"]["1.0"]["cosine_with_judge_gradient"] is not None
    assert len(list((tmp_path / "p").glob("estimate-*.png"))) == 4


def test_training_logs_the_tuning_pictures_before_and_every_n_steps(tmp_path):
    (tmp_path / "tuning.txt").write_text("a_wolf__x__a_husky\na_crow__x__a_raven\n")
    pid.main(["train", "--dry-run", "--steps", "5", "--log-pairs-file", str(tmp_path / "tuning.txt"), "--log-seeds",
              "12001", "12002", "--log-every-steps", "2", "--out", str(tmp_path / "t")])
    logged = [json.loads(x) for x in (tmp_path / "t/logged.jsonl").read_text().splitlines()]
    assert [r["step"] for r in logged] == [0, 2, 4, 5]
    assert all(r["n"] == 4 and (tmp_path / f"t/logged/step-{r['step']:05d}/sheet.png").exists() for r in logged)
    summary = json.loads((tmp_path / "t/summary.json").read_text())
    assert summary["memory"] == {"on_card": False} and len(summary["logged"]) == 4
    with pytest.raises(SystemExit, match="go together"):
        pid.main(["train", "--dry-run", "--log-pairs-file", str(tmp_path / "tuning.txt"), "--out", str(tmp_path / "u")])


# ---- the trainer's strength-0 rollout is CO3's own row, bit for bit (step 2's rollout check, run here on the stand-in)

def _as_array(pil):
    import numpy as np
    return np.asarray(pil.convert("RGB"))


@pytest.mark.parametrize("seed", [12001, 12002])
def test_strength_0_rollout_and_finish_equal_the_models_own_sample_loop_bit_for_bit(seed):
    """CO3's row is seed, then the model's sample_loop (every step through denoise_step, then its own decode). The
    trainer's is its rollout with a dial attached at strength 0, then finish. The dial carries weights that would move
    the picture if they acted, so equality also shows strength 0 skips the adapter."""
    import numpy as np
    row = pid.stand_in_sampler(0)
    row.set_pair("donkey", "pony")
    reference = row.model.sample_loop(row.start(seed))[0]
    trainer = pid.stand_in_sampler(0)
    trainer.set_pair("donkey", "pony")
    dial = pid.attach_dial(trainer.unet)
    randomise(dial)
    x = trainer.rollout(trainer.start(seed), 0, trainer.n_steps, dial, trainer.n_steps, 0.0)
    assert np.array_equal(_as_array(pid.to_pil(trainer.finish(x))), _as_array(reference))
    # and at strength 1 from step 2 the same dial does move it, so the check above is not blind
    y = trainer.rollout(trainer.start(seed), 0, trainer.n_steps, dial, 2, 1.0)
    assert not np.array_equal(_as_array(pid.to_pil(trainer.finish(y))), _as_array(reference))


def test_finish_is_the_models_decode_and_restores_its_steps():
    s = pid.stand_in_sampler(0)
    steps = s.model.scheduler.timesteps.clone()
    x = s.rollout(s.start(1), 0, s.n_steps, None, s.n_steps, 0.0)
    calls_before = len(s.model.calls)
    img = s.finish(x)
    assert len(s.model.calls) == calls_before                      # no step ran inside finish
    assert torch.equal(s.model.scheduler.timesteps, steps)
    assert _as_array(pid.to_pil(img)).tobytes() == _as_array(pid.to_pil(s.model.decode(x.float()))).tobytes()

    class Broken(pid.StandInModel):
        def sample_loop(self, x):
            raise RuntimeError("decode failed")
    b = Broken(0)
    before = b.scheduler.timesteps.clone()
    with pytest.raises(RuntimeError):
        pid.finish_by_model(b, x)
    assert torch.equal(b.scheduler.timesteps, before)              # restored even when the decode fails


def test_drawn_rows_are_the_models_own_pictures(tmp_path):
    """draw-co3 (no adapter) and draw with an untrained dial (up at zero) both give, for each pair and seed, exactly the
    picture the model's own sample_loop gives."""
    import numpy as np
    from PIL import Image
    (tmp_path / "pairs.txt").write_text("a_wolf__x__a_husky\na_crow__x__a_raven\n")
    assert run("draw-co3", "--dry-run", "--pairs-file", str(tmp_path / "pairs.txt"), "--seeds", "13001:13003",
               "--out", str(tmp_path / "co3")).returncode == 0
    dial = pid.attach_dial(pid.stand_in_sampler(0).unet)
    pid.save_dial(dial.state(), tmp_path / "zero.pt", {"dial_from_step": 2})
    pid.main(["draw", "--dry-run", "--ckpt", str(tmp_path / "zero.pt"), "--pairs-file", str(tmp_path / "pairs.txt"),
              "--seeds", "13001:13003", "--out", str(tmp_path / "dial")])
    for folder in ("co3", "dial"):
        assert json.loads((tmp_path / folder / "draw.json").read_text())["finish"] == pid.FINISH
    for slug, (a, b) in (("a_wolf__x__a_husky", ("wolf", "husky")), ("a_crow__x__a_raven", ("crow", "raven"))):
        for seed in (13001, 13002):
            ref = pid.stand_in_sampler(0)
            ref.set_pair(a, b)
            want = _as_array(ref.model.sample_loop(ref.start(seed))[0])
            for folder in ("co3", "dial"):
                got = np.asarray(Image.open(tmp_path / folder / slug / f"seed-{seed}.png").convert("RGB"))
                assert np.array_equal(got, want), (folder, slug, seed)


def test_co3_sampler_keeps_co3s_own_vae_and_decodes_finished_pictures_with_it():
    """co3_sampler cannot run here (it loads SDXL); its source must leave CO3's VAE in its own dtype, keep the graph
    decode on a copy, and give every finished picture to finish rather than to the graph decode."""
    src = DIAL.read_text()
    body = src[src.index("def co3_sampler("):src.index("# ---------------------------------------------------------------- inputs")]
    assert "model.vae.to(torch.float32)" not in body and "copy.deepcopy(model.vae).to(torch.float32)" in body
    for fn in ("def draw_rows(", "def log_tuning_pictures("):
        part = src[src.index(fn):]
        part = part[:part.index("\ndef ", 1)]
        assert "sampler.finish(x)" in part and "sampler.decode(x" not in part, fn


# ---- the closeness term's gradient survives fp16 (the smoke of 2026-11-28 stopped with the term at 6.05e-7, gradient 0.0)

def _fp16_world(up=1e-3, n=4096, size=100.0):
    """The stand-in U-Net under CPU fp16 autocast with a large output over many latent numbers, as SDXL's is, a dial whose
    change weights are `up` in size, and one cached state with its frozen prediction."""
    from types import SimpleNamespace
    torch.manual_seed(0)
    unet = pid.StandInUNet()
    for p in unet.parameters():
        p.requires_grad_(False)

    def eps(x, k, prompt):
        with torch.autocast("cpu", dtype=torch.float16):
            return size * unet(x)
    x = torch.randn(1, n, 8)
    with torch.no_grad():
        frozen = eps(x, 0, "a cat").float()
    dial = pid.attach_dial(unet)
    g = torch.Generator().manual_seed(1)
    for m in dial.layers.values():
        m.lora_up.weight.data = up * torch.randn(m.lora_up.weight.shape, generator=g)
    params = dial.parameters()
    for p in params:
        p.requires_grad_(True)
    dial.set_strength(1.0)
    return SimpleNamespace(unet=unet, prompt_eps=eps), params, [{"x": x, "k": 0, "prompt": "a cat", "eps": frozen}]


def _norm(params):
    return float(sum(p.grad.double().pow(2).sum() for p in params).sqrt())


def test_closeness_gradient_underflows_unscaled_and_survives_scaled():
    s, params, batch = _fp16_world()
    drift, _ = pid.closeness_backward(s, params, batch, 1.0, scale=1.0)
    assert 0 < drift < 1e-5 and _norm(params) == 0.0           # the smoke's failure: a term, and no gradient
    s, params, batch = _fp16_world()
    drift, used = pid.closeness_backward(s, params, batch, 1.0)
    assert used == pid.CLOSENESS_GRAD_SCALE
    # against the same term computed in fp32 throughout
    from types import SimpleNamespace
    ref_s, ref_p, ref_b = _fp16_world()
    plain = SimpleNamespace(unet=ref_s.unet, prompt_eps=lambda x, k, p: 100.0 * ref_s.unet(x))
    pid.closeness_backward(plain, ref_p, ref_b, 1.0, scale=1.0)
    assert _norm(params) == pytest.approx(_norm(ref_p), rel=0.1)


def test_closeness_backward_halves_an_overflowing_scale_and_keeps_the_task_gradient():
    s, params, batch = _fp16_world()
    for p in params:
        p.grad = torch.full_like(p, 1e-6)                         # the task term's gradient, already there
    _, used = pid.closeness_backward(s, params, batch, 1.0, scale=2.0 ** 100)
    assert used < 2.0 ** 100 and all(bool(torch.isfinite(p.grad).all()) for p in params)
    s2, params2, batch2 = _fp16_world()
    pid.closeness_backward(s2, params2, batch2, 1.0)
    for p, q in zip(params, params2):
        assert torch.allclose(p.grad - 1e-6, q.grad, rtol=0.05, atol=1e-10)


def _fp16_training(tmp_path, monkeypatch):
    """A dry-run training whose closeness term runs as on the card: fp16 autocast, a large output, 4,096 latent numbers."""
    real = pid.stand_in_sampler

    def fp16_sampler(seed=0):
        s = real(seed)
        m = s.model

        def prompt_eps(x, k, prompt):
            with torch.autocast("cpu", dtype=torch.float16):
                return 100.0 * m.eps(x, prompt)
        s.prompt_eps = prompt_eps
        return s
    monkeypatch.setattr(pid, "stand_in_sampler", fp16_sampler)
    frozen = fp16_sampler(0)
    entries = []
    for i, k in enumerate((2, 3, 4, 2)):
        x = torch.randn(1, 4096, 8, generator=torch.Generator().manual_seed(i))
        with torch.no_grad():
            entries.append({"prompt": "a cat", "seed": 21001 + i, "k": k, "x": x,
                            "eps": frozen.prompt_eps(x, k, "a cat").float()})
    torch.save({"kind": "stand-in", "dial_attached": False, "window": [2, 5], "entries": entries}, tmp_path / "cache.pt")
    return tmp_path / "cache.pt"


def test_closeness_gradient_reaches_the_dial_after_one_step_in_fp16(tmp_path, monkeypatch):
    cache = _fp16_training(tmp_path, monkeypatch)
    pid.main(["train", "--dry-run", "--steps", "3", "--closeness-cache", str(cache), "--out", str(tmp_path / "run")])
    log = [json.loads(x) for x in (tmp_path / "run/train-log.jsonl").read_text().splitlines()]
    assert log[0]["closeness_term"] == 0                          # a new dial: no drift at step 1
    assert log[1]["closeness_term"] > 0 and log[1]["closeness_grad_norm"] > 0
    assert log[1]["closeness_grad_scale"] == pid.CLOSENESS_GRAD_SCALE


def test_unscaled_fp16_closeness_stops_the_run_as_the_smoke_did(tmp_path, monkeypatch):
    cache = _fp16_training(tmp_path, monkeypatch)
    with pytest.raises(SystemExit, match=r"closeness term is exactly 0 at step 2, after the dial moved at step 1 \(term "
                                         r"[0-9.e-]+, gradient norm 0.0\)"):
        pid.main(["train", "--dry-run", "--steps", "3", "--closeness-cache", str(cache), "--closeness-grad-scale", "1",
                  "--out", str(tmp_path / "run")])


def test_a_cache_off_the_window_is_refused(tmp_path):
    cache = tmp_path / "cache.pt"
    assert run("closeness-cache", "--dry-run", "--window", "2", "4", "--out", str(cache)).returncode == 0
    r = run("train", "--dry-run", "--window", "4", "6", "--closeness-cache", str(cache), "--out", str(tmp_path / "run"))
    assert r.returncode != 0 and "outside the window" in r.stdout + r.stderr
