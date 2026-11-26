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
    prompts = pid.read_prompts(PAPER2 / "paper2_closeness_prompts.txt")
    plan = (REPO / "plans/paper-2-dials/plans/02-species-judge.md").read_text()
    table = plan[plan.index("### The pairs"):plan.index("### The labelled check set")]
    known = {frozenset(m) for row in table.splitlines() if row.startswith("| ")
             for m in re.findall(r"(\w+) and (\w+)", row.split("|")[1])}
    pooled = json.loads((REPO / "artifacts/notes/progress/leakage/10-co-occurrence-dial/pairs.json").read_text())
    known |= {frozenset(re.findall(r"(?:an?) (\w+)", p)) for p in pooled["pairs"]}
    singles = [p for p in prompts if " and " not in p]
    pairs = [p for p in prompts if " and " in p]
    assert singles and pairs
    for p in prompts:
        assert is_allowed(p), p
    for p in pairs:
        assert frozenset(re.findall(r"(?:an?) (\w+)", p)) not in known, p
