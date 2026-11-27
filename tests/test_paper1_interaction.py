"""Paper 1's Method A, the learned interaction term (2-classifier-guidance/scripts/paper1/paper1_interaction.py), on stand-in experts.

Needs torch, so it runs on the cluster (co3 or co3_bw python) or with a CPU build of torch; without torch it is skipped.

  python -m pytest tests/test_paper1_interaction.py
"""
import json
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "2-classifier-guidance" / "scripts" / "paper1"))
import paper1_sample as ps  # noqa: E402
import paper1_interaction as pi  # noqa: E402

SMALL = 8   # delta's base width in these tests, so a CPU runs them in seconds


def _stub_nets():
    """Three stand-in experts that differ from each other, read their label, and depend on sigma like a denoiser."""
    def expert(f):
        def net(x, s, labels):
            skip = 0.25 / (s.view(-1, 1, 1, 1) ** 2 + 0.25)
            shift = 0.0 if labels is None else 0.05 * labels.argmax(1).view(-1, 1, 1, 1).float()
            return torch.tanh(x * skip * f) + shift
        return net
    return [expert(f) for f in (1.0, 0.8, 0.6)]


def _random_delta(width=SMALL, seed=0):
    torch.manual_seed(seed)
    model = pi.InteractionNet(width)
    for p in model.parameters():
        torch.nn.init.normal_(p, std=0.05)
    return model.eval()


# ---------------------------------------------------------------------------------------------------------------------------------------
# With delta zeroed, D_new is the plain sum exactly


def test_untrained_delta_is_zero_and_d_new_is_the_plain_sum_exactly():
    nets, n = _stub_nets(), 4
    x = torch.randn(n, 3, 64, 64, generator=torch.Generator().manual_seed(1)) * 5
    plain = ps.plainsum_denoiser(nets, ps.one_hot(2, n), ps.one_hot(2, n))
    for step, s in ((0, ps.sigmas("cpu")[0]), (16, ps.sigmas("cpu")[16]), (31, ps.sigmas("cpu")[31])):
        fresh = pi.interaction_denoiser(nets, pi.InteractionNet(SMALL).eval(), 2, n)
        assert torch.equal(fresh(x, s, step), plain(x, s, step))
        trained_zeroed = pi.interaction_denoiser(nets, _random_delta(), 2, n, zero_delta=True)
        assert torch.equal(trained_zeroed(x, s, step), plain(x, s, step))
        trained = pi.interaction_denoiser(nets, _random_delta(), 2, n)
        assert not torch.equal(trained(x, s, step), plain(x, s, step))      # a non-zero delta does change D
    out = pi.InteractionNet(SMALL)(x, torch.full((n,), 1.0), torch.full((n,), 2), x, x, x)
    assert out.shape == (n, 3, 64, 64) and out.abs().max() == 0


def test_loss_settings_are_the_experts_and_the_weight_is_one_over_c_out_squared():
    assert (pi.P_MEAN, pi.P_STD, pi.SIGMA_DATA) == (-0.8, 1.6, 0.5)
    s = torch.tensor([0.002, 0.1, 0.5, 2.9, 80.0], dtype=torch.float64)
    assert torch.allclose(pi.loss_weight(s), 1 / pi.c_out(s) ** 2)
    # lambda (D_new - x0)^2 is the plan's target distance (F - (x0 - D_sum) / c_out)^2
    g = torch.Generator().manual_seed(2)
    x0, d_sum, f = (torch.randn(5, 3, 4, 4, generator=g, dtype=torch.float64) for _ in range(3))
    s4 = s.view(-1, 1, 1, 1)
    d_new = d_sum + pi.c_out(s4) * f
    assert torch.allclose(pi.loss_weight(s4) * (d_new - x0) ** 2, (f - (x0 - d_sum) / pi.c_out(s4)) ** 2)
    sig = pi.draw_sigma(200000, torch.Generator().manual_seed(3)).log()
    assert abs(float(sig.mean()) + 0.8) < 0.02 and abs(float(sig.std()) - 1.6) < 0.02


# ---------------------------------------------------------------------------------------------------------------------------------------
# sample writes byte-identical pictures to paper1_sample.py plainsum at the same seeds


class StubEncoder:
    """EDM2's StandardRGBEncoder arithmetic (training/encoders.py)."""

    def encode(self, x):
        return x.to(torch.float32) / 127.5 - 1

    def decode(self, x):
        return (x.to(torch.float32) * 127.5 + 128).clip(0, 255).to(torch.uint8)


def _seed_noise(seeds):
    """Stand-in for EDM2's per-seed generator: seed s gives its own fixed Gaussian noise."""
    return torch.stack([torch.randn(3, 64, 64, generator=torch.Generator().manual_seed(s)) for s in seeds]), None


@pytest.fixture
def stub_models(monkeypatch):
    nets = _stub_nets()
    monkeypatch.setattr(ps, "load_three", lambda args: (nets, StubEncoder()))
    monkeypatch.setattr(ps, "unit_noise", _seed_noise)
    return nets


def _bytes(folder, k, seeds):
    return [(Path(folder) / f"count-{k}" / f"{s:06d}.png").read_bytes() for s in seeds]


def test_sample_with_delta_zeroed_writes_the_plain_sums_pictures_byte_for_byte(tmp_path, stub_models):
    seeds, counts = [5, 6, 7], (1, 3)
    ps.main(["plainsum", "--run", "plain", "--root", str(tmp_path), "--counts", "1,3", "--seeds", "5-7", "--batch", "2"])
    fresh = tmp_path / "fresh.pt"
    pi.save_checkpoint(fresh, pi.InteractionNet(SMALL), {"note": "untrained"}, 0, 1.0)
    trained = tmp_path / "trained.pt"
    pi.save_checkpoint(trained, _random_delta(), {"note": "random"}, 7, 0.5)
    common = ["--root", str(tmp_path), "--counts", "1,3", "--seeds", "5-7", "--batch", "2"]
    pi.main(["sample", "--run", "untrained", "--checkpoint", str(fresh)] + common)
    pi.main(["sample", "--run", "zeroed", "--checkpoint", str(trained), "--zero-delta"] + common)
    pi.main(["sample", "--run", "with-delta", "--checkpoint", str(trained)] + common)
    for k in counts:
        plain = _bytes(tmp_path / "plain", k, seeds)
        assert _bytes(tmp_path / "untrained", k, seeds) == plain
        assert _bytes(tmp_path / "zeroed", k, seeds) == plain
        assert _bytes(tmp_path / "with-delta", k, seeds) != plain
    timing = json.loads((tmp_path / "zeroed" / "timing.jsonl").read_text())
    assert timing["zero_delta"] is True and timing["pictures"] == 6 and timing["checkpoint_update"] == 7


def test_sample_takes_seeds_from_a_split_file_as_region_does(tmp_path, stub_models):
    split = tmp_path / "split.json"
    split.write_text(json.dumps({"evaluation": {"count-2": {"touching": [9, 4], "apart": [5]}}}))
    ckpt = tmp_path / "c.pt"
    pi.save_checkpoint(ckpt, pi.InteractionNet(SMALL), {}, 0, 1.0)
    pi.main(["sample", "--run", "s", "--root", str(tmp_path), "--checkpoint", str(ckpt), "--counts", "2", "--split", str(split),
             "--set", "evaluation", "--side", "touching"])
    assert sorted(p.name for p in (tmp_path / "s" / "count-2").iterdir()) == ["000004.png", "000009.png"]


def test_sample_needs_a_checkpoint_unless_delta_is_zeroed(tmp_path, stub_models):
    with pytest.raises(SystemExit):
        pi.main(["sample", "--run", "s", "--root", str(tmp_path), "--counts", "1", "--seeds", "1"])


# ---------------------------------------------------------------------------------------------------------------------------------------
# Training on a stub target


def _positives(folder, n=20):
    """A pack-layout folder: scene i holds (i mod 3) + 1 of each colour, drawn as squares on grey, named by render index."""
    rng = np.random.default_rng(0)
    for i in range(n):
        k = i % 3 + 1
        img = np.full((64, 64, 3), 128, np.uint8)
        for j in range(k):
            r, c = rng.integers(4, 44, 2)
            img[r:r + 12, c:c + 12] = (174, 36, 36)
            r, c = rng.integers(4, 44, 2)
            img[r:r + 12, c:c + 12] = (41, 74, 214)
        (folder / f"count-{k}").mkdir(parents=True, exist_ok=True)
        Image.fromarray(img, "RGB").save(folder / f"count-{k}" / f"{i:06d}.png")
    return folder


def test_training_loss_falls_on_a_stub_target(tmp_path, stub_models):
    pos = _positives(tmp_path / "pack" / "k20-s1")
    pi.main(["train", "--run", "t", "--root", str(tmp_path), "--positives", str(pos), "--updates", "100", "--batch", "8",
             "--width", str(SMALL), "--lr", "2e-3", "--val-every", "25", "--no-wandb"])
    out = json.loads((tmp_path / "t" / "train.json").read_text())
    curve, val = out["loss_curve"], [v["validation_loss"] for v in out["validation_curve"]]
    print(json.dumps({"loss_curve": curve, "validation_curve": val}))
    assert len(curve) == 2 and curve[-1] < curve[0]
    assert [v["update"] for v in out["validation_curve"]] == [0, 25, 50, 75, 100]
    assert out["validation_loss_plain_sum"] == val[0] and min(val[1:]) < val[0]
    assert out["best_update"] > 0 and out["best_validation_loss"] == pytest.approx(min(val), abs=1e-6)
    s = out["settings"]
    assert (s["P_mean"], s["P_std"], s["sigma_data"]) == (-0.8, 1.6, 0.5)
    assert s["train_scenes"] == 18 and s["validation_scenes"] == 2 and s["delta_weights"] == pi.weight_count(pi.InteractionNet(SMALL))
    model, record = pi.load_interaction(tmp_path / "t" / "interaction.pt")
    assert record["update"] == out["best_update"]


def test_only_delta_is_trained(stub_models):
    model = pi.InteractionNet(SMALL)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    g = torch.Generator().manual_seed(4)
    x0 = torch.rand(4, 3, 64, 64, generator=g) * 2 - 1
    loss = pi.training_loss(stub_models, model, x0, torch.tensor([1, 2, 3, 1]), pi.draw_sigma(4, g), torch.randn(x0.shape, generator=g))
    loss.backward()
    opt.step()
    assert model.conv_out.weight.abs().max() > 0                              # the zero layer moves on the first update


# ---------------------------------------------------------------------------------------------------------------------------------------
# The held-out split and the checkpoint


def test_validation_split_never_shares_a_scene_with_training():
    import hashlib
    g = torch.Generator().manual_seed(5)
    pics = torch.rand(40, 3, 64, 64, generator=g)
    names = [f"count-{i % 3 + 1}/{i + 100:06d}.png" for i in range(40)]
    pics[30] = pics[0]                                                        # the same picture under two render indices
    pics[11] = pics[2]
    train, val = pi.split_scenes(pics, names)
    assert sorted(train + val) == list(range(40)) and not set(train) & set(val)
    digest = lambda i: hashlib.sha256(pics[i].numpy().tobytes()).hexdigest()
    assert not {digest(i) for i in train} & {digest(i) for i in val}         # no picture on both sides
    assert {0, 30} <= set(val) and {2, 11} <= set(train)                     # a duplicate follows its first copy
    assert len(val) == 5                                                      # 38 groups: ranks 0, 10, 20, 30 held out, one with its copy
    pics2 = torch.rand(384, 3, 64, 64, generator=g)
    names2 = [f"count-{i % 3 + 1}/{i:06d}.png" for i in range(384)]
    train2, val2 = pi.split_scenes(pics2, names2)
    assert len(val2) == 39 and val2 == list(range(0, 384, 10))                # 10 percent, every tenth render index
    counts = [int(names2[i].split("/")[0][-1]) for i in val2]
    assert all(counts.count(k) >= 12 for k in (1, 2, 3))                      # every count is held out


def test_checkpoint_round_trips(tmp_path):
    model = _random_delta(seed=6)
    path = tmp_path / "interaction.pt"
    pi.save_checkpoint(path, model, {"lr": 2e-4, "positives": "p"}, 250, 0.123)
    loaded, record = pi.load_interaction(path)
    assert record["update"] == 250 and record["validation_loss"] == 0.123 and record["settings"]["lr"] == 2e-4
    assert record["width"] == SMALL
    a, b = model.state_dict(), loaded.state_dict()
    assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)
    g = torch.Generator().manual_seed(7)
    x, d = torch.randn(2, 3, 64, 64, generator=g), [torch.randn(2, 3, 64, 64, generator=g) for _ in range(3)]
    s, c = torch.tensor([0.3, 4.0]), torch.tensor([1, 3])
    with torch.no_grad():
        assert torch.equal(model(x, s, c, *d), loaded(x, s, c, *d))
    assert not any(p.requires_grad for p in loaded.parameters())


def test_weight_count_at_the_default_width():
    model = pi.InteractionNet()
    print(json.dumps({"width": pi.WIDTH, "delta_weights": pi.weight_count(model),
                      "uncertainty_weights": pi.weight_count(model, True) - pi.weight_count(model)}))
    assert pi.WIDTH == 96 and pi.weight_count(model) == 2_968_739 and pi.weight_count(model, True) - pi.weight_count(model) == 33
