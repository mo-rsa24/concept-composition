"""Paper 1's head (2-classifier-guidance/scripts/paper1/paper1_disc.py) on a stand-in network with EDM2's decoder block names.

Needs torch, so it runs on the cluster (co3 or co3_bw python); on the laptop, which has no torch, it is skipped.

  python -m pytest tests/test_paper1_disc.py
"""
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "2-classifier-guidance" / "scripts" / "paper1"))
import paper1_disc as pd  # noqa: E402


class Block(nn.Module):
    def __init__(self, cin, cout, scale):
        super().__init__()
        self.conv, self.scale = nn.Conv2d(cin, cout, 1), scale

    def forward(self, x):
        if self.scale == "up":
            x = nn.functional.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class StandIn(nn.Module):
    """Decoder blocks named as edm2-img64-xs's: 8x8 at 512 channels, then 16x16 at 384, then 32x32 and 64x64."""

    def __init__(self):
        super().__init__()
        self.unet = nn.Module()
        dec = nn.ModuleDict()
        dec["8x8_in0"], dec["8x8_in1"] = Block(3, 512, None), Block(512, 512, None)
        for i in range(4):
            dec[f"8x8_block{i}"] = Block(512, 512, None)
        dec["16x16_up"] = Block(512, 384, "up")
        for i in range(4):
            dec[f"16x16_block{i}"] = Block(384, 384, None)
        dec["32x32_up"] = Block(384, 256, "up")
        self.unet.dec = dec
        self.calls = []

    def forward(self, x, sigma, labels):
        h = nn.functional.avg_pool2d(x, 8)
        for name, block in self.unet.dec.items():
            self.calls.append(name)
            h = block(h)
        return h


def test_features_stop_after_last_block_and_sampler_call_runs_through():
    net = StandIn()
    feats = pd.ExpertFeatures(net, ("8x8_block3", "16x16_block3"))
    x = torch.randn(2, 3, 64, 64)
    out = feats(x, torch.ones(2), torch.zeros(2, 3))
    assert out["8x8_block3"].shape == (2, 512, 8, 8) and out["16x16_block3"].shape == (2, 384, 16, 16)
    assert "32x32_up" not in net.calls                      # stopped after 16x16_block3
    net.calls.clear()
    net(x, torch.ones(2), torch.zeros(2, 3))                # an ordinary call (the sampler's) is not stopped
    assert "32x32_up" in net.calls


def test_logit_shape_weight_count_and_gradient_reaches_x():
    disc = pd.Discriminator(StandIn(), StandIn(), "8x8_block3", "16x16_block3")
    x = torch.randn(3, 3, 64, 64, requires_grad=True)
    count = torch.tensor([1, 2, 3])
    feats = disc.features(x, torch.full((3,), 2.0), count)
    assert feats.shape == (3, 2 * (384 + 512), 16, 16)
    disc.head = pd.Head(feats.shape[1])
    logit = disc.logit(x, torch.full((3,), 2.0), count)
    assert logit.shape == (3,)
    (g,) = torch.autograd.grad(logit.sum(), x)
    assert g.abs().sum() > 0
    assert pd.weight_count(disc.head) > 0


def test_balanced_accuracy():
    logits = torch.tensor([1.0, 1.0, -1.0, -1.0, 1.0])
    labels = torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0])
    assert pd.balanced_accuracy(logits, labels) == pytest.approx(0.5 * (1 + 2 / 3))


def test_folds_put_each_training_scene_in_one_fold_with_sizes_balanced():
    names = [f"count-{1 + i % 3}/{i:06d}.png" for i in (5, 0, 3, 1, 4, 2, 7, 6)]     # out of order, as count folders list them
    folds = pd.folds_of(names, 4)
    by_index = {pd.file_number(n): int(f) for n, f in zip(names, folds)}
    assert by_index == {0: 0, 1: 1, 2: 2, 3: 3, 4: 0, 5: 1, 6: 2, 7: 3}
    assert pd.CHOOSE_FOLDS == 4 and not hasattr(pd, "CHOOSE_HELDOUT")


def test_shifts_are_whole_pixels_with_edges_repeated_and_never_mirrored():
    x = torch.arange(64.0).view(1, 1, 1, 64).expand(3, 1, 64, 64).clone()
    out = pd.shifted(x, 4, torch.Generator().manual_seed(0))
    for row in out[:, 0, 0]:
        d = row[1:] - row[:-1]
        assert set(d.tolist()) <= {0.0, 1.0}                 # increasing left to right: shifted, never mirrored
        assert abs(row[32].item() - 32) <= 4


def test_logit_is_bounded():
    head = pd.Head(8, width=16, bound=4.0)
    with torch.no_grad():
        head.out.bias.fill_(1000.0)
    logit = head(torch.randn(2, 8, 16, 16), torch.tensor([1.0, 2.0]), torch.tensor([1, 2]))
    assert logit.abs().max() <= 4.0


def test_negatives_are_weighted_to_the_positives_touching_share(tmp_path):
    import json
    (tmp_path / "touching.json").write_text(json.dumps({"files": {}, "per_count": {"count-1": {"touching_share": 0.5}}}))
    split = tmp_path / "split.json"
    split.write_text(json.dumps({"negatives": {"count-1": {"touching": [1], "apart": [2, 3, 4]}}}))
    names = [f"count-1/{s:06d}.png" for s in (1, 2, 3, 4)]
    w, report = pd.touching_weights(tmp_path, names, torch.tensor([1, 1, 1, 1]), split)
    share = float(w[0] / w.sum())
    assert share == pytest.approx(0.5) and report["count-1"]["negatives_touching_share"] == 0.25


# --------------------------------------------------------------------------------------------------------------------------------------
# The alignment loss: the aligned rerun the first result's direction check can call for


class DStandIn(StandIn):
    """A stand-in that also denoises: its decoder blocks run (so the head's hooks see them), and D(x) = factor * x."""

    def __init__(self, factor):
        super().__init__()
        self.factor = factor

    def forward(self, x, sigma, labels):
        super().forward(x, sigma, labels)
        return self.factor * x


def _aligned_setup():
    torch.manual_seed(0)
    red, blue, empty = DStandIn(0.7), DStandIn(0.5), DStandIn(0.2)
    disc = pd.Discriminator(red, blue, "8x8_block3", "16x16_block3")
    disc.head = pd.Head(2 * (384 + 512), width=16)
    x0 = torch.randn(2, 3, 64, 64)
    sigma, count = torch.tensor([0.5, 1.5]), torch.tensor([1, 2])
    m_red, m_blue = torch.zeros(2, 1, 64, 64), torch.zeros(2, 1, 64, 64)
    m_red[..., 10:30, 10:30] = 1
    m_blue[..., 25:45, 25:45] = 1
    return disc, [red, blue, empty], x0, sigma, count, m_red, m_blue


def test_alignment_term_is_positive_and_trains_the_head_through_the_gradient():
    disc, nets3, x0, sigma, count, mr, mb = _aligned_setup()
    loss = pd.alignment_loss(disc, nets3, x0, sigma, count, mr, mb, torch.Generator().manual_seed(1))
    assert loss.item() > 0
    loss.backward()
    assert sum(p.grad.abs().sum().item() for p in disc.head.parameters() if p.grad is not None) > 0


def test_alignment_term_equals_one_minus_the_cosine_by_hand():
    disc, nets3, x0, sigma, count, mr, mb = _aligned_setup()
    loss = pd.alignment_loss(disc, nets3, x0, sigma, count, mr, mb, torch.Generator().manual_seed(1))
    noise = torch.randn(x0.shape, generator=torch.Generator().manual_seed(1))
    xt = x0 + sigma.view(-1, 1, 1, 1) * noise
    d_region = mr * 0.7 * xt + mb * 0.5 * xt + (1 - mr - mb) * 0.2 * xt
    target = x0 - d_region
    xx = xt.clone().requires_grad_(True)
    (g,) = torch.autograd.grad(disc.logit(xx, sigma, count).sum(), xx)
    push = sigma.view(-1, 1, 1, 1) ** 2 * g
    by_hand = (1 - torch.nn.functional.cosine_similarity(push.flatten(1), target.flatten(1), dim=1)).mean()
    assert loss.item() == pytest.approx(by_hand.item(), abs=1e-5)


def test_masks_are_dilated_by_one_pixel_and_a_missing_one_stops(tmp_path):
    from PIL import Image
    import numpy as np
    (tmp_path / "masks" / "count-1").mkdir(parents=True)
    for colour in ("red", "blue"):
        m = np.zeros((64, 64), np.uint8)
        m[20, 20] = 255
        Image.fromarray(m).save(tmp_path / "masks" / "count-1" / f"000003-{colour}.png")
    red, blue = pd.load_masks(tmp_path, ["count-1/000003.png"])
    assert red[0, 0].sum().item() == 9 and red[0, 0, 19:22, 19:22].min().item() == 1
    with pytest.raises(SystemExit, match="mask files missing"):
        pd.load_masks(tmp_path, ["count-1/000004.png"])


def test_with_weight_zero_the_term_is_absent_and_no_empty_model_loads(monkeypatch, tmp_path):
    import argparse
    loaded = []
    monkeypatch.setattr(pd.ps, "use_edm2", lambda path: None)
    monkeypatch.setattr(pd.ps, "load_net", lambda path: (loaded.append(str(path)) or DStandIn(0.5), None))
    args = argparse.Namespace(align_weight=0.0, edm2="x", red_net="red.pkl", blue_net="blue.pkl", empty_net=str(tmp_path / "none.pkl"))
    assert len(pd.load_nets_for(args)) == 2 and loaded == ["red.pkl", "blue.pkl"]
    args.align_weight = pd.ALIGN_WEIGHT_ON
    with pytest.raises(SystemExit, match="no empty-scene model"):
        pd.load_nets_for(args)
    assert pd.ALIGN_WEIGHT_ON == 1.0 and pd.ALIGN_WEIGHT == 0.0


def test_training_records_the_alignment_term_only_when_on():
    import argparse
    disc, nets3, x0, sigma, count, mr, mb = _aligned_setup()
    args = argparse.Namespace(width=16, logit_bound=4.0, lr=1e-3, weight_decay=0.01, batch_pos=4, batch_neg=4, shift=1,
                              sigma_lo=0.28, sigma_hi=10.5, align_weight=0.0)
    pos = (torch.randn(4, 3, 64, 64), torch.tensor([1, 2, 3, 1]))
    neg = (torch.randn(4, 3, 64, 64), torch.tensor([1, 2, 3, 1]))
    off = pd.train_head(disc, pos, neg, args, 2, 0)
    assert off["align_loss_mean"] is None
    args.align_weight = 1.0
    masks = (torch.ones(4, 1, 64, 64), torch.zeros(4, 1, 64, 64))
    on = pd.train_head(disc, pos, neg, args, 2, 0, align=(nets3, masks[0], masks[1]))
    assert on["align_loss_mean"] is not None and on["align_loss_mean"] > 0


def test_direction_cosines_give_real_and_shuffled_per_band(tmp_path):
    from PIL import Image
    import numpy as np
    disc, nets3, *_ = _aligned_setup()
    (tmp_path / "count-1").mkdir(); (tmp_path / "masks" / "count-1").mkdir(parents=True)
    for i in range(3):
        Image.fromarray(np.random.default_rng(i).integers(0, 255, (64, 64, 3), dtype=np.uint8)).save(tmp_path / "count-1" / f"{i:06d}.png")
        for colour in ("red", "blue"):
            Image.fromarray(np.full((64, 64), 255 if colour == "red" else 0, np.uint8)).save(tmp_path / "masks" / "count-1" / f"{i:06d}-{colour}.png")
    out = pd.direction_cosines(disc, nets3, pd.load_folder(tmp_path), tmp_path, seed=0, draws=2)
    assert list(out) == [f"{hi}-{lo}" for hi, lo in pd.BANDS]
    assert all(len(v["real"]) == 6 and len(v["shuffled"]) == 6 for v in out.values())
    assert all(-1 <= c <= 1 for v in out.values() for c in v["real"])
