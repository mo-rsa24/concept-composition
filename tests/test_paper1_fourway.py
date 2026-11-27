"""Paper 1's Method B, the four-way classifier and its push (2-classifier-guidance/scripts/paper1/paper1_fourway.py), on stand-in experts.

Needs torch, so it runs on the cluster (co3 or co3_bw python) or with a CPU build of torch; without torch it is skipped.

  CUDA_VISIBLE_DEVICES="" python -m pytest -q tests/test_paper1_fourway.py
"""
import io
import json
import sys
import zipfile
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
import numpy as np  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "2-classifier-guidance" / "scripts" / "paper1"))
import paper1_sample as ps  # noqa: E402
import paper1_disc as pdisc  # noqa: E402
import paper1_fourway as pf  # noqa: E402

SMALL = 16   # the head's width in these tests (a multiple of its 8 GroupNorm groups)
RED, BLUE = (174, 36, 36), (41, 74, 214)


# ---------------------------------------------------------------------------------------------------------------------------------------
# Stand-ins


def _stub_nets():
    """Three stand-in experts that differ, read their label, and depend on sigma like a denoiser (as test_paper1_interaction's)."""
    def expert(f):
        def net(x, s, labels):
            skip = 0.25 / (s.view(-1, 1, 1, 1) ** 2 + 0.25)
            shift = 0.0 if labels is None else 0.05 * labels.argmax(1).view(-1, 1, 1, 1).float()
            return torch.tanh(x * skip * f) + shift
        return net
    return [expert(f) for f in (1.0, 0.8, 0.6)]


def _stub_features(red, blue, block8, block16):
    """In place of the experts' decoder features: the picture averaged to 16 by 16, with a count channel."""
    def features(x, sigma, count):
        h = F.avg_pool2d(x.float(), 4)
        return torch.cat([h, count.float().view(-1, 1, 1, 1).expand(len(x), 1, 16, 16) / 3], 1)
    return features


class StubEncoder:
    """EDM2's StandardRGBEncoder arithmetic (training/encoders.py)."""

    def encode(self, x):
        return x.to(torch.float32) / 127.5 - 1

    def decode(self, x):
        return (x.to(torch.float32) * 127.5 + 128).clip(0, 255).to(torch.uint8)


def _seed_noise(seeds):
    return torch.stack([torch.randn(3, 64, 64, generator=torch.Generator().manual_seed(s)) for s in seeds]), None


@pytest.fixture
def stubs(monkeypatch):
    nets = _stub_nets()
    monkeypatch.setattr(ps, "load_three", lambda args: (nets, StubEncoder()))
    monkeypatch.setattr(ps, "unit_noise", _seed_noise)
    monkeypatch.setattr(pdisc, "load_pair", lambda args: (nets[0], nets[1]))
    monkeypatch.setattr(pf, "make_features", _stub_features)
    return nets


def _scene(rng, n_red, n_blue):
    img = np.full((64, 64, 3), 128, np.uint8) + rng.integers(0, 3, (64, 64, 3)).astype(np.uint8)   # backgrounds differ slightly
    for colour, n in ((RED, n_red), (BLUE, n_blue)):
        for _ in range(n):
            r, c = rng.integers(4, 48, 2)
            img[r:r + 10, c:c + 10] = colour
    return img


def _write_zip(path, pictures, labels=None):
    """An EDM2 dataset zip in clevr_data.py's layout: <index // 1000:05d>/img<index:08d>.png, and dataset.json when labelled."""
    entries = []
    with zipfile.ZipFile(path, "w") as z:
        for i, img in enumerate(pictures):
            name = f"{i // 1000:05d}/img{i:08d}.png"
            buf = io.BytesIO()
            Image.fromarray(img, "RGB").save(buf, format="PNG")
            z.writestr(name, buf.getvalue())
            entries.append([name, labels[i]] if labels is not None else None)
        if labels is not None:
            z.writestr("dataset.json", json.dumps({"labels": entries}))
    return path


def _data(tmp_path, per_count=10, empties=20, joints=30, duplicate=False):
    """Stub training sets: red and blue zips (per_count pictures at each count), an empty zip, and a joint pack folder."""
    rng = np.random.default_rng(0)
    red = [_scene(rng, k, 0) for k in (1, 2, 3) for _ in range(per_count)]
    blue = [_scene(rng, 0, k) for k in (1, 2, 3) for _ in range(per_count)]
    labels = [k - 1 for k in (1, 2, 3) for _ in range(per_count)]
    empty = [_scene(rng, 0, 0) for _ in range(empties)]
    if duplicate:                       # picture 1 repeated at 11: the pair must sit on one side of the split
        red[11] = red[1].copy()
    pack = tmp_path / "pack" / "k-s1"
    for i in range(joints):
        k = i % 3 + 1
        (pack / f"count-{k}").mkdir(parents=True, exist_ok=True)
        Image.fromarray(_scene(rng, k, k), "RGB").save(pack / f"count-{k}" / f"{i:06d}.png")
    return {"red": _write_zip(tmp_path / "r.zip", red, labels), "blue": _write_zip(tmp_path / "b.zip", blue, labels),
            "empty": _write_zip(tmp_path / "e.zip", empty), "pack": pack}


def _train_args(tmp_path, data, *extra):
    return ["train", "--run", "clf", "--root", str(tmp_path), "--positives", str(data["pack"]), "--red-zip", str(data["red"]),
            "--blue-zip", str(data["blue"]), "--empty-zip", str(data["empty"]), "--width", str(SMALL), "--per-class", "4",
            "--no-wandb", *extra]


# ---------------------------------------------------------------------------------------------------------------------------------------
# The ratio and its gradient


class LinearHead(nn.Module):
    """logits = A x: four linear scores of the flattened picture."""

    def __init__(self, a):
        super().__init__()
        self.a = a

    def forward(self, feats, sigma, count):
        return feats.flatten(1) @ self.a.T


def _hand_gradient(a, x):
    """d/dx [log q11 - log(q10 + q11) - log(q01 + q11)] by hand, with q = softmax(A x)."""
    q = torch.softmax(x.flatten(1) @ a.T, 1)
    q00, q10, q01, q11 = q.unbind(1)
    coef = torch.stack([q00, q10 - q10 / (q10 + q11), q01 - q01 / (q01 + q11),
                        1 + q11 - q11 / (q10 + q11) - q11 / (q01 + q11)], 1)
    return (coef @ a).view_as(x), q


def test_the_ratio_and_its_gradient_equal_the_hand_computed_ones():
    g = torch.Generator().manual_seed(3)
    a = torch.randn(4, 3 * 4 * 4, generator=g, dtype=torch.float64)
    x = torch.randn(5, 3, 4, 4, generator=g, dtype=torch.float64, requires_grad=True)
    clf = pf.FourWay(lambda x, s, c: x, LinearHead(a))
    s, c = torch.full((5,), 2.0, dtype=torch.float64), torch.full((5,), 2)
    ratio = clf.logit(x, s, c)
    grad, = torch.autograd.grad(ratio.sum(), x)
    hand, q = _hand_gradient(a, x.detach())
    q00, q10, q01, q11 = q.unbind(1)
    assert torch.allclose(ratio.detach(), q11.log() - (q10 + q11).log() - (q01 + q11).log(), atol=1e-12)
    assert torch.allclose(grad, hand, atol=1e-10)


def test_the_push_adds_w_sigma_squared_times_the_ratio_gradient(stubs):
    nets, n = stubs, 2
    g = torch.Generator().manual_seed(4)
    a = torch.randn(4, 3 * 64 * 64, generator=g) * 0.01
    clf = pf.FourWay(lambda x, s, c: x, LinearHead(a))
    m = torch.zeros(n, 1, 64, 64)
    m[..., 20:30, 20:30] = 1
    x = torch.randn(n, 3, 64, 64, generator=g)
    s = ps.sigmas("cpu")[16]
    out = ps.push_denoiser(nets, clf, 2, m, m, 3.0, window=ps.parse_window("0-31"))(x, s, 16)
    base = ps.region_denoiser(nets, ps.one_hot(2, n), ps.one_hot(2, n), m, m)(x, s, 16)
    hand, _ = _hand_gradient(a.double(), x.double())
    assert torch.allclose((out - base).double(), 3.0 * float(s) ** 2 * hand, rtol=1e-3, atol=1e-4)


def test_each_logit_and_the_ratio_stay_bounded():
    head = pf.FourWayHead(4, SMALL)
    for p in head.parameters():
        nn.init.normal_(p, std=50.0)                           # raw scores far beyond the bound
    logits = head(torch.randn(6, 4, 16, 16), torch.full((6,), 0.5), torch.tensor([1, 2, 3, 1, 2, 3]))
    assert logits.shape == (6, 4) and logits.abs().max() <= pf.LOGIT_BOUND
    corners = torch.cartesian_prod(*[torch.tensor([-4.0, 4.0], dtype=torch.float64)] * 4)   # every extreme of four bounded logits
    assert pf.interaction_ratio(corners).abs().max() <= pf.RATIO_BOUND


# ---------------------------------------------------------------------------------------------------------------------------------------
# w = 0 draws the base sampler's pictures byte for byte


def _lone_folders(tmp_path, seeds, counts):
    rng = np.random.default_rng(5)
    for colour, folder in (("red", "lone-red"), ("blue", "lone-blue")):
        for k in counts:
            out = tmp_path / folder / f"count-{k}"
            out.mkdir(parents=True, exist_ok=True)
            for s in seeds:
                img = _scene(rng, k if colour == "red" else 0, k if colour == "blue" else 0)
                Image.fromarray(img, "RGB").save(out / f"{s:06d}.png")
    torch.save(torch.full((64, 64, 3), 128 / 255), tmp_path / "mean-empty.pt")
    return ["--red-dir", str(tmp_path / "lone-red"), "--blue-dir", str(tmp_path / "lone-blue"), "--mean-empty", str(tmp_path / "mean-empty.pt")]


def _random_fourway(path, in_channels=4):
    torch.manual_seed(0)
    head = pf.FourWayHead(in_channels, SMALL)
    for p in head.parameters():
        nn.init.normal_(p, std=0.3)
    torch.save({"head": head.state_dict(), "block8": "b8", "block16": "b16", "in_channels": in_channels, "width": SMALL,
                "logit_bound": pf.LOGIT_BOUND, "classes": list(pf.CLASSES), "update": 0, "settings": {}}, path)
    return path


def _bytes(folder, k, seeds):
    return [(Path(folder) / f"count-{k}" / f"{s:06d}.png").read_bytes() for s in seeds]


class LinearDisc:
    def logit(self, x, s, count):
        return (x * 0.01).flatten(1).sum(1)


def test_w_zero_draws_the_existing_push_and_the_region_samplers_pictures_byte_for_byte(tmp_path, stubs, monkeypatch):
    seeds, counts = [5, 6, 7], (1, 3)
    lone = _lone_folders(tmp_path, seeds, counts)
    common = ["--root", str(tmp_path), "--counts", "1,3", "--seeds", "5-7", "--batch", "2"] + lone
    head = _random_fourway(tmp_path / "fourway.pt")
    ps.main(["region", "--run", "region"] + common)
    monkeypatch.setattr(pdisc, "load_discriminator", lambda path, red, blue: LinearDisc())
    ps.main(["push", "--run", "old-push", "--head", "unused", "--weights", "0", "--window", "0-31"] + common)
    pf.main(["push", "--run", "b-w0", "--head", str(head), "--weights", "0"] + common)
    pf.main(["push", "--run", "b-w64", "--head", str(head), "--weights", "64"] + common)
    for k in counts:
        region = _bytes(tmp_path / "region", k, seeds)
        assert _bytes(tmp_path / "old-push", k, seeds) == region
        assert _bytes(tmp_path / "b-w0", k, seeds) == region
        assert _bytes(tmp_path / "b-w64", k, seeds) != region           # a non-zero weight does move the pictures
    push = json.loads((tmp_path / "b-w64" / "push.json").read_text())
    assert push["base"] == "region" and push["pushed_evaluations"] == 63 and push["pictures"] == 6
    trace = push["weights"]["64"]["classifier_along_trajectory"]
    assert set(trace) == {pf.band_name(b) for b in pf.VAL_BANDS}
    assert sum(v["evaluations"] for v in trace.values()) == 63 * 6
    assert all(abs(sum(v["mean_probability"].values()) - 1) < 1e-3 for v in trace.values())


def test_plainsum_base_at_w_zero_is_the_plain_sum_byte_for_byte(tmp_path, stubs):
    seeds = [5, 6, 7]
    lone = _lone_folders(tmp_path, seeds, (2,))
    head = _random_fourway(tmp_path / "fourway.pt")
    ps.main(["plainsum", "--run", "plain", "--root", str(tmp_path), "--counts", "2", "--seeds", "5-7", "--batch", "2"])
    pf.main(["push", "--run", "b-plain", "--root", str(tmp_path), "--head", str(head), "--weights", "0", "--base", "plainsum",
             "--counts", "2", "--seeds", "5-7", "--batch", "2"] + lone)
    assert _bytes(tmp_path / "b-plain", 2, seeds) == _bytes(tmp_path / "plain", 2, seeds)


def test_push_takes_seeds_from_the_split_file_and_writes_one_folder_per_weight(tmp_path, stubs):
    seeds = [4, 9]
    lone = _lone_folders(tmp_path, seeds, (2,))
    split = tmp_path / "split.json"
    split.write_text(json.dumps({"evaluation": {"count-2": {"touching": [9, 4], "apart": [5]}}}))
    head = _random_fourway(tmp_path / "fourway.pt")
    pf.main(["push", "--run", "w", "--root", str(tmp_path), "--head", str(head), "--weights", "1,4", "--counts", "2",
             "--split", str(split), "--set", "evaluation", "--side", "touching"] + lone)
    for w in ("w-1", "w-4"):
        assert sorted(p.name for p in (tmp_path / "w" / w / "count-2").iterdir()) == ["000004.png", "000009.png"]


# ---------------------------------------------------------------------------------------------------------------------------------------
# Data: balanced batches and held-out pictures


class _Args:
    def __init__(self, tmp_path, data):
        self.root, self.positives = str(tmp_path), str(data["pack"])
        self.red_zip, self.blue_zip, self.empty_zip = str(data["red"]), str(data["blue"]), str(data["empty"])


def test_every_batch_holds_each_class_equally_with_the_same_counts(tmp_path):
    pools = pf.load_pools(_Args(tmp_path, _data(tmp_path)))
    assert [p["name"] for p in pools] == list(pf.CLASSES)
    gen = torch.Generator().manual_seed(0)
    for _ in range(20):
        pics, counts, labels, picks = pf.draw_batch(pools, 8, gen)
        assert pics.shape == (32, 3, 64, 64) and pics.dtype == torch.uint8
        assert torch.bincount(labels, minlength=4).tolist() == [8, 8, 8, 8]
        per_class = counts.view(4, 8)
        assert all(torch.equal(per_class[0], per_class[c]) for c in range(4))     # the count says nothing about the class
        for cls in (pf.RED, pf.BLUE, pf.BOTH):                                     # each picture shows the count it is labelled with
            assert torch.equal(pools[cls]["counts"][picks[cls]], per_class[cls])


def test_held_out_pictures_are_every_tenth_and_never_drawn(tmp_path):
    pools = pf.load_pools(_Args(tmp_path, _data(tmp_path, duplicate=True)))
    red, empty, both = pools[pf.RED], pools[pf.NEITHER], pools[pf.BOTH]
    assert empty["val"] == [0, 10] and len(empty["train"]) == 18
    # picture 11 repeats picture 1: the twins form one group at rank 1, so both train, and the held-out ranks 0, 10, 20 are pictures
    # 0, 10 and 21
    assert red["val"] == [0, 10, 21] and {1, 11} <= set(red["train"])
    assert sorted(pdisc.file_number(both["names"][i]) for i in both["val"]) == [0, 10, 20]     # render indices, Method A's split
    gen = torch.Generator().manual_seed(1)
    drawn = [set() for _ in pools]
    for _ in range(300):
        for seen, p in zip(drawn, pf.draw_batch(pools, 8, gen)[3]):
            seen.update(p.tolist())
    for pool, seen in zip(pools, drawn):
        assert not seen & set(pool["val"])
        assert seen == set(pool["train"])                  # and every training picture does get drawn
        hashes = lambda idx: {pool["pics"][i].numpy().tobytes() for i in idx}
        assert not hashes(pool["train"]) & hashes(pool["val"])


# ---------------------------------------------------------------------------------------------------------------------------------------
# Training end to end on stubs


def test_training_writes_validation_per_band_and_a_checkpoint_the_push_loads(tmp_path, stubs):
    data = _data(tmp_path)
    pf.main(_train_args(tmp_path, data, "--updates", "40", "--val-every", "20", "--lr", "2e-3"))
    out = json.loads((tmp_path / "clf" / "train.json").read_text())
    assert [v["update"] for v in out["validation_curve"]] == [20, 40]
    bands = out["validation_final"]["bands"]
    assert [b["band"] for b in bands] == [pf.band_name(b) for b in pf.VAL_BANDS]
    assert bands[0]["band"] == "80-10.5" and bands[-1]["band"] == "0.28-0.002"
    assert bands[-1]["examples_per_class"] == {"neither": 4, "red": 6, "blue": 6, "both": 6}   # 2 draws of 2, 3, 3, 3 held out
    assert all(v == 0 for v in out["training"]["drawn_held_out"].values())
    curve = out["training"]["loss_curve"]
    assert len(out["training"]["train_accuracy_curve"]) == 1 and curve[0] > 0
    d = out["settings"]["data"]
    assert d["red"]["per_count"]["count-2"] == {"pictures": 10, "held_out": 1} and d["neither"]["held_out"] == 2
    assert out["settings"]["sigma_lo"] == 0.002 and out["settings"]["sigma_hi"] == 80.0
    clf, record = pf.load_fourway(tmp_path / "clf" / "fourway.pt", *stubs[:2])
    assert record["update"] == 40 and record["classes"] == list(pf.CLASSES)
    x = torch.zeros(2, 3, 64, 64)
    assert clf.logits(x, torch.tensor(1.0), torch.tensor([1, 3])).shape == (2, 4)


def test_training_loss_falls_on_stub_classes(tmp_path, stubs):
    data = _data(tmp_path, per_count=20, empties=40)
    pf.main(_train_args(tmp_path, data, "--updates", "150", "--val-every", "150", "--lr", "3e-3", "--sigma-hi", "1",
                        "--per-class", "8", "--shift", "0"))
    out = json.loads((tmp_path / "clf" / "train.json").read_text())
    curve = out["training"]["loss_curve"]
    print(json.dumps({"loss_curve": curve, "accuracy": out["training"]["train_accuracy_curve"]}))
    assert curve[-1] < curve[0] < 1.6                      # starts near log 4 = 1.386 and falls


def test_training_stops_past_the_gpu_hour_limit(tmp_path, stubs, monkeypatch):
    monkeypatch.setattr(pdisc, "TIMING_UPDATES", 3)
    data = _data(tmp_path)
    with pytest.raises(SystemExit) as stop:
        pf.main(_train_args(tmp_path, data, "--updates", "10", "--other-gpu-hours", "100"))
    assert stop.value.code == 3
    rec = json.loads((tmp_path / "clf" / "stop.json").read_text())
    assert rec["stage"] == "fourway-train" and rec["first_gpu_hours_stop"] == 4 and rec["over"]
