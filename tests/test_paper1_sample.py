"""Paper 1's sampler pieces that need no model (2-classifier-guidance/scripts/paper1/paper1_sample.py).

Needs torch, so it runs on the cluster (co3 or co3_bw python); on the laptop, which has no torch, it is skipped.

  python -m pytest tests/test_paper1_sample.py
"""
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "2-classifier-guidance" / "scripts" / "paper1"))
import paper1_sample as ps  # noqa: E402


def test_pushed_evaluations_stay_inside_the_window():
    sig = ps.sigmas("cpu")
    default = ps.pushed_evaluations()
    assert len(default) == 15 and default == ps.PUSHED_EVALUATIONS            # steps 15 to 22: 8 first, 7 corrections
    sig_d = [float(sig[i + int(sec)]) for i, sec in default]
    assert max(sig_d) == pytest.approx(2.90, abs=0.01) and min(sig_d) == pytest.approx(0.283, abs=0.001)
    wide = ps.pushed_evaluations(ps.parse_window("10-22"))
    assert len(wide) == 25
    sig_w = [float(sig[i + int(sec)]) for i, sec in wide]
    assert max(sig_w) == pytest.approx(10.52, abs=0.01) and min(sig_w) == pytest.approx(0.283, abs=0.001)
    assert not ps.pushed(22, True) and not ps.pushed(14, True) and not ps.pushed(14, False)


def test_every_pushed_sigma_has_a_band():
    sig = ps.sigmas("cpu")
    for i, sec in ps.pushed_evaluations(ps.parse_window("10-22")):
        assert ps.band_of(float(sig[i + int(sec)])) != "below the window"
    assert ps.band_of(10.52) == "10.5-3.8" and ps.band_of(0.283) == "0.6-0.28"


class LinearDisc:
    """logit = sum(W * x): its gradient is W wherever x is."""

    def __init__(self, weight):
        self.weight = weight

    def logit(self, x, s, count):
        return (x * self.weight).flatten(1).sum(1)


def _push_setup(w, random_push):
    torch.manual_seed(0)
    n = 2
    nets = [lambda x, s, l, f=f: x * f for f in (0.3, 0.4, 0.5)]
    weight = torch.zeros(1, 3, 64, 64)
    weight[..., :, :32] = torch.randn(1, 3, 64, 32)                         # the real push lives on the left half only
    m = torch.zeros(n, 1, 64, 64)
    m[..., 20:30, 20:30] = 1
    gen = torch.Generator().manual_seed(1) if random_push else None

    class Gen:
        def randn_like(self, x):
            return torch.randn(x.shape, generator=gen)
    stats = ps.PushStats()
    denoise = ps.push_denoiser(nets, LinearDisc(weight), 1, m, m, w, generator=Gen() if random_push else None, record=stats)
    x = torch.randn(n, 3, 64, 64)
    s = ps.sigmas("cpu")[16]
    base = ps.region_denoiser(nets, ps.one_hot(1, n), ps.one_hot(1, n), m, m)(x, s, 16)
    return denoise(x, s, 16, False), base, weight, s, stats


def test_ours_adds_w_sigma_squared_grad_and_the_sign_check_reads_plus_one():
    out, base, weight, s, stats = _push_setup(2.0, False)
    assert torch.allclose(out - base, 2.0 * s ** 2 * weight.expand_as(out), atol=1e-5)
    band = next(iter(stats.summary().values()))
    assert band["sign_check_cosine_smallest"] == pytest.approx(1.0, abs=1e-5)


def test_w_zero_is_the_region_sampler_exactly():
    out, base, _, _, _ = _push_setup(0.0, False)
    assert torch.equal(out, base)


def test_random_push_matches_length_and_stays_in_the_real_push_footprint():
    out, base, weight, s, stats = _push_setup(2.0, True)
    push = out - base
    real_len = (2.0 * s ** 2 * weight).flatten(1).norm(dim=1)
    assert torch.allclose(push.flatten(1).norm(dim=1), real_len.expand(2), rtol=1e-4)
    assert push[..., :, 32:].abs().max() == 0                               # nothing where the real push is zero
    assert next(iter(stats.summary().values()))["largest_length_mismatch"] < 1e-4


def test_weight_grid_takes_the_ladder_weight_nearest_five_grey_levels(tmp_path):
    import json
    import argparse
    smoke = tmp_path / "push.json"
    smoke.write_text(json.dumps({"weights": {"0": {"grey_change": {"median_change_in_band_grey": 0.0}},
                                             "1": {"grey_change": {"median_change_in_band_grey": 1.2}},
                                             "4": {"grey_change": {"median_change_in_band_grey": 4.1}},
                                             "16": {"grey_change": {"median_change_in_band_grey": 19.0}}}}))
    ps.cmd_weight_grid(argparse.Namespace(root=str(tmp_path), run="grid", smoke=str(smoke)))
    out = json.loads((tmp_path / "grid" / "weight-grid.json").read_text())
    assert out["w1"] == 4.0 and out["grid"] == [4.0, 8.0, 16.0, 32.0]


def test_region_with_all_ones_masks_is_the_plain_sum_bit_for_bit():
    g = torch.Generator().manual_seed(0)
    d_r, d_b, d_e = (torch.randn(4, 3, 64, 64, generator=g) for _ in range(3))
    nets = [lambda x, s, l, d=d: d for d in (d_r, d_b, d_e)]
    ones = torch.ones(4, 1, 64, 64)
    region = ps.region_denoiser(nets, None, None, ones, ones)(d_r, torch.tensor(1.0), 0)
    plain = ps.plainsum_denoiser(nets, None, None)(d_r, torch.tensor(1.0), 0)
    assert torch.equal(region, plain)


def test_contact_kept_on_drawn_squares():
    empty = torch.full((64, 64, 3), 0.5)
    pics = empty.expand(3, 64, 64, 3).clone()
    red, blue = torch.tensor([0.68, 0.14, 0.14]), torch.tensor([0.16, 0.29, 0.84])
    pics[0, 10:20, 10:20] = red; pics[0, 10:20, 20:30] = blue        # touching
    pics[1, 10:20, 10:20] = red; pics[1, 10:20, 40:50] = blue        # apart
    pics[2, 10:20, 10:20] = red; pics[2, 10:20, 20:30] = torch.tensor([0.45, 0.2, 0.45])   # purple neighbour is neither colour
    assert ps.contact(pics, empty).tolist() == [True, False, False]


def test_lower_on_top_paste_puts_the_lower_object_in_front():
    empty = torch.full((64, 64, 3), 0.5)
    red_col, blue_col = torch.tensor([0.68, 0.14, 0.14]), torch.tensor([0.16, 0.29, 0.84])
    red, blue = empty.expand(2, 64, 64, 3).clone(), empty.expand(2, 64, 64, 3).clone()
    red[0, 30:50, 10:30] = red_col; blue[0, 20:40, 20:40] = blue_col          # seed 0: red's base (row 49) is lower: red in front
    red[1, 10:30, 10:30] = red_col; blue[1, 20:45, 20:40] = blue_col          # seed 1: blue's base (row 44) is lower: blue in front
    out, _ = ps.paste_lower_on_top(red, blue, empty)
    plain, _ = ps.paste(red, blue, empty)
    assert torch.equal(out[0, 35, 25], red_col) and torch.equal(plain[0, 35, 25], blue_col)   # overlap: lower-on-top differs from paste
    assert torch.equal(out[1, 25, 25], blue_col)
    assert torch.equal(out[0, 22, 35], blue_col) and torch.equal(out[0, 45, 12], red_col)   # outside the overlap each keeps its own


def test_lower_owns_overlap_gives_the_overlap_to_the_lower_object():
    import torch
    m_r = torch.zeros(1, 1, 4, 4); m_b = torch.zeros(1, 1, 4, 4)
    m_r[..., 0:3, 0:3] = 1; m_b[..., 1:4, 1:4] = 1          # overlap is rows 1-2, columns 1-2
    base_r = torch.where(m_r > 0, torch.tensor(2.0), torch.tensor(-1.0))
    base_b = torch.where(m_b > 0, torch.tensor(3.0), torch.tensor(-1.0))   # blue's base is lower in the frame
    r, b = ps.lower_owns_overlap(m_r, m_b, base_r, base_b)
    assert r[..., 1:3, 1:3].sum() == 0 and b[..., 1:3, 1:3].sum() == 4       # blue owns the overlap
    assert r[..., 0, 0] == 1 and b[..., 3, 3] == 1                           # outside the overlap nothing changes
    r2, b2 = ps.lower_owns_overlap(m_r, m_b, base_b, base_r)                  # now red is lower
    assert b2[..., 1:3, 1:3].sum() == 0 and r2[..., 1:3, 1:3].sum() == 4
    r3, b3 = ps.lower_owns_overlap(m_r, m_b, base_r, base_r)                  # a tie keeps blue, as paste does
    assert r3[..., 1:3, 1:3].sum() == 0


# ---------------------------------------------------------------------------------------------------------------------------------------
# heun's start argument, separate starting noise per expert (--blue-seed-offset), harmonise, and paper1_overlap_stats.py

import json  # noqa: E402

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402


def _heun_before_start(denoise, x, t, after_step=None):
    """heun as it was before the start argument, kept verbatim as the reference."""
    for i in range(ps.STEPS):
        t_cur, t_next = t[i], t[i + 1]
        d_cur = (x - denoise(x, t_cur, i, False)) / t_cur
        x_next = x + (t_next - t_cur) * d_cur
        if i < ps.STEPS - 1:
            d_eval = denoise(x_next, t_next, i, True)
            x_next = x + (t_next - t_cur) * 0.5 * (d_cur + (x_next - d_eval) / t_next)
        x = x_next
        if after_step is not None and i < ps.STEPS - 1:
            x = after_step(x, t_next)
    return x


def _stub_denoise(x, s, step, second):
    return torch.tanh(x) * (0.5 + 0.01 * step) + 0.1 * second


def test_heun_start_zero_is_todays_heun_exactly():
    t = ps.sigmas("cpu")
    x = torch.randn(2, 3, 8, 8, generator=torch.Generator().manual_seed(3)) * t[0]
    shift = lambda z, s: z * 0.999
    assert torch.equal(ps.heun(_stub_denoise, x, t), _heun_before_start(_stub_denoise, x, t))
    assert torch.equal(ps.heun(_stub_denoise, x, t, start=0), _heun_before_start(_stub_denoise, x, t))
    assert torch.equal(ps.heun(_stub_denoise, x, t, after_step=shift), _heun_before_start(_stub_denoise, x, t, after_step=shift))


def test_heun_from_step_s_finishes_the_same_trajectory():
    t = ps.sigmas("cpu")
    x = torch.randn(2, 3, 8, 8, generator=torch.Generator().manual_seed(4)) * t[0]
    states = []
    full = ps.heun(_stub_denoise, x, t, after_step=lambda z, s: states.append(z.clone()) or z)
    for S in (10, 15):
        steps = []
        out = ps.heun(lambda z, s, step, second: steps.append(step) or _stub_denoise(z, s, step, second), states[S - 1], t, start=S)
        assert torch.equal(out, full) and min(steps) == S                 # states[S-1] is x at t[S], after step S-1


def test_compose_noise_takes_each_experts_noise_inside_its_mask():
    n_r, n_b, n_e = (torch.full((1, 3, 6, 6), v) for v in (1.0, 2.0, 3.0))
    m_r, m_b = torch.zeros(1, 1, 6, 6), torch.zeros(1, 1, 6, 6)
    m_r[..., 0:3, 0:3] = 1; m_b[..., 2:5, 2:5] = 1                          # overlap at row 2, column 2
    x = ps.compose_noise(n_r, n_b, n_e, m_r, m_b)
    assert x.shape == (1, 3, 6, 6)
    assert (x[0, :, 0, 0] == 1).all() and (x[0, :, 4, 4] == 2).all() and (x[0, :, 5, 5] == 3).all()
    assert (x[0, :, 2, 2] == 2).all()                                       # --overlap both: the overlap takes blue's noise
    base_r = torch.where(m_r > 0, torch.tensor(5.0), torch.tensor(-1.0))   # red's base lower: red owns the overlap
    base_b = torch.where(m_b > 0, torch.tensor(4.0), torch.tensor(-1.0))
    lr, lb = ps.lower_owns_overlap(m_r, m_b, base_r, base_b)
    assert (ps.compose_noise(n_r, n_b, n_e, lr, lb)[0, :, 2, 2] == 1).all()   # --overlap lower: the owner's noise


# Synthetic pictures on the 0-255 scale; the mean empty scene is grey 128 everywhere.
GREY, RED, BLUE, PURPLE = (128, 128, 128), (174, 36, 36), (41, 74, 214), (115, 51, 115)


def _picture(*boxes):
    img = np.full((64, 64, 3), GREY, np.uint8)
    for (r0, r1, c0, c1), colour in boxes:
        img[r0:r1, c0:c1] = colour
    return img


def _write(folder, k, seed, img):
    (folder / f"count-{k}").mkdir(parents=True, exist_ok=True)
    Image.fromarray(img, "RGB").save(folder / f"count-{k}" / f"{seed:06d}.png")


def _lone_folders(tmp_path, offset=0):
    """Seed 5: red and blue squares overlapping on columns 18-19; seed 6: apart."""
    red, blue = tmp_path / "lone-red", tmp_path / "lone-blue"
    _write(red, 1, 5, _picture(((10, 20, 10, 20), RED))); _write(blue, 1, 5, _picture(((10, 20, 18, 28), BLUE)))
    _write(red, 1, 6, _picture(((10, 20, 10, 20), RED))); _write(blue, 1, 6, _picture(((10, 20, 40, 50), BLUE)))
    if offset:
        (blue / ps.BLUE_OFFSET_FILE).write_text(json.dumps({"blue_seed_offset": offset}))
    empty = tmp_path / "mean-empty.pt"
    torch.save(torch.full((64, 64, 3), 128 / 255), empty)
    return red, blue, empty


class StubEncoder:
    """EDM2's StandardRGBEncoder arithmetic (training/encoders.py)."""

    def encode(self, x):
        return x.to(torch.float32) / 127.5 - 1

    def decode(self, x):
        return (x.to(torch.float32) * 127.5 + 128).clip(0, 255).to(torch.uint8)


def _seed_noise(seeds):
    """Stand-in for EDM2's per-seed generator: seed s gives the constant s, so each pixel shows which seed it came from."""
    return torch.stack([torch.full((3, 64, 64), float(s)) for s in seeds]), None


def _patch_models(monkeypatch):
    nets = [lambda x, s, l: x * 0.0 for _ in range(3)]
    monkeypatch.setattr(ps, "load_three", lambda args: (nets, StubEncoder()))
    monkeypatch.setattr(ps, "unit_noise", _seed_noise)
    seen = []
    real = ps.heun

    def heun(denoise, x, t, after_step=None, start=0):
        seen.append((x.clone(), start))
        return real(denoise, x, t, after_step=after_step, start=start)
    monkeypatch.setattr(ps, "heun", heun)
    return seen


@pytest.mark.parametrize("offset,overlap", [(0, "both"), (100, "both"), (100, "lower")])
def test_region_starting_noise_with_and_without_a_blue_seed_offset(tmp_path, monkeypatch, offset, overlap):
    red, blue, empty = _lone_folders(tmp_path, offset)
    seen = _patch_models(monkeypatch)
    argv = ["region", "--run", "r", "--root", str(tmp_path), "--red-dir", str(red), "--blue-dir", str(blue), "--mean-empty", str(empty),
            "--counts", "1", "--seeds", "5-6", "--overlap", overlap] + (["--blue-seed-offset", str(offset)] if offset else [])
    ps.main(argv)
    x0, start = seen[0]
    s80 = float(ps.sigmas("cpu")[0])
    assert start == 0
    assert torch.allclose(x0[0, :, 0, 0], torch.tensor(5 * s80)) and torch.allclose(x0[0, :, 15, 12], torch.tensor(5 * s80))  # floor, red
    blue_only, in_overlap = x0[0, :, 15, 25], x0[0, :, 15, 19]
    if offset == 0:
        assert torch.equal(x0, _seed_noise([5, 6])[0] * ps.sigmas("cpu")[0])   # today's shared noise, unchanged
    else:
        assert torch.allclose(blue_only, torch.tensor((5 + offset) * s80))
        assert torch.allclose(in_overlap, torch.tensor((5 + offset) * s80))    # both: blue's; lower: tie on base row 19 keeps blue
    timing = json.loads((tmp_path / "r" / "timing.jsonl").read_text().splitlines()[-1])
    band = json.loads((tmp_path / "r" / "band-distance.json").read_text())
    assert ("blue_seed_offset" in timing) == bool(offset) and ("blue_seed_offset" in band) == bool(offset)
    if offset:
        assert timing["blue_seed_offset"] == offset and band["blue_seed_offset"] == offset


def test_region_refuses_a_blue_folder_drawn_at_another_offset(tmp_path, monkeypatch):
    red, blue, empty = _lone_folders(tmp_path, offset=100)
    _patch_models(monkeypatch)
    with pytest.raises(SystemExit):
        ps.main(["region", "--run", "r", "--root", str(tmp_path), "--red-dir", str(red), "--blue-dir", str(blue),
                 "--mean-empty", str(empty), "--counts", "1", "--seeds", "5-6"])


@pytest.mark.parametrize("offset", [0, 100])
def test_harmonise_noises_the_picture_to_step_s_along_its_seeds_noise_and_finishes_from_there(tmp_path, monkeypatch, offset):
    red, blue, empty = _lone_folders(tmp_path, offset)
    pics = tmp_path / "paste-lower"
    _write(pics, 1, 5, _picture(((10, 20, 10, 20), RED), ((10, 20, 20, 28), BLUE)))
    _write(pics, 1, 6, _picture(((10, 20, 10, 20), RED), ((10, 20, 40, 50), BLUE)))
    seen = _patch_models(monkeypatch)
    argv = ["harmonise", "--run", "h", "--root", str(tmp_path), "--pictures", "paste-lower", "--red-dir", str(red),
            "--blue-dir", str(blue), "--mean-empty", str(empty), "--counts", "1", "--seeds", "5-6", "--from-step", "15"]
    ps.main(argv + (["--blue-seed-offset", str(offset)] if offset else []))
    x_s, start = seen[0]
    t = ps.sigmas("cpu")
    assert start == 15
    clean = StubEncoder().encode(torch.from_numpy(np.stack([np.asarray(Image.open(pics / "count-1" / f"{s:06d}.png")) for s in (5, 6)]))
                                 .permute(0, 3, 1, 2))
    eps = _seed_noise([5, 6])[0]
    if offset:
        blue_mask = torch.zeros(2, 1, 64, 64)
        blue_mask[0, :, 9:21, 17:29] = 1; blue_mask[1, :, 9:21, 39:51] = 1     # blue's dilated masks (the overlap's tie keeps blue)
        eps = torch.where(blue_mask > 0, _seed_noise([5 + offset, 6 + offset])[0], eps)
    assert torch.allclose(x_s, clean + t[15] * eps)
    assert sorted(p.name for p in (tmp_path / "h" / "count-1").iterdir()) == ["000005.png", "000006.png"]
    timing = json.loads((tmp_path / "h" / "timing.jsonl").read_text())
    assert timing["from_step"] == 15 and timing["sigma_from_step"] == pytest.approx(2.90, abs=0.01)
    assert timing["blue_seed_offset"] == offset and timing["overlap"] == "lower"


def test_harmonise_defaults_to_step_10_and_the_lower_overlap():
    args = ps.build_parser().parse_args(["harmonise", "--run", "h", "--pictures", "p", "--red-dir", "r", "--blue-dir", "b",
                                         "--mean-empty", "m", "--seeds", "1"])
    assert args.from_step == 10 and args.overlap == "lower" and args.blue_seed_offset == 0


def test_every_command_given_the_offset_defaults_to_zero():
    for argv in (["lone", "--run", "r", "--expert", "blue", "--seeds", "1"],
                 ["paste", "--run", "r", "--red-dir", "a", "--blue-dir", "b", "--mean-empty", "m", "--seeds", "1"],
                 ["split", "--run", "r", "--set", "evaluation", "--red-dir", "a", "--blue-dir", "b", "--mean-empty", "m", "--seeds", "1"],
                 ["region", "--run", "r", "--mean-empty", "m", "--identity", "--seeds", "1"]):
        assert ps.build_parser().parse_args(argv).blue_seed_offset == 0


def test_lone_offset_is_for_blue_only_and_one_folder_holds_one_offset(tmp_path):
    with pytest.raises(SystemExit):
        ps.claim_blue_offset(tmp_path, "red", 100)
    ps.claim_blue_offset(tmp_path / "fresh", "blue", 0)                       # offset 0 writes nothing
    (tmp_path / "fresh").mkdir()
    ps.claim_blue_offset(tmp_path / "fresh", "blue", 0)
    assert not (tmp_path / "fresh" / ps.BLUE_OFFSET_FILE).exists()
    ps.claim_blue_offset(tmp_path / "fresh", "blue", 100)
    assert ps.recorded_blue_offset(tmp_path / "fresh") == 100
    with pytest.raises(SystemExit):
        ps.claim_blue_offset(tmp_path / "fresh", "blue", 0)
    old = tmp_path / "old"
    _write(old, 1, 5, _picture())                                             # pictures and no file: drawn at offset 0
    with pytest.raises(SystemExit):
        ps.claim_blue_offset(old, "blue", 100)


def test_paste_and_split_write_what_they_wrote_before_at_offset_zero(tmp_path):
    red, blue, empty = _lone_folders(tmp_path)
    common = ["--root", str(tmp_path), "--red-dir", str(red), "--blue-dir", str(blue), "--mean-empty", str(empty), "--counts", "1",
              "--seeds", "5-6"]
    ps.main(["paste", "--run", "p"] + common)
    assert json.loads((tmp_path / "p" / "timing.jsonl").read_text()) == {"stage": "paste", "order": "blue-on-top", "counts": [1],
                                                                        "pictures": 2}
    ps.main(["split", "--run", "s", "--set", "evaluation"] + common)
    entry = json.loads((tmp_path / "s" / "split.json").read_text())["evaluation"]["count-1"]
    assert set(entry) == {"touching", "apart", "touching_share", "object_pixels"}
    assert entry["touching"] == [5] and entry["apart"] == [6]


def test_split_records_the_offset_and_never_merges_across_offsets(tmp_path):
    red, blue, empty = _lone_folders(tmp_path, offset=100)
    common = ["--root", str(tmp_path), "--red-dir", str(red), "--blue-dir", str(blue), "--mean-empty", str(empty), "--counts", "1",
              "--seeds", "5-6", "--set", "evaluation"]
    with pytest.raises(SystemExit):                                           # the folder says 100, the command says 0
        ps.main(["split", "--run", "s"] + common)
    ps.main(["split", "--run", "s", "--blue-seed-offset", "100"] + common)
    entry = json.loads((tmp_path / "s" / "split.json").read_text())["evaluation"]["count-1"]
    assert entry["blue_seed_offset"] == 100 and entry["touching"] == [5]
    (blue / ps.BLUE_OFFSET_FILE).unlink()
    with pytest.raises(SystemExit):                                           # offset 0 pairs into the offset-100 lists
        ps.main(["split", "--run", "s"] + common)


def test_overlap_stats_counts_the_overlap_and_its_purple_share(tmp_path):
    import paper1_overlap_stats as pos
    red, blue, empty = _lone_folders(tmp_path)
    pics = tmp_path / "pics"
    _write(pics, 1, 5, _picture(((10, 20, 10, 18), RED), ((10, 20, 18, 20), PURPLE), ((10, 20, 20, 28), BLUE)))
    _write(pics, 1, 6, _picture(((10, 20, 10, 20), RED), ((10, 20, 40, 50), BLUE)))
    pos.main(["--run", "st", "--root", str(tmp_path), "--pictures", "pics", "--red-dir", str(red), "--blue-dir", str(blue),
              "--mean-empty", str(empty), "--counts", "1", "--seeds", "5-6"])
    out = json.loads((tmp_path / "st" / "overlap-stats.json").read_text())["pics"]
    s5, s6 = out["per_count"]["count-1"]["per_seed"]
    # dilated masks: red rows 9-20 columns 9-20, blue rows 9-20 columns 17-28; overlap 12 rows by columns 17-20 = 48 pixels.
    # Picture foreground inside it: rows 10-19 by columns 17-20 = 40 pixels, of which columns 18-19 (20 pixels) are purple.
    assert s5 == {"seed": 5, "overlap_pixels": 48, "touching": True, "purple_share": 0.5}
    assert s6 == {"seed": 6, "overlap_pixels": 0, "touching": False, "purple_share": None}
    pooled = out["pooled"]
    assert pooled["touching"] == 1 and pooled["median_purple_share"] == 0.5 and pooled["seeds_mostly_purple"] == 0
    assert pooled["median_overlap_pixels_touching"] == 48 and out["blue_seed_offset"] == 0
