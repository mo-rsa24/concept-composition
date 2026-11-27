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
