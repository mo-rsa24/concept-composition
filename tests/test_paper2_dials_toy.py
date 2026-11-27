"""Paper 2's miniature toy rung (2-classifier-guidance/scripts/paper2/toy/): reader, targets, dials, sampler, correction.

  /home/molef/micromamba/envs/voice-agent/bin/python -m pytest tests/test_paper2_dials_toy.py -q
"""
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("scipy")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "2-classifier-guidance" / "scripts" / "paper2" / "toy"))
import toy_canvas as tc  # noqa: E402
import toy_model as tm  # noqa: E402


def _canvas(objs):
    """objs: list of (combination name, top, left)."""
    conds = [tuple(tc.NAMES.index(n) for n, _, _ in objs)]
    pos = [np.array([(r, q) for _, r, q in objs])]
    return tc.to_rgb(tc.render(conds, pos)), conds[0]


def test_template_margin_and_shapes_read_exactly():
    x, c = _canvas([("RS", 0, 0), ("BD", 8, 9), ("YD", 0, 10)])
    r = tc.read(x)[0]
    assert r["specks"] == 0
    assert sorted((s, col) for s, col, *_ in r["objects"]) == sorted((tc.shape_of(k), tc.colour_of(k)) for k in c)
    for s in (0, 1):
        m = np.zeros((16, 16), bool)
        m[3:8, 3:8] = tc.TEMPLATES[s]
        ious = tc.shape_ious(m)
        assert ious[s] == 1.0 and ious[s] - ious[1 - s] >= 0.15


def test_training_canvases_never_mix_shapes_or_hold_held_out_combinations():
    idx, conds = tc.training_canvases(500, seed=3)
    for c in conds:
        assert len({tc.shape_of(k) for k in c}) == 1
        assert not any(k in tc.HELD_OUT for k in c)
    R = tc.read(tc.to_rgb(idx))
    assert all(tc.asked_exactly(r, c) for r, c in zip(R, conds))


def test_boxes_keep_their_gap():
    rng = np.random.default_rng(0)
    for p in tc.sample_positions([3] * 200, rng):
        for a in range(3):
            for b in range(a + 1, 3):
                assert abs(p[a, 0] - p[b, 0]) >= 7 or abs(p[a, 1] - p[b, 1]) >= 7


def test_targets_edit_only_their_own_attribute():
    cond = tc.parse("YS+RD")
    base = {"objects": [(0, 0, 3.0, 3.0, 25), (0, 0, 10.0, 10.0, 25)], "specks": 0}      # two red squares
    # identity: within red, the asked shape is the diamond; one red square becomes a red diamond, the other is left as drawn
    assert tc.target_pairs(base, cond, {"identity": 1.0}) == Counter([tc.parse("RS")[0], tc.parse("RD")[0]])
    # binding: within squares, the asked colour is yellow; one red square becomes yellow
    assert tc.target_pairs(base, cond, {"binding": 1.0}) == Counter([tc.parse("YS")[0], tc.parse("RS")[0]])
    # both: the asked objects
    assert tc.target_pairs(base, cond, {"identity": 1.0, "binding": 1.0}) == Counter(cond)
    # count adds one copy of the asked blue object, untouched otherwise
    c2 = tc.parse("RS+BD")
    b2 = {"objects": [(0, 0, 3.0, 3.0, 25), (0, 2, 10.0, 10.0, 25)], "specks": 0}          # red square, blue square
    assert tc.target_pairs(b2, c2, {"count": 1.0}) == Counter([0, 2, tc.parse("BD")[0]])
    assert tc.target_pairs(b2, c2, {"count": 0.5}) is None


def test_meets_needs_positions_kept_without_relation_and_the_bin_with_it():
    x, c = _canvas([("RS", 2, 0), ("BS", 2, 9)])
    r = tc.read(x)[0]
    assert tc.meets(r, r, c, {}) is True
    assert abs(tc.m_gap(r, c) - 9 / 16) < 1e-9
    assert tc.meets(r, r, c, {"relation": 1.0}) is True
    assert tc.meets(r, r, c, {"relation": 0.5}) is False
    x2, _ = _canvas([("RS", 2, 1), ("BS", 2, 9)])
    assert tc.meets(tc.read(x2)[0], r, c, {}) is False          # the red square moved one pixel


def _tiny_model():
    torch.manual_seed(0)
    m = tm.Denoiser(w=16).to("cpu")
    for p in m.parameters():
        p.data.normal_(0, 0.05)
    tm.freeze_base(m)
    m.eval()
    return m


def test_strength_zero_and_switched_off_dials_reproduce_the_frozen_model_exactly():
    m = _tiny_model()
    x = torch.randn(4, 3, 16, 16)
    cv = tc.cond_vector([tc.parse("RS+BD")] * 4)
    ref = m(x, torch.full((4,), 1.3), cv)
    tm.wrap(m)
    assert torch.equal(m(x, torch.full((4,), 1.3), cv), ref)
    tm.add_dial(m, "identity", seed=1)
    tm.add_dial(m, "count", seed=2)
    for _, lay in tm.layers(m):
        for a in lay.adapters.values():
            a.up.weight.data.normal_(0, 0.1)
    tm.set_active(m, {"identity": 0.0})
    assert torch.equal(m(x, torch.full((4,), 1.3), cv), ref)
    tm.set_active(m, {})
    assert torch.equal(m(x, torch.full((4,), 1.3), cv), ref)
    tm.set_active(m, {"identity": 1.0})
    one = m(x, torch.full((4,), 1.3), cv)
    tm.set_active(m, {"identity": 2.0})
    two = m(x, torch.full((4,), 1.3), cv)
    assert (two - ref).abs().mean() > (one - ref).abs().mean() > 0


def test_random_dial_matches_norm_per_layer():
    m = _tiny_model()
    tm.wrap(m)
    tm.add_dial(m, "count", seed=2)
    for _, lay in tm.layers(m):
        lay.adapters["count"].up.weight.data.normal_(0, 0.1)
    s = tm.dial_state(m, "count")
    r = tm.random_like(s, seed=7)
    for n in s:
        a = (s[n]["up.weight"].flatten(1) @ s[n]["down.weight"].flatten(1)).norm()
        b = (r[n]["up.weight"].flatten(1) @ r[n]["down.weight"].flatten(1)).norm()
        assert torch.allclose(a, b, rtol=1e-4)


def test_heun_range_matches_the_squares_toy_sampler():
    m = _tiny_model()
    cv = tc.cond_vector([tc.parse("RS")] * 2)
    x = tm.st.starting_noise([0, 1], device="cpu")
    den = tm.denoiser_for(m, cv)
    a = tm.st.heun(den, x)
    b = tm.heun_range(den, x)
    mid = tm.heun_range(den, tm.heun_range(den, x, 0, 10), 10)
    assert torch.equal(a, b) and torch.equal(a, mid)


def test_correction_starts_at_the_plain_stack():
    m = _tiny_model()
    tm.wrap(m)
    for d in ("identity", "count"):
        tm.add_dial(m, d, seed=3)
        for _, lay in tm.layers(m):
            lay.adapters[d].up.weight.data.normal_(0, 0.1)
    names = [n for n, _ in tm.layers(m)]
    li = {n: i for i, n in enumerate(names)}
    corr = tm.Correction(len(names))
    static = tm.Correction.static_features(m, ["identity", "count"])
    colw = corr(static, {"identity": 1.0, "count": 1.0}, li)
    x = torch.randn(2, 3, 16, 16)
    cv = tc.cond_vector([tc.parse("RS+BD")] * 2)
    tm.set_active(m, {"identity": 1.0, "count": 1.0})
    plain = m(x, torch.full((2,), 0.7), cv)
    tm.set_colw(m, colw)
    assert torch.allclose(m(x, torch.full((2,), 0.7), cv), plain, atol=1e-6)


def test_soft_identity_loss_is_zero_on_a_clean_render_and_positive_on_the_wrong_shape():
    x, c = _canvas([("RS", 1, 1), ("RD", 9, 9)])
    tgt, win, col, used = tm.targets_from(x, [c], "identity")
    assert used == 1
    p = tm.soft_probs(x)
    assert float(tm.soft_loss(x, tgt, win, col, "identity")) < 1e-3
    xw, _ = _canvas([("RS", 1, 1), ("RS", 9, 9)])
    tgt, win, col, used = tm.targets_from(xw, [c], "identity")
    assert float(tm.soft_loss(xw, tgt, win, col, "identity")) > 0.05
    assert p.shape == (1, 5, 16, 16)
