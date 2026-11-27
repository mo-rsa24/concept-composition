"""The squares toy for Paper 1 (2-classifier-guidance/scripts/paper1/toy/squares_toy.py): exact denoiser, sampler, zero delta.

  /home/molef/micromamba/envs/voice-agent/bin/python -m pytest tests/test_squares_toy.py
"""
import math
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "2-classifier-guidance" / "scripts" / "paper1" / "toy"))
import squares_toy as st  # noqa: E402


def test_exact_denoiser_two_pictures_matches_hand_computation():
    """Set {a, b}; at x the weights are softmax(-||x - a||^2 / 2s^2, -||x - b||^2 / 2s^2)."""
    a = torch.full((1, 3, 2, 2), -1.0)
    b = torch.full((1, 3, 2, 2), 1.0)
    x = torch.full((1, 3, 2, 2), 0.1)
    s = 2.0
    da, db = 12 * (0.1 + 1) ** 2, 12 * (0.1 - 1) ** 2          # 12 values each
    wa = math.exp(-da / (2 * s * s)) / (math.exp(-da / (2 * s * s)) + math.exp(-db / (2 * s * s)))
    want = wa * -1.0 + (1 - wa) * 1.0
    got = st.Exact(torch.cat([a, b]), device="cpu")(x, s)
    assert torch.allclose(got, torch.full_like(x, want), atol=1e-6)


def test_sampler_with_exact_joint_returns_a_member():
    """32 Heun steps down to sigma 0.002 with the exact denoiser of a small joint subset end on a member of it."""
    pics, _ = st.joint_set()
    sub = pics[:: 97][:64]
    ex = st.Exact(sub, device="cpu")
    x = st.starting_noise([0, 1, 2, 3], device="cpu")
    out = st.heun(lambda xx, s, i, sec: ex(xx, s), x)
    d = torch.cdist(out.flatten(1).double(), sub.flatten(1).double()).min(1).values
    assert (d < 1e-3).all(), d


def test_zero_delta_reproduces_plain_sum_byte_for_byte():
    """A delta with nonzero random weights, multiplied by 0 inside the interaction denoiser, gives the plain sum's pictures exactly."""
    torch.manual_seed(0)
    nets = [st.Expert(w=8).eval() for _ in range(3)]
    delta = st.Delta(w=8).eval()
    for p in delta.parameters():
        torch.nn.init.normal_(p, std=0.05)
    x = st.starting_noise([5, 6], device="cpu")
    with torch.no_grad():
        plain = st.heun(st.plainsum_denoiser(*nets), x)
        zero = st.heun(st.interaction_denoiser(*nets, delta, zero_delta=True), x)
        live = st.heun(st.interaction_denoiser(*nets, delta), x)
    assert torch.equal(plain, zero)
    assert not torch.equal(plain, live)


def test_joint_set_rules():
    """Every joint picture has red and blue pixels, no purple, and red within 1 pixel of blue; sizes per kind are the hand counts."""
    pics, meta = st.joint_set()
    red, blue, purple = st.masks(pics)
    assert not purple.any()
    assert red.flatten(1).any(1).all() and blue.flatten(1).any(1).all()
    assert len(pics) == 2844 + 2 * 2760 + 2 * 2112


def test_expert_features_leave_the_output_unchanged():
    """Asking an expert for its inner decoder maps returns the same denoised picture, and maps at 4, 8 and 16 pixels."""
    torch.manual_seed(1)
    ex = st.Expert(w=8).eval()
    x = torch.randn(3, 3, 16, 16) * 5
    with torch.no_grad():
        plain = ex(x, 2.5)
        out, fs = ex(x, 2.5, feats=True)
    assert torch.equal(plain, out)
    assert [f.shape[-1] for f in fs] == [4, 8, 16]
    assert [f.shape[1] for f in fs] == [16, 16, 8]


def test_method_d_starts_at_the_plain_sum_and_its_zero_control_is_byte_identical():
    """DeltaD's last convolution starts at zero, so before training the composed sampler is the plain sum; with random weights
    and delta multiplied by 0 it is still the plain sum byte for byte, through the feature path."""
    torch.manual_seed(0)
    nets = [st.Expert(w=8).eval() for _ in range(3)]
    fresh = st.DeltaD(w=8, proj=8, expert_w=8).eval()
    x = st.starting_noise([5, 6], device="cpu")
    with torch.no_grad():
        plain = st.heun(st.plainsum_denoiser(*nets), x)
        start = st.heun(st.interaction_denoiser(*nets, fresh), x)
        for p in fresh.parameters():
            torch.nn.init.normal_(p, std=0.05)
        zero = st.heun(st.interaction_denoiser(*nets, fresh, zero_delta=True), x)
        live = st.heun(st.interaction_denoiser(*nets, fresh), x)
    assert torch.equal(plain, start)
    assert torch.equal(plain, zero)
    assert not torch.equal(plain, live)


def test_recording_the_push_does_not_change_the_samples():
    torch.manual_seed(0)
    nets = [st.Expert(w=8).eval() for _ in range(3)]
    head = st.Head(1, w=8).eval()
    x = st.starting_noise([3], device="cpu")
    rec = []
    with torch.no_grad():
        a = st.heun(st.push_denoiser(st.plainsum_denoiser(*nets), lambda xx, s: head(xx, s)[:, 0], 4.0), x)
        b = st.heun(st.push_denoiser(st.plainsum_denoiser(*nets), lambda xx, s: head(xx, s)[:, 0], 4.0, record=rec), x)
    assert torch.equal(a, b)
    assert len(rec) == st.STEPS and rec[0][0] == 0


def test_a_new_run_refuses_an_existing_key_or_file(tmp_path, monkeypatch):
    monkeypatch.setattr(st, "RESULTS", tmp_path / "results")
    monkeypatch.setattr(st, "OUT", tmp_path / "out")
    st.save_results("rung9", {"x": 1})
    with pytest.raises(SystemExit):
        st.require_new("rung9", [])
    taken = tmp_path / "out" / "taken.pt"
    taken.write_text("")
    with pytest.raises(SystemExit):
        st.require_new("rung10", [taken])
    st.require_new("rung10", [tmp_path / "out" / "free.pt"])


def test_corrector_with_zero_steps_reproduces_the_plain_sampler_byte_for_byte():
    """Rung 6 at 0 corrector steps runs the corrector path and gives the plain sampler's pictures exactly, for the plain sum,
    Method A's composed denoiser and a pushed one; 5 steps changes them."""
    torch.manual_seed(0)
    nets = [st.Expert(w=8).eval() for _ in range(3)]
    delta = st.Delta(w=8).eval()
    for p in delta.parameters():
        torch.nn.init.normal_(p, std=0.05)
    head = st.Head(1, w=8).eval()
    seeds = [5, 6]
    x = st.starting_noise(seeds, device="cpu")
    for denoise in (st.plainsum_denoiser(*nets), st.interaction_denoiser(*nets, delta),
                    st.push_denoiser(st.plainsum_denoiser(*nets), lambda xx, s: head(xx, s)[:, 0], 4.0)):
        with torch.no_grad():
            plain = st.heun(denoise, x)
            zero = st.sample_corrected(denoise, 0, seeds=seeds, device="cpu")
            five = st.sample_corrected(denoise, 5, seeds=seeds, device="cpu")
        assert torch.equal(plain, zero)
        assert not torch.equal(plain, five)


def test_langevin_step_matches_hand_computation():
    """Stand-in denoiser D(x, s) = 0.5 x + 0.3. At x = 0.7, sigma 2, size 0.05, noise 0.4:
    score = (0.65 - 0.7) / 4, eta = 0.05 * 4, x' = x + eta score + sqrt(2 eta) noise."""
    x = torch.full((1, 3, 2, 2), 0.7)
    noise = torch.full_like(x, 0.4)
    s = torch.tensor(2.0)
    got = st.langevin_step(lambda xx, ss, i, sec: 0.5 * xx + 0.3, x, s, 0.05, noise)
    score = (0.5 * 0.7 + 0.3 - 0.7) / 4.0
    eta = 0.05 * 4.0
    want = 0.7 + eta * score + math.sqrt(2 * eta) * 0.4
    assert torch.allclose(got, torch.full_like(x, want), atol=1e-6)


def test_corrector_draws_each_seeds_own_noise():
    """Two corrector steps at one level for seeds 3 and 9: each seed's noise comes from its own generator, seeded 200,000 + seed,
    drawn as (steps, 3, 16, 16), so the result equals a hand loop over those draws, and a seed's result does not depend on its batch."""
    def denoise(xx, ss, i, sec):
        return 0.5 * xx + 0.3
    x = torch.randn(2, 3, st.H, st.H, generator=torch.Generator().manual_seed(1))
    s = torch.tensor(1.5)
    got = st.corrector(denoise, [3, 9], 2)(0, x, s)
    want = x.clone()
    draws = [torch.Generator().manual_seed(st.CORRECTOR_SEED_OFFSET + sd) for sd in (3, 9)]
    noise = torch.stack([torch.randn(2, 3, st.H, st.H, generator=g) for g in draws])
    for k in range(2):
        d = 0.5 * want + 0.3
        want = want + 0.05 * 1.5 ** 2 * (d - want) / 1.5 ** 2 + math.sqrt(2 * 0.05 * 1.5 ** 2) * noise[:, k]
    assert torch.allclose(got, want, atol=1e-6)
    alone = st.corrector(denoise, [9], 2)(0, x[1:], s)
    assert torch.equal(alone, got[1:])


def test_rung6_looks_for_the_saved_models_by_their_training_names(tmp_path, monkeypatch):
    """The files rung 6 loads are the ones rungs 2c, 5, 3c and 4b wrote at N = 1,024 and 20,000 updates; with none present, every
    neural row is listed as missing and the two exact rows need nothing."""
    monkeypatch.setattr(st, "OUT", tmp_path)
    rows = st.rung6_rows()
    need = {name: {p.name for p in files} for name, files, _, _ in rows}
    assert need["exact joint"] == set() and need["exact plain sum"] == set()
    names = set().union(*need.values())
    for f in ("expert-red.pt", "expert-blue.pt", "expert-empty.pt", "delta-N1024-u20000.pt", "deltaD-N1024-u20000.pt",
              "head-Cb-N1024-u20000.pt", "head-B-N1024-u20000.pt"):
        assert f in names
    assert len(rows) == 9
    assert {r[3] for r in rows} >= {"methodA-N1024-u20000", "methodD-N1024-u20000", "methodCb-N1024-u20000-w16",
                                    "methodB-N1024-u20000-w64"}
