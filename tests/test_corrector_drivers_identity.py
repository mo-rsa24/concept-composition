"""The two new correctors must reduce to the samplers they extend.

A corrector switched off has to be the plain sampler, or every comparison against that sampler
measures the wrapper as well as the corrector. These tests pin it down on a deterministic stand-in
for the UNet and the real DDIM scheduler with SDXL's settings, on the CPU, in seconds.

  1. run_joint_langevin at k = 0 equals run_cfg with the same embedding, bit for bit.
  2. run_poe_uhmc at k = 0 equals run_cfg_poe, bit for bit.
  3. run_poe_uhmc with L = 1 takes the same steps as run_poe_langevin at the same c, k and noise
     seed (equal up to float rounding, since x + d*s + sqrt(2d)*z and x + e*(p + e/2*s) round
     differently).
  4. The call counter: a level costs k + 1 calls for Langevin and k * L + 2 for unadjusted HMC.

The same identity on real SDXL is checked on the cluster by scripts/check_corrector_identity.py.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

diffusers = pytest.importorskip("diffusers")

from poe_repair.methods import _sampling  # noqa: E402
from poe_repair.methods._corrector_drivers import run_joint_langevin, run_poe_uhmc  # noqa: E402
from poe_repair.methods._poe_langevin import run_poe_langevin  # noqa: E402

STEPS = 10
DEV, DT = torch.device("cpu"), torch.float32


class StubUNet:
    """Deterministic, depends on the latent, the timestep and the prompt, like the real one."""

    def __call__(self, latent_input, timestep, *, encoder_hidden_states, added_cond_kwargs,
                 timestep_cond=None):
        prompt = encoder_hidden_states.mean(dim=(1, 2)).view(-1, 1, 1, 1)
        t = float(timestep) / 1000.0
        sample = 0.3 * latent_input + 0.05 * torch.tanh(latent_input * prompt) + 0.01 * t
        return SimpleNamespace(sample=sample)


def scheduler():
    return diffusers.DDIMScheduler(
        beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
        num_train_timesteps=1000, clip_sample=False, set_alpha_to_one=False,
        steps_offset=1, timestep_spacing="leading",
    )


def embeds(seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(1, 4, 8, generator=g), torch.randn(1, 8, generator=g)


@pytest.fixture
def common(monkeypatch):
    monkeypatch.setattr(_sampling, "decode_latents", lambda models, latents: latents)
    init = torch.randn(1, 4, 8, 8, generator=torch.Generator().manual_seed(9))
    return dict(init_latents=init, models={"unet": StubUNet()}, guidance_scale=7.5,
                num_inference_steps=STEPS, height=64, width=64, euler_init_noise_sigma=1.0,
                device=DEV, dtype=DT)


def test_joint_langevin_at_k0_is_run_cfg(common):
    (sj, pj), (se, pe) = embeds(1), embeds(0)
    ref = _sampling.run_cfg(scheduler=scheduler(), seq_cond=sj, pool_cond=pj, seq_e=se,
                            pool_e=pe, **common)
    out = run_joint_langevin(scheduler=scheduler(), seq_j=sj, pool_j=pj, seq_e=se, pool_e=pe,
                             k=0, c=3.0, decode=False, **common)
    assert torch.equal(ref.latents, out.latents)


def test_poe_uhmc_at_k0_is_run_cfg_poe(common):
    (sa, pa), (sb, pb), (se, pe) = embeds(1), embeds(2), embeds(0)
    ref = _sampling.run_cfg_poe(scheduler=scheduler(), seq_a=sa, pool_a=pa, seq_b=sb, pool_b=pb,
                                seq_e=se, pool_e=pe, **common)
    out = run_poe_uhmc(scheduler=scheduler(), seq_a=sa, pool_a=pa, seq_b=sb, pool_b=pb,
                       seq_e=se, pool_e=pe, k=0, c=3.0, decode=False, **common)
    assert torch.equal(ref.latents, out.latents)


def test_uhmc_with_one_leapfrog_step_is_langevin(common):
    (sa, pa), (sb, pb), (se, pe) = embeds(1), embeds(2), embeds(0)
    kw = dict(seq_a=sa, pool_a=pa, seq_b=sb, pool_b=pb, seq_e=se, pool_e=pe, k=3, c=3.0,
              noise_seed=9, decode=False, **common)
    lang = run_poe_langevin(scheduler=scheduler(), **kw)
    hmc = run_poe_uhmc(scheduler=scheduler(), n_leapfrog=1, **kw)
    assert not torch.equal(lang.latents, common["init_latents"])
    assert torch.allclose(lang.latents, hmc.latents, rtol=1e-5, atol=1e-6)


def test_call_counts(common):
    (sa, pa), (sb, pb), (se, pe) = embeds(1), embeds(2), embeds(0)
    j = run_joint_langevin(scheduler=scheduler(), seq_j=sa, pool_j=pa, seq_e=se, pool_e=pe,
                           k=3, c=3.0, decode=False, **common)
    h = run_poe_uhmc(scheduler=scheduler(), seq_a=sa, pool_a=pa, seq_b=sb, pool_b=pb,
                     seq_e=se, pool_e=pe, k=2, c=3.0, n_leapfrog=5, decode=False, **common)
    assert j.extras["nfe"] == STEPS * (3 + 1)
    assert h.extras["nfe"] == STEPS * (2 * 5 + 2)
