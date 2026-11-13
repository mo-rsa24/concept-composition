#!/usr/bin/env python
"""Can a gradient flow back through SDXL's down and middle blocks, and what does it cost?

One forward to the middle block at batch 8 with no gradient (what training pays per step), then one
forward and backward to the latent at batch 1 (what each guided sampler step pays), timed and with
peak memory, on the card the run will use.
"""
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from poe_repair.experiments.presence_head.strong_head import (  # noqa: E402
    SpeciesHead, StrongPresenceHead, UNetMidFeatures,
)
from poe_repair.run import make_ctx  # noqa: E402
from poe_repair.runtime import encode_prompt_sdxl  # noqa: E402


def main() -> int:
    if not torch.cuda.is_available():
        raise SystemExit("guard: no CUDA")
    ctx = make_ctx()
    unet = ctx.models["unet"]
    unet.requires_grad_(False)
    seq_e, pool_e = encode_prompt_sdxl("", models=ctx.models, device=ctx.device, dtype=ctx.dtype)
    feats = UNetMidFeatures(unet, seq_e, pool_e)
    model = StrongPresenceHead(feats, SpeciesHead(200).to(ctx.device),
                               species=["cat", "dog"] + [f"s{i}" for i in range(198)])
    t = torch.tensor([500], device=ctx.device)

    torch.cuda.reset_peak_memory_stats()
    x = torch.randn(8, 4, 128, 128, device=ctx.device)
    with torch.no_grad():
        torch.cuda.synchronize(); t0 = time.time()
        f = feats(x, t.expand(8))
        torch.cuda.synchronize(); fwd = time.time() - t0
    print(f"middle-block features: {tuple(f.shape)}; batch 8 forward without gradient {fwd:.2f}s, "
          f"peak {torch.cuda.max_memory_allocated() / 1024**3:.1f} GiB", flush=True)

    torch.cuda.reset_peak_memory_stats()
    z = torch.randn(1, 4, 128, 128, device=ctx.device, requires_grad=True)
    torch.cuda.synchronize(); t0 = time.time()
    lp = model.log_prob_both(z, t)
    g, = torch.autograd.grad(lp, z)
    torch.cuda.synchronize(); bwd = time.time() - t0
    print(f"gradient to the latent: {tuple(g.shape)}, finite {bool(torch.isfinite(g).all())}, "
          f"norm {g.norm().item():.4g}; forward and backward {bwd:.2f}s, "
          f"peak {torch.cuda.max_memory_allocated() / 1024**3:.1f} GiB", flush=True)
    print(f"a 50-step render guided on 20 steps costs about {20 * bwd:.0f}s on top of the plain render",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
