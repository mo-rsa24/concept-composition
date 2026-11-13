#!/usr/bin/env python
"""What does the frozen presence head believe about pictures it is about to steer?

Reads its "dog present" probability on the no-prompt pictures, on "a dog" under ordinary guidance,
and on the most-pushed classifier-guided pictures, each encoded and noised to six sampler steps and
averaged over four noise draws. If it calls the no-prompt paintings dogs, it was never shown them,
and the push had no reason to find one.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from poe_repair import paths  # noqa: E402
from poe_repair.experiments.presence_head.strong_head import (  # noqa: E402
    SpeciesHead, StrongPresenceHead, UNetMidFeatures,
)
from poe_repair.run import make_ctx  # noqa: E402
from poe_repair.runtime import encode_prompt_sdxl  # noqa: E402
from PIL import Image  # noqa: E402

TILES = paths.MOUNT_ROOT / "outputs/classifier_guidance_from_the_paper/tiles"
HEAD = paths.MOUNT_ROOT / "outputs/strong_presence_head/strong-01/checkpoints/head_step_004000.pt"
STEPS = (0, 10, 20, 30, 40, 49)
DRAWS = 4


def main() -> int:
    ctx = make_ctx()
    seq_e, pool_e = encode_prompt_sdxl("", models=ctx.models, device=ctx.device, dtype=ctx.dtype)
    state = torch.load(HEAD, map_location="cpu")
    h = SpeciesHead(len(state["species"])).to(ctx.device)
    h.load_state_dict(state["state_dict"])
    model = StrongPresenceHead(UNetMidFeatures(ctx.models["unet"], seq_e, pool_e), h,
                               state["species"], ("dog",)).eval()
    dog = model.species.index("dog")
    ctx.scheduler.set_timesteps(ctx.num_inference_steps)
    ts = ctx.scheduler.timesteps
    vae = ctx.models["vae"]
    gen = torch.Generator().manual_seed(0)
    rows = {}
    print(f"{'picture':24s}" + "".join(f"  step {s:>2}" for s in STEPS))
    for seed in (9, 10, 11, 12):
        for key, label in (("no_prompt", "no prompt"), ("a_dog", "\"a dog\""), ("push30", "most pushed")):
            with Image.open(TILES / f"seed{seed:02d}__{key}.png") as im:
                x = torch.from_numpy(np.asarray(im.convert("RGB"), dtype=np.uint8).copy()).float()
            x = (x.permute(2, 0, 1) / 127.5 - 1).unsqueeze(0).to(ctx.device, vae.dtype)
            with torch.no_grad():
                z0 = (vae.encode(x).latent_dist.mean * vae.config.scaling_factor).float()
                probs = []
                for s in STEPS:
                    t = ts[s].reshape(1).to(ctx.device)
                    p = [torch.sigmoid(model(ctx.scheduler.add_noise(
                        z0, torch.randn(z0.shape, generator=gen).to(ctx.device), t), t))[0, dog].item()
                        for _ in range(DRAWS)]
                    probs.append(sum(p) / DRAWS)
            name = f"seed {seed} {label}"
            rows[name] = probs
            print(f"{name:24s}" + "".join(f"  {p:7.2f}" for p in probs), flush=True)
    out = paths.MOUNT_ROOT / "outputs/dog_classifier"
    out.mkdir(parents=True, exist_ok=True)
    (out / "what_the_head_believes.json").write_text(json.dumps(
        {"steps": list(STEPS), "timesteps": [int(ts[s]) for s in STEPS], "p_dog": rows}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
