#!/usr/bin/env python
"""Push, then pick: does choosing among a few pushed pictures give a clean cat and dog more often?

The readout checker's push gives two animals on most held-out seeds but a clean cat and dog on one or two
(checker-loop.md, iteration 6). Its draws vary with the starting noise, so for each held-out seed (9 to 16)
this draws four tries from four starting noises (the seed's own, then fresh draws from generator
40,000 + 10 * seed + k), each once as plain PoE and once with the identity push (iteration 6's checker,
readout-catdog-identity, strength 100, cap 0.1). The CLIP checker fine-tuned on finished pictures
(finetune/clip/best.pt, 0.900) scores every finished picture and picks the best of the four in each set, so
plain picking gets the same number of chances as pushed picking.

Writes, under MOUNT_ROOT/outputs/classifier_guidance_ladder/rung2/push-then-pick/:
    tiles/seedNN__<plain|pushed>_tryK.png, picks.json, sheet.png
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE.parent))  # scripts/, for classifier_guidance_from_the_paper
sys.path.insert(0, str(HERE))
from poe_repair import paths  # noqa: E402
from poe_repair.methods._classifier_guided import run_poe_classifier_guided  # noqa: E402
from poe_repair.methods._sampling import run_cfg_poe, write_decoded_image  # noqa: E402
from poe_repair.run import make_ctx  # noqa: E402
from poe_repair.runtime import encode_prompt_sdxl  # noqa: E402
from poe_repair.composers._helpers import init_latents_for_cell  # noqa: E402
import classifier_guidance_from_the_paper as single  # noqa: E402
from finetune_checker import SPECS, Checker  # noqa: E402
from train_readout_pair import PairHead, ReadoutPairClassifier  # noqa: E402

RUNG2 = paths.MOUNT_ROOT / "outputs/classifier_guidance_ladder/rung2"
OUT = RUNG2 / "push-then-pick"
HEAD = RUNG2 / "readout_pair/readout-catdog-identity/head_step_002000.pt"
PICKER = RUNG2 / "finetune/clip/best.pt"
SEEDS, TRIES, STRENGTH = list(range(9, 17)), 4, 100.0


def main() -> int:
    if not torch.cuda.is_available():
        raise SystemExit("guard: torch sees no CUDA card")
    (OUT / "tiles").mkdir(parents=True, exist_ok=True)
    ctx = make_ctx(output_root=OUT / "raw")
    enc = lambda p: encode_prompt_sdxl(p, models=ctx.models, device=ctx.device, dtype=ctx.dtype)
    seq_e, pool_e = enc("")
    seq_a, pool_a = enc("a cat")
    seq_b, pool_b = enc("a dog")
    common = dict(models=ctx.models, scheduler=ctx.scheduler, seq_a=seq_a, pool_a=pool_a, seq_b=seq_b, pool_b=pool_b,
                  seq_e=seq_e, pool_e=pool_e, guidance_scale=ctx.guidance_scale,
                  num_inference_steps=ctx.num_inference_steps, height=1024, width=1024,
                  device=ctx.device, dtype=ctx.dtype)
    state = torch.load(HEAD, map_location="cpu")
    seq_r, pool_r = enc("a cat and a dog") if state.get("pair_prompt") else (seq_e, pool_e)
    head = PairHead(n_out=state.get("heads", 1))
    head.load_state_dict(state["head"])
    disc = ReadoutPairClassifier(ctx.models["unet"], seq_r, pool_r, head.float().eval()).to(ctx.device)

    pick_state = torch.load(PICKER, map_location="cpu")
    picker = Checker(pick_state["model"])
    picker.load_state_dict(pick_state["state"])
    picker = picker.to(ctx.device).eval()
    spec = SPECS[pick_state["model"]]
    mean = torch.tensor(spec["mean"], device=ctx.device).view(1, 3, 1, 1)
    std = torch.tensor(spec["std"], device=ctx.device).view(1, 3, 1, 1)

    @torch.no_grad()
    def judge(image):  # image: (1, 3, H, W) in [0, 1]
        x = F.interpolate(image.float().to(ctx.device), size=(spec["res"], spec["res"]), mode="bicubic",
                          antialias=True, align_corners=False)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return float(picker((x - mean) / std).float()[0])

    def noise(seed, k):
        if k == 0:
            return init_latents_for_cell(single.cell_for(seed), ctx)
        g = torch.Generator(device="cpu").manual_seed(40_000 + 10 * seed + k)
        return torch.randn(1, 4, 128, 128, generator=g).to(device=ctx.device, dtype=ctx.dtype), 1.0

    record = {"head": str(HEAD), "picker": str(PICKER), "strength": STRENGTH, "tries": TRIES, "seeds": {}}
    t0 = time.time()
    for seed in SEEDS:
        rec = {"plain": {}, "pushed": {}}
        for k in range(TRIES):
            init, sigma = noise(seed, k)
            o = run_cfg_poe(init_latents=init, euler_init_noise_sigma=sigma, **common)
            write_decoded_image(o.image, OUT / "tiles" / f"seed{seed:02d}__plain_try{k}.png")
            rec["plain"][k] = judge(o.image)
            init, sigma = noise(seed, k)
            o = run_poe_classifier_guided(init_latents=init, euler_init_noise_sigma=sigma, head=disc,
                                          strength=STRENGTH, **common)
            write_decoded_image(o.image, OUT / "tiles" / f"seed{seed:02d}__pushed_try{k}.png")
            rec["pushed"][k] = judge(o.image)
        rec["plain_pick"] = max(rec["plain"], key=rec["plain"].get)
        rec["pushed_pick"] = max(rec["pushed"], key=rec["pushed"].get)
        record["seeds"][seed] = rec
        (OUT / "picks.json").write_text(json.dumps(record, indent=1))
        print(f"seed {seed}: plain picks try {rec['plain_pick']} ({rec['plain'][rec['plain_pick']]:+.2f}), pushed picks "
              f"try {rec['pushed_pick']} ({rec['pushed'][rec['pushed_pick']]:+.2f})  ({time.time() - t0:.0f} s)", flush=True)

    T, font = 256, ImageFont.load_default(size=18)
    cols = ["plain_try0", "plain pick"] + [f"pushed_try{k}" for k in range(TRIES)] + ["pushed pick"]
    sheet = Image.new("RGB", (len(cols) * T, 30 + len(SEEDS) * (T + 24)), "white")
    d = ImageDraw.Draw(sheet)
    for i, c in enumerate(cols):
        d.text((i * T + 4, 6), c.replace("_try", " try "), fill="black", font=font)
    for r, seed in enumerate(SEEDS):
        rec = record["seeds"][seed]
        names = ["plain_try0", f"plain_try{rec['plain_pick']}"] + [f"pushed_try{k}" for k in range(TRIES)] + \
                [f"pushed_try{rec['pushed_pick']}"]
        for i, n in enumerate(names):
            y = 30 + r * (T + 24)
            sheet.paste(Image.open(OUT / "tiles" / f"seed{seed:02d}__{n}.png").convert("RGB").resize((T, T)), (i * T, y))
            kind, k = n.split("_try")
            d.text((i * T + 4, y + T + 2), f"seed {seed}  CLIP {rec[kind][int(k)]:+.1f}", fill="black", font=font)
    sheet.save(OUT / "sheet.png")
    print("PUSH THEN PICK DONE", OUT / "sheet.png", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
