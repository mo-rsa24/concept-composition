"""SynGen ported to SDXL (poe_repair/baselines/syngen_sdxl.py) over Table 1's 1,104 cells, one tile per cell.

Plan: 1-lora-correction/artifacts/drips/answer-iclr-reviews/routes/plan-sd15-baselines-and-syngen-sdxl.md, Plan 2. Same cells, starting
latents and sampler as the joint, PoE, ours and CO3 columns: 1024 by 1024, 50 DDIM steps, guidance 7.5, fp16, the seed's
latent from the a_cat__x__a_dog cache path (`init_latents_for_cell`, which rendered r1_joint) checked byte for byte against
the saved paper_review/latents/seedNN.pt. SynGen's step size is its released 20.0 (decision D4); the parse is read from
data/prompt-sets/attend-and-excite/syngen_indices_sdxl.json (1-lora-correction/scripts/review_syngen_indices.py).

    python 1-lora-correction/scripts/review_syngen_sdxl_cell.py --shard 0/2      # production, every 2nd cell from the first
    python 1-lora-correction/scripts/review_syngen_sdxl_cell.py --smoke          # the identity check, the loss check, the contact sheet

Production writes <OUT>/syngen-sdxl_default/r1/<pair slug>/seedNN/syngen-sdxl.png and .json (shard 0 adds Figure 5's two
rows outside R1 under r6/ instead of r1/), skipping any cell whose PNG
and JSON both exist. A cell whose gradient goes non-finite writes no tile and is listed in aborted_<i>of<n>.json. The smoke
run writes under <OUT>/syngen-sdxl_smoke/ only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import torch

os.environ.setdefault("HF_HUB_OFFLINE", "1")
REPO = Path(os.environ.get("POE_REPO", Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
from review_baseline_cell import UNetCounter, make_dir, row, slug, tile  # noqa: E402

OUT = Path(os.environ.get("REVIEW_BASELINE_OUT", "/datasets/mmolefe/poe_repair_min/outputs/paper_review"))
LATENTS = OUT / "latents"
PAIRS = REPO / "scripts" / "review_benchmark_pairs.txt"
INDICES = REPO / "data" / "prompt-sets" / "attend-and-excite" / "syngen_indices_sdxl.json"
SD15 = OUT / "sd15" / "syngen_default" / "r1"
NOISE_PAIR = "a_cat__x__a_dog"
SEEDS = (9, 10, 11, 12)
METHOD = "syngen-sdxl"
SAMPLER = "DDIM with SynGen latent update (port)"
# Figure 5's two rows outside R1, rendered by shard 0 after its R1 cells, under group r6 as CO3's tiles for them are.
FIG5_EXTRA = (("a tent", "a snowy mountain", "and", 10), ("a lighthouse", "a stormy sea", "and", 10))
SMOKE_IDENTITY = ("a cat", "a dog", "and", 9)
SMOKE_LOSS = (("a pink crown", "a purple bow", "and", 9), ("a monkey", "a red car", "and", 9))
MAX_GREY_DIFF = 3.0  # grey levels, the fp16-against-fp32 gap measured on SDXL, used only if the bytes differ


def benchmark() -> list[tuple[str, str, str]]:
    rows = [ln.strip().split("|")[:3] for ln in PAIRS.read_text().splitlines() if ln.strip() and not ln.startswith("#")]
    assert len(rows) == 276, f"{len(rows)} benchmark prompts, not 276"
    return [tuple(r) for r in rows]


def md5(x: torch.Tensor) -> str:
    return hashlib.md5(x.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


class Port:
    def __init__(self) -> None:
        from poe_repair.baselines import syngen_sdxl
        from poe_repair.run import make_ctx
        self.sx = syngen_sdxl
        self.ctx = make_ctx()
        self.unet = self.ctx.models["unet"]
        self.unet.requires_grad_(False)  # gradients flow to the latent only
        self.counter = UNetCounter(self.unet)
        sgp = syngen_sdxl.import_syngen()
        self.loss_fn = syngen_sdxl.SynGenLoss(sgp, self.ctx.models["tokenizer"])
        self.indices = json.loads(INDICES.read_text())
        self.saved = {}

    def check_tokenizers(self, prompt: str) -> None:
        t1, t2 = self.ctx.models["tokenizer"], self.ctx.models["tokenizer_2"]
        i1, i2 = t1(prompt)["input_ids"], t2(prompt)["input_ids"]
        assert i1 == i2, f"tokenizer and tokenizer_2 differ on {prompt!r}: {i1} {i2}"

    def draw(self, a, b, joiner, seed, step_size, group="r1"):
        from poe_repair._sdxl.runtime import encode_prompt_sdxl
        from poe_repair.composers._helpers import init_latents_for_cell
        from poe_repair.experiments.interaction_term.cell import cell_from_slug
        ctx = self.ctx
        joint = f"{a} {joiner} {b}"
        self.check_tokenizers(joint)
        init, sigma = init_latents_for_cell(cell_from_slug(NOISE_PAIR, seed), ctx)
        if seed not in self.saved:
            self.saved[seed] = torch.load(LATENTS / f"seed{seed:02d}.pt", map_location="cpu")
        assert torch.equal(init.cpu(), self.saved[seed].to(init.dtype)), f"seed {seed}: x_T differs from the saved latent"
        enc = lambda p: encode_prompt_sdxl(p, models=ctx.models, device=ctx.device, dtype=ctx.dtype)
        seq_e, pool_e = enc("")
        seq_c, pool_c = enc(joint)
        idx = self.indices[slug(a, b)]["syngen_indices"]
        self.counter.reset()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        image, losses = self.sx.run_cfg_syngen(
            init_latents=init, models=ctx.models, scheduler=ctx.scheduler, seq_cond=seq_c, pool_cond=pool_c,
            seq_e=seq_e, pool_e=pool_e, guidance_scale=ctx.guidance_scale,
            num_inference_steps=ctx.num_inference_steps, height=1024, width=1024, euler_init_noise_sigma=sigma,
            device=ctx.device, dtype=ctx.dtype, prompt=joint, indices=idx, loss_fn=self.loss_fn, step_size=step_size)
        torch.cuda.synchronize()
        secs = time.perf_counter() - t0
        knobs = {"syngen_step_size": step_size, "attn_res": list(self.sx.ATTN_RES),
                 "num_intervention_steps": self.sx.NUM_INTERVENTION_STEPS, "include_entities": False,
                 "parser": "en_core_web_trf", "precision": "U-Net fp16, maps, loss and latent update fp32"}
        rec = row(METHOD, SAMPLER, ctx.num_inference_steps, ctx.guidance_scale, a, b, seed, "default", self.counter,
                  f"seed latent {LATENTS / f'seed{seed:02d}.pt'} md5 {md5(self.saved[seed])}", knobs, secs, group, joint)
        rec.update({"syngen_indices": idx, "loss_per_step": losses, "peak_gb": round(torch.cuda.max_memory_allocated()
                    / 2 ** 30, 2), "commit": self.sx.SYNGEN_COMMIT, "device": torch.cuda.get_device_name(0),
                    "python": sys.executable, "torch": torch.__version__})
        return image, rec


def write(image, rec, png: Path) -> None:
    from poe_repair.methods._sampling import write_decoded_image
    make_dir(png.parent)
    write_decoded_image(image, png)
    png.with_suffix(".json").write_text(json.dumps(rec, indent=1))


def produce(shard: str, step_size: float) -> None:
    from poe_repair.baselines.syngen_sdxl import SynGenNonFinite
    i, n = (int(x) for x in shard.split("/"))
    cells = [("r1", a, b, j, s) for a, b, j in benchmark() for s in SEEDS][i::n]
    if i == 0:
        cells += [("r6", *c) for c in FIG5_EXTRA]
    port, aborted = Port(), []
    root = OUT / f"{METHOD}_default"
    print(f"{len(cells)} cells, shard {shard}, {torch.cuda.get_device_name(0)}", flush=True)
    for group, a, b, joiner, seed in cells:
        png = tile(root, group, a, b, seed, METHOD)
        if png.exists() and png.with_suffix(".json").exists():
            print(f"{slug(a, b)} seed {seed}: already rendered", flush=True)
            continue
        try:
            image, rec = port.draw(a, b, joiner, seed, step_size, group)
        except SynGenNonFinite as e:
            aborted.append({"pair": slug(a, b), "seed": seed, "error": str(e)})
            print(f"{slug(a, b)} seed {seed}: ABORTED {e}", flush=True)
            continue
        write(image, rec, png)
        print(f"{slug(a, b)} seed {seed}: {rec['seconds']:.1f} s, peak {rec['peak_gb']:.1f} GB, nfe {rec['nfe']}, "
              f"loss {rec['loss_per_step'][0]:.3f} -> {rec['loss_per_step'][-1]:.3f} -> {png}", flush=True)
    (root / f"aborted_{i}of{n}.json").write_text(json.dumps(aborted, indent=1))
    print(f"done: {len(aborted)} aborted cells", flush=True)


def smoke(step_size: float) -> int:
    import numpy as np
    from PIL import Image, ImageDraw
    port, root, ok, checks = Port(), OUT / f"{METHOD}_smoke", True, []

    def verdict(name, passed, detail):
        nonlocal ok
        ok &= bool(passed)
        checks.append({"check": name, "pass": bool(passed), "detail": detail})
        print(f"{'PASS' if passed else 'FAIL'}  {name}: {detail}", flush=True)

    a, b, j, s = SMOKE_IDENTITY
    image, rec = port.draw(a, b, j, s, 0.0)
    png = root / "step0" / slug(a, b) / f"seed{s:02d}" / f"{METHOD}.png"
    write(image, rec, png)
    ref = OUT / "r1_joint" / slug(a, b) / f"seed{s:02d}" / "mono.png"
    x, y = (np.asarray(Image.open(p).convert("RGB"), dtype=np.float64) for p in (png, ref))
    same, d = bool((x == y).all()), float(np.abs(x - y).mean())
    verdict("step size 0 reproduces the joint tile", same or d < MAX_GREY_DIFF,
            f"{'byte-identical' if same else f'mean |difference| {d:.4f} grey levels'} against {ref}; "
            f"{rec['seconds']:.1f} s, peak {rec['peak_gb']:.1f} GB on {rec['device']}")
    tiles = [(f"step 0  {a} / {b}", png), ("joint (r1_joint)", ref),
             ("PoE (r1_poe)", OUT / "r1_poe" / slug(a, b) / f"seed{s:02d}" / "poe.png")]
    for a, b, j, s in SMOKE_LOSS:
        image, rec = port.draw(a, b, j, s, step_size)
        png = root / f"step{step_size:g}" / slug(a, b) / f"seed{s:02d}" / f"{METHOD}.png"
        write(image, rec, png)
        L = rec["loss_per_step"]
        sd15 = SD15 / slug(a, b) / f"seed{s:02d}" / "syngen.json"
        L15 = json.loads(sd15.read_text())["loss_per_step"] if sd15.exists() else None
        verdict(f"the loss falls on {a} / {b}", len(L) == 25 and L[-1] < L[0],
                f"SDXL step 0 {L[0]:.4f}, step 24 {L[-1]:.4f}; SD 1.5 "
                + (f"{L15[0]:.4f} -> {L15[-1]:.4f}" if L15 else "not on file")
                + f"; indices {rec['syngen_indices']}; {rec['seconds']:.1f} s, peak {rec['peak_gb']:.1f} GB")
        print("  SDXL  " + " ".join(f"{v:.3f}" for v in L), flush=True)
        if L15:
            print("  SD1.5 " + " ".join(f"{v:.3f}" for v in L15), flush=True)
        tiles += [(f"step {step_size:g}  {a} / {b}", png),
                  ("joint", OUT / "r1_joint" / slug(a, b) / f"seed{s:02d}" / "mono.png"),
                  ("PoE", OUT / "r1_poe" / slug(a, b) / f"seed{s:02d}" / "poe.png")]
    W = 320
    sheet = Image.new("RGB", (3 * W, 3 * (W + 18)), "white")
    dr = ImageDraw.Draw(sheet)
    for k, (label, p) in enumerate(tiles):
        r, c = divmod(k, 3)
        if Path(p).exists():
            sheet.paste(Image.open(p).convert("RGB").resize((W, W)), (c * W, r * (W + 18) + 18))
        dr.text((c * W + 4, r * (W + 18) + 3), label, fill="black")
    sheet.save(root / "contact_sheet.png")
    (root / "check.json").write_text(json.dumps({"pass": ok, "checks": checks}, indent=1))
    print(f"{'ALL PASS' if ok else 'SOME FAILED'} -> {root / 'check.json'}, {root / 'contact_sheet.png'}", flush=True)
    return 0 if ok else 1


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--shard", default="0/1")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--step-size", type=float, default=20.0, help="SynGen's released default (decision D4)")
    args = p.parse_args()
    assert torch.cuda.is_available(), "torch sees no CUDA"
    if args.smoke:
        sys.exit(smoke(args.step_size))
    produce(args.shard, args.step_size)


if __name__ == "__main__":
    main()
