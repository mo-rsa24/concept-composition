"""Compute job 10: Attend-and-Excite, SynGen, Divide-and-Bind, PoE and the joint prompt on SD 1.5, one tile per cell.

Plan: _archive/1-lora-correction/artifacts/drips/answer-iclr-reviews/routes/plan-sd15-baselines-and-syngen-sdxl.md, Plan 1. Every method runs on
the SD 1.5 snapshot 451f4fe at 512 by 512, guidance 7.5, 50 DDIM steps (decision D1: the scheduler stored in the snapshot,
PNDM, is swapped for DDIMScheduler.from_config of its config, here in the runner; no method code is edited), fp32 (D3),
and every cell starts from the saved latent of its seed (1-lora-correction/scripts/review_sd15_prep.py), passed through each pipeline's
own `latents=` argument. SynGen keeps its released include_entities=False (D2).

The top level imports only the standard library and torch, and each method's imports sit in its loader, so this one
file runs under all three environments:
    joint, poe   co3                                    the repo's rule, as render_unet in same_rule_four_models.py
    aae          /datasets/mmolefe/envs/sd15-aae         composition/aae (= yuval-alaluf/Attend-and-Excite@163efdf)
    dnb          /datasets/mmolefe/envs/sd15-aae         boschresearch/Divide-and-Bind@644dfda, loss tv_bind
    syngen       /datasets/mmolefe/envs/syngen           RoyiRa/Linguistic-Binding-in-Diffusion-Models@50b8eab

    python 1-lora-correction/scripts/review_sd15_cell.py --method aae --shard 0/2      # production, every 2nd cell from the first
    python 1-lora-correction/scripts/review_sd15_cell.py --method syngen --smoke        # the smoke cells plus the PNDM tile
    python 1-lora-correction/scripts/review_sd15_cell.py --check                        # the smoke checks (co3), after every method

Writes <OUT>/sd15/<method>_default/r1/<pair slug>/seedNN/<method>.png and .json (protocol rule 9's fields). A tile whose
PNG and JSON both exist is skipped, so a stopped job resumes. The smoke run writes its extra variants beside the cat and
dog tile as <method>_pndm.png and <method>_standard_sd.png, with no JSON, so the row builder never reads them, and a
per-method summary to <OUT>/sd15/smoke/<method>.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

REPO = Path(os.environ.get("POE_REPO", Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(REPO / "scripts"))
from review_baseline_cell import UNetCounter, make_dir, slug  # noqa: E402

OUT_ROOT = Path(os.environ.get("REVIEW_SD15_OUT", "/datasets/mmolefe/poe_repair_min/outputs/paper_review"))
OUT = OUT_ROOT / "sd15"
LATENTS = OUT_ROOT / "latents_sd15"
SNAPSHOT = ("/home-mscluster/mmolefe/.cache/huggingface/hub/models--runwayml--stable-diffusion-v1-5/snapshots/"
            "451f4fe16113bff5a5d2269ed5ad43b0592e9a14")
AAE_DIR = REPO / "composition" / "aae"
SYNGEN_DIR = Path("/datasets/mmolefe/repos/Linguistic-Binding-in-Diffusion-Models")
DNB_DIR = Path("/datasets/mmolefe/repos/Divide-and-Bind")
PINNED = {"aae": "163efdfd341bf3590df3c0c2b582935fbc8e8343", "syngen": "50b8eab175dc9b31ce9c7c4b2eba28c817fc440e",
          "dnb": "644dfda4ee4e3fbf8ac7128d47623b7274322042"}
PAIRS = REPO / "scripts" / "review_benchmark_pairs.txt"
INDICES = REPO / "data" / "prompt-sets" / "attend-and-excite" / "token_indices_sd15.json"
STEPS, GUIDANCE, RESOLUTION = 50, 7.5, 512
SEEDS = (9, 10, 11, 12)
SYNGEN_STEP_SIZE, SYNGEN_ATTN_RES = 20.0, (16, 16)
# The smoke cells: one per benchmark set, seed 9. "a cat and a dog" carries the identity checks (no adjective, so
# SynGen's loss is zero); the other two give SynGen a prompt with modifiers in the same process (the reset check)
# and give every method a measured time on an animals-with-objects and an objects prompt.
SMOKE = (("a cat", "a dog", "and", 9), ("a pink crown", "a purple bow", "and", 9), ("a monkey", "a red car", "and", 9))
# The bar for "the method took the saved latent and the DDIM swap": mean absolute difference from the joint tile, in
# grey levels of 0 to 255. 3 is the fp16-against-fp32 gap measured on SDXL; the stacks here differ in torch and diffusers.
MAX_GREY_DIFF = 3.0


def benchmark() -> list[tuple[str, str, str]]:
    rows = []
    for ln in PAIRS.read_text().splitlines():
        if ln.strip() and not ln.startswith("#"):
            a, b, joiner, _ = ln.strip().split("|")
            rows.append((a, b, joiner))
    assert len(rows) == 276, f"{len(rows)} benchmark prompts, not 276"
    return rows


def md5(x: torch.Tensor) -> str:
    return hashlib.md5(x.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def load_latent(seed: int) -> tuple[torch.Tensor, dict]:
    path = LATENTS / f"seed{seed:02d}.pt"
    x = torch.load(path, map_location="cpu")
    assert tuple(x.shape) == (1, 4, 64, 64) and x.dtype == torch.float32, (path, x.shape, x.dtype)
    return x, {"file": str(path), "md5": md5(x)}


def git_head(d: Path) -> str:
    try:
        return subprocess.run(["git", "-C", str(d), "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception as e:  # recorded, never fatal
        return f"unknown ({type(e).__name__})"


def to_pil(img: torch.Tensor):
    from PIL import Image
    arr = ((img[0].float().clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(arr)


# ---- the five methods. Each engine loads once and draws one image per call; variant is "ddim" (the column),
# "pndm" (the released scheduler, smoke only) or "standard_sd" (aae and dnb with their own loss switched off).

class Plain:
    """Joint prompt and PoE: eps = eps_u + w * sum_c (eps_c - eps_u), the rule of render_unet, in fp32."""

    variants = ("pndm",)

    def __init__(self) -> None:
        import diffusers
        from diffusers import AutoencoderKL, DDIMScheduler, PNDMScheduler, UNet2DConditionModel
        from transformers import CLIPTextModel, CLIPTokenizer
        self.unet = UNet2DConditionModel.from_pretrained(SNAPSHOT, subfolder="unet").to("cuda").eval()
        self.vae = AutoencoderKL.from_pretrained(SNAPSHOT, subfolder="vae").to("cuda").eval()
        self.text_encoder = CLIPTextModel.from_pretrained(SNAPSHOT, subfolder="text_encoder").to("cuda").eval()
        self.tokenizer = CLIPTokenizer.from_pretrained(SNAPSHOT, subfolder="tokenizer")
        self.pndm = PNDMScheduler.from_pretrained(SNAPSHOT, subfolder="scheduler")
        self.ddim = DDIMScheduler.from_config(self.pndm.config)
        self.versions = {"diffusers": diffusers.__version__}

    @torch.no_grad()
    def enc(self, text: str) -> torch.Tensor:
        ids = self.tokenizer(text, padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True,
                             return_tensors="pt").input_ids.to("cuda")
        return self.text_encoder(ids)[0]

    @torch.no_grad()
    def draw(self, method, a, b, joint, seed, lat, variant):
        sch = self.pndm if variant == "pndm" else self.ddim
        conds = [self.enc(joint)] if method == "joint" else [self.enc(a), self.enc(b)]
        emb = torch.cat([self.enc("")] + conds)
        n = emb.shape[0]
        sch.set_timesteps(STEPS, device="cuda")
        x = lat.to("cuda") * sch.init_noise_sigma
        for t in sch.timesteps:
            eps = self.unet(sch.scale_model_input(x.repeat(n, 1, 1, 1), t), t, encoder_hidden_states=emb).sample
            e0, *ec = eps.chunk(n)
            x = sch.step(e0 + GUIDANCE * sum(c - e0 for c in ec), t, x).prev_sample
        img = self.vae.decode(x / self.vae.config.scaling_factor).sample
        return to_pil(img), {}

    def knobs(self, method):
        return {"rule": "joint prompt with CFG" if method == "joint" else "eps_u + w(eps_a-eps_u) + w(eps_b-eps_u)",
                "precision": "fp32"}

    def commit(self, method):
        return f"poe_repair_min {git_head(REPO)}"


class AttendAndExcite:
    variants = ("pndm", "standard_sd")

    def __init__(self) -> None:
        sys.path.insert(0, str(AAE_DIR))
        import diffusers
        from config import RunConfig
        from diffusers import DDIMScheduler
        from pipeline_attend_and_excite import AttendAndExcitePipeline
        from utils import ptp_utils
        from utils.ptp_utils import AttentionStore
        self.ptp_utils, self.AttentionStore = ptp_utils, AttentionStore
        self.pipe = AttendAndExcitePipeline.from_pretrained(SNAPSHOT, safety_checker=None,
                                                            requires_safety_checker=False).to("cuda")
        self.pndm = self.pipe.scheduler
        self.ddim = DDIMScheduler.from_config(self.pndm.config)
        self.unet = self.pipe.unet
        # config.py's defaults, read from the released file rather than restated here.
        self.cfg = RunConfig(prompt="", output_path=OUT / "_scratch_aae")
        self.indices = json.loads(INDICES.read_text())
        self.versions = {"diffusers": diffusers.__version__}

    def draw(self, method, a, b, joint, seed, lat, variant):
        c = self.cfg
        self.pipe.scheduler = self.pndm if variant == "pndm" else self.ddim
        controller = self.AttentionStore()
        self.ptp_utils.register_attention_control(self.pipe, controller)  # as run.py's run_on_prompt does
        idx = self.indices[slug(a, b)]["token_indices"]
        out = self.pipe(prompt=joint, attention_store=controller, indices_to_alter=idx,
                        attention_res=c.attention_res, guidance_scale=c.guidance_scale,
                        generator=torch.Generator("cuda").manual_seed(seed), latents=lat,
                        num_inference_steps=c.n_inference_steps, max_iter_to_alter=c.max_iter_to_alter,
                        run_standard_sd=(variant == "standard_sd") or c.run_standard_sd, thresholds=c.thresholds,
                        scale_factor=c.scale_factor, scale_range=c.scale_range, smooth_attentions=c.smooth_attentions,
                        sigma=c.sigma, kernel_size=c.kernel_size, sd_2_1=c.sd_2_1)
        self.pipe.scheduler = self.ddim
        return out.images[0], {"token_indices": idx}

    def knobs(self, method):
        c = self.cfg
        return {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(c).items() if k not in ("prompt", "output_path")}

    def commit(self, method):
        return f"{PINNED['aae']} (vendored at composition/aae, poe_repair_min {git_head(REPO)})"


class DivideAndBind:
    variants = ("pndm", "standard_sd")

    def __init__(self) -> None:
        sys.path.insert(0, str(DNB_DIR))
        import diffusers
        from diffusers import DDIMScheduler
        from divide_and_bind.config import RunConfig
        from divide_and_bind.pipeline_divide_and_bind import DivideAndBindPipeline
        from utils import ptp_utils
        from utils.ptp_utils import AttentionStore
        from utils.test_utils import default_param
        self.ptp_utils, self.AttentionStore = ptp_utils, AttentionStore
        self.pipe = DivideAndBindPipeline.from_pretrained(SNAPSHOT, safety_checker=None,
                                                          requires_safety_checker=False).to("cuda")
        self.pndm = self.pipe.scheduler
        self.ddim = DDIMScheduler.from_config(self.pndm.config)
        self.unet = self.pipe.unet
        # generate_images.ipynb: loss_mode tv_bind, max_iter_to_alter 25, default_param('tv_bind'), RunConfig defaults.
        self.loss_mode = "tv_bind"
        self.prm = default_param(self.loss_mode)
        self.cfg = RunConfig(prompt="", output_path=OUT / "_scratch_dnb", max_iter_to_alter=25,
                             thresholds=self.prm["thresholds"])
        self.indices = json.loads(INDICES.read_text())
        self.versions = {"diffusers": diffusers.__version__}

    def draw(self, method, a, b, joint, seed, lat, variant):
        c, prm = self.cfg, self.prm
        self.pipe.scheduler = self.pndm if variant == "pndm" else self.ddim
        controller = self.AttentionStore()  # its defaults save nothing to disk
        self.ptp_utils.register_attention_control(self.pipe, controller)
        rec = self.indices[slug(a, b)]
        colours = rec["color_indices"] if any(rec["color_indices"]) else None  # the notebook: None without attributes
        out = self.pipe(prompt=[joint], attention_store=controller, indices_to_alter=rec["token_indices"],
                        attention_res=c.attention_res, guidance_scale=c.guidance_scale,
                        generator=torch.Generator("cuda").manual_seed(seed), latents=lat,
                        num_inference_steps=c.n_inference_steps, max_iter_to_alter=c.max_iter_to_alter,
                        run_standard_sd=(variant == "standard_sd") or c.run_standard_sd, thresholds=c.thresholds,
                        scale_factor=c.scale_factor, scale_range=c.scale_range, smooth_attentions=c.smooth_attentions,
                        sigma=c.sigma, kernel_size=c.kernel_size, save_attn_everystep=False, loss_mode=self.loss_mode,
                        max_refinement_steps_init_step=prm["max_refinement_steps_init_step"],
                        color_index_list=colours, threshold_indicator=prm["threshold_indicator"])
        self.pipe.scheduler = self.ddim
        return out.images[0], {"token_indices": rec["token_indices"], "color_indices": colours}

    def knobs(self, method):
        c = self.cfg
        k = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(c).items() if k not in ("prompt", "output_path")}
        return {**k, "loss_mode": self.loss_mode, **self.prm}

    def commit(self, method):
        return git_head(DNB_DIR)


class SynGen:
    variants = ("pndm",)

    def __init__(self) -> None:
        sys.path.insert(0, str(SYNGEN_DIR))
        import diffusers
        from diffusers import DDIMScheduler
        from syngen_diffusion_pipeline import SynGenDiffusionPipeline
        # run.py's load_model, with include_entities passed as the released default (its argparse type=bool flag
        # reads any value given on the command line as True, so it is set here directly).
        self.pipe = SynGenDiffusionPipeline.from_pretrained(SNAPSHOT, include_entities=False, safety_checker=None,
                                                            requires_safety_checker=False).to("cuda")
        self.pndm = self.pipe.scheduler
        self.ddim = DDIMScheduler.from_config(self.pndm.config)
        self.unet = self.pipe.unet
        self.losses: list[float] = []
        released = self.pipe._compute_loss

        def logged(attention_maps, prompt):  # records the loss each update step sees; returns it unchanged
            loss = released(attention_maps=attention_maps, prompt=prompt)
            self.losses.append(float(loss))
            return loss

        self.pipe._compute_loss = logged
        self.versions = {"diffusers": diffusers.__version__}

    def draw(self, method, a, b, joint, seed, lat, variant):
        self.pipe.scheduler = self.pndm if variant == "pndm" else self.ddim
        # The pipeline keeps the first prompt's parse in subtrees_indices and only reparses when it is empty; the
        # released run.py builds one pipeline per prompt, so resetting it here reproduces that.
        self.pipe.subtrees_indices = None
        self.losses = []
        out = self.pipe(prompt=joint, generator=torch.Generator("cuda").manual_seed(seed), latents=lat,
                        num_inference_steps=STEPS, guidance_scale=GUIDANCE, syngen_step_size=SYNGEN_STEP_SIZE,
                        attn_res=SYNGEN_ATTN_RES)
        self.pipe.scheduler = self.ddim
        return out.images[0], {"syngen_indices": self.pipe.subtrees_indices, "loss_per_step": self.losses}

    def knobs(self, method):
        return {"syngen_step_size": SYNGEN_STEP_SIZE, "attn_res": list(SYNGEN_ATTN_RES), "num_intervention_steps": 25,
                "include_entities": False, "parser": "en_core_web_trf"}

    def commit(self, method):
        return git_head(SYNGEN_DIR)


ENGINES = {"joint": Plain, "poe": Plain, "aae": AttendAndExcite, "dnb": DivideAndBind, "syngen": SynGen}


def tile(method: str, a: str, b: str, seed: int) -> Path:
    return OUT / f"{method}_default" / "r1" / slug(a, b) / f"seed{seed:02d}" / f"{method}.png"


def render(methods: list[str], cells: list[tuple[str, str, str, int]], smoke: bool) -> None:
    engine = None
    for method in methods:
        if engine is None or not isinstance(engine, ENGINES[method]):
            engine = ENGINES[method]()
            counter = UNetCounter(engine.unet)
        summary = []
        print(f"=== {method}: {len(cells)} cells, {sys.executable}, torch {torch.__version__}, {engine.versions}, "
              f"{torch.cuda.get_device_name(0)}", flush=True)
        for a, b, joiner, seed in cells:
            joint = f"{a} {joiner} {b}"
            png = tile(method, a, b, seed)
            if not smoke and png.exists() and png.with_suffix(".json").exists():
                print(f"{slug(a, b)} seed {seed}: already rendered", flush=True)
                continue
            make_dir(png.parent)
            lat, noise = load_latent(seed)
            variants = ["ddim"] + (list(engine.variants) if smoke and (a, b) == SMOKE[0][:2] else [])
            for variant in variants:
                counter.reset()
                torch.cuda.reset_peak_memory_stats()
                t0 = time.perf_counter()
                img, extra = engine.draw(method, a, b, joint, seed, lat, variant)
                torch.cuda.synchronize()
                secs = time.perf_counter() - t0
                peak = torch.cuda.max_memory_allocated() / 2 ** 30
                out = png if variant == "ddim" else png.with_name(f"{method}_{variant}.png")
                img.save(out)
                rec = {"model": "SD 1.5", "method": method, "adapter": "none",
                       "sampler": "DDIM" if variant != "pndm" else "PNDM (released default, smoke only)",
                       "steps": STEPS, "guidance": GUIDANCE, "resolution": RESOLUTION, "pair": slug(a, b),
                       "seed": seed, "group": "r1", "setting": "default" if variant == "ddim" else variant,
                       "nfe": counter.rows, "unet_calls": counter.calls, "prompt_a": a, "prompt_b": b,
                       "joint_prompt": joint, "noise": noise, "knobs": engine.knobs(method), **extra,
                       "seconds": round(secs, 2), "peak_gb": round(peak, 2), "commit": engine.commit(method),
                       "python": sys.executable, "torch": torch.__version__, **engine.versions,
                       "device": torch.cuda.get_device_name(0), "model_path": SNAPSHOT}
                if variant == "ddim":
                    png.with_suffix(".json").write_text(json.dumps(rec, indent=1))
                summary.append({**rec, "tile": str(out)})
                print(f"{method} {variant} {slug(a, b)} seed {seed}: {secs:.1f} s, peak {peak:.1f} GB, "
                      f"nfe {counter.rows} ({counter.calls} calls) -> {out}", flush=True)
        if smoke:
            make_dir(OUT / "smoke")
            (OUT / "smoke" / f"{method}.json").write_text(json.dumps(summary, indent=1))


# ---- the smoke checks, run in co3 after every method's smoke cells exist.

def check() -> int:
    import numpy as np
    from PIL import Image, ImageDraw

    a, b, _, seed = SMOKE[0]
    s = slug(a, b)
    grey = lambda p: np.asarray(Image.open(p).convert("RGB"), dtype=np.float64)
    diff = lambda p, q: float(np.abs(grey(p) - grey(q)).mean())
    results, ok = [], True

    def verdict(name, passed, detail):
        nonlocal ok
        ok &= bool(passed)
        results.append({"check": name, "pass": bool(passed), "detail": detail})
        print(f"{'PASS' if passed else 'FAIL'}  {name}: {detail}", flush=True)

    methods = ("joint", "poe", "aae", "syngen", "dnb")
    have = {m: tile(m, a, b, seed).exists() and tile(m, a, b, seed).with_suffix(".json").exists() for m in methods}
    verdict("one tile and one sidecar per method", all(have.values()), have)
    if not all(have.values()):
        (OUT / "smoke" / "check.json").write_text(json.dumps(results, indent=1))
        return 1
    joint = tile("joint", a, b, seed)
    for m, variant in (("aae", "standard_sd"), ("dnb", "standard_sd"), ("syngen", None)):
        p = tile(m, a, b, seed) if variant is None else tile(m, a, b, seed).with_name(f"{m}_{variant}.png")
        d = diff(p, joint)
        verdict(f"{m}{'_' + variant if variant else ''} against the joint tile", d < MAX_GREY_DIFF,
                f"mean |difference| {d:.3f} grey levels (bar {MAX_GREY_DIFF})")
    for m in ("poe", "aae", "dnb"):
        print(f"info  {m} (method on) against joint: {diff(tile(m, a, b, seed), joint):.2f} grey levels", flush=True)
    for m in methods:
        p = tile(m, a, b, seed).with_name(f"{m}_pndm.png")
        print(f"info  {m} PNDM against DDIM: {diff(p, tile(m, a, b, seed)):.2f} grey levels" if p.exists()
              else f"info  {m}: no PNDM tile", flush=True)
    # SynGen: the cat and dog parse, then a prompt with modifiers in the same process.
    idx = []
    for pa, pb, _, sd in SMOKE:
        j = tile("syngen", pa, pb, sd).with_suffix(".json")
        idx.append(json.loads(j.read_text()).get("syngen_indices") if j.exists() else "missing")
    verdict("SynGen reparses per prompt", idx[0] == [] and idx[1] not in ([], "missing", None) and idx[1] != idx[2],
            dict(zip([slug(x[0], x[1]) for x in SMOKE], idx)))
    # The shared starting latent: every sidecar names the seed's file and the md5 that file has now.
    lat, noise = load_latent(seed)
    names = {}
    for m in methods:
        for pa, pb, _, sd in SMOKE:
            j = tile(m, pa, pb, sd).with_suffix(".json")
            if j.exists() and sd == seed:
                names[f"{m} {slug(pa, pb)}"] = json.loads(j.read_text())["noise"] == noise
    verdict("every sidecar names latents_sd15/seed09.pt with its md5", all(names.values()), names)
    # Time and memory, from the smoke summaries.
    rows = []
    for m in methods:
        f = OUT / "smoke" / f"{m}.json"
        for r in (json.loads(f.read_text()) if f.exists() else []):
            rows.append({k: r[k] for k in ("method", "pair", "setting", "seconds", "peak_gb", "nfe", "unet_calls")})
            print(f"time  {r['method']:6s} {r['setting']:12s} {r['pair']:32s} {r['seconds']:7.1f} s  "
                  f"{r['peak_gb']:5.1f} GB  nfe {r['nfe']}", flush=True)
    # The contact sheet: one row per smoke prompt, one column per method; then the variant row for cat and dog.
    cols = list(methods)
    W = 256
    sheet = Image.new("RGB", (W * len(cols), (W + 18) * (len(SMOKE) + 1)), "white")
    dr = ImageDraw.Draw(sheet)
    for r, (pa, pb, _, sd) in enumerate(SMOKE):
        for c, m in enumerate(cols):
            p = tile(m, pa, pb, sd)
            if p.exists():
                sheet.paste(Image.open(p).convert("RGB").resize((W, W)), (c * W, r * (W + 18) + 18))
            dr.text((c * W + 4, r * (W + 18) + 3), f"{m}  {pa} / {pb}", fill="black")
    r = len(SMOKE)
    for c, m in enumerate(cols):
        v = "standard_sd" if m in ("aae", "dnb") else "pndm"
        p = tile(m, a, b, seed).with_name(f"{m}_{v}.png")
        if p.exists():
            sheet.paste(Image.open(p).convert("RGB").resize((W, W)), (c * W, r * (W + 18) + 18))
        dr.text((c * W + 4, r * (W + 18) + 3), f"{m} {v}", fill="black")
    sheet.save(OUT / "smoke" / "contact_sheet.png")
    (OUT / "smoke" / "check.json").write_text(json.dumps({"pass": ok, "checks": results, "times": rows}, indent=1))
    print(f"{'ALL PASS' if ok else 'SOME FAILED'} -> {OUT / 'smoke' / 'check.json'}, {OUT / 'smoke' / 'contact_sheet.png'}",
          flush=True)
    return 0 if ok else 1


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--method", choices=sorted(ENGINES), nargs="+")
    p.add_argument("--shard", default="0/1", help="i/n: every n-th cell from the i-th")
    p.add_argument("--smoke", action="store_true", help="the three smoke cells at seed 9, plus the variants")
    p.add_argument("--check", action="store_true", help="the smoke checks and the contact sheet (co3)")
    args = p.parse_args()
    if args.check:
        sys.exit(check())
    assert args.method, "--method is required unless --check"
    assert torch.cuda.is_available(), "torch sees no CUDA"
    if args.smoke:
        cells = list(SMOKE)
    else:
        i, n = (int(x) for x in args.shard.split("/"))
        cells = [(a, b, j, s) for a, b, j in benchmark() for s in SEEDS][i::n]
    render(args.method, cells, args.smoke)


if __name__ == "__main__":
    main()
