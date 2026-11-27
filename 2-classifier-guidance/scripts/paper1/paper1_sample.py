#!/usr/bin/env python3
"""Paper 1's sampler: the colour-expert run's Heun sampler, paste, apart and the mean empty scene, ported to the cluster.

Ported from 2-classifier-guidance/scripts/method_search/m10_pod/m10_compose.py with the behaviour unchanged: the same 32-step Heun
sampler (sigma 80 to 0.002, rho 7, no added noise), the same seeding (EDM2's StackedRandomGenerator, one generator per seed), the
same 0.1 foreground rule against the mean empty scene, the same paste (blue's foreground over red) and the same 1-pixel apart rule.
What changed is only where things come from: every path, count and seed is an argument, and nothing reads the RunPod machine's paths.

Settings this file implements: plans/paper-1-clevr-touching/plans/03-full-plan.md, "The models" ("The sampler", "The region
sampler", "The push"); the first result's sizes: plans/paper-1-clevr-touching/plans/02-first-result.md, "Seeds and scenes".

Layout. A run writes under PAPER1_ROOT/<run-name>/. Pictures go in sub-folders named count-<k> (k = 1, 2, 3, the count each
expert was asked; the scene holds k red and k blue objects, so 2k in all) or empty, one PNG per seed named <seed:06d>.png. Small
outputs (json, txt, sheets) go at the run folder's top level, which the job scripts copy to outputs/paper1/<run-name>/.

Subcommands:
  lone        one expert's (or the empty-scene model's) pictures at the given counts and seeds
  mean-empty  the mean empty scene from a folder of empty-scene pictures (the first 1,024 seeds), saved as mean-empty.pt
  plainsum    the plain sum D_red + D_blue - D_empty at the given counts and seeds
  paste       mask paste of two lone folders on the CPU: blue's foreground over red; --order lower-on-top puts the object whose base
              sits lower in the frame on top where they overlap (the lower-on-top paste row)
  split       touching or apart per seed per count from two lone folders, written to split.json under one seed set's name
  port-sheet  the plain sum beside archived pictures at the same seeds, for the port check
  region      the region sampler from two lone folders' masks; --identity sets both masks to all ones and checks it against the plain sum
  pair-sheet  two folders side by side at the same seeds (the region sampler beside mask paste)
  mask-sheet  lone samples with each object's dilated mask outlined, to check shadows fall inside
  push        ours (the region sampler plus the head's push on sampler steps 10 to 22) or --random-push, at one or more weights;
              push.json with the push length per sigma band and the grey-level change from the region sampler; grad-maps.png
  contact     contact kept on the touching seeds of picture runs, per count and pooled; contact.json
  weight-grid the tuning weights w1 x (1, 2, 4, 8) from the push smoke's in-band change, written before tuning; weight-grid.json

Usage (from the repository root, on the cluster, with the co3 python):
  python paper1_sample.py lone --run paper1-first-lone-red --expert red --red-net <pkl> --counts 1,2,3 --seeds 20000-20255,10000-10255
  python paper1_sample.py lone --run paper1-first-lone-red --expert empty --empty-net <pkl> --seeds 0-1023
  python paper1_sample.py mean-empty --run paper1-first-lone-red --empty-dir <run>/empty
  python paper1_sample.py plainsum --run paper1-first-sampler-port --red-net <pkl> --blue-net <pkl> --empty-net <pkl> --counts 1,2,3 --seeds 0-7
  python paper1_sample.py paste --run paper1-first-row-paste --red-dir <lone red run> --blue-dir <lone blue run> --mean-empty <pt> --counts 1,2,3 --seeds 20000-20255
  python paper1_sample.py split --run paper1-first-lone-red --set evaluation --red-dir <lone red run> --blue-dir <lone blue run> --mean-empty <pt> --counts 1,2,3 --seeds 20000-20255
  python paper1_sample.py region --run paper1-first-region-check --identity --empty-net <pkl> --mean-empty <pt> --counts 1,2,3 --seeds 30000-30007
  python paper1_sample.py region --run paper1-first-region-check --empty-net <pkl> --red-dir <lone red run> --blue-dir <lone blue run> --mean-empty <pt> --counts 2 --split <split.json> --set check --side touching --limit 64
  python paper1_sample.py push --run paper1-first-push-smoke --head paper1-first-disc-k64/head.pt --weights ladder --red-dir <lone red run> --blue-dir <lone blue run> --mean-empty <pt> --split <split.json> --set check --side touching --limit 16 --compare-to paper1-first-region-check --grad-maps 8
  python paper1_sample.py contact --run paper1-first-tune-weight --folders paper1-first-tune-weight/w-0.5,paper1-first-tune-weight/w-1 --mean-empty <pt> --split <split.json> --set tuning --side touching
  python paper1_sample.py port-sheet --run paper1-first-sampler-port --archive <folder holding c11/, c22/, c33/> --counts 1,2,3 --seeds 0-7
"""
import argparse
import json
import math
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

# ---------------------------------------------------------------------------------------------------------------------------------------
# Paths. PAPER1_ROOT is the one constant every large output is named under; nothing large goes to /home-mscluster.

PAPER1_ROOT = Path("/datasets/mmolefe/poe_repair_min/paper1")
EXPERT_DIR = Path("/datasets/mmolefe/poe_repair_min/loop-6/m10/full/models")
RED_NET = EXPERT_DIR / "r-full" / "network-snapshot-0002000-0.100.pkl"
BLUE_NET = EXPERT_DIR / "b-full" / "network-snapshot-0002000-0.100.pkl"
# The empty-scene model: the M7 run's final 0.100 snapshot, 248,538,254 bytes, sha256 recorded in environment/clevr-and-edm2.md
EMPTY_NET = Path("/datasets/mmolefe/poe_repair_min/loop-6/m7/full/models/empty/network-snapshot-0002000-0.100.pkl")
EDM2_DIR = Path("/home-mscluster/mmolefe/Playground/PhD/projective-composition/reference/edm2")

# ---------------------------------------------------------------------------------------------------------------------------------------
# The sampler and the pixel rules, as in m10_compose.py.

STEPS, SIGMA_MIN, SIGMA_MAX, RHO = 32, 0.002, 80.0, 7.0
MASK_THRESHOLD = 0.1          # on a 0-to-1 scale, against the mean empty scene, on any channel
MEAN_EMPTY_SEEDS = 1024       # the mean empty scene is the mean over empty-scene seeds 0 to 1,023
COUNTS = (1, 2, 3)            # each expert's count label; the scene sizes are 2, 4 and 6 objects
BATCH = 128
SEED_SETS = ("evaluation", "tuning", "check", "negatives", "heldout-negatives")
PUSH_WINDOW_DEFAULT = "15-22"  # the push acts on sampler steps 15 to 22 (sigma 2.90 to 0.283); "10-22" is the other tuning setting
from paper1_first_bars import FIRST_W_ANCHOR_GREY as W1_TARGET_GREY   # guardrail 4b: w1's in-band change, kept with the bars
W_LADDER = (0.25, 0.5, 1, 2, 4, 8, 16)   # the weights the push smoke steps through to find w1
WEIGHT_GRID_FACTORS = (1, 2, 4, 8)       # the tuning grid is w1 times these
SIGMA_BANDS = ((10.5, 3.8), (3.8, 1.6), (1.6, 0.6), (0.6, 0.28))   # the four bands the push length is reported in
GRAD_MAP_STEPS = (15, 18, 22) # the gradient-map sheet shows the push's gradient at these steps (first evaluations in the window)
COLOUR_MARGIN = 0.1           # contact kept: a foreground pixel is red when R - B > 0.1 and blue when B - R > 0.1, on 0 to 1
IDENTITY_TOL = 1e-5          # the region sampler with all-ones masks must equal the plain sum to within this per pixel, 0-to-1 scale

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sigmas(device=None):
    """The 32 noise levels of EDM2's Heun sampler, then 0."""
    i = torch.arange(STEPS, dtype=torch.float64)
    t = (SIGMA_MAX ** (1 / RHO) + i / (STEPS - 1) * (SIGMA_MIN ** (1 / RHO) - SIGMA_MAX ** (1 / RHO))) ** RHO
    return torch.cat([t, torch.zeros(1, dtype=torch.float64)]).float().to(device or DEV)


def heun(denoise, x, t, after_step=None):
    """EDM2's deterministic Heun sampler. denoise(x, sigma, step, second) returns the denoised picture D(x; sigma) at sampler step
    `step`; second is False for the step's first evaluation (at sigma_step) and True for its correction (at sigma_step+1). The
    arithmetic is m10_compose.py's; the flag only lets the push choose which evaluations it acts on."""
    for i in range(STEPS):
        t_cur, t_next = t[i], t[i + 1]
        d_cur = (x - denoise(x, t_cur, i, False)) / t_cur
        x_next = x + (t_next - t_cur) * d_cur
        if i < STEPS - 1:
            d_eval = denoise(x_next, t_next, i, True)
            x_next = x + (t_next - t_cur) * 0.5 * (d_cur + (x_next - d_eval) / t_next)
        x = x_next
        if after_step is not None and i < STEPS - 1:
            x = after_step(x, t_next)
    return x


def use_edm2(edm2_dir):
    """EDM2's folder on the path, so its pickles (dnnlib, torch_utils) load and StackedRandomGenerator imports."""
    edm2_dir = str(edm2_dir)
    if edm2_dir not in sys.path:
        sys.path.insert(0, edm2_dir)
    if torch.distributed.is_available() and not torch.distributed.is_initialized():
        torch.distributed.init_process_group("gloo", init_method=f"file:///tmp/paper1-sample-{os.getpid()}", rank=0, world_size=1)


def load_net(path):
    """One frozen EDM2 network (its EMA weights) and its encoder."""
    with open(path, "rb") as f:
        data = pickle.load(f)
    net = data["ema"].to(DEV).eval().requires_grad_(False)
    data["encoder"].init(DEV)
    return net, data["encoder"]


def one_hot(k, n):
    """The count label k (1 to 3) as EDM2's one-hot class vector."""
    return F.one_hot(torch.full((n,), k - 1, device=DEV), 3).float()


def starting_noise(seeds):
    """x at sigma 80 for each seed, from EDM2's per-seed generator, so every method starts from the same noise at a seed."""
    from generate_images import StackedRandomGenerator
    rnd = StackedRandomGenerator(DEV, seeds)
    return rnd.randn([len(seeds), 3, 64, 64], device=DEV) * sigmas()[0], rnd


def to_uint8(encoder, x):
    return encoder.decode(x).permute(0, 2, 3, 1).cpu().numpy()


def save(images, folder, seeds):
    folder.mkdir(parents=True, exist_ok=True)
    for seed, image in zip(seeds, images):
        Image.fromarray(image, "RGB").save(folder / f"{seed:06d}.png")


def parse_seeds(text):
    """'20000-20255,10000-10255' to a list of ints."""
    seeds = []
    for part in text.split(","):
        a, _, b = part.strip().partition("-")
        seeds += list(range(int(a), int(b or a) + 1))
    return seeds


def parse_counts(text):
    counts = [int(c) for c in text.split(",")]
    if any(c not in COUNTS for c in counts):
        raise SystemExit(f"counts must be among {COUNTS}")
    return counts


def run_dir(args):
    folder = Path(args.root) / args.run
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def log_json(folder, name, record):
    """Print one record and append it to <run>/<name>.jsonl."""
    print(json.dumps(record), flush=True)
    with open(folder / f"{name}.jsonl", "a") as f:
        f.write(json.dumps(record) + "\n")


def batches(seeds, size):
    for start in range(0, len(seeds), size):
        yield seeds[start:start + size]


# ---------------------------------------------------------------------------------------------------------------------------------------
# Pixel rules on saved pictures, images in [0, 1] as (n, 64, 64, 3)


def load_pictures(folder, seeds):
    return torch.from_numpy(np.stack([np.asarray(Image.open(Path(folder) / f"{s:06d}.png").convert("RGB")) for s in seeds])).float() / 255.0


def foreground(images, empty):
    """Pixels differing from the mean empty scene by more than 0.1 on any channel."""
    return (images - empty).abs().amax(-1) > MASK_THRESHOLD


def paste(red, blue, empty):
    """Mask paste: the blue lone sample's foreground laid over the red lone sample."""
    mask = foreground(blue, empty)
    return torch.where(mask[..., None], blue, red), mask


def components(mask):
    """Connected objects (8-neighbour) of a boolean mask (n, 64, 64): each pixel gets its object's label, 0 off the mask.

    Labels spread by repeated 3 by 3 max pooling inside the mask until nothing changes, so no scipy is needed."""
    n, h, w = mask.shape
    ids = (torch.arange(1, h * w + 1, dtype=torch.float32).view(1, h, w).expand(n, h, w) * mask).clone()
    while True:
        grown = F.max_pool2d(ids[:, None], 3, 1, 1)[:, 0] * mask
        if torch.equal(grown, ids):
            return ids.long()
        ids = grown


def base_row(labels):
    """Each pixel's object base: the lowest row (largest row index) any pixel of its object reaches; -1 off the objects."""
    n, h, w = labels.shape
    rows = torch.arange(h).view(1, h, 1).expand(n, h, w)
    out = torch.full(labels.shape, -1, dtype=torch.long)
    for i in range(n):
        for lab in labels[i].unique():
            if lab == 0:
                continue
            sel = labels[i] == lab
            out[i][sel] = rows[i][sel].max()
    return out


def paste_lower_on_top(red, blue, empty, lower=True):
    """Guardrail 5b: the same lone samples as mask paste, but where a red and a blue object overlap the object whose base sits lower
    in the frame (nearer the camera) is on top; a tie keeps blue on top, as plain paste. Elsewhere the same as paste.
    lower=False puts the higher-based object on top instead (the upper-on-top paste the depth-order preference compares against)."""
    fg_r, fg_b = foreground(red, empty), foreground(blue, empty)
    base_r, base_b = base_row(components(fg_r)), base_row(components(fg_b))
    red_wins = (base_r > base_b) if lower else (base_r < base_b)
    blue_shows = fg_b & ~(fg_r & red_wins)
    return torch.where(blue_shows[..., None], blue, red), blue_shows


def apart(red, blue, empty):
    """True when no blue foreground pixel lies within 1 pixel of a red foreground pixel (3 by 3 neighbourhood)."""
    near_red = F.max_pool2d(foreground(red, empty).float()[:, None], 3, 1, 1)[:, 0] > 0
    return ~(near_red & foreground(blue, empty)).flatten(1).any(1)


def load_mean_empty(path):
    return torch.load(path, map_location="cpu")


# ---------------------------------------------------------------------------------------------------------------------------------------
# Sheets


def sheet(paths, columns, scale=1, labels=None, gap=2):
    """A plain grid of pictures, each scaled up by nearest neighbour, with an optional label under each."""
    tiles = [Image.open(p).convert("RGB") for p in paths]
    w, h = tiles[0].size
    w, h = w * scale, h * scale
    label_h = 12 if labels else 0
    rows = (len(tiles) + columns - 1) // columns
    out = Image.new("RGB", (columns * (w + gap), rows * (h + gap + label_h)), "white")
    draw = ImageDraw.Draw(out)
    for i, tile in enumerate(tiles):
        x, y = (i % columns) * (w + gap), (i // columns) * (h + gap + label_h)
        out.paste(tile.resize((w, h), Image.NEAREST), (x, y))
        if labels:
            draw.text((x + 1, y + h), str(labels[i]), fill="black")
    return out


# ---------------------------------------------------------------------------------------------------------------------------------------
# Subcommands


@torch.no_grad()
def cmd_lone(args):
    use_edm2(args.edm2)
    folder = run_dir(args)
    path = {"red": args.red_net, "blue": args.blue_net, "empty": args.empty_net}[args.expert]
    if not Path(path).is_file():
        raise SystemExit(f"no {args.expert} model at {path}")
    net, encoder = load_net(path)
    t = sigmas()
    seeds = parse_seeds(args.seeds)
    counts = [None] if args.expert == "empty" else parse_counts(args.counts)
    begin = time.time()
    for k in counts:
        out = folder / ("empty" if k is None else f"count-{k}")
        for batch in batches(seeds, args.batch):
            x, _ = starting_noise(batch)
            labels = None if k is None else one_hot(k, len(batch))
            x = heun(lambda z, s, step, second: net(z, s.expand(len(z)), labels), x, t)
            save(to_uint8(encoder, x), out, batch)
    seconds = time.time() - begin
    log_json(folder, "timing", {"stage": "lone", "expert": args.expert, "counts": counts, "seeds": len(seeds),
                                "pictures": len(seeds) * len(counts), "seconds": round(seconds, 1),
                                "seconds_per_picture": round(seconds / (len(seeds) * len(counts)), 4), "device": device_name()})


def cmd_mean_empty(args):
    folder = run_dir(args)
    seeds = list(range(MEAN_EMPTY_SEEDS))
    mean = load_pictures(args.empty_dir, seeds).mean(0)
    torch.save(mean, folder / "mean-empty.pt")
    Image.fromarray((mean * 255).round().clamp(0, 255).byte().numpy(), "RGB").save(folder / "mean-empty.png")
    log_json(folder, "timing", {"stage": "mean-empty", "pictures": len(seeds), "file": str(folder / "mean-empty.pt")})


def plainsum_denoiser(nets, labels_red, labels_blue):
    red, blue, empty = nets

    def denoise(x, s, step, second=False):
        sv = s.expand(len(x))
        return (red(x, sv, labels_red) + blue(x, sv, labels_blue) - empty(x, sv, None)).float()
    return denoise


def load_three(args):
    use_edm2(args.edm2)
    if not Path(args.empty_net).is_file():
        raise SystemExit(f"no empty-scene model at {args.empty_net} (environment/clevr-and-edm2.md records where it is)")
    loaded = [load_net(p) for p in (args.red_net, args.blue_net, args.empty_net)]
    return [n for n, _ in loaded], loaded[0][1]


@torch.no_grad()
def cmd_plainsum(args):
    folder = run_dir(args)
    nets, encoder = load_three(args)
    t = sigmas()
    seeds, counts = parse_seeds(args.seeds), parse_counts(args.counts)
    begin = time.time()
    for k in counts:
        for batch in batches(seeds, args.batch):
            x, _ = starting_noise(batch)
            denoise = plainsum_denoiser(nets, one_hot(k, len(batch)), one_hot(k, len(batch)))
            save(to_uint8(encoder, heun(denoise, x, t)), folder / f"count-{k}", batch)
    seconds = time.time() - begin
    log_json(folder, "timing", {"stage": "plainsum", "counts": counts, "pictures": len(seeds) * len(counts), "seconds": round(seconds, 1),
                                "seconds_per_picture": round(seconds / (len(seeds) * len(counts)), 4), "device": device_name()})


# ---------------------------------------------------------------------------------------------------------------------------------------
# The region sampler (03-full-plan.md, "The region sampler")


def seeds_for(args, k):
    """The seeds for count k: --seeds as given, or from split.json, one seed set's touching or apart seeds in seed order."""
    if getattr(args, "seeds", None):
        return parse_seeds(args.seeds)
    if not getattr(args, "split", None):
        raise SystemExit("give --seeds, or --split with --set and --side")
    entry = json.loads(Path(args.split).read_text())[args.set][f"count-{k}"]
    chosen = sorted(entry[args.side])
    return chosen[:args.limit] if args.limit else chosen


def object_masks(lone_dir, k, seeds, empty):
    """Each lone sample's object mask: pixels more than 0.1 from the mean empty scene, dilated by 1 pixel; (n, 1, 64, 64) on DEV."""
    pictures = load_pictures(Path(lone_dir) / f"count-{k}", seeds)
    mask = foreground(pictures, empty).float()[:, None]
    return (F.max_pool2d(mask, 3, 1, 1) > 0).float().to(DEV)


def lower_owns_overlap(mask_red, mask_blue, base_red, base_blue):
    """Where the two masks overlap, keep only the mask of the object whose base sits lower in the frame (nearer the camera); a tie
    keeps blue, as paste does. base_red and base_blue are each pixel's object base row (base_row, -1 off the objects), dilated like
    the masks. Added 2026-11-27 for the diagnosis of the first result's no: in the shared overlap the region sampler is the plain sum,
    which draws purple; this gives the overlap to one expert."""
    both = (mask_red > 0) & (mask_blue > 0)
    red_wins = base_red > base_blue
    return (torch.where(both & ~red_wins, torch.zeros_like(mask_red), mask_red),
            torch.where(both & red_wins, torch.zeros_like(mask_blue), mask_blue))


def object_bases(lone_dir, k, seeds, empty):
    """Each lone sample's object base rows (base_row of its foreground objects), dilated by 1 pixel like object_masks; on DEV."""
    pictures = load_pictures(Path(lone_dir) / f"count-{k}", seeds)
    base = base_row(components(foreground(pictures, empty))).float()[:, None]
    return F.max_pool2d(base, 3, 1, 1).to(DEV)


def region_denoiser(nets, labels_red, labels_blue, mask_red, mask_blue):
    """D = D_empty + M_red (D_red - D_empty) + M_blue (D_blue - D_empty), written as M_red D_red + M_blue D_blue + (1 - M_red - M_blue)
    D_empty. The two forms are the same sum; the second, with both masks all ones, performs exactly the plain sum's floating-point
    operations ((D_red + D_blue) - D_empty), so the identity check compares like with like."""
    red, blue, empty = nets

    def denoise(x, s, step, second=False):
        sv = s.expand(len(x))
        d_r, d_b, d_e = red(x, sv, labels_red), blue(x, sv, labels_blue), empty(x, sv, None)
        return (mask_red * d_r + mask_blue * d_b + (1 - mask_red - mask_blue) * d_e).float()
    return denoise


@torch.no_grad()
def cmd_region(args):
    """The region sampler at the given seeds, each from the same starting noise as its two lone samples.

    With --identity both masks are all ones, the plain sum is drawn at the same seeds in the same process, and identity.txt records
    the largest per-pixel difference between the two sampler outputs on the 0-to-1 scale, against IDENTITY_TOL.
    """
    if not args.identity and not (args.red_dir and args.blue_dir):
        raise SystemExit("--red-dir and --blue-dir are required unless --identity")
    folder = run_dir(args)
    nets, encoder = load_three(args)
    empty = load_mean_empty(args.mean_empty)
    t = sigmas()
    counts = parse_counts(args.counts)
    worst, pictures, begin, distances = 0.0, 0, time.time(), []
    for k in counts:
        seeds = seeds_for(args, k)
        for batch in batches(seeds, args.batch):
            x0, _ = starting_noise(batch)
            labels = one_hot(k, len(batch))
            if args.identity:
                m_r = m_b = torch.ones(len(batch), 1, 64, 64, device=DEV)
            else:
                m_r = object_masks(args.red_dir, k, batch, empty)
                m_b = object_masks(args.blue_dir, k, batch, empty)
                if getattr(args, "overlap", "both") == "lower":
                    m_r, m_b = lower_owns_overlap(m_r, m_b, object_bases(args.red_dir, k, batch, empty),
                                                  object_bases(args.blue_dir, k, batch, empty))
            x = heun(region_denoiser(nets, labels, labels, m_r, m_b), x0, t)
            images = to_uint8(encoder, x)
            save(images, folder / f"count-{k}", batch)
            if not args.identity:
                distances += band_distance(images, args, k, batch, empty, m_r, m_b)
            if args.identity:
                plain = heun(plainsum_denoiser(nets, labels, labels), x0, t)
                worst = max(worst, float(((x - plain).abs() / 2).max()))   # sampler output is on -1 to 1; halve for the 0-to-1 scale
            pictures += len(batch)
    seconds = time.time() - begin
    log_json(folder, "timing", {"stage": "region", "identity": bool(args.identity), "overlap": getattr(args, "overlap", "both"),
                                "counts": counts, "pictures": pictures,
                                "seconds": round(seconds, 1), "seconds_per_picture": round(seconds / max(1, pictures), 4),
                                "device": device_name()})
    if distances:
        measured = [d["median_distance_grey"] for d in distances if d["median_distance_grey"] is not None]
        result = {"rule": "for each overlap-band pixel (where the dilated red and blue masks meet), the mean absolute difference over "
                          "the three channels, in grey levels, to the nearer of the seed's red and blue object colours, each the median "
                          "colour of that seed's own lone sample's object pixels; per seed the median over its band pixels",
                  "median_over_seeds_grey": round(float(np.median(measured)), 3) if measured else None,
                  "seeds_with_band": len(measured), "seeds": distances}
        (folder / "band-distance.json").write_text(json.dumps(result, indent=1) + "\n")
        print(json.dumps({"band_distance_median_over_seeds_grey": result["median_over_seeds_grey"], "seeds_with_band": len(measured)}),
              flush=True)
    if args.identity:
        passes = worst <= IDENTITY_TOL
        (folder / "identity.txt").write_text(
            f"region sampler with both masks all ones against the plain sum, same seeds, same process\n"
            f"pictures {pictures}\nlargest per-pixel difference (0-to-1 scale) {worst:.3e}\ntolerance {IDENTITY_TOL:.0e}\n"
            f"passes {passes}\n")
        print(json.dumps({"identity_largest_difference": worst, "tolerance": IDENTITY_TOL, "passes": passes}), flush=True)
        if not passes:
            raise SystemExit(1)


def band_distance(images, args, k, seeds, empty, m_r, m_b):
    """Guardrail 4a, print only: how far the region sampler's overlap band sits from pure red or pure blue, per seed."""
    red = load_pictures(Path(args.red_dir) / f"count-{k}", seeds)
    blue = load_pictures(Path(args.blue_dir) / f"count-{k}", seeds)
    band = (overlap_band(m_r, m_b)[:, 0] > 0).cpu()
    pics = torch.from_numpy(images).float()
    out = []
    for i, seed in enumerate(seeds):
        cols = []
        for lone in (red[i], blue[i]):
            fg = foreground(lone[None], empty)[0]
            cols.append((lone[fg].median(0).values * 255) if fg.any() else None)
        pix = pics[i][band[i]]                                                   # (m, 3) band pixels, grey levels
        if pix.numel() == 0 or any(c is None for c in cols):
            out.append({"seed": seed, "count": k, "band_pixels": int(band[i].sum()), "median_distance_grey": None})
            continue
        dist = torch.stack([(pix - c).abs().mean(1) for c in cols]).min(0).values
        out.append({"seed": seed, "count": k, "band_pixels": int(band[i].sum()), "median_distance_grey": round(float(dist.median()), 3),
                    "red_colour": [round(float(v), 1) for v in cols[0]], "blue_colour": [round(float(v), 1) for v in cols[1]]})
    return out


def cmd_pair_sheet(args):
    """Two folders side by side at the same seeds, one row per count (for example the region sampler beside mask paste)."""
    folder = run_dir(args)
    paths, labels = [], []
    for k in parse_counts(args.counts):
        for s in seeds_for(args, k):
            paths += [Path(args.left) / f"count-{k}" / f"{s:06d}.png", Path(args.right) / f"count-{k}" / f"{s:06d}.png"]
            labels += [f"{args.left_label} {k}+{k} s{s}", f"{args.right_label} {k}+{k} s{s}"]
    sheet(paths, args.columns * 2, scale=4, labels=labels).save(folder / args.name)
    print(json.dumps({"sheet": str(folder / args.name), "pairs": len(paths) // 2}), flush=True)


def cmd_mask_sheet(args):
    """Lone samples at 4 times size with each object's dilated mask outlined in yellow, to check shadows fall inside the outline."""
    folder = run_dir(args)
    empty = load_mean_empty(args.mean_empty)
    tiles, labels = [], []
    for k in parse_counts(args.counts):
        seeds = seeds_for(args, k)
        for colour, lone in (("red", args.red_dir), ("blue", args.blue_dir)):
            pictures = load_pictures(Path(lone) / f"count-{k}", seeds)
            mask = F.max_pool2d(foreground(pictures, empty).float()[:, None], 3, 1, 1)[:, 0] > 0
            edge = mask & ~(-F.max_pool2d(-mask.float()[:, None], 3, 1, 1)[:, 0] > 0)   # mask minus its 1-pixel erosion
            for s, pic, e in zip(seeds, pictures, edge):
                big = (pic * 255).round().byte().numpy().repeat(4, 0).repeat(4, 1)
                e4 = e.numpy().repeat(4, 0).repeat(4, 1)
                big[e4] = (255, 255, 0)
                tiles.append(Image.fromarray(big, "RGB"))
                labels.append(f"{colour} {k} s{s}")
    out = Image.new("RGB", (8 * 258, ((len(tiles) + 7) // 8) * 270), "white")
    draw = ImageDraw.Draw(out)
    for i, (tile, label) in enumerate(zip(tiles, labels)):
        x, y = (i % 8) * 258, (i // 8) * 270
        out.paste(tile, (x, y))
        draw.text((x + 1, y + 257), label, fill="black")
    out.save(folder / args.name)
    print(json.dumps({"sheet": str(folder / args.name), "tiles": len(tiles)}), flush=True)


# ---------------------------------------------------------------------------------------------------------------------------------------
# The push (03-full-plan.md, "The push", with the guard note's window): D' = D + w * sigma^2 * grad_x logit on the window's steps


def parse_window(text):
    """'15-22' to range(15, 23): the sampler steps the push acts on."""
    a, _, b = str(text).partition("-")
    first, last = int(a), int(b or a)
    if not 0 <= first <= last < STEPS:
        raise SystemExit(f"window {text} is outside sampler steps 0 to {STEPS - 1}")
    return range(first, last + 1)


def pushed(step, second, window=None):
    """True for the Heun evaluations the push acts on: the first evaluation of every window step and the second (correction)
    evaluation of every window step but the last, so every pushed sigma is one of the window's steps' sigmas. With the default window
    15 to 22 that is sigma 2.90 down to 0.283; with 10 to 22, 10.52 down to 0.283. The last step's correction (sigma 0.189) is not
    pushed. m10_compose.py pushed both evaluations of every step in its window."""
    window = window or parse_window(PUSH_WINDOW_DEFAULT)
    return step in window and (not second or step + 1 in window)


def pushed_evaluations(window=None):
    return [(i, sec) for i in range(STEPS) for sec in (False, True) if pushed(i, sec, window)]


PUSHED_EVALUATIONS = pushed_evaluations()   # default window: 8 first + 7 second = 15


def overlap_band(mask_red, mask_blue):
    """The strip where the two dilated object masks meet: their intersection, about 2 pixels wide at a contact; (n, 1, 64, 64)."""
    return mask_red * mask_blue


def push_denoiser(nets, disc, k, mask_red, mask_blue, w, window=None, generator=None, record=None):
    """The region sampler's D plus the push at the pushed evaluations of the window.

    The head's gradient is taken at the current noisy picture x and its sigma, through both frozen experts, each told its own count k,
    and through the head's bounded logit. Ours adds w * sigma^2 * grad. The random push (guardrail 5a) adds Gaussian noise from the
    seed's own generator multiplied pixel by pixel by the real push's per-pixel size (|push| summed over channels, broadcast to the
    three channels), then scaled per seed to the real push's length, so it disturbs the same places by the same amount. At w = 0 the
    gradient is still computed and 0 is added, so the output is the region sampler's exactly.
    """
    n = len(mask_red)
    labels = one_hot(k, n)
    base = region_denoiser(nets, labels, labels, mask_red, mask_blue)
    count = torch.full((n,), k, device=DEV)
    band = overlap_band(mask_red, mask_blue)

    def denoise(x, s, step, second=False):
        d = base(x, s, step, second)
        if not pushed(step, second, window):
            return d
        with torch.enable_grad():
            xx = x.detach().requires_grad_(True)
            g, = torch.autograd.grad(disc.logit(xx, s.expand(n), count).sum(), xx)
        g = g.float()
        real = w * s ** 2 * g
        length = real.flatten(1).norm(dim=1)
        push = real
        if generator is not None:
            size = real.abs().sum(1, keepdim=True).expand_as(real)
            r = generator.randn_like(x) * size
            push = r * (length / r.flatten(1).norm(dim=1).clamp_min(1e-12))[:, None, None, None]
        out = d + push
        if record is not None:
            applied = out - d
            record(step=step, second=second, sigma=float(s), real=real, push=push, applied=applied, g=g, band=band)
        return out
    return denoise


def band_of(sigma):
    """The band a pushed sigma falls in, by lower edges: step 10's sigma (10.52) counts in the top band, step 22's (0.283) in the last."""
    for hi, lo in SIGMA_BANDS:
        if sigma >= lo - 1e-3:
            return f"{hi}-{lo}"
    return "below the window"


class PushStats:
    """Per sigma band, over seeds and pushed evaluations (guardrails 5a and 6, item 9 of the guard note's adoption):
    the push length; the sign check (cosine between the change the code applied to D and the gradient: +1 for ours); the share of
    gradient entries that are zero or not finite (the gradient is float32); the band ratio (share of the push's squared length inside
    the overlap band, over the band's share of the picture); for the random push, the largest relative length mismatch against the
    real push and both pushes' in-band shares."""

    def __init__(self):
        self.rows = {}

    def __call__(self, step, second, sigma, real, push, applied, g, band, grads=None):
        row = self.rows.setdefault(band_of(sigma), {k: [] for k in ("length", "sign", "bad_share", "band_ratio", "in_band_real",
                                                                    "in_band_push", "mismatch")})
        flat = lambda v: v.flatten(1)
        band3 = band.expand_as(push)
        area = flat(band3).mean(1)
        sq_real, sq_push = flat(real) ** 2, flat(push) ** 2
        in_real = (sq_real * flat(band3)).sum(1) / sq_real.sum(1).clamp_min(1e-30)
        in_push = (sq_push * flat(band3)).sum(1) / sq_push.sum(1).clamp_min(1e-30)
        has_band = area > 0
        row["length"] += flat(push).norm(dim=1).tolist()
        row["sign"] += torch.nn.functional.cosine_similarity(flat(applied), flat(g), dim=1).tolist()
        row["bad_share"] += ((flat(g) == 0) | ~torch.isfinite(flat(g))).float().mean(1).tolist()
        row["band_ratio"] += (in_push[has_band] / area[has_band]).tolist()
        row["in_band_real"] += in_real[has_band].tolist()
        row["in_band_push"] += in_push[has_band].tolist()
        lr, lp = flat(real).norm(dim=1), flat(push).norm(dim=1)
        row["mismatch"] += ((lp - lr).abs() / lr.clamp_min(1e-30)).tolist()

    def summary(self):
        med = lambda v: round(float(np.median(v)), 6) if v else None
        out = {}
        for b, r in sorted(self.rows.items()):
            out[b] = {"push_length_median": med(r["length"]), "push_length_largest": round(float(np.max(r["length"])), 6),
                      "sign_check_cosine_median": med(r["sign"]), "sign_check_cosine_smallest": round(float(np.min(r["sign"])), 6),
                      "gradient_zero_or_not_finite_share": med(r["bad_share"]), "band_ratio_median": med(r["band_ratio"]),
                      "in_band_share_real_median": med(r["in_band_real"]), "in_band_share_applied_median": med(r["in_band_push"]),
                      "largest_length_mismatch": round(float(np.max(r["mismatch"])), 8), "evaluations": len(r["length"])}
        return out


@torch.no_grad()
def draw_pushed(nets, disc, encoder, args, k, batch, w, window, empty, stats=None, grads=None):
    x0, rnd = starting_noise(batch)
    m_r = object_masks(args.red_dir, k, batch, empty)
    m_b = object_masks(args.blue_dir, k, batch, empty)

    def record(**kw):
        if stats is not None:
            stats(**kw)
        if grads is not None and not kw["second"] and kw["step"] in GRAD_MAP_STEPS:
            grads[kw["step"]] = kw["g"][:args.grad_maps].abs().sum(1).cpu()
    denoise = push_denoiser(nets, disc, k, m_r, m_b, w, window=window, generator=rnd if args.random_push else None, record=record)
    return to_uint8(encoder, heun(denoise, x0, sigmas()))


@torch.no_grad()
def cmd_push(args):
    """Ours (the region sampler plus the head's push) or the random push, at one or more weights w, on the window's steps.

    One weight writes <run>/count-<k>/; several write <run>/w-<w>/count-<k>/. push.json holds the window and, per weight, PushStats
    per sigma band and, with --compare-to, the grey-level change from the region sampler (whole picture, inside and outside the
    overlap band). --grad-maps N draws the gradient maps of the first N seeds of the first count; --repeat-check N redraws the first N
    seeds of the first count at the first weight in the same job and records the largest difference (guardrail 9).
    """
    import paper1_disc as pdisc
    folder = run_dir(args)
    nets, encoder = load_three(args)
    disc = pdisc.load_discriminator(args.head, nets[0], nets[1])
    empty = load_mean_empty(args.mean_empty)
    window = parse_window(args.window)
    weights = load_weights(args)
    counts = parse_counts(args.counts)
    summary = {"random_push": bool(args.random_push), "head": args.head, "window": [window[0], window[-1]],
               "window_sigmas": [round(float(sigmas("cpu")[window[0]]), 3), round(float(sigmas("cpu")[window[-1]]), 3)],
               "pushed_evaluations": len(pushed_evaluations(window)), "weights": {}}
    begin, pictures, maps = time.time(), 0, []
    for w in weights:
        out = folder if len(weights) == 1 else folder / f"w-{w:g}"
        stats = PushStats()
        for k in counts:
            for batch in batches(seeds_for(args, k), args.batch):
                grads = {} if (args.grad_maps and k == counts[0] and w == weights[0] and not maps) else None
                images = draw_pushed(nets, disc, encoder, args, k, batch, w, window, empty, stats=stats, grads=grads)
                save(images, out / f"count-{k}", batch)
                pictures += len(batch)
                if grads is not None:
                    maps = [(batch[i], images[i], [grads[s][i] for s in GRAD_MAP_STEPS if s in grads])
                            for i in range(min(args.grad_maps, len(batch)))]
        entry = {"by_band": stats.summary()}
        if args.compare_to:
            region = Path(args.compare_to) if Path(args.compare_to).is_absolute() else Path(args.root) / args.compare_to
            entry["grey_change"] = grey_change(out, region, counts, args, empty)
        summary["weights"][f"{w:g}"] = entry
        print(json.dumps({"w": w, **entry}), flush=True)
    if args.repeat_check:
        k, w = counts[0], weights[0]
        seeds = seeds_for(args, k)[:args.repeat_check]
        again = draw_pushed(nets, disc, encoder, args, k, seeds, w, window, empty)
        first = load_pictures((folder if len(weights) == 1 else folder / f"w-{w:g}") / f"count-{k}", seeds) * 255
        largest = float((torch.from_numpy(again).float() - first).abs().max())
        summary["repeat_check"] = {"seeds": seeds, "count": k, "w": w, "largest_difference_grey": round(largest, 3)}
        print(json.dumps({"repeat_check_largest_difference_grey": largest}), flush=True)
    seconds = time.time() - begin
    summary.update({"pictures": pictures, "seconds": round(seconds, 1), "seconds_per_picture": round(seconds / max(1, pictures), 4),
                    "device": device_name()})
    (folder / "push.json").write_text(json.dumps(summary, indent=1) + "\n")
    if maps:
        grad_map_sheet(maps, folder / "grad-maps.png")


def load_weights(args):
    """--weights '0,1'; --weights ladder for 0 and W_LADDER (the push smoke); or --weights-file weight-grid.json (its 'grid')."""
    if args.weights_file:
        return [float(v) for v in json.loads(Path(args.weights_file).read_text())["grid"]]
    if args.weights == "ladder":
        return [0.0] + [float(v) for v in W_LADDER]
    return [float(v) for v in args.weights.split(",")]


def grey_change(folder, region_folder, counts, args, empty):
    """Grey-level change (0 to 255) of each picture from the region sampler's at the same seed and count.

    median_change_grey: the median over pictures of each picture's mean absolute change over all its pixels and channels, the number
    FIRST_PUSH_MIN_GREY is read against. median_largest_change_grey: the median over pictures of each picture's largest change.
    largest_change_grey: the largest change anywhere. median_change_in_band_grey and median_change_off_band_grey: the same mean change
    over the overlap band's pixels and over the rest, for pictures with a non-empty band (guardrail 4b).
    """
    means, peaks, inside, outside = [], [], [], []
    for k in counts:
        for batch in batches(seeds_for(args, k), 512):
            a = load_pictures(folder / f"count-{k}", batch) * 255
            b = load_pictures(region_folder / f"count-{k}", batch) * 255
            diff = (a - b).abs().mean(-1)                                       # (n, 64, 64), mean over channels
            band = overlap_band(object_masks(args.red_dir, k, batch, empty), object_masks(args.blue_dir, k, batch, empty))[:, 0].cpu() > 0
            means += diff.flatten(1).mean(1).tolist()
            peaks += (a - b).abs().flatten(1).amax(1).tolist()
            for d, m in zip(diff, band):
                if m.any():
                    inside.append(float(d[m].mean()))
                    outside.append(float(d[~m].mean()))
    med = lambda v: round(float(np.median(v)), 4) if v else None
    return {"median_change_grey": med(means), "median_largest_change_grey": med(peaks), "largest_change_grey": round(float(np.max(peaks)), 4),
            "median_change_in_band_grey": med(inside), "median_change_off_band_grey": med(outside), "pictures_with_band": len(inside),
            "compared_pictures": len(means), "compared_to": str(region_folder)}


def cmd_weight_grid(args):
    """Guardrail 4b: the tuning weights, fixed from the smoke run before tuning runs.

    w1 is the weight of the smoke run's ladder whose median in-band change (median_change_in_band_grey) is nearest W1_TARGET_GREY on a
    log scale; the grid is w1 times WEIGHT_GRID_FACTORS. Weights with no in-band change measured are skipped.
    """
    folder = run_dir(args)
    smoke = json.loads(Path(args.smoke).read_text())
    ladder = []
    for w, entry in smoke["weights"].items():
        change = (entry.get("grey_change") or {}).get("median_change_in_band_grey")
        if float(w) > 0 and change and change > 0:
            ladder.append((float(w), change))
    if not ladder:
        raise SystemExit("the smoke run has no weight above 0 with an in-band change; run push with --compare-to and the ladder")
    w1, change = min(ladder, key=lambda wc: abs(math.log(wc[1] / W1_TARGET_GREY)))
    grid = [w1 * f for f in WEIGHT_GRID_FACTORS]
    out = {"rule": f"w1 = the ladder weight whose median in-band change is nearest {W1_TARGET_GREY} grey levels on a log scale; "
                   f"grid = w1 x {list(WEIGHT_GRID_FACTORS)}", "ladder": [{"w": w, "median_change_in_band_grey": c} for w, c in ladder],
           "w1": w1, "w1_in_band_change_grey": change, "grid": grid, "smoke": args.smoke}
    (folder / "weight-grid.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({"w1": w1, "grid": grid}), flush=True)


def grad_map_sheet(maps, path):
    """One row per seed: the picture, then |grad| summed over channels at sampler steps GRAD_MAP_STEPS inside the window, each map
    scaled to its own largest value, all at 4 times size."""
    rows = []
    for seed, image, grads in maps:
        tiles = [Image.fromarray(image, "RGB").resize((256, 256), Image.NEAREST)]
        for g in grads:
            g = (g / g.max().clamp_min(1e-12) * 255).round().byte().numpy()
            tiles.append(Image.fromarray(g, "L").convert("RGB").resize((256, 256), Image.NEAREST))
        rows.append((seed, tiles))
    shown = len(rows[0][1]) - 1
    out = Image.new("RGB", ((shown + 1) * 258, len(rows) * 270 + 14), "white")
    draw = ImageDraw.Draw(out)
    steps = [s for s in GRAD_MAP_STEPS][:shown]
    for j, label in enumerate(["picture"] + [f"|grad| step {s} (sigma {float(sigmas('cpu')[s]):.2f})" for s in steps]):
        draw.text((j * 258 + 2, 0), label, fill="black")
    for i, (seed, tiles) in enumerate(rows):
        for j, tile in enumerate(tiles):
            out.paste(tile, (j * 258, 14 + i * 270))
        draw.text((2, 14 + i * 270 + 257), f"seed {seed}", fill="black")
    out.save(path)


# ---------------------------------------------------------------------------------------------------------------------------------------
# Contact kept (03-full-plan.md, "Words this uses")


def colour_masks(images, empty):
    """Red and blue object pixels of a composed picture: foreground by the 0.1 rule, then red where the red channel exceeds the blue
    by more than COLOUR_MARGIN and blue the other way round. A purple blend whose two channels are within the margin is neither."""
    fg = foreground(images, empty)
    r, b = images[..., 0], images[..., 2]
    return fg & (r - b > COLOUR_MARGIN), fg & (b - r > COLOUR_MARGIN)


def contact(images, empty):
    """True where some red pixel lies within 1 pixel (3 by 3) of some blue pixel."""
    red, blue = colour_masks(images, empty)
    near_red = F.max_pool2d(red.float()[:, None], 3, 1, 1)[:, 0] > 0
    return (near_red & blue).flatten(1).any(1)


def cmd_contact(args):
    """contact kept per count and pooled over the touching seeds of one or more picture folders; written to contact.json."""
    folder = run_dir(args)
    empty = load_mean_empty(args.mean_empty)
    path = folder / "contact.json"
    result = json.loads(path.read_text()) if path.exists() else {}
    for pics in args.folders.split(","):
        src = Path(pics) if Path(pics).is_absolute() else Path(args.root) / pics
        per, kept_all, n_all = {}, 0, 0
        for k in parse_counts(args.counts):
            seeds = seeds_for(args, k)
            kept = sum(int(contact(load_pictures(src / f"count-{k}", b), empty).sum()) for b in batches(seeds, 512))
            per[f"count-{k}"] = {"seeds": len(seeds), "contact_kept": kept, "share": round(kept / max(1, len(seeds)), 4)}
            kept_all, n_all = kept_all + kept, n_all + len(seeds)
        result[pics] = {"per_count": per, "pooled_share": round(kept_all / max(1, n_all), 4), "seeds": n_all,
                        "rule": f"foreground > {MASK_THRESHOLD} from the mean empty scene; red where R - B > {COLOUR_MARGIN}, blue where "
                                f"B - R > {COLOUR_MARGIN}; some red pixel within 1 pixel of some blue pixel"}
        print(json.dumps({pics: result[pics]["pooled_share"], **{k: v["share"] for k, v in per.items()}}), flush=True)
    path.write_text(json.dumps(result, indent=1) + "\n")


def cmd_paste(args):
    folder = run_dir(args)
    empty = load_mean_empty(args.mean_empty)
    seeds, counts = parse_seeds(args.seeds), parse_counts(args.counts)
    for k in counts:
        for batch in batches(seeds, 512):
            red = load_pictures(Path(args.red_dir) / f"count-{k}", batch)
            blue = load_pictures(Path(args.blue_dir) / f"count-{k}", batch)
            pasted, _ = (paste_lower_on_top if args.order == "lower-on-top" else paste)(red, blue, empty)
            save((pasted * 255).round().clamp(0, 255).byte().numpy(), folder / f"count-{k}", batch)
    log_json(folder, "timing", {"stage": "paste", "order": args.order, "counts": counts, "pictures": len(seeds) * len(counts)})


def cmd_split(args):
    """split.json: {<set>: {"count-k": {"touching": [seeds], "apart": [seeds], "touching_share": x}}}.

    Each seed set (evaluation, tuning, check, negatives, heldout-negatives) keeps its own lists, so "the first 32 touching evaluation
    seeds in seed order" can never pick a tuning or check seed (consensus rule 12).
    """
    folder = run_dir(args)
    empty = load_mean_empty(args.mean_empty)
    seeds, counts = parse_seeds(args.seeds), parse_counts(args.counts)
    path = folder / "split.json"
    whole = json.loads(path.read_text()) if path.exists() else {}
    result = whole.setdefault(args.set, {})
    for k in counts:
        touching, separate, sizes = [], [], []
        for batch in batches(seeds, 512):
            red = load_pictures(Path(args.red_dir) / f"count-{k}", batch)
            blue = load_pictures(Path(args.blue_dir) / f"count-{k}", batch)
            is_apart = apart(red, blue, empty)
            for fg in (foreground(red, empty), foreground(blue, empty)):         # object pixel sizes, for composition beside the pack's
                labels = components(fg)
                for i in range(len(labels)):
                    sizes += [int((labels[i] == lab).sum()) for lab in labels[i].unique().tolist() if lab]
            for seed, a in zip(batch, is_apart.tolist()):
                (separate if a else touching).append(seed)
        entry = result.setdefault(f"count-{k}", {"touching": [], "apart": []})
        entry["touching"] = sorted(set(entry["touching"]) | set(touching))
        entry["apart"] = sorted(set(entry["apart"]) | set(separate))
        entry["touching_share"] = round(len(entry["touching"]) / max(1, len(entry["touching"]) + len(entry["apart"])), 4)
        if sizes:
            entry["object_pixels"] = {"median": int(np.median(sizes)), "p10": int(np.percentile(sizes, 10)),
                                      "p90": int(np.percentile(sizes, 90)), "objects": len(sizes)}
        print(json.dumps({"set": args.set, "count": k, "objects": 2 * k, "seeds": len(seeds), "touching": len(touching),
                          "apart": len(separate), "touching_share_these_seeds": round(len(touching) / max(1, len(seeds)), 4)}), flush=True)
    path.write_text(json.dumps(whole, indent=1) + "\n")


def cmd_port_sheet(args):
    """One row per count: the archived picture and the ported plain sum side by side at each seed."""
    folder = run_dir(args)
    seeds, counts = parse_seeds(args.seeds), parse_counts(args.counts)
    paths, labels = [], []
    for k in counts:
        for s in seeds:
            paths += [Path(args.archive) / f"c{k}{k}" / f"{s:06d}.png", folder / f"count-{k}" / f"{s:06d}.png"]
            labels += [f"old {k}+{k} s{s}", f"new {k}+{k} s{s}"]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise SystemExit(f"{len(missing)} pictures missing, first {missing[0]}")
    sheet(paths, 2 * len(seeds), scale=4, labels=labels).save(folder / "port-sheet.png")
    print(json.dumps({"sheet": str(folder / "port-sheet.png"), "pairs": len(paths) // 2}), flush=True)


def device_name():
    return torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------------------------------------------------------------------


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="stage", required=True)

    def common(p, nets=False):
        p.add_argument("--run", required=True, help="the run name; outputs go to <root>/<run>/")
        p.add_argument("--root", default=str(PAPER1_ROOT), help="where runs write (default %(default)s)")
        if nets:
            p.add_argument("--red-net", default=str(RED_NET))
            p.add_argument("--blue-net", default=str(BLUE_NET))
            p.add_argument("--empty-net", default=str(EMPTY_NET), help="the empty-scene model's snapshot (default %(default)s)")
            p.add_argument("--edm2", default=str(EDM2_DIR), help="EDM2 checkout, for its pickles and seeding")
            p.add_argument("--batch", type=int, default=BATCH)
        return p

    p = common(sub.add_parser("lone"), nets=True)
    p.add_argument("--expert", choices=["red", "blue", "empty"], required=True)
    p.add_argument("--counts", default="1,2,3")
    p.add_argument("--seeds", required=True)

    p = common(sub.add_parser("mean-empty"))
    p.add_argument("--empty-dir", required=True, help="folder of empty-scene pictures at seeds 0 to 1,023")

    p = common(sub.add_parser("plainsum"), nets=True)
    p.add_argument("--counts", default="1,2,3")
    p.add_argument("--seeds", required=True)

    for name in ("paste", "split"):
        p = common(sub.add_parser(name))
        p.add_argument("--red-dir", required=True, help="the red lone run's folder, holding count-1/ to count-3/")
        p.add_argument("--blue-dir", required=True)
        p.add_argument("--mean-empty", required=True, help="mean-empty.pt")
        p.add_argument("--counts", default="1,2,3")
        p.add_argument("--seeds", required=True)
        if name == "split":
            p.add_argument("--set", required=True, choices=SEED_SETS, help="which seed set these seeds are; each keeps its own lists")
        else:
            p.add_argument("--order", choices=["blue-on-top", "lower-on-top"], default="blue-on-top",
                           help="blue-on-top: mask paste; lower-on-top: the object whose base is lower in the frame on top (guardrail 5b)")

    def seed_choice(p):
        p.add_argument("--seeds", help="seed ranges such as 30000-30127; or choose them from a split file below")
        p.add_argument("--split", help="split.json from the split subcommand")
        p.add_argument("--set", choices=SEED_SETS, help="the seed set to read from the split file")
        p.add_argument("--side", choices=["touching", "apart"], default="touching")
        p.add_argument("--limit", type=int, help="the first N seeds in seed order, per count")

    p = common(sub.add_parser("region"), nets=True)
    p.add_argument("--red-dir", help="the red lone run's folder (count-1/ to count-3/), same seeds and counts")
    p.add_argument("--blue-dir")
    p.add_argument("--mean-empty", required=True)
    p.add_argument("--counts", default="1,2,3")
    p.add_argument("--identity", action="store_true", help="both masks all ones; also draws the plain sum and writes identity.txt")
    p.add_argument("--overlap", choices=("both", "lower"), default="both",
                   help="both: where the masks overlap both experts act (the plan's sampler); lower: the overlap goes to the object whose "
                        "base sits lower in the frame")
    seed_choice(p)

    p = common(sub.add_parser("pair-sheet"))
    p.add_argument("--left", required=True); p.add_argument("--right", required=True)
    p.add_argument("--left-label", default="left"); p.add_argument("--right-label", default="right")
    p.add_argument("--columns", type=int, default=8, help="pairs per sheet row")
    p.add_argument("--name", default="pair-sheet.png")
    p.add_argument("--counts", default="1,2,3")
    seed_choice(p)

    p = common(sub.add_parser("mask-sheet"))
    p.add_argument("--red-dir", required=True); p.add_argument("--blue-dir", required=True)
    p.add_argument("--mean-empty", required=True)
    p.add_argument("--name", default="mask-sheet.png")
    p.add_argument("--counts", default="1,2,3")
    seed_choice(p)

    p = common(sub.add_parser("push"), nets=True)
    p.add_argument("--head", required=True, help="head.pt from paper1_disc.py train")
    p.add_argument("--weights", default="1", help="one or more w, such as 1,2; or 'ladder' for 0 and W_LADDER (the push smoke)")
    p.add_argument("--random-push", action="store_true", help="a random direction of the head's push length at each pushed evaluation")
    p.add_argument("--red-dir", required=True); p.add_argument("--blue-dir", required=True)
    p.add_argument("--mean-empty", required=True)
    p.add_argument("--counts", default="1,2,3")
    p.add_argument("--compare-to", help="the region sampler's run at the same seeds, for the grey-level change")
    p.add_argument("--grad-maps", type=int, default=0, help="draw the gradient maps of this many seeds (8 in the first result)")
    p.add_argument("--window", default=PUSH_WINDOW_DEFAULT, help="sampler steps the push acts on (default %(default)s; the other is 10-22)")
    p.add_argument("--weights-file", help="weight-grid.json from the weight-grid subcommand; its grid replaces --weights")
    p.add_argument("--repeat-check", type=int, default=0, help="redraw this many seeds in the same job and record the largest difference")
    seed_choice(p)

    p = common(sub.add_parser("weight-grid"))
    p.add_argument("--smoke", required=True, help="push.json of the push smoke run, drawn at 0 and the ladder W_LADDER with --compare-to")

    p = common(sub.add_parser("contact"))
    p.add_argument("--folders", required=True, help="comma-separated picture runs (or w-<w> sub-folders), each holding count-<k>/")
    p.add_argument("--mean-empty", required=True)
    p.add_argument("--counts", default="1,2,3")
    seed_choice(p)

    p = common(sub.add_parser("port-sheet"))
    p.add_argument("--archive", required=True, help="the archived plain-sum pictures, a folder holding c11/, c22/ and c33/")
    p.add_argument("--counts", default="1,2,3")
    p.add_argument("--seeds", default="0-7")
    return ap


COMMANDS = {"lone": cmd_lone, "mean-empty": cmd_mean_empty, "plainsum": cmd_plainsum, "paste": cmd_paste, "split": cmd_split,
            "port-sheet": cmd_port_sheet, "region": cmd_region, "pair-sheet": cmd_pair_sheet, "mask-sheet": cmd_mask_sheet,
            "push": cmd_push, "contact": cmd_contact, "weight-grid": cmd_weight_grid}


def main(argv=None):
    args = build_parser().parse_args(argv)
    COMMANDS[args.stage](args)


if __name__ == "__main__":
    main()
