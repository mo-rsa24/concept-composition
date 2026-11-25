#!/usr/bin/env python3
"""Step M10's comparison on one RunPod RTX 4090: two colour experts composed by their product, with a count checker's push or a corrector.

The settings are in artifacts/notes/progress/method-search/05-two-experts-by-colour/design.md, "Running the comparison". Run from the EDM2
folder's parent with EDM2 on the path; every path below is under /workspace/m10.

  sources        each expert's samples at counts 1 to 3 and the empty-scene model's, for given seeds (paste sources and the mask paste)
  train-checker  the count checker on mask pastes of the paste sources (seeds 100,000 to 100,511), against a held-out balanced accuracy of 0.75
  row            one row (plain, checker, zero, corrector) over all 9 cells for one block of 256 seeds
  paste          the mask paste row for one block, on the CPU, from the sources at seeds 0 to 1,023
  report         scores one block of every finished row with the count scorer and draws its grid

Usage: python m10_compose.py sources --colour r|b|empty --seeds 0-1023
       python m10_compose.py train-checker
       python m10_compose.py row --row plain|checker|zero|corrector --block 0..3
       python m10_compose.py paste --block 0..3
       python m10_compose.py report --block 0..3
"""
import argparse
import glob
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

W = Path("/workspace/m10")
sys.path.insert(0, str(W / "edm2"))
sys.path.insert(0, str(W / "scripts"))
NETS = {"r": W / "runs/full-quarters/r-full/network-snapshot-0002000-0.100.pkl",
        "b": W / "models/b-full/network-snapshot-0002000-0.100.pkl",
        "empty": W / "models/empty/network-snapshot-0002000-0.100.pkl"}
CELLS = [(x, y) for x in (1, 2, 3) for y in (1, 2, 3)]
BLOCK = 256
STEPS, SIGMA_MIN, SIGMA_MAX, RHO = 32, 0.002, 80.0, 7.0
PUSH_STEPS = range(10, 15)          # sigma 10.5 to 3.8, the digit test's window by noise level (13.1 to 3.1)
PUSH_B = 0.4                        # the digit test's frozen push length, not tuned here
CORRECTOR_STEPS, CORRECTOR_EPS = 2, 0.05
MASK_THRESHOLD = 0.1                # on a 0-to-1 scale, against the mean empty scene
SOURCE_TRAIN, SOURCE_HELD = range(100_000, 100_410), range(100_410, 100_512)
CHECKER_UPDATES, CHECKER_BATCH, CHECKER_LR = 3000, 64, 1e-3
MIN_CHECKER_BA, FUTILE_CHECKER_BA = 0.75, 0.55
DEV = torch.device("cuda")


def sigmas():
    i = torch.arange(STEPS, dtype=torch.float64)
    t = (SIGMA_MAX ** (1 / RHO) + i / (STEPS - 1) * (SIGMA_MIN ** (1 / RHO) - SIGMA_MAX ** (1 / RHO))) ** RHO
    return torch.cat([t, torch.zeros(1, dtype=torch.float64)]).float().to(DEV)


def load_net(name):
    with open(NETS[name], "rb") as f:
        data = pickle.load(f)
    net = data["ema"].to(DEV).eval().requires_grad_(False)
    data["encoder"].init(DEV)
    return net, data["encoder"]


def one_hot(k, n):
    return F.one_hot(torch.full((n,), k - 1, device=DEV), 3).float()


def to_uint8(encoder, x):
    return encoder.decode(x).permute(0, 2, 3, 1).cpu().numpy()


def save(images, folder, seeds):
    folder.mkdir(parents=True, exist_ok=True)
    for seed, image in zip(seeds, images):
        Image.fromarray(image, "RGB").save(folder / f"{seed:06d}.png")


def parse_seeds(text):
    a, _, b = text.partition("-")
    return list(range(int(a), int(b or a) + 1))


# ---------------------------------------------------------------------------------------------------------------------------------------
# The count checker


class Checker(nn.Module):
    """The scorer's four convolution blocks over the two maps and log sigma, then the asked counts, then one logit."""

    def __init__(self, channels=7):
        super().__init__()
        layers = []
        for width in (32, 64, 128, 128):
            layers += [nn.Conv2d(channels, width, 3, padding=1), nn.GroupNorm(8, width), nn.SiLU(), nn.MaxPool2d(2)]
            channels = width
        self.features = nn.Sequential(*layers, nn.Flatten())
        self.head = nn.Sequential(nn.Linear(128 * 4 * 4 + 6, 128), nn.SiLU(), nn.Linear(128, 1))

    def forward(self, maps, sigma, asked):
        log_sigma = sigma.log().view(-1, 1, 1, 1).expand(-1, 1, *maps.shape[2:])
        return self.head(torch.cat([self.features(torch.cat([maps, log_sigma], 1)), asked], 1))[:, 0]


def maps_of(nets, x, sigma, x_count, y_count):
    r, b, e = nets
    s = sigma.view(-1)
    d_e = e(x, s, None)
    return torch.cat([r(x, s, x_count) - d_e, b(x, s, y_count) - d_e], 1), d_e


def asked_vector(x_count, y_count):
    return torch.cat([x_count, y_count], 1)


# ---------------------------------------------------------------------------------------------------------------------------------------
# Sources and pastes


@torch.no_grad()
def sources(args):
    from generate_images import StackedRandomGenerator
    net, encoder = load_net(args.colour)
    seeds = parse_seeds(args.seeds)
    t = sigmas()
    counts = [None] if args.colour == "empty" else [1, 2, 3]
    for k in counts:
        out = W / "sources" / (args.colour if k is None else f"{args.colour}-count-{k}")
        for start in range(0, len(seeds), 128):
            batch = seeds[start:start + 128]
            rnd = StackedRandomGenerator(DEV, batch)
            x = rnd.randn([len(batch), 3, 64, 64], device=DEV) * t[0]
            labels = None if k is None else one_hot(k, len(batch))
            x = heun(lambda z, s: net(z, s.expand(len(z)), labels), x, t)
            save(to_uint8(encoder, x), out, batch)
        print(json.dumps({"stage": "sources", "folder": str(out), "seeds": [seeds[0], seeds[-1]]}), flush=True)


def load_source(folder, seeds):
    return torch.from_numpy(np.stack([np.asarray(Image.open(folder / f"{s:06d}.png").convert("RGB")) for s in seeds])).float() / 255.0


def mean_empty():
    path = W / "sources/mean-empty.pt"
    if not path.exists():
        torch.save(load_source(W / "sources/empty", range(1024)).mean(0), path)
    return torch.load(path)


def foreground(images, empty):
    """Pixels differing from the mean empty scene by more than 0.1 on any channel, images in [0, 1] as (n, 64, 64, 3)."""
    return ((images - empty).abs().amax(-1) > MASK_THRESHOLD)


def paste(red, blue, empty):
    mask = foreground(blue, empty)
    return torch.where(mask[..., None], blue, red), mask


def apart(red, blue, empty):
    """True when no blue foreground pixel lies within 1 pixel of a red foreground pixel."""
    near_red = F.max_pool2d(foreground(red, empty).float()[:, None], 3, 1, 1)[:, 0] > 0
    return ~(near_red & foreground(blue, empty)).flatten(1).any(1)


# ---------------------------------------------------------------------------------------------------------------------------------------
# Training the checker


def checker_examples(pool, n, gen, empty):
    """n pastes: half positives (apart, asked their sources' counts), a quarter touching, a quarter apart but asked one count off by 1."""
    kinds = torch.tensor([0] * (n // 2) + [1] * (n // 4) + [2] * (n - n // 2 - n // 4))
    xs, asked, labels = [], [], []
    for kind in kinds.tolist():
        for _ in range(200):
            x = int(torch.randint(1, 4, (1,), generator=gen)); y = int(torch.randint(1, 4, (1,), generator=gen))
            r = pool[("r", x)][int(torch.randint(len(pool[("r", x)]), (1,), generator=gen))][None]
            b = pool[("b", y)][int(torch.randint(len(pool[("b", y)]), (1,), generator=gen))][None]
            sep = bool(apart(r, b, empty)[0])
            if (kind in (0, 2) and sep) or (kind == 1 and not sep):
                break
        if kind == 2:
            if int(torch.randint(2, (1,), generator=gen)):
                x = x + 1 if x == 1 else x - 1 if x == 3 else x + (1 if int(torch.randint(2, (1,), generator=gen)) else -1)
            else:
                y = y + 1 if y == 1 else y - 1 if y == 3 else y + (1 if int(torch.randint(2, (1,), generator=gen)) else -1)
        xs.append(paste(r, b, empty)[0][0]); asked.append((x, y)); labels.append(1.0 if kind == 0 else 0.0)
    x0 = torch.stack(xs).permute(0, 3, 1, 2) * 2 - 1
    return x0, torch.tensor(asked), torch.tensor(labels), kinds


SIGMA_WINDOW = None   # (lo, hi) for the diagnosis; None is the push window
READER = "maps"       # "maps": the two experts' maps (6 channels); "raw": the noisy picture itself (3 channels), question 116's second checker


def batch_inputs(nets, x0, asked, gen):
    lo, hi = SIGMA_WINDOW or (float(sigmas()[PUSH_STEPS[-1]]), float(sigmas()[PUSH_STEPS[0]]))
    sigma = torch.exp(torch.empty(len(x0)).uniform_(np.log(lo), np.log(hi), generator=gen)).to(DEV)
    x = x0.to(DEV) + sigma.view(-1, 1, 1, 1) * torch.randn(x0.shape, generator=gen).to(DEV)
    xc = F.one_hot(asked[:, 0].to(DEV) - 1, 3).float(); yc = F.one_hot(asked[:, 1].to(DEV) - 1, 3).float()
    if READER == "raw":
        return x, sigma, asked_vector(xc, yc)
    with torch.no_grad():
        maps, _ = maps_of(nets, x, sigma, xc, yc)
    return maps, sigma, asked_vector(xc, yc)


def balanced_accuracy(logits, labels):
    called = logits > 0
    pos, neg = labels > 0.5, labels < 0.5
    return 0.5 * (float((called & pos).sum()) / max(1, int(pos.sum())) + float((~called & neg).sum()) / max(1, int(neg.sum())))


def train_checker(args):
    begin = time.time()
    empty = mean_empty()
    pool_train = {(c, k): load_source(W / "sources" / f"{c}-count-{k}", SOURCE_TRAIN) for c in "rb" for k in (1, 2, 3)}
    pool_held = {(c, k): load_source(W / "sources" / f"{c}-count-{k}", SOURCE_HELD) for c in "rb" for k in (1, 2, 3)}
    nets = [load_net(n)[0] for n in ("r", "b", "empty")]
    held_gen = torch.Generator().manual_seed(7)
    hx, ha, hl, hk = checker_examples(pool_held, 1024, held_gen, empty)
    held = [batch_inputs(nets, hx[i:i + 128], ha[i:i + 128], held_gen) for i in range(0, 1024, 128)]
    model = Checker(4 if READER == "raw" else 7).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=CHECKER_LR, weight_decay=1e-4)
    gen = torch.Generator().manual_seed(20261125)
    out = W / (args.out_name or "checker"); out.mkdir(parents=True, exist_ok=True)
    log = open(out / "train.jsonl", "w")

    def evaluate():
        model.eval()
        with torch.no_grad():
            logits = torch.cat([model(m, s, a) for m, s, a in held]).cpu()
        model.train()
        return balanced_accuracy(logits, hl), logits

    updates = args.updates or CHECKER_UPDATES
    for step in range(1, updates + 1):
        x0, asked, labels, _ = checker_examples(pool_train, CHECKER_BATCH, gen, empty)
        maps, sigma, av = batch_inputs(nets, x0, asked, gen)
        loss = F.binary_cross_entropy_with_logits(model(maps, sigma, av), labels.to(DEV))
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 250 == 0 or step == updates // 2:
            ba, _ = evaluate()
            rec = {"update": step, "loss": round(float(loss), 4), "held_out_balanced_accuracy": round(ba, 4), "seconds": round(time.time() - begin)}
            log.write(json.dumps(rec) + "\n"); log.flush(); print(json.dumps(rec), flush=True)
            if step == updates // 2 and ba <= FUTILE_CHECKER_BA and not args.sigma_window:   # the diagnosis runs to its end
                (out / "checker.json").write_text(json.dumps({"stopped_at_half": True, **rec}, indent=1) + "\n")
                raise SystemExit(f"stopped at half: held-out balanced accuracy {ba:.3f} at or below {FUTILE_CHECKER_BA}")
    ba, logits = evaluate()
    by_kind = {name: round(float(((logits > 0).float() == hl)[hk == k].float().mean()), 4) for k, name in
               enumerate(("apart, right counts", "touching", "apart, one count off"))}
    torch.save({"state": model.state_dict(), "push_steps": list(PUSH_STEPS), "b": PUSH_B, "reader": READER}, out / "checker.pt")
    result = {"reader": READER, "updates": updates, "batch": CHECKER_BATCH, "sigma_window": SIGMA_WINDOW, "held_out_pastes": 1024, "held_out_balanced_accuracy": round(ba, 4),
              "right_by_kind": by_kind, "min_balanced_accuracy": MIN_CHECKER_BA, "passes": ba >= MIN_CHECKER_BA,
              "seconds": round(time.time() - begin), "device": torch.cuda.get_device_name()}
    (out / "checker.json").write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result), flush=True)
    raise SystemExit(0 if result["passes"] else 1)


# ---------------------------------------------------------------------------------------------------------------------------------------
# Sampling


def heun(denoise, x, t, after_step=None):
    for i in range(STEPS):
        t_cur, t_next = t[i], t[i + 1]
        d_cur = (x - denoise(x, t_cur, i)) / t_cur if denoise.__code__.co_argcount == 3 else (x - denoise(x, t_cur)) / t_cur
        x_next = x + (t_next - t_cur) * d_cur
        if i < STEPS - 1:
            d_eval = denoise(x_next, t_next, i) if denoise.__code__.co_argcount == 3 else denoise(x_next, t_next)
            x_next = x + (t_next - t_cur) * 0.5 * (d_cur + (x_next - d_eval) / t_next)
        x = x_next
        if after_step is not None and i < STEPS - 1:
            x = after_step(x, t_next)
    return x


def product_denoiser(nets, xc, yc, checker=None, b=0.0):
    r, bl, e = nets

    def denoise(x, s, step):
        sv = s.expand(len(x))
        if checker is not None and step in PUSH_STEPS:
            with torch.enable_grad():
                xx = x.detach().requires_grad_(True)
                if READER == "raw":
                    logit = checker(xx, sv, asked_vector(xc, yc))
                else:
                    d_e = e(xx, sv, None)
                    d_r, d_b = r(xx, sv, xc), bl(xx, sv, yc)
                    logit = checker(torch.cat([d_r - d_e, d_b - d_e], 1), sv, asked_vector(xc, yc))
                g, = torch.autograd.grad(logit.sum(), xx)
            with torch.no_grad():
                d = (r(x, sv, xc) + bl(x, sv, yc) - e(x, sv, None)).float()
            size = (x - d).flatten(1).norm(dim=1)[:, None, None, None]
            return d + b * size * g / g.flatten(1).norm(dim=1).clamp_min(1e-12)[:, None, None, None]
        with torch.no_grad():
            return (r(x, sv, xc) + bl(x, sv, yc) - e(x, sv, None)).float()
    return denoise


def row(args):
    from generate_images import StackedRandomGenerator
    nets_enc = [load_net(n) for n in ("r", "b", "empty")]
    nets, encoder = [n for n, _ in nets_enc], nets_enc[0][1]
    checker = None
    if args.row in ("checker", "zero"):
        checker = Checker(4 if READER == "raw" else 7).to(DEV)
        checker.load_state_dict(torch.load(W / (args.out_name or "checker") / "checker.pt", map_location=DEV)["state"])
        checker.eval().requires_grad_(False)
    b = {"checker": PUSH_B, "zero": 0.0}.get(args.row, 0.0)
    t = sigmas()
    seeds = list(range(args.block * BLOCK, (args.block + 1) * BLOCK))
    if args.limit:
        seeds = seeds[:args.limit]
    cells = CELLS[:1] if args.row == "zero" else CELLS
    batch_size = 64 if checker is not None else 128
    begin = time.time()
    for x_count, y_count in cells:
        out = W / "rows" / (args.row if READER == "maps" else f"{args.row}-raw") / f"c{x_count}{y_count}"
        for start in range(0, len(seeds), batch_size):
            batch = seeds[start:start + batch_size]
            rnd = StackedRandomGenerator(DEV, batch)
            x = rnd.randn([len(batch), 3, 64, 64], device=DEV) * t[0]
            xc, yc = one_hot(x_count, len(batch)), one_hot(y_count, len(batch))
            denoise = product_denoiser(nets, xc, yc, checker, b)
            after = None
            if args.row == "corrector":
                def after(z, s, denoise=denoise, rnd=rnd):
                    for _ in range(CORRECTOR_STEPS):
                        eps = CORRECTOR_EPS * s ** 2
                        z = z + eps * (denoise(z, s, -1) - z) / s ** 2 + (2 * eps).sqrt() * rnd.randn_like(z)
                    return z
            with torch.no_grad() if checker is None else torch.enable_grad():
                x = heun(denoise, x, t, after)
            save(to_uint8(encoder, x.detach()), out, batch)
    rec = {"row": args.row, "block": args.block, "seeds": [seeds[0], seeds[-1]], "cells": len(cells), "images": len(cells) * len(seeds),
           "seconds": round(time.time() - begin, 1), "device": torch.cuda.get_device_name()}
    with open(W / "rows" / "timing.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec), flush=True)
    if args.row == "zero":
        diffs = [np.abs(np.asarray(Image.open(p), dtype=np.int16) -
                        np.asarray(Image.open(W / "rows/plain/c11" / Path(p).name), dtype=np.int16)).max()
                 for p in sorted(glob.glob(str(W / "rows" / ("zero" if READER == "maps" else "zero-raw") / "c11/*.png")))]
        print(json.dumps({"zero_push_largest_pixel_difference": int(max(diffs)), "images": len(diffs)}), flush=True)


def paste_row(args):
    empty = mean_empty()
    seeds = list(range(args.block * BLOCK, (args.block + 1) * BLOCK))
    for x_count, y_count in CELLS:
        red = load_source(W / "sources" / f"r-count-{x_count}", seeds)
        blue = load_source(W / "sources" / f"b-count-{y_count}", seeds)
        pasted, _ = paste(red, blue, empty)
        save((pasted * 255).round().clamp(0, 255).byte().numpy(), W / "rows/paste" / f"c{x_count}{y_count}", seeds)
    print(json.dumps({"row": "paste", "block": args.block, "images": 9 * len(seeds)}), flush=True)


# ---------------------------------------------------------------------------------------------------------------------------------------
# Reporting


def report(args):
    from m10_scorer import load_scorer, counts
    scorer = load_scorer(W / "scorer")
    rows = sorted(p.name for p in (W / "rows").iterdir() if p.is_dir() and p.name != "zero")
    seeds = list(range(args.block * BLOCK, (args.block + 1) * BLOCK))
    result = {"block": args.block, "seeds": [seeds[0], seeds[-1]], "rows": {}}
    tiles = {}
    for name in rows:
        per_cell, done = {}, True
        for x_count, y_count in CELLS:
            folder = W / "rows" / name / f"c{x_count}{y_count}"
            paths = [folder / f"{s:06d}.png" for s in seeds]
            if not all(p.exists() for p in paths):
                done = False
                break
            x = torch.from_numpy(np.stack([np.asarray(Image.open(p).convert("RGB")) for p in paths])).float().permute(0, 3, 1, 2) / 127.5 - 1
            called = counts(scorer, x)
            per_cell[f"c{x_count}{y_count}"] = round(float((called == torch.tensor([x_count, y_count])).all(1).float().mean()), 4)
            tiles[(name, x_count, y_count)] = [Image.open(p).convert("RGB") for p in paths[:8]]
        if done:
            result["rows"][name] = {"exact_share_mean_over_cells": round(float(np.mean(list(per_cell.values()))), 4), "per_cell": per_cell}
    out = W / "reports"; out.mkdir(exist_ok=True)
    (out / f"block-{args.block}.json").write_text(json.dumps(result, indent=1) + "\n")
    names = list(result["rows"])
    grid = Image.new("RGB", (9 * 8 * 64 + 8 * 8, len(names) * 64), "white")
    for i, name in enumerate(names):
        for j, (x_count, y_count) in enumerate(CELLS):
            for k, tile in enumerate(tiles[(name, x_count, y_count)]):
                grid.paste(tile, (j * (8 * 64 + 8) + k * 64, i * 64))
    grid.save(out / f"block-{args.block}.png")
    print(json.dumps({n: result["rows"][n]["exact_share_mean_over_cells"] for n in names}), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("stage", choices=["sources", "train-checker", "row", "paste", "report"])
    ap.add_argument("--colour", choices=["r", "b", "empty"])
    ap.add_argument("--seeds")
    ap.add_argument("--row", choices=["plain", "checker", "zero", "corrector"])
    ap.add_argument("--block", type=int, default=0)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--sigma-window", help="lo,hi: the diagnosis's noise window for train-checker")
    ap.add_argument("--out-name", help="train-checker's output folder under /workspace/m10, for the diagnosis")
    ap.add_argument("--updates", type=int)
    ap.add_argument("--reader", choices=["maps", "raw"], default="maps")
    args = ap.parse_args()
    global READER, PUSH_STEPS
    READER = args.reader
    if READER == "raw":
        PUSH_STEPS = range(14, 18)  # sigma 3.8 to 1.6: question 116's checker, written into the design before it trained
    if args.sigma_window:
        global SIGMA_WINDOW
        SIGMA_WINDOW = tuple(float(v) for v in args.sigma_window.split(","))
    torch.distributed.init_process_group("gloo", init_method=f"file:///tmp/m10-compose-{os.getpid()}", rank=0, world_size=1)
    {"sources": sources, "train-checker": train_checker, "row": row, "paste": paste_row, "report": report}[args.stage](args)


if __name__ == "__main__":
    main()
