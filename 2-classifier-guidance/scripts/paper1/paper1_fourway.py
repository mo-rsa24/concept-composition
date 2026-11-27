#!/usr/bin/env python3
"""Paper 1's Method B: a noise-conditioned four-way classifier on real pictures, whose push corrects the composed CLEVR sampler.

The four classes, for a noisy 64-pixel picture x at noise level sigma, each told the count k (k red objects, k blue objects):
  0  neither  (q00)  an empty scene
  1  red      (q10)  k red objects and no blue
  2  blue     (q01)  k blue objects and no red
  3  both     (q11)  k red and k blue objects in one scene
q = softmax of the four logits. p(red present | x) = q10 + q11, p(blue present | x) = q01 + q11, p(both | x) = q11.

The push direction is the gradient of the interaction ratio
    r(x, sigma) = grad_x [ log q11 - log(q10 + q11) - log(q01 + q11) ]
                = grad_x [ log p(both | x) - log p(red | x) - log p(blue | x) ],
the part of "both" that each expert alone does not already say. With l the four logits it is computed without forming q:
    ratio = l11 - logaddexp(l10, l11) - logaddexp(l01, l11) + logsumexp(l00, l10, l01, l11).
During sampling it is added exactly as paper1_sample.py push adds the head's gradient: D' = D + w * sigma^2 * r on the pushed
evaluations of the window, D the region sampler's D_empty + M_red (D_red - D_empty) + M_blue (D_blue - D_empty) (--base region, the
default, the same masks and starting noise as paper1_sample.py push), or the plain sum D_red + D_blue - D_empty (--base plainsum,
both masks all ones, which is the plain sum's own floating-point arithmetic). The Heun schedule is paper1_sample.heun's (32 steps,
sigma 80 to 0.002, rho 7). The default window is 0-31, every one of the 63 evaluations, because this classifier is trained over the
whole sampler range. FourWay.logit returns the ratio under the name the push calls, so paper1_sample.push_denoiser, draw_pushed and
PushStats take it unchanged; at w = 0 the gradient is still taken and 0 is added, so the pictures are the base sampler's exactly.

The bound. Each of the four logits is LOGIT_BOUND * tanh(raw / LOGIT_BOUND), LOGIT_BOUND = 4 (paper1_disc's head's bound, applied to
each class). l11 - logaddexp(l10, l11) is then in [-8.0003, -0.0003] and logsumexp(all) - logaddexp(l01, l11) in [0.0003, 8.0003], so
the ratio stays within plus or minus 8.0004 and its gradient flattens wherever a logit saturates. The gradient itself is not capped,
as paper1_sample.py push does not cap the head's.

The network: paper1_disc.py's head on the frozen red and blue experts' decoder features (red 16, red 8 upsampled, blue 16, blue 8
upsampled; the blocks from paper1_disc.py choose's blocks.json, or --block8 and --block16), two [3 by 3 convolution, GroupNorm, scale
and shift from (log sigma, count one-hot), SiLU] blocks, the grid average and a linear layer, here to four raw scores instead of one.

Training data, real rendered pictures only (train):
  both      the joint scenes of a paper1_pack.py folder (--positives; Method A trains on paper1-diag-pack-k384/k384-s1)
  red       the red expert's own training set, r-full.zip (EDM2 zip, dataset.json label = count - 1), 5,000 per count
  blue      the blue expert's, b-full.zip, 5,000 per count
  neither   the empty-scene model's, empty.zip, 5,000 pictures, no label
Every tenth picture per class is held out (pictures with identical pixels grouped first, groups in index order, ranks 0, 10, 20, ...;
for the joint scenes this is paper1_interaction.split_scenes, so Method A and Method B hold out the same scenes). Batches hold the same
number of each class; one count per slot is drawn uniformly from 1 to 3 and shared by the four classes' examples in that slot, so the
count carries no class information (an empty scene takes the slot's count as the experts' label). sigma is log-uniform on
[SIGMA_LO, SIGMA_HI] = [0.002, 80], the sampler's whole range, fresh per example; every example is shifted by up to 4 whole pixels
each way, edges repeated, never mirrored (paper1_disc.shifted). Loss: cross-entropy over the four classes; AdamW as paper1_disc.
At update 200 the measured rate is projected to the full run and paper1_disc.stop_if_over stops it (exit 3) past 4 GPU-hours.
Validation is read only, never chooses: every VAL_EVERY updates and at the end, each held-out picture noised VAL_DRAWS times per sigma
band, giving per band the four-way accuracy per class, their mean (balanced accuracy), the confusion counts and the median ratio per
class, written to train.json. The checkpoint is the last update's, fourway.pt.

Outputs under /datasets/mmolefe/poe_repair_min/paper1/<run>/. train: fourway.pt, train.json, train.jsonl, projection.jsonl.
push: count-<k>/<seed>.png (one weight) or w-<w>/count-<k>/ (several), push.json (paper1_sample's PushStats per band, the grey-level
change from --compare-to, and the classifier's mean class probabilities and median ratio along the pushed trajectory per band).

Usage (cluster, through paper1_gpu.sbatch; folders absolute or relative to /datasets/mmolefe/poe_repair_min/paper1/):
  python paper1_fourway.py train --run paper1-diag-fourway-k384 --positives paper1-diag-pack-k384/k384-s1 \
      --blocks-file paper1-first-disc-k64/blocks.json --seed 0
  python paper1_fourway.py push --run paper1-diag-push-fourway --head paper1-diag-fourway-k384/fourway.pt --weights 1,4,16,64 \
      --red-dir <lone red run> --blue-dir <lone blue run> --mean-empty <pt> --counts 1,2,3 \
      --split paper1-first-lone-red/split.json --set evaluation --side touching --compare-to paper1-first-row-region
"""
import argparse
import io
import json
import math
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import paper1_sample as ps  # noqa: E402
import paper1_disc as pdisc  # noqa: E402
import paper1_interaction as pi  # noqa: E402

DEV = ps.DEV

# ---------------------------------------------------------------------------------------------------------------------------------------
# Settings

CLASSES = ("neither", "red", "blue", "both")      # q00, q10, q01, q11, in logit order
NEITHER, RED, BLUE, BOTH = range(4)
RED_ZIP = Path("/datasets/mmolefe/poe_repair_min/loop-6/m10/data/r-full.zip")     # the red expert's training set (r1, r2, r3)
BLUE_ZIP = Path("/datasets/mmolefe/poe_repair_min/loop-6/m10/data/b-full.zip")    # the blue expert's (b1, b2, b3)
EMPTY_ZIP = Path("/datasets/mmolefe/poe_repair_min/loop-6/m7/full/data/empty.zip")  # the empty-scene model's
SIGMA_LO, SIGMA_HI = ps.SIGMA_MIN, ps.SIGMA_MAX    # 0.002 to 80, the sampler's whole range
PER_CLASS = 32                                     # batch 128: 32 of each class
UPDATES = 4000
LR, WEIGHT_DECAY = pdisc.LR, pdisc.WEIGHT_DECAY
WIDTH = pdisc.WIDTH
LOGIT_BOUND = pdisc.LOGIT_BOUND
SHIFT_MAX = pdisc.SHIFT_MAX
VAL_NTH = 10                                       # every tenth picture per class is held out
VAL_EVERY = 500
VAL_DRAWS = 2                                      # noise draws per held-out picture per band
VAL_SEED = 4321
VAL_BATCH = 256
# The validation bands cover the whole sampler range, highest noise first; the middle four are paper1_disc's check bands.
VAL_BANDS = ((80.0, 10.5), (10.5, 3.8), (3.8, 1.6), (1.6, 0.6), (0.6, 0.28), (0.28, 0.002))
PUSH_WINDOW = "0-31"
RATIO_BOUND = 8.0004                               # |ratio| under LOGIT_BOUND = 4 (docstring, "The bound")


def band_name(band):
    return f"{band[0]:g}-{band[1]:g}"


def band_of(sigma):
    """The validation band a sigma falls in, by lower edges (0.002 counts in the last band)."""
    for band in VAL_BANDS:
        if sigma >= band[1] - 1e-6:
            return band_name(band)
    return band_name(VAL_BANDS[-1])


# ---------------------------------------------------------------------------------------------------------------------------------------
# The classifier


class FourWayHead(pdisc.Head):
    """paper1_disc.Head with four raw scores, each bounded as the head's one logit is."""

    def __init__(self, in_channels, width=WIDTH, bound=LOGIT_BOUND):
        super().__init__(in_channels, width, bound=bound)
        self.out = nn.Linear(width, len(CLASSES))

    def forward(self, feats, sigma, count):
        """feats (n, C, 16, 16); sigma (n,); count (n,) 1 to 3. Returns the four bounded logits (n, 4)."""
        c = torch.cat([sigma.float().log()[:, None], F.one_hot(count.long() - 1, 3).float()], 1)
        s1, b1, s2, b2 = (v[:, :, None, None] for v in self.cond(c).chunk(4, 1))
        h = F.silu(self.norm1(self.conv1(feats)) * (1 + s1) + b1)
        h = F.silu(self.norm2(self.conv2(h)) * (1 + s2) + b2)
        raw = self.out(h.mean((2, 3)))
        return self.bound * torch.tanh(raw / self.bound) if self.bound > 0 else raw


def interaction_ratio(logits):
    """log q11 - log(q10 + q11) - log(q01 + q11) from the four logits (n, 4), q = softmax(logits). (n,)"""
    l10, l01, l11 = logits[:, RED], logits[:, BLUE], logits[:, BOTH]
    return l11 - torch.logaddexp(l10, l11) - torch.logaddexp(l01, l11) + torch.logsumexp(logits, 1)


class FourWay:
    """The frozen experts' features plus the four-way head. logits(x, sigma, count) -> (n, 4); logit(x, sigma, count) -> the interaction
    ratio (n,), named as paper1_disc.Discriminator.logit is so paper1_sample.push_denoiser takes it as the push's scalar.

    trace: None, or a callable trace(sigma, logits) that logit() hands every evaluation's detached logits (ProbTrace)."""

    def __init__(self, features, head, trace=None):
        self.features, self.head, self.trace = features, head, trace

    def logits(self, x, sigma, count):
        s = sigma.expand(len(x)) if sigma.dim() == 0 else sigma
        return self.head(self.features(x, s, count), s, count)

    def logit(self, x, sigma, count):
        out = self.logits(x, sigma, count)
        if self.trace is not None:
            self.trace(sigma, out.detach())
        return interaction_ratio(out)


def make_features(red_net, blue_net, block8, block16):
    """paper1_disc's frozen-expert feature extractor: features(x, sigma, count) -> (n, 1792, 16, 16) for edm2-img64-xs."""
    return pdisc.Discriminator(red_net, blue_net, block8, block16).features


def load_fourway(path, red_net, blue_net):
    """fourway.pt to a FourWay on DEV, head in eval mode with no gradients on its weights; and the checkpoint's record."""
    ckpt = torch.load(path, map_location=DEV)
    head = FourWayHead(ckpt["in_channels"], ckpt["width"], bound=ckpt["logit_bound"]).to(DEV)
    head.load_state_dict(ckpt["head"])
    head.eval().requires_grad_(False)
    clf = FourWay(make_features(red_net, blue_net, ckpt["block8"], ckpt["block16"]), head)
    return clf, {k: v for k, v in ckpt.items() if k != "head"}


# ---------------------------------------------------------------------------------------------------------------------------------------
# Data: four pools of real pictures, each split into training and held-out


def load_zip(path):
    """An EDM2 dataset zip: (pictures uint8 (n, 3, 64, 64), counts (n,) = label + 1 or 0 with no dataset.json, names), in name order."""
    with zipfile.ZipFile(path) as z:
        names = sorted(n for n in z.namelist() if n.endswith(".png"))
        labels = dict(json.loads(z.read("dataset.json"))["labels"]) if "dataset.json" in z.namelist() else None
        pics = np.stack([np.asarray(Image.open(io.BytesIO(z.read(n))).convert("RGB")) for n in names])
    counts = [labels[n] + 1 for n in names] if labels is not None else [0] * len(names)
    return torch.from_numpy(pics).permute(0, 3, 1, 2).contiguous(), torch.tensor(counts), names


def split_nth(pics, nth=VAL_NTH):
    """(train indices, held-out indices) for pictures in index order: identical pixels grouped, every nth group held out (ranks 0,
    nth, 2 nth, ...); the rule of paper1_interaction.split_scenes with the index as the order."""
    import hashlib
    groups = {}
    for i in range(len(pics)):
        groups.setdefault(hashlib.sha256(pics[i].numpy().tobytes()).hexdigest(), []).append(i)
    ordered = sorted(groups.values(), key=min)
    val = sorted(i for rank, g in enumerate(ordered) if rank % nth == 0 for i in g)
    return sorted(set(range(len(pics))) - set(val)), val


def make_pool(name, source, pics, counts, names, train, val):
    """One class's pictures (uint8) with its training indices by count (0 for the empty scenes, which carry no count)."""
    train_t = torch.tensor(train, dtype=torch.long)
    by_count = {int(k): train_t[counts[train_t] == k] for k in counts[train_t].unique().tolist()}
    return {"name": name, "source": str(source), "pics": pics, "counts": counts, "names": names, "train": train, "val": val,
            "by_count": by_count}


def load_pools(args):
    """The four pools in class order: neither, red, blue, both."""
    pools = []
    for cls, path in ((NEITHER, args.empty_zip), (RED, args.red_zip), (BLUE, args.blue_zip)):
        pics, counts, names = load_zip(pdisc.resolve(args.root, path))
        if cls == NEITHER and counts.any():
            raise SystemExit(f"{path} carries count labels; the empty scenes should carry none")
        if cls != NEITHER and not set(counts.tolist()) <= set(ps.COUNTS):
            raise SystemExit(f"{path} holds counts {sorted(set(counts.tolist()))}, not among {ps.COUNTS}")
        train, val = split_nth(pics)
        pools.append(make_pool(CLASSES[cls], path, pics, counts, names, train, val))
    pos_dir = pdisc.resolve(args.root, args.positives)
    x, counts, names = pdisc.load_folder(pos_dir)
    train, val = pi.split_scenes(x, names)                      # the same held-out scenes as Method A
    pics = ((x + 1) * 127.5).round().clamp(0, 255).to(torch.uint8)
    pools.append(make_pool("both", args.positives, pics, counts, names, train, val))
    return pools


def draw_batch(pools, per_class, gen):
    """One balanced batch: per_class examples of each class in class order, one count per slot (uniform on 1 to 3) shared by the four
    classes, each example drawn with replacement from its class's training pictures at that count (any empty scene for neither).
    Returns (pictures uint8 (4 per_class, 3, 64, 64), counts, labels, the pool indices drawn per class)."""
    slot_counts = torch.randint(1, 4, (per_class,), generator=gen)
    xs, picks = [], []
    for cls, pool in enumerate(pools):
        idx = torch.empty(per_class, dtype=torch.long)
        for k in ps.COUNTS:
            slots = (slot_counts == k).nonzero()[:, 0]
            if len(slots) == 0:
                continue
            avail = pool["by_count"][0] if cls == NEITHER else pool["by_count"].get(k)
            if avail is None or len(avail) == 0:
                raise SystemExit(f"class {pool['name']} has no training pictures at count {k}")
            idx[slots] = avail[torch.randint(len(avail), (len(slots),), generator=gen)]
        xs.append(pool["pics"][idx])
        picks.append(idx)
    labels = torch.arange(len(pools)).repeat_interleave(per_class)
    return torch.cat(xs), slot_counts.repeat(len(pools)), labels, picks


def validation_set(pools):
    """Every held-out picture of every class: (pictures uint8, counts, labels); an empty scene's count cycles 1, 2, 3 by its order."""
    xs, cs, ys = [], [], []
    for cls, pool in enumerate(pools):
        idx = torch.tensor(pool["val"], dtype=torch.long)
        xs.append(pool["pics"][idx])
        cs.append(torch.arange(len(idx)) % 3 + 1 if cls == NEITHER else pool["counts"][idx])
        ys.append(torch.full((len(idx),), cls))
    return torch.cat(xs), torch.cat(cs), torch.cat(ys)


def to_float(pics):
    return pics.float() / 127.5 - 1


@torch.no_grad()
def validate(clf, val, seed=VAL_SEED, draws=VAL_DRAWS, batch=VAL_BATCH):
    """Per sigma band: four-way accuracy per class, balanced accuracy (their mean), confusion counts (rows true, columns called) and the
    median interaction ratio per true class. The noise is drawn from a generator seeded the same way every call."""
    pics, counts, labels = val
    gen = torch.Generator().manual_seed(seed)
    out = []
    for band in VAL_BANDS:
        hi, lo = band
        called, truth, ratios = [], [], []
        for _ in range(draws):
            for i in range(0, len(pics), batch):
                x0 = to_float(pics[i:i + batch])
                sigma = pdisc.log_uniform(len(x0), lo, hi, gen)
                x = pdisc.noised(x0, sigma, gen)
                logits = clf.logits(x.to(DEV), sigma.to(DEV), counts[i:i + batch].to(DEV)).float().cpu()
                called.append(logits.argmax(1))
                ratios.append(interaction_ratio(logits))
                truth.append(labels[i:i + batch])
        called, truth, ratios = torch.cat(called), torch.cat(truth), torch.cat(ratios)
        confusion = torch.zeros(len(CLASSES), len(CLASSES), dtype=torch.long)
        confusion.index_put_((truth, called), torch.ones_like(truth), accumulate=True)
        per_class = {CLASSES[c]: round(float(confusion[c, c]) / max(1, int(confusion[c].sum())), 4) for c in range(len(CLASSES))}
        out.append({"band": band_name(band), "accuracy_per_class": per_class,
                    "balanced_accuracy": round(float(np.mean(list(per_class.values()))), 4),
                    "confusion_rows_true_columns_called": confusion.tolist(),
                    "median_ratio_per_class": {CLASSES[c]: round(float(ratios[truth == c].median()), 4) if (truth == c).any() else None
                                               for c in range(len(CLASSES))},
                    "examples_per_class": {CLASSES[c]: int((truth == c).sum()) for c in range(len(CLASSES))}})
    return out


# ---------------------------------------------------------------------------------------------------------------------------------------
# train


def train_classifier(clf, pools, args, log=None, on_validate=None):
    """Train clf.head from a fresh start on balanced batches. Returns the training record (curves and rate)."""
    torch.manual_seed(args.seed)
    gen = torch.Generator().manual_seed(args.seed)
    probe = to_float(pools[BOTH]["pics"][:1]).to(DEV)
    in_channels = clf.features(probe, torch.tensor([1.0], device=DEV), torch.tensor([1], device=DEV)).shape[1]
    clf.head = FourWayHead(in_channels, args.width, bound=args.logit_bound).to(DEV).train()
    opt = torch.optim.AdamW(clf.head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    folder = ps.run_dir(args)
    begin, losses, accs, rate, drawn = time.time(), [], [], None, [set() for _ in pools]
    for update in range(1, args.updates + 1):
        pics, count, label, picks = draw_batch(pools, args.per_class, gen)
        for seen, p in zip(drawn, picks):
            seen.update(p.tolist())
        x0 = pdisc.shifted(to_float(pics), args.shift, gen).to(DEV)
        sigma = pdisc.log_uniform(len(x0), args.sigma_lo, args.sigma_hi, gen).to(DEV)
        x = pdisc.noised(x0, sigma, gen)
        count, label = count.to(DEV), label.to(DEV)
        with torch.no_grad():
            feats = clf.features(x, sigma, count)
        logits = clf.head(feats, sigma, count)
        loss = F.cross_entropy(logits, label)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
        accs.append(float((logits.detach().argmax(1) == label).float().mean()))
        if log is not None and update % 10 == 0:
            log({"update": update, "loss": float(np.mean(losses[-10:])), "train_accuracy": float(np.mean(accs[-10:]))})
        if update == pdisc.TIMING_UPDATES:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            rate = (time.time() - begin) / update
            pdisc.stop_if_over(folder, {"stage": "fourway-train", "seconds_per_update": round(rate, 5),
                                        "hours_per_1000_updates": round(rate * 1000 / 3600, 4), "updates": args.updates,
                                        "already_gpu_hours": args.already_gpu_hours, "other_gpu_hours": args.other_gpu_hours,
                                        "projected_gpu_hours": pdisc.projection(rate, args.updates, args.already_gpu_hours,
                                                                                args.other_gpu_hours)})
        if on_validate is not None and (update % args.val_every == 0 or update == args.updates):
            clf.head.eval()
            on_validate(update)
            clf.head.train()
    return {"updates": args.updates, "seconds": round(time.time() - begin, 1), "seconds_per_update_first_200": rate,
            "in_channels": in_channels, "head_weights": pdisc.weight_count(clf.head),
            "loss_curve_every": pi.LOSS_CURVE_EVERY,
            "loss_curve": [round(float(np.mean(losses[i:i + pi.LOSS_CURVE_EVERY])), 5) for i in range(0, len(losses), pi.LOSS_CURVE_EVERY)],
            "train_accuracy_curve": [round(float(np.mean(accs[i:i + pi.LOSS_CURVE_EVERY])), 4) for i in range(0, len(accs), pi.LOSS_CURVE_EVERY)],
            "distinct_pictures_drawn": {pools[c]["name"]: len(drawn[c]) for c in range(len(pools))},
            "drawn_held_out": {pools[c]["name"]: len(drawn[c] & set(pools[c]["val"])) for c in range(len(pools))}}


def cmd_train(args):
    folder = ps.run_dir(args)
    if args.blocks_file:
        blocks = json.loads(pdisc.resolve(args.root, args.blocks_file).read_text())["chosen"]
        block8, block16 = blocks["block8"], blocks["block16"]
    else:
        block8, block16 = args.block8, args.block16
    pools = load_pools(args)
    for pool in pools:
        if set(pool["train"]) & set(pool["val"]):
            raise SystemExit(f"class {pool['name']} has pictures on both sides of the split")
    val = validation_set(pools)
    red, blue = pdisc.load_pair(args)
    clf = FourWay(make_features(red, blue, block8, block16), None)

    data = {}
    for pool in pools:
        per_count = {}
        for k in sorted(set(pool["counts"].tolist())):
            sel = pool["counts"] == k
            idx = sel.nonzero()[:, 0].tolist()
            per_count["any" if k == 0 else f"count-{k}"] = {"pictures": len(idx), "held_out": len(set(idx) & set(pool["val"]))}
        data[pool["name"]] = {"source": pool["source"], "pictures": len(pool["names"]), "trained": len(pool["train"]),
                              "held_out": len(pool["val"]), "per_count": per_count}
    settings = {k: getattr(args, k) for k in ("sigma_lo", "sigma_hi", "per_class", "lr", "weight_decay", "updates", "width", "shift",
                                              "logit_bound", "seed", "val_every")}
    settings.update({"classes": list(CLASSES), "batch": args.per_class * len(CLASSES), "block8": block8, "block16": block16,
                     "mirror": False, "val_nth": VAL_NTH, "val_draws": VAL_DRAWS, "val_seed": VAL_SEED,
                     "val_bands": [band_name(b) for b in VAL_BANDS], "data": data,
                     "split_rule": f"per class, identical pictures grouped, groups in index order (zip name, or render index for the "
                                   f"joint scenes, paper1_interaction.split_scenes), every {VAL_NTH}th group (ranks 0, {VAL_NTH}, ...) held out",
                     "count_rule": "one count per batch slot, uniform on 1 to 3, shared by the four classes' examples in that slot; an "
                                   "empty scene takes the slot's count as the experts' label",
                     "red_net": args.red_net, "blue_net": args.blue_net})
    print(json.dumps({"settings": {k: v for k, v in settings.items()}}), flush=True)
    wb = None
    if args.wandb:
        try:
            import wandb
            wb = wandb.init(entity=pdisc.WANDB_ENTITY, project=pdisc.WANDB_PROJECT, name=args.run, config=settings,
                            job_type="paper1-methodB")
        except Exception as error:   # the curves still go to train.json
            print(json.dumps({"wandb": f"not logging: {error}"}), flush=True)

    def log(rec):
        with open(folder / "train.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        if wb is not None:
            wb.log(rec, step=rec["update"])

    val_curve = []

    def on_validate(update):
        bands = validate(clf, val)
        val_curve.append({"update": update, "bands": bands})
        print(json.dumps({"update": update, "balanced_accuracy_by_band": {b["band"]: b["balanced_accuracy"] for b in bands}}), flush=True)
        if wb is not None:
            wb.log({f"val/balanced_accuracy/{b['band']}": b["balanced_accuracy"] for b in bands}, step=update)

    record = train_classifier(clf, pools, args, log=log, on_validate=on_validate)
    if any(record["drawn_held_out"].values()):
        raise SystemExit(f"held-out pictures were drawn for training: {record['drawn_held_out']}")
    torch.save({"head": clf.head.state_dict(), "block8": block8, "block16": block16, "in_channels": record["in_channels"],
                "width": args.width, "logit_bound": args.logit_bound, "classes": list(CLASSES), "update": args.updates,
                "settings": settings}, folder / "fourway.pt")
    result = {"settings": settings, "training": record, "validation_curve": val_curve,
              "validation_final": val_curve[-1] if val_curve else None,
              "validation_rule": "read only: never chooses the checkpoint, which is the last update's",
              "checkpoint": str(folder / "fourway.pt"), "device": ps.device_name(), "wandb_run": wb.url if wb is not None else None}
    (folder / "train.json").write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({"updates": args.updates, "seconds": record["seconds"],
                      "final_balanced_accuracy_by_band": {b["band"]: b["balanced_accuracy"] for b in val_curve[-1]["bands"]}
                      if val_curve else None}), flush=True)
    if wb is not None:
        wb.finish()


# ---------------------------------------------------------------------------------------------------------------------------------------
# push


class ProbTrace:
    """Along the pushed trajectory, per validation band: the mean of each class probability and the median interaction ratio, over
    every evaluation the push took the gradient at (the logits at the current x, before the push is added)."""

    def __init__(self):
        self.rows = {}

    def __call__(self, sigma, logits):
        s = float(sigma.flatten()[0])
        row = self.rows.setdefault(band_of(s), {"probs": [], "ratio": []})
        row["probs"].append(logits.float().softmax(1).cpu())
        row["ratio"].append(interaction_ratio(logits.float()).cpu())

    def summary(self):
        out = {}
        for band in VAL_BANDS:
            row = self.rows.get(band_name(band))
            if not row:
                continue
            probs, ratio = torch.cat(row["probs"]), torch.cat(row["ratio"])
            out[band_name(band)] = {"mean_probability": {CLASSES[c]: round(float(probs[:, c].mean()), 4) for c in range(len(CLASSES))},
                                    "median_ratio": round(float(ratio.median()), 4), "evaluations": len(ratio)}
        return out


@torch.no_grad()
def draw(nets, clf, encoder, args, k, batch, w, window, empty, stats=None, grads=None):
    """One batch at weight w. --base region: paper1_sample.draw_pushed itself (its masks, --overlap and starting noise). --base
    plainsum: both masks all ones (the plain sum's arithmetic) from starting_noise(seed), with the same push and record."""
    if args.base == "region":
        return ps.draw_pushed(nets, clf, encoder, args, k, batch, w, window, empty, stats=stats, grads=grads)
    x0, rnd = ps.starting_noise(batch)
    ones = torch.ones(len(batch), 1, 64, 64, device=x0.device)

    def record(**kw):
        if stats is not None:
            stats(**kw)
        if grads is not None and not kw["second"] and kw["step"] in ps.GRAD_MAP_STEPS:
            grads[kw["step"]] = kw["g"][:args.grad_maps].abs().sum(1).cpu()
    denoise = ps.push_denoiser(nets, clf, k, ones, ones, w, window=window, generator=rnd if args.random_push else None, record=record)
    return ps.to_uint8(encoder, ps.heun(denoise, x0, ps.sigmas()))


@torch.no_grad()
def cmd_push(args):
    """Method B's push at one or more weights; push.json as paper1_sample.py push writes it, plus the classifier's trace per band."""
    offset = getattr(args, "blue_seed_offset", 0)
    if args.base == "plainsum" and (offset or args.overlap != "both"):
        raise SystemExit("--base plainsum starts every pair from starting_noise(seed) with no masks; it takes no --blue-seed-offset or "
                         "--overlap")
    ps.check_blue_offset(args.blue_dir, offset)
    folder = ps.run_dir(args)
    nets, encoder = ps.load_three(args)
    clf, record = load_fourway(pdisc.resolve(args.root, args.head), nets[0], nets[1])
    empty = ps.load_mean_empty(args.mean_empty)
    window = ps.parse_window(args.window)
    weights = ps.load_weights(args)
    counts = ps.parse_counts(args.counts)
    t_cpu = ps.sigmas("cpu")
    summary = {"method": "B: four-way classifier, push w sigma^2 grad_x [log q11 - log(q10 + q11) - log(q01 + q11)]",
               "base": args.base, "random_push": bool(args.random_push), "head": args.head, "head_update": record.get("update"),
               "window": [window[0], window[-1]], "window_sigmas": [round(float(t_cpu[window[0]]), 4), round(float(t_cpu[window[-1]]), 4)],
               "pushed_evaluations": len(ps.pushed_evaluations(window)), "ratio_bound": RATIO_BOUND,
               "seed_choice": {"seeds": args.seeds, "split": args.split, "set": args.set, "side": args.side, "limit": args.limit},
               "weights": {}}
    if offset:
        summary["blue_seed_offset"] = offset
    if args.overlap != "both":
        summary["overlap"] = args.overlap
    begin, pictures, maps = time.time(), 0, []
    for w in weights:
        out = folder if len(weights) == 1 else folder / f"w-{w:g}"
        stats, clf.trace = ps.PushStats(), ProbTrace()
        for k in counts:
            for batch in ps.batches(ps.seeds_for(args, k), args.batch):
                grads = {} if (args.grad_maps and k == counts[0] and w == weights[0] and not maps) else None
                images = draw(nets, clf, encoder, args, k, batch, w, window, empty, stats=stats, grads=grads)
                ps.save(images, out / f"count-{k}", batch)
                pictures += len(batch)
                if grads is not None:
                    maps = [(batch[i], images[i], [grads[s][i] for s in ps.GRAD_MAP_STEPS if s in grads])
                            for i in range(min(args.grad_maps, len(batch)))]
        entry = {"by_band": stats.summary(), "classifier_along_trajectory": clf.trace.summary()}
        clf.trace = None
        if args.compare_to:
            entry["grey_change"] = ps.grey_change(out, pdisc.resolve(args.root, args.compare_to), counts, args, empty)
        summary["weights"][f"{w:g}"] = entry
        print(json.dumps({"w": w, "grey_change": entry.get("grey_change"),
                          "classifier_along_trajectory": entry["classifier_along_trajectory"]}), flush=True)
    if args.repeat_check:
        k, w = counts[0], weights[0]
        seeds = ps.seeds_for(args, k)[:args.repeat_check]
        again = draw(nets, clf, encoder, args, k, seeds, w, window, empty)
        first = ps.load_pictures((folder if len(weights) == 1 else folder / f"w-{w:g}") / f"count-{k}", seeds) * 255
        largest = float((torch.from_numpy(again).float() - first).abs().max())
        summary["repeat_check"] = {"seeds": seeds, "count": k, "w": w, "largest_difference_grey": round(largest, 3)}
        print(json.dumps({"repeat_check_largest_difference_grey": largest}), flush=True)
    seconds = time.time() - begin
    summary.update({"pictures": pictures, "seconds": round(seconds, 1), "seconds_per_picture": round(seconds / max(1, pictures), 4),
                    "device": ps.device_name()})
    (folder / "push.json").write_text(json.dumps(summary, indent=1) + "\n")
    if maps:
        ps.grad_map_sheet(maps, folder / "grad-maps.png")


# ---------------------------------------------------------------------------------------------------------------------------------------


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="stage", required=True)

    def common(p, batch):
        p.add_argument("--run", required=True, help="the run name; outputs go to <root>/<run>/")
        p.add_argument("--root", default=str(ps.PAPER1_ROOT), help="where runs write (default %(default)s)")
        p.add_argument("--red-net", default=str(ps.RED_NET))
        p.add_argument("--blue-net", default=str(ps.BLUE_NET))
        p.add_argument("--empty-net", default=str(ps.EMPTY_NET))
        p.add_argument("--edm2", default=str(ps.EDM2_DIR), help="EDM2 checkout, for its pickles and seeding")
        p.add_argument("--batch", type=int, default=batch)
        return p

    p = common(sub.add_parser("train"), ps.BATCH)
    p.add_argument("--positives", required=True, help="the joint scenes (class both): a paper1_pack.py folder holding count-<k>/")
    p.add_argument("--red-zip", default=str(RED_ZIP), help="class red: the red expert's training zip (default %(default)s)")
    p.add_argument("--blue-zip", default=str(BLUE_ZIP), help="class blue: the blue expert's training zip (default %(default)s)")
    p.add_argument("--empty-zip", default=str(EMPTY_ZIP), help="class neither: the empty-scene model's training zip (default %(default)s)")
    p.add_argument("--blocks-file", help="blocks.json from paper1_disc.py choose")
    p.add_argument("--block8", default=pdisc.CANDIDATES_8[-1])
    p.add_argument("--block16", default=pdisc.CANDIDATES_16[-1])
    p.add_argument("--sigma-lo", type=float, default=SIGMA_LO)
    p.add_argument("--sigma-hi", type=float, default=SIGMA_HI)
    p.add_argument("--per-class", type=int, default=PER_CLASS, help="examples of each class per batch (default %(default)s, batch 128)")
    p.add_argument("--lr", type=float, default=LR)
    p.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    p.add_argument("--updates", type=int, default=UPDATES)
    p.add_argument("--width", type=int, default=WIDTH)
    p.add_argument("--shift", type=int, default=SHIFT_MAX, help="largest whole-pixel shift each way, every class; 0 turns it off")
    p.add_argument("--logit-bound", type=float, default=LOGIT_BOUND, help="each logit = c tanh(raw / c); 0 turns the bound off")
    p.add_argument("--seed", type=int, default=0, help="the random start and the order of examples")
    p.add_argument("--val-every", type=int, default=VAL_EVERY, help="updates between validation readings (default %(default)s)")
    p.add_argument("--already-gpu-hours", type=float, default=0.0, help="GPU-hours already spent, counted toward the stop")
    p.add_argument("--other-gpu-hours", type=float, default=0.0, help="GPU-hours still to come after training, counted toward the stop")
    p.add_argument("--no-wandb", dest="wandb", action="store_false", help="do not log to W&B")

    p = common(sub.add_parser("push"), ps.BATCH)
    p.add_argument("--head", required=True, help="fourway.pt from train")
    p.add_argument("--weights", default="1", help="one or more w, such as 1,4,16,64; or 'ladder' for 0 and paper1_sample.W_LADDER")
    p.add_argument("--base", choices=("region", "plainsum"), default="region",
                   help="region: the region sampler from the lone folders' masks, as paper1_sample.py push (default); plainsum: "
                        "D_red + D_blue - D_empty everywhere")
    p.add_argument("--random-push", action="store_true", help="a random direction of the real push's length at each pushed evaluation")
    p.add_argument("--red-dir", required=True); p.add_argument("--blue-dir", required=True)
    p.add_argument("--mean-empty", required=True)
    p.add_argument("--counts", default="1,2,3")
    p.add_argument("--compare-to", help="a run at the same seeds (such as the region sampler's), for the grey-level change")
    p.add_argument("--grad-maps", type=int, default=0, help="draw the gradient maps of this many seeds")
    p.add_argument("--window", default=PUSH_WINDOW, help="sampler steps the push acts on (default %(default)s: all 63 evaluations)")
    p.add_argument("--repeat-check", type=int, default=0, help="redraw this many seeds in the same job and record the largest difference")
    p.add_argument("--overlap", choices=("both", "lower"), default="both", help="as paper1_sample.py region (region base only)")
    p.add_argument("--blue-seed-offset", type=int, default=0, help="as paper1_sample.py push (region base only)")
    p.add_argument("--seeds", help="seed ranges such as 30000-30127; or choose them from a split file below")
    p.add_argument("--split", help="split.json from paper1_sample.py split")
    p.add_argument("--set", choices=ps.SEED_SETS, help="the seed set to read from the split file")
    p.add_argument("--side", choices=["touching", "apart"], default="touching")
    p.add_argument("--limit", type=int, help="the first N seeds in seed order, per count")
    p.set_defaults(weights_file=None)
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    {"train": cmd_train, "push": cmd_push}[args.stage](args)


if __name__ == "__main__":
    main()
