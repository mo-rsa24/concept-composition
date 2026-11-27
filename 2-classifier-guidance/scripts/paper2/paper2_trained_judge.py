"""Paper 2's own species judge: a small head trained on the hand-called pictures, on frozen CLIP ViT-L/14 features.

Run name `paper2-judge-trained`. The six off-the-shelf judges missed every bar of the species judge plan
(plans/paper-2-dials/plans/02-species-judge.md; artifacts/results/can-a-judge-name-the-species-in-a-composed-picture/),
so this judge learns the six kinds from the labelled set instead. It is scored by paper2_score.py, unchanged, on the
same 240 read test pictures, against the same bars.

What the head sees, per picture and its two asked names A and B. Every view is a 224 by 224 square cut from the picture
by torch operations (so the dial trainer's gradient reaches the pixels) and embedded by frozen CLIP ViT-L/14 into its
joint image and text space:

  views    the whole picture, and its four quarters
  bodies   each body the body counter found (bodies.json), cropped with the 10% margin and every other body's box
           painted out in the picture's mean colour, as the crop judges and the dial trainer's crop reward do (at most
           MAX_BODIES, the largest; the true count is kept)
  names    A, B and the 43 "neither" names (paper2_common.NEITHER_FILE, minus A and B), each the four-phrasing CLIP
           text embedding the CLIP judge uses

For every view and body the head reads the CLIP judge's shares (A against B against every neither name), the three
cosines, a 32-number projection of the image embedding, how alike A and B are, and for a body its box. It pools the
views and the bodies, adds the body count, and returns the six kinds' logits; it is run with A and B in both orders and
averaged, so swapping the names changes nothing. The two-clear score the dial trainer needs (01-setup.md: "the score is
the judge's two-clear score") is the probability of two clear.

Which pictures train it. Never a read or confirmation picture (the 240 sealed for scoring and the 60 sealed for step
0.23), never crow and raven (reported apart), never "other". `--pool unsealed` (the default) trains on every settled
picture on a choosing pair plus the spare pictures on test pairs, and drops any spare picture that shares its file or
its pair and seed with a read or confirmation picture (renders of one seed by different methods look alike), and the
unselected sample on test pairs (the scorer reads two-clear precision on it at its natural share). `--pool choosing`
trains on choosing pairs only, which keeps the plan's claim that the judge is read on pairs it never saw. A fixed tenth
of the training pictures, by pair and seed so renders of one seed never straddle the two sides, is held out as the
development part; the checkpoint is the epoch with the highest development agreement averaged over kinds.

Three subcommands, in this order:

  features  embeds every training-pool picture (as drawn and mirrored) into <run>/features/pool.pt and every picture
            of paper2_judge_lists' test list (read, spare and apart; never a confirmation picture) into
            <run>/features/test.pt. confirm.json is never opened
  train     reads pool.pt only, splits off the development part, trains the head, keeps the best development epoch.
            Writes head.pt, train-log.json, split-counts.json and settings-chosen.json (the head's checksum, its epoch
            and the pool, for paper2_score.py's settings check)
  score     the sealed read, once: the kept head on test.pt, written as <run>/test/calls.json and test/settings.json in
            the format paper2_score.py reads. Refuses to run a second time

    "$PY" paper2_trained_judge.py features --labelled-set <paper2-label-set>/labelled-set.csv \\
        --split <paper2-label-set>/split.json --bodies <paper2-find-bodies-split>/bodies.json \\
        --run /datasets/mmolefe/poe_repair_min/paper2/paper2-judge-trained
    "$PY" paper2_trained_judge.py train --labelled-set ... --split ... --run <the same>
    "$PY" paper2_trained_judge.py score --labelled-set ... --split ... --run <the same>
    "$PY" paper2_score.py --labelled-set ... --split ... --judge trained=<the same> --out <...>/paper2-judge-score-trained

The dial trainer takes the score through TrainedJudgeReward(head_path, device)(image, a, b), which has the same call
as paper2_identity_dial.CropShareReward: a (3, H, W) picture in [0, 1] in, (two-clear probability with its graph,
{"kind", "n_bodies", "probs"}) out, the boxes found on the detached picture and held fixed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
import zlib
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from paper2_common import NEITHER_FILE, load_bodies  # noqa: E402
from paper2_identity_dial import resize_center, torch_crops  # noqa: E402
from paper2_judge_lists import make_lists, read_labelled, refuse_confirm  # noqa: E402
from paper2_kinds import KINDS, TWO_CLEAR  # noqa: E402

CLIP_NAME = "openai/clip-vit-large-patch14"
SIZE = 224                      # CLIP ViT-L/14's input side
N_VIEWS = 5                     # the whole picture and its four quarters
MAX_BODIES = 6
SEALED_PARTS = ("read", "confirmation")
POOLS = ("unsealed", "choosing")
DEV_SHARE = 0.10
DEV_SEED = 0
TRAIN_DEFAULTS = {"epochs": 300, "patience": 60, "lr": 1e-3, "weight_decay": 0.05, "batch": 64, "hidden": 128,
                  "appearance": 32, "dropout": 0.2, "seed": 0}


def norm_kind(k) -> str:
    return "_".join(str(k or "").strip().lower().replace("-", " ").replace("_", " ").split())


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---------------------------------------------------------------- which pictures train the head

def training_pool(rows: list[dict], split: dict, pool: str) -> tuple[list[dict], Counter]:
    """The rows the head may learn from, and why each other candidate was dropped. No read or confirmation row, no
    row sharing a file or a pair and seed with one, ever comes out."""
    if pool not in POOLS:
        raise SystemExit(f"--pool must be one of {POOLS}, not {pool!r}")
    choosing, test = make_lists(rows, split)
    sealed = [r for r in rows if r["part"] in SEALED_PARTS]
    sealed_files = {r["picture"] for r in sealed}
    sealed_seeds = {(r["pair"], r["seed"]) for r in sealed}
    cands = list(choosing) + ([r for r in test if r["part"] == "spare"] if pool == "unsealed" else [])
    out, dropped = [], Counter()
    for r in cands:
        kind = norm_kind(r["kind"])
        if kind not in KINDS:
            dropped["not one of the six kinds"] += 1
        elif r["picture"] in sealed_files:
            dropped["same file as a read or confirmation picture"] += 1
        elif (r["pair"], r["seed"]) in sealed_seeds:
            dropped["same pair and seed as a read or confirmation picture"] += 1
        elif r["part"] != "choosing" and r.get("unselected_sample") == "True":
            dropped["the unselected sample on a test pair"] += 1
        else:
            out.append({**r, "kind": kind})
    bad = [r["item"] for r in out if r["part"] in SEALED_PARTS or r["picture"] in sealed_files]
    if bad:
        raise SystemExit(f"{len(bad)} sealed pictures reached the training pool; nothing written")
    return out, dropped


def dev_split(rows: list[dict], share: float = DEV_SHARE, seed: int = DEV_SEED) -> tuple[list[dict], list[dict]]:
    """A fixed tenth held out by (pair, seed), so look-alike renders of one seed never straddle the two sides. Groups are
    drawn within strata named by each group's rarest kind, so every kind with two or more groups reaches both sides."""
    groups = defaultdict(list)
    for r in rows:
        groups[(r["pair"], r["seed"])].append(r)
    total = Counter(r["kind"] for r in rows)
    strata = defaultdict(list)
    for g, rs in groups.items():
        strata[min({r["kind"] for r in rs}, key=lambda k: (total[k], k))].append(g)
    rng, dev_groups = random.Random(seed), set()
    for k in sorted(strata):
        gs = sorted(strata[k])
        rng.shuffle(gs)
        if len(gs) > 1:
            dev_groups.update(gs[:max(1, round(share * len(gs)))])
    train = [r for r in rows if (r["pair"], r["seed"]) not in dev_groups]
    dev = [r for r in rows if (r["pair"], r["seed"]) in dev_groups]
    return train, dev


def side_counts(rows: list[dict]) -> dict:
    return {k: {"all": sum(r["kind"] == k for r in rows),
                "look-alike": sum(r["kind"] == k and r["pair_type"] == "look-alike" for r in rows)} for k in KINDS}


def print_sides(sides: dict[str, list[dict]]) -> None:
    print("pictures per kind (look-alike in brackets): " + ", ".join(f"{n} {len(rs)}" for n, rs in sides.items()))
    for k in KINDS:
        print(f"  {k:16s} " + "   ".join(f"{n} {sum(r['kind'] == k for r in rs):4d} "
                                         f"({sum(r['kind'] == k and r['pair_type'] == 'look-alike' for r in rs):3d})"
                                         for n, rs in sides.items()))


# ---------------------------------------------------------------- the frozen backbone

class ClipBackbone:
    """CLIP ViT-L/14, frozen: image_embed takes (N, 3, 224, 224) in [0, 1] with its graph; text_embed is the CLIP judge's
    four-phrasing ensemble, without graph."""

    name = CLIP_NAME

    def __init__(self, device):
        from paper2_judge_clip import ClipNamer
        self.namer = ClipNamer(device)
        self.model = self.namer.model
        for p in self.model.parameters():
            p.requires_grad_(False)
        ip = self.namer.proc.image_processor
        self.mean = torch.tensor(ip.image_mean, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(ip.image_std, device=device).view(1, 3, 1, 1)
        self.logit_scale = float(self.model.logit_scale.exp())
        self.dim = int(self.model.config.projection_dim)

    def image_embed(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model.get_image_features(pixel_values=((x - self.mean) / self.std).to(self.model.dtype))
        if not torch.is_tensor(out):                   # newer transformers return an output object
            out = out.image_embeds if getattr(out, "image_embeds", None) is not None else out.pooler_output
        return F.normalize(out.float(), dim=-1)

    def text_embed(self, name: str) -> torch.Tensor:
        return self.namer.text_embedding(name).float()


class StubBackbone(nn.Module):
    """Tests and dry runs: a fixed random convolution in place of CLIP, differentiable, and hashed text vectors."""

    name = "stub"
    logit_scale = 10.0
    dim = 16

    def __init__(self, device="cpu"):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            self.conv = nn.Conv2d(3, self.dim, 16, stride=16)
        self.to(device)
        self.device = device
        for p in self.parameters():
            p.requires_grad_(False)

    def image_embed(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.conv(x).mean(dim=(2, 3)), dim=-1)

    def text_embed(self, name: str) -> torch.Tensor:
        g = torch.Generator().manual_seed(zlib.crc32(name.encode()))
        return F.normalize(torch.randn(self.dim, generator=g), dim=0).to(self.device)


def make_backbone(which: str, device):
    return StubBackbone(device) if which == "stub" else ClipBackbone(device)


# ---------------------------------------------------------------- one picture's features (the same code with or without graph)

def picture_views(image: torch.Tensor, boxes: list[list[int]]) -> tuple[torch.Tensor, list[list[int]]]:
    """(N_VIEWS + bodies, 3, 224, 224) for a (3, H, W) picture in [0, 1]: the whole, the four quarters, then up to
    MAX_BODIES body crops (the largest), each crop painted out against every body found. Returns the kept boxes too."""
    _, H, W = image.shape
    h2, w2 = H // 2, W // 2
    views = [resize_center(image, SIZE)] + [resize_center(image[:, y:y + h2, x:x + w2], SIZE)
                                            for y in (0, h2) for x in (0, w2)]
    keep = sorted(range(len(boxes)), key=lambda i: -(boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1]))
    keep = sorted(keep[:MAX_BODIES])
    crops = torch_crops(image, boxes) if boxes else []
    views += [resize_center(crops[i], SIZE) for i in keep]
    return torch.stack(views), [boxes[i] for i in keep]


def picture_features(backbone, image: torch.Tensor, boxes: list[list[int]]) -> dict:
    """The head's inputs for one picture: views (5, D), bodies (MAX_BODIES, D), mask, geometry, and the true count."""
    _, H, W = image.shape
    x, kept = picture_views(image, boxes)
    e = backbone.image_embed(x)
    D = e.shape[-1]
    bodies = e.new_zeros(MAX_BODIES, D)
    geom = e.new_zeros(MAX_BODIES, 5)
    mask = torch.zeros(MAX_BODIES, dtype=torch.bool, device=e.device)
    for i, (x0, y0, x1, y1) in enumerate(kept):
        bodies[i] = e[N_VIEWS + i]
        geom[i] = torch.tensor([x0 / W, y0 / H, x1 / W, y1 / H, (x1 - x0) * (y1 - y0) / (W * H)], device=e.device)
        mask[i] = True
    return {"views": e[:N_VIEWS], "bodies": bodies, "mask": mask, "geom": geom, "n_bodies": len(boxes)}


def mirror(boxes: list[list[int]], W: int) -> list[list[int]]:
    return [[W - x1, y0, W - x0, y1] for x0, y0, x1, y1 in boxes]


def load_picture(path: str, device) -> torch.Tensor:
    import numpy as np
    from PIL import Image
    arr = np.asarray(Image.open(path).convert("RGB"), dtype="float32") / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous().to(device)


# ---------------------------------------------------------------- the head

class JudgeHead(nn.Module):
    def __init__(self, dim: int, hidden: int = 128, appearance: int = 32, dropout: float = 0.2):
        super().__init__()
        self.config = {"dim": dim, "hidden": hidden, "appearance": appearance, "dropout": dropout}
        self.app = nn.Linear(dim, appearance)
        tok = 6 + appearance + 1                         # shares and cosines, appearance, how alike A and B are

        def mlp(n_in):
            return nn.Sequential(nn.Linear(n_in, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, hidden),
                                 nn.GELU())
        self.view_mlp = mlp(tok + 2)                     # + whole picture or quarter
        self.body_mlp = mlp(tok + 5)                     # + the box
        self.out = nn.Sequential(nn.Linear(4 * hidden + 7, hidden), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, len(KINDS)))

    def tokens(self, v, ta, tb, bank, bank_mask, scale):
        """v (N, T, D) unit vectors; ta, tb (N, D); bank (M, D); bank_mask (N, M), True where a neither name counts."""
        ca, cb = (v * ta[:, None]).sum(-1), (v * tb[:, None]).sum(-1)
        cn = (v @ bank.T).masked_fill(~bank_mask[:, None, :], float("-inf"))
        p = (torch.cat([ca[..., None], cb[..., None], cn], -1) * scale).softmax(-1)
        cn_max = cn.max(-1).values
        cn_max = torch.where(torch.isfinite(cn_max), cn_max, torch.zeros_like(cn_max))
        pn = p[..., 2:].max(-1).values if bank.shape[0] else torch.zeros_like(ca)
        pair = (ta * tb).sum(-1)[:, None].expand_as(ca)
        return torch.cat([torch.stack([p[..., 0], p[..., 1], pn, ca, cb, cn_max], -1), self.app(v), pair[..., None]], -1)

    def one_order(self, f, ta, tb, bank, bank_mask, scale):
        N = ta.shape[0]
        tv = self.tokens(f["views"], ta, tb, bank, bank_mask, scale)
        kind = torch.zeros(N, N_VIEWS, 2, device=tv.device)
        kind[:, 0, 0], kind[:, 1:, 1] = 1, 1
        hv = self.view_mlp(torch.cat([tv, kind], -1))
        tb_ = self.tokens(f["bodies"], ta, tb, bank, bank_mask, scale)
        hb = self.body_mlp(torch.cat([tb_, f["geom"]], -1))
        m = f["mask"][..., None]
        mean = (hb * m).sum(1) / m.sum(1).clamp_min(1)
        mx = hb.masked_fill(~m, float("-inf")).max(1).values
        mx = torch.where(torch.isfinite(mx), mx, torch.zeros_like(mx))
        count = F.one_hot(f["n_bodies"].long().clamp(0, 5), 6).float()
        pair = (ta * tb).sum(-1, keepdim=True)
        return self.out(torch.cat([hv[:, 0], hv[:, 1:].mean(1), mean, mx, count, pair], -1))

    def forward(self, f: dict, ta, tb, bank, bank_mask, scale: float) -> torch.Tensor:
        """The six kinds' logits, the same for (A, B) and (B, A)."""
        return 0.5 * (self.one_order(f, ta, tb, bank, bank_mask, scale) + self.one_order(f, tb, ta, bank, bank_mask, scale))


def bank_mask_for(rows: list[dict], bank_names: list[str], device) -> torch.Tensor:
    return torch.tensor([[n not in (r["a"], r["b"]) for n in bank_names] for r in rows], dtype=torch.bool, device=device)


# ---------------------------------------------------------------- features

def neither_bank() -> list[str]:
    return list(json.loads(Path(NEITHER_FILE).read_text())["list"])


def embed_pictures(backbone, pictures: list[str], bodies: dict, flips: int, device) -> dict:
    out = {k: [] for k in ("views", "bodies", "mask", "geom", "n_bodies", "boxes")}
    with torch.no_grad():
        for i, pic in enumerate(pictures):
            im = load_picture(pic, device)
            per = {k: [] for k in out}
            for flip in range(flips):
                img, bx = (im, bodies[pic]) if flip == 0 else (im.flip(-1), mirror(bodies[pic], im.shape[-1]))
                f = picture_features(backbone, img, bx)
                for k in ("views", "bodies", "mask", "geom"):
                    per[k].append(f[k].half().cpu() if f[k].is_floating_point() else f[k].cpu())
            for k in ("views", "bodies", "mask", "geom"):
                out[k].append(torch.stack(per[k]))
            out["n_bodies"].append(len(bodies[pic]))
            out["boxes"].append(bodies[pic])
            if (i + 1) % 100 == 0:
                print(f"  {i + 1} of {len(pictures)} pictures", flush=True)
    return {"pictures": list(pictures), "views": torch.stack(out["views"]), "bodies": torch.stack(out["bodies"]),
            "mask": torch.stack(out["mask"]), "geom": torch.stack(out["geom"]),
            "n_bodies": torch.tensor(out["n_bodies"]), "boxes": out["boxes"], "flips": flips}


def cmd_features(a) -> int:
    refuse_confirm(a.labelled_set, a.split, a.bodies)
    rows = read_labelled(a.labelled_set)
    split = json.loads(Path(a.split).read_text())
    pool, dropped = training_pool(rows, split, a.pool)
    _, test = make_lists(rows, split)
    bodies = load_bodies(a.bodies)
    pool_pics = sorted({r["picture"] for r in pool})
    test_pics = sorted({r["picture"] for r in test})
    missing = [p for p in pool_pics + test_pics if p not in bodies]
    if missing:
        raise SystemExit(f"{len(missing)} pictures have no boxes in {a.bodies}, for example {missing[:2]}")
    device = torch.device(a.device)
    backbone = make_backbone(a.backbone, device)
    bank = neither_bank()
    names = sorted({r["a"] for r in pool + test} | {r["b"] for r in pool + test} | set(bank))
    with torch.no_grad():
        text = {n: backbone.text_embed(n).cpu() for n in names}
    folder = a.run / "features"
    folder.mkdir(parents=True, exist_ok=True)
    print(f"training pool ({a.pool}): {len(pool)} rows on {len(pool_pics)} picture files; dropped {dict(dropped)}")
    print_sides({"pool": pool})
    for name, pics, flips in (("pool", pool_pics, 2), ("test", test_pics, 1)):
        t0 = time.time()
        f = embed_pictures(backbone, pics, bodies, flips, device)
        secs = time.time() - t0
        torch.save({**f, "text": text, "bank": bank, "backbone": backbone.name, "logit_scale": backbone.logit_scale,
                    "pool": a.pool, "written": now(), "seconds": round(secs, 1)}, folder / f"{name}.pt")
        print(f"{name}.pt: {len(pics)} pictures x {flips} ({'as drawn and mirrored' if flips == 2 else 'as drawn'}) in "
              f"{secs:.1f} s, {secs / max(len(pics), 1):.2f} s a picture")
    return 0


# ---------------------------------------------------------------- training

def batch_inputs(feat: dict, idx: torch.Tensor, flip: torch.Tensor, device) -> dict:
    return {"views": feat["views"][idx, flip].float().to(device), "bodies": feat["bodies"][idx, flip].float().to(device),
            "mask": feat["mask"][idx, flip].to(device), "geom": feat["geom"][idx, flip].float().to(device),
            "n_bodies": feat["n_bodies"][idx].to(device)}


def text_inputs(feat: dict, rows: list[dict], device):
    ta = torch.stack([feat["text"][r["a"]] for r in rows]).float().to(device)
    tb = torch.stack([feat["text"][r["b"]] for r in rows]).float().to(device)
    bank = torch.stack([feat["text"][n] for n in feat["bank"]]).float().to(device)
    return ta, tb, bank, bank_mask_for(rows, feat["bank"], device)


def per_kind(pred: list[str], truth: list[str]) -> dict:
    out = {}
    for k in KINDS:
        idx = [i for i, t in enumerate(truth) if t == k]
        out[k] = {"right": sum(pred[i] == k for i in idx), "of": len(idx)}
    return out


def balanced(pk: dict) -> float:
    rates = [v["right"] / v["of"] for v in pk.values() if v["of"]]
    return round(sum(rates) / len(rates), 4) if rates else 0.0


def cmd_train(a) -> int:
    refuse_confirm(a.labelled_set, a.split)
    rows = read_labelled(a.labelled_set)
    split = json.loads(Path(a.split).read_text())
    feat_path = a.run / "features" / "pool.pt"
    feat = torch.load(feat_path, map_location="cpu", weights_only=False)
    pool, dropped = training_pool(rows, split, feat["pool"])
    sealed_files = {r["picture"] for r in rows if r["part"] in SEALED_PARTS}
    leaked = [p for p in feat["pictures"] if p in sealed_files]
    if leaked:
        raise SystemExit(f"pool.pt holds {len(leaked)} read or confirmation pictures, for example {leaked[0]}; not training")
    where = {p: i for i, p in enumerate(feat["pictures"])}
    absent = [r["picture"] for r in pool if r["picture"] not in where]
    if absent:
        raise SystemExit(f"{len(absent)} training-pool pictures are not in {feat_path}; run features again")
    train, dev = dev_split(pool)
    if not train or not dev:
        raise SystemExit(f"the training pool split into {len(train)} training and {len(dev)} development pictures")
    print(f"training pool ({feat['pool']}): {len(pool)}; dropped {dict(dropped)}")
    print_sides({"train": train, "dev": dev})

    hp = {k: getattr(a, k) for k in TRAIN_DEFAULTS}
    torch.manual_seed(hp["seed"])
    random.seed(hp["seed"])
    device = torch.device(a.device)
    head = JudgeHead(feat["views"].shape[-1], hp["hidden"], hp["appearance"], hp["dropout"]).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
    n_k = Counter(r["kind"] for r in train)
    weight = torch.tensor([len(train) / (len(KINDS) * n_k[k]) if n_k[k] else 0.0 for k in KINDS], device=device)
    scale = feat["logit_scale"]

    def tensors(rs):
        idx = torch.tensor([where[r["picture"]] for r in rs])
        y = torch.tensor([KINDS.index(r["kind"]) for r in rs], device=device)
        return idx, y, text_inputs(feat, rs, device)

    tr_idx, tr_y, (tr_a, tr_b, bank, tr_m) = tensors(train)
    dv_idx, dv_y, (dv_a, dv_b, _, dv_m) = tensors(dev)
    dv_in = batch_inputs(feat, dv_idx, torch.zeros_like(dv_idx), device)
    log, best, since, secs = [], None, 0, []
    for epoch in range(1, hp["epochs"] + 1):
        t0 = time.time()
        head.train()
        order = torch.randperm(len(train))
        total = 0.0
        for s in range(0, len(train), hp["batch"]):
            j = order[s:s + hp["batch"]]
            flip = torch.randint(0, feat["flips"], (len(j),))
            logits = head(batch_inputs(feat, tr_idx[j], flip, device), tr_a[j], tr_b[j], bank, tr_m[j], scale)
            loss = F.cross_entropy(logits, tr_y[j], weight=weight)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(j)
        head.eval()
        with torch.no_grad():
            dl = head(dv_in, dv_a, dv_b, bank, dv_m, scale)
            dev_loss = float(F.cross_entropy(dl, dv_y, weight=weight)) if len(dev) else float("nan")
            pk = per_kind([KINDS[i] for i in dl.argmax(-1).tolist()], [r["kind"] for r in dev])
        secs.append(time.time() - t0)
        rec = {"epoch": epoch, "train_loss": round(total / len(train), 4), "dev_loss": round(dev_loss, 4),
               "dev_balanced": balanced(pk), "dev_right": sum(v["right"] for v in pk.values()), "dev_of": len(dev),
               "dev_per_kind": pk, "seconds": round(secs[-1], 3)}
        log.append(rec)
        if best is None or (rec["dev_balanced"], -rec["dev_loss"]) > (best["dev_balanced"], -best["dev_loss"]):
            best, since = rec, 0
            torch.save({"state": {k: v.detach().cpu() for k, v in head.state_dict().items()}, "config": head.config,
                        "kinds": list(KINDS), "bank": feat["bank"], "backbone": feat["backbone"],
                        "logit_scale": scale, "epoch": epoch, "dev": rec, "pool": feat["pool"], "hyper": hp},
                       a.run / "head.pt")
        else:
            since += 1
        if epoch % 10 == 0 or epoch == 1:
            print(f"epoch {epoch}: train loss {rec['train_loss']}, dev loss {rec['dev_loss']}, dev agreement "
                  f"{rec['dev_right']} of {rec['dev_of']}, averaged over kinds {rec['dev_balanced']} "
                  f"({secs[-1]:.2f} s)", flush=True)
        if since >= hp["patience"]:
            print(f"stopped at epoch {epoch}: no better development epoch in {hp['patience']}")
            break
    spe = sum(secs) / len(secs)
    print(f"kept epoch {best['epoch']}: development agreement {best['dev_right']} of {best['dev_of']}, averaged over "
          f"kinds {best['dev_balanced']}; {spe:.3f} seconds an epoch over {len(secs)} epochs")
    for k, v in best["dev_per_kind"].items():
        print(f"  dev {k:16s} {v['right']} of {v['of']}")
    (a.run / "train-log.json").write_text(json.dumps({"hyper": hp, "seconds_per_epoch": round(spe, 4),
                                                      "kept_epoch": best["epoch"], "epochs": log}, indent=1))
    (a.run / "split-counts.json").write_text(json.dumps({
        "pool": feat["pool"], "dropped": dict(dropped), "dev_share": DEV_SHARE, "dev_seed": DEV_SEED,
        "dev_unit": "pair and seed", "train": side_counts(train), "dev": side_counts(dev),
        "train_pairs": sorted({r["pair"] for r in train}), "dev_groups": sorted({f"{r['pair']} seed {r['seed']}" for r in dev})},
        indent=1))
    chosen = {"checkpoint": "head.pt", "head_sha256": sha256(a.run / "head.pt"), "epoch": best["epoch"],
              "pool": feat["pool"], "backbone": feat["backbone"]}
    (a.run / "settings-chosen.json").write_text(json.dumps({
        "judge": "trained", "chosen": chosen, "agreement": best["dev_balanced"],
        "rule": "the epoch with the highest development agreement averaged over kinds (ties to the lower development "
                "loss); the development part is a tenth of the training pool by pair and seed; no read or confirmation "
                "picture is opened", "written": now()}, indent=1))
    print(f"wrote {a.run / 'head.pt'}, train-log.json, split-counts.json and settings-chosen.json")
    return 0


# ---------------------------------------------------------------- the sealed read

def cmd_score(a) -> int:
    refuse_confirm(a.labelled_set, a.split)
    out = a.run / "test"
    if (out / "calls.json").exists():
        raise SystemExit(f"{out / 'calls.json'} exists: the sealed read runs once, and it has run")
    chosen = json.loads((a.run / "settings-chosen.json").read_text())
    if sha256(a.run / "head.pt") != chosen["chosen"]["head_sha256"]:
        raise SystemExit("head.pt is not the head settings-chosen.json names; not scoring")
    ckpt = torch.load(a.run / "head.pt", map_location="cpu", weights_only=False)
    feat = torch.load(a.run / "features" / "test.pt", map_location="cpu", weights_only=False)
    if feat["bank"] != ckpt["bank"] or feat["backbone"] != ckpt["backbone"]:
        raise SystemExit("test.pt was embedded with another backbone or neither list than the head was trained on")
    rows = read_labelled(a.labelled_set)
    _, test = make_lists(rows, json.loads(Path(a.split).read_text()))
    device = torch.device(a.device)
    head = JudgeHead(**ckpt["config"]).to(device)
    head.load_state_dict(ckpt["state"])
    head.eval()
    where = {p: i for i, p in enumerate(feat["pictures"])}
    t0 = time.time()
    with torch.no_grad():
        idx = torch.tensor([where[r["picture"]] for r in test])
        ta, tb, bank, m = text_inputs(feat, test, device)
        probs = head(batch_inputs(feat, idx, torch.zeros_like(idx), device), ta, tb, bank, m,
                     ckpt["logit_scale"]).softmax(-1).cpu()
    pictures = []
    for r, p, i in zip(test, probs.tolist(), idx.tolist()):
        boxes = feat["boxes"][i]
        pictures.append({"picture": r["picture"], "a": r["a"], "b": r["b"], "kind": KINDS[max(range(len(KINDS)), key=p.__getitem__)],
                         "two_clear_score": round(p[KINDS.index(TWO_CLEAR)], 6),
                         "probs": {k: round(v, 6) for k, v in zip(KINDS, p)}, "n_bodies": len(boxes),
                         "bodies": [{"box": b, "call": "whole picture", "flag": False} for b in boxes]})
    settings = dict(chosen["chosen"])
    out.mkdir(parents=True, exist_ok=True)
    stamp = now()
    (out / "settings.json").write_text(json.dumps({"judge": "trained", "settings": settings, "dry_run": False,
                                                   "written": stamp}, indent=1))
    (out / "calls.json").write_text(json.dumps({"judge": "trained", "settings": settings, "dry_run": False,
                                                "seconds": round(time.time() - t0, 2), "written": stamp,
                                                "pictures": pictures}, indent=1))
    print(f"wrote {out / 'calls.json'}: {len(pictures)} pictures ({dict(Counter(r['part'] for r in test))}); "
          f"score them with paper2_score.py --judge trained={a.run}")
    return 0


# ---------------------------------------------------------------- the dial trainer's reward

class TrainedJudgeReward:
    """paper2_identity_dial's reward call: (image (3, H, W) in [0, 1], a, b) -> (two-clear probability with its graph,
    info). The boxes come from the body counter on the detached picture (or `boxes_fn`) and are held fixed."""

    def __init__(self, head_path: Path, device, backbone=None, boxes_fn=None):
        ckpt = torch.load(head_path, map_location=device, weights_only=False)
        self.head = JudgeHead(**ckpt["config"]).to(device)
        self.head.load_state_dict(ckpt["state"])
        self.head.eval()
        for p in self.head.parameters():
            p.requires_grad_(False)
        self.backbone = backbone if backbone is not None else make_backbone(
            "stub" if ckpt["backbone"] == "stub" else "clip", device)
        if self.backbone.name != ckpt["backbone"]:
            raise SystemExit(f"the head was trained on {ckpt['backbone']} features, not {self.backbone.name}")
        self.scale, self.names, self.device = ckpt["logit_scale"], ckpt["bank"], device
        with torch.no_grad():
            self.bank = torch.stack([self.backbone.text_embed(n) for n in self.names]).float().to(device)
        if boxes_fn is None:
            from paper2_identity_dial import BodyBoxes
            boxes_fn = BodyBoxes(device)
        self.boxes = boxes_fn

    def probs(self, image: torch.Tensor, a: str, b: str, boxes: list[list[int]]) -> torch.Tensor:
        f = picture_features(self.backbone, image, boxes)
        f = {k: (v[None] if torch.is_tensor(v) else torch.tensor([v], device=image.device)) for k, v in f.items()}
        with torch.no_grad():
            ta, tb = self.backbone.text_embed(a).float()[None], self.backbone.text_embed(b).float()[None]
        m = bank_mask_for([{"a": a, "b": b}], self.names, image.device)
        return self.head(f, ta, tb, self.bank, m, self.scale).softmax(-1)[0]

    def __call__(self, image: torch.Tensor, a: str, b: str):
        boxes = self.boxes(image.detach())
        p = self.probs(image, a, b, boxes)
        return p[KINDS.index(TWO_CLEAR)], {"kind": KINDS[int(p.argmax())], "n_bodies": len(boxes),
                                           "probs": {k: float(v) for k, v in zip(KINDS, p.detach())}}


# ---------------------------------------------------------------- command line

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("features", "train", "score"):
        s = sub.add_parser(name)
        s.add_argument("--labelled-set", type=Path, required=True)
        s.add_argument("--split", type=Path, required=True)
        s.add_argument("--run", type=Path, required=True, help="the run folder: features/, head.pt, test/")
        s.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
        if name == "features":
            s.add_argument("--bodies", type=Path, required=True, help="paper2-find-bodies-split's bodies.json")
            s.add_argument("--pool", choices=POOLS, default="unsealed")
            s.add_argument("--backbone", choices=("clip", "stub"), default="clip")
        if name == "train":
            for k, v in TRAIN_DEFAULTS.items():
                s.add_argument(f"--{k.replace('_', '-')}", type=type(v), default=v)
    a = ap.parse_args(argv)
    a.run.mkdir(parents=True, exist_ok=True)
    return {"features": cmd_features, "train": cmd_train, "score": cmd_score}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
