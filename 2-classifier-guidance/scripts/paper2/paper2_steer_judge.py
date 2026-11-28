"""A steering judge for the identity dial: does a picture show its two asked animals as two clear bodies, or not?

Run name `paper2-judge-steer`. The dial's reward is "two clear" against everything else, and the trained species head
(paper2_trained_judge.py) disagreed with the blind reviewers on 53% of the test sheet's pictures. This script builds
candidate judges for that one yes-or-no decision and measures each one on pairs it never trained on.

Three subcommands, in this order:

  table     every blind-settled picture available, one row each, as <run>/table.csv: the label set's settled rows
            (paper2-label-set/labelled-set.csv), the dial's pick sheet and its test sheet (settled the way
            paper2_first_read.py settles them: the first two reviewers where they agree, the third where they split).
            The label set's sealed confirmation rows are listed with sealed=True and never embedded, trained on or
            scored. Prints the counts per kind and source. CPU only
  features  GPU. Finds each picture's bodies (the existing bodies.json files, and the project's body counter on any
            picture none of them covers), cuts the whole picture and up to MAX_CROPS body crops (10% margin, other
            bodies painted out, paper2_common.crop_body, padded square in the mean colour), and embeds every view with
            each frozen backbone in BACKBONES. Text-and-image backbones also embed every asked name and the 43
            "neither" names (four phrasings, as the CLIP judge does). Writes <run>/features/<backbone>.pt,
            boxes.json and timing.json (seconds per picture for the counter and for each backbone)
  fit       CPU. Builds each candidate's features, runs the split rule below, and writes <run>/fit/report.json,
            report.txt and predictions.csv

The split rule, fixed here before any score: pictures are grouped by the species pair their picture was drawn for
(unordered, so cat|fox and fox|cat are one group; a renamed picture stays with the pair it was drawn for). The groups
are dealt into N_FOLDS folds, largest group first, each to the fold holding the fewest pictures so far (ties to the
lower fold number). Each fold in turn is held out; the judge is trained on the other folds only, and its settings (the
regularisation strength and the yes-or-no threshold) are chosen by an inner grouped cross-validation inside those
training folds. So every picture gets exactly one call, from a judge that never saw its pair.

The bar, also fixed here: a judge is usable as a steering signal when it disagrees with the blind calls on two clear or
not for at most BAR_MAX_DISAGREE of the held-out pictures. Each kind's agreement is reported beside it: for two clear,
the share called two clear; for every other kind, the share called not two clear.

    PY=/home-mscluster/mmolefe/miniforge3/envs/co3_bw/bin/python
    $PY paper2_steer_judge.py table    --run /datasets/mmolefe/poe_repair_min/paper2/paper2-judge-steer
    $PY paper2_steer_judge.py features --run <the same>     (inside an sbatch job on bigbatch)
    $PY paper2_steer_judge.py fit      --run <the same>
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from paper2_common import NEITHER_FILE, crop_body, species_from_slug, with_article  # noqa: E402
from paper2_first_read import read_calls, settle  # noqa: E402
from paper2_kinds import ALL_KINDS, TWO_CLEAR  # noqa: E402

P2 = Path("/datasets/mmolefe/poe_repair_min/paper2")
LABEL_SET = P2 / "paper2-label-set/labelled-set.csv"
CALLS = HERE.parents[1] / "data/paper2/first-result"
SHEETS = {  # source: (key.json, first two reviewers' calls, the third reviewer's calls, the old head's bodies.json)
    "pick": (P2 / "paper2-first-identity-pick/blind/key/key.json",
             [CALLS / "pick-calls-1.csv", CALLS / "pick-calls-2.csv"], CALLS / "pick-calls-3.csv"),
    "test": (P2 / "paper2-first-identity-test/key.json",
             [CALLS / "calls-1.csv", CALLS / "calls-2.csv"], CALLS / "calls-3.csv"),
}
BODY_FILES = [P2 / "paper2-find-bodies-split/bodies.json", P2 / "paper2-first-identity-pick/trained/bodies.json",
              P2 / "paper2-first-identity-test/trained/bodies.json"]
SEALED_PARTS = ("confirmation",)

# ---------------------------------------------------------------- the split rule and the bar, fixed before scoring
N_FOLDS = 5
INNER_FOLDS = 4
C_GRID = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0)
BAR_MAX_DISAGREE = 0.10          # at most 1 in 10 held-out pictures called on the wrong side of two clear or not

MAX_CROPS = 4
SIZE = 224
PHRASINGS = ("a photo of {}", "a close-up photo of {}", "a drawing of {}", "an engraving of {}")
BIGG_WEIGHTS = Path("/home-mscluster/mmolefe/.cache/huggingface/hub/models--laion--CLIP-ViT-bigG-14-laion2B-39B-b160k"
                    "/snapshots")
BACKBONES = ("clip_l", "dinov2_l", "bioclip", "clip_bigg")
TEXT_BACKBONES = ("clip_l", "bioclip", "clip_bigg")

TABLE_COLUMNS = ["item", "picture", "a", "b", "pair", "group", "seed", "kind", "two_clear", "source", "part", "row",
                 "rules_version", "renamed", "settled_by", "sealed"]


def norm_kind(k: str) -> str:
    return "_".join(str(k or "").strip().lower().replace("-", " ").replace("_", " ").split())


def group_of(x: str, y: str) -> str:
    return "|".join(sorted((x.strip(), y.strip())))


# ---------------------------------------------------------------- table

def build_table() -> list[dict]:
    rows = []
    with open(LABEL_SET, newline="") as f:
        for i, r in enumerate(csv.DictReader(f)):
            pa, pb = species_from_slug(r["pair"])
            kind = norm_kind(r["kind"])
            sealed = r["part"] in SEALED_PARTS     # a sealed row carries no kind in this file
            assert sealed or kind in ALL_KINDS, kind
            rows.append({"item": f"label-{i:04d}", "picture": r["picture"], "a": r["a"], "b": r["b"],
                         "pair": f"{pa}|{pb}", "group": group_of(pa, pb), "seed": r["seed"], "kind": kind,
                         "two_clear": int(kind == TWO_CLEAR), "source": "label set", "part": r["part"],
                         "row": r["source"], "rules_version": r["rules_version"], "renamed": r["renamed"],
                         "settled_by": "both reviewers" if r["agreed"] == "True" else "the third reviewer",
                         "sealed": sealed})
    for source, (key_path, calls, third) in SHEETS.items():
        key = json.loads(key_path.read_text())["pictures"]
        first, second = read_calls(calls[0]), read_calls(calls[1])
        settled, by, unresolved = settle(key, first, second, read_calls(third))
        if unresolved:
            raise SystemExit(f"{source}: {len(unresolved)} pictures have no settled kind: {unresolved[:5]}")
        for name, e in key.items():
            a, b = e["pair"].split("|")
            rows.append({"item": f"{source}-{name}", "picture": e["picture_path"], "a": e["a"], "b": e["b"],
                         "pair": e["pair"], "group": group_of(a, b), "seed": e["seed"], "kind": settled[name],
                         "two_clear": int(settled[name] == TWO_CLEAR), "source": source + " sheet",
                         "part": e.get("set", ""), "row": e["row"], "rules_version": "4", "renamed": "False",
                         "settled_by": by[name], "sealed": False})
    missing = [r["picture"] for r in rows if not Path(r["picture"]).exists()]
    if missing:
        raise SystemExit(f"{len(missing)} pictures are missing on disk, e.g. {missing[:3]}")
    return rows


def read_table(run: Path, unsealed_only: bool = True) -> list[dict]:
    with open(run / "table.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["two_clear"] = int(r["two_clear"])
        r["sealed"] = r["sealed"] == "True"
    return [r for r in rows if not (unsealed_only and r["sealed"])]


def cmd_table(a) -> int:
    rows = build_table()
    a.run.mkdir(parents=True, exist_ok=True)
    with open(a.run / "table.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=TABLE_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    live = [r for r in rows if not r["sealed"]]
    sources = sorted({r["source"] for r in live})
    print(f"wrote {a.run / 'table.csv'}: {len(rows)} rows, {len(rows) - len(live)} sealed (never used), "
          f"{len(live)} used, {len({r['picture'] for r in live})} distinct picture files, "
          f"{len({r['group'] for r in live})} pair groups")
    print(f"{'kind':16s} " + " ".join(f"{s:>14s}" for s in sources) + f" {'all':>6s}")
    for k in ALL_KINDS:
        print(f"{k:16s} " + " ".join(f"{sum(r['kind'] == k and r['source'] == s for r in live):14d}" for s in sources)
              + f" {sum(r['kind'] == k for r in live):6d}")
    print(f"{'total':16s} " + " ".join(f"{sum(r['source'] == s for r in live):14d}" for s in sources)
          + f" {len(live):6d}")
    return 0


# ---------------------------------------------------------------- features (GPU)

def pad_square(im):
    from PIL import Image, ImageStat
    w, h = im.size
    s = max(w, h)
    mean = tuple(int(v) for v in ImageStat.Stat(im).mean[:3])
    out = Image.new("RGB", (s, s), mean)
    out.paste(im, ((s - w) // 2, (s - h) // 2))
    return out


def load_backbone(name: str, device):
    """(image_fn, text_fn or None, logit_scale). image_fn takes (N, 3, SIZE, SIZE) floats in [0, 1]."""
    import torch
    import torch.nn.functional as F
    if name == "clip_l":
        from transformers import CLIPModel, CLIPTokenizer
        m = CLIPModel.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=torch.float16).to(device).eval()
        tok = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        mean, std = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)

        def text(prompts):
            t = tok(prompts, padding=True, return_tensors="pt").to(device)
            return m.get_text_features(**t).float()

        def image(x):
            return m.get_image_features(pixel_values=x.half()).float()
        scale = float(m.logit_scale.exp())
    elif name == "dinov2_l":
        from transformers import AutoModel
        m = AutoModel.from_pretrained("facebook/dinov2-large", torch_dtype=torch.float16).to(device).eval()
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        text, scale = None, 1.0

        def image(x):
            h = m(pixel_values=x.half()).last_hidden_state.float()
            return torch.cat([h[:, 0], h[:, 1:].mean(1)], dim=-1)
    else:
        import open_clip
        if name == "bioclip":
            m, _, pre = open_clip.create_model_and_transforms("hf-hub:imageomics/bioclip")
            tok = open_clip.get_tokenizer("hf-hub:imageomics/bioclip")
        else:
            weights = sorted(BIGG_WEIGHTS.glob("*/open_clip_model.safetensors"))[0]
            m, _, pre = open_clip.create_model_and_transforms("ViT-bigG-14", pretrained=str(weights))
            tok = open_clip.get_tokenizer("ViT-bigG-14")
        m = m.half().to(device).eval()
        norm = [t for t in pre.transforms if t.__class__.__name__ == "Normalize"][0]
        mean, std = tuple(norm.mean), tuple(norm.std)

        def text(prompts):
            return m.encode_text(tok(prompts).to(device)).float()

        def image(x):
            return m.encode_image(x.half()).float()
        scale = float(m.logit_scale.exp())
    mu = torch.tensor(mean, device=device).view(1, 3, 1, 1)
    sd = torch.tensor(std, device=device).view(1, 3, 1, 1)

    def image_fn(x):
        return image((x - mu) / sd)

    def text_fn(name_):
        e = F.normalize(text([p.format(with_article(name_)) for p in PHRASINGS]), dim=-1).mean(0)
        return F.normalize(e, dim=-1)
    return image_fn, (text_fn if text is not None else None), scale


def cmd_features(a) -> int:
    import numpy as np
    import torch
    from PIL import Image
    rows = read_table(a.run)
    pictures = sorted({r["picture"] for r in rows})[: a.limit]
    names = sorted({r["a"] for r in rows} | {r["b"] for r in rows} | set(json.loads(NEITHER_FILE.read_text())["list"]))
    out = a.out or a.run / "features"
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device(a.device)
    sync = torch.cuda.synchronize if dev.type == "cuda" else (lambda: None)
    timing = {}

    # bodies: the existing counter output where it covers a picture, the counter itself elsewhere
    boxes = {}
    for bf in BODY_FILES:
        for pic, v in json.loads(bf.read_text())["pictures"].items():
            bx = [[int(round(c)) for c in (d["box"] if isinstance(d, dict) else d)] for d in v["boxes"]]
            boxes.setdefault(pic, {"n": v.get("n", len(bx)), "boxes": bx, "from": str(bf)})
    todo = [p for p in pictures if p not in boxes]
    sys.path.insert(0, str(HERE.parents[2]))
    from poe_repair.experiments.compose_scorer_validation import detection_scorer as ds
    sample = pictures[:: max(1, len(pictures) // 32)][:32]
    ds.count_instances(Path(sample[0]), query="animal", device=dev)   # warm-up, loads the detector
    t0 = time.time()
    for p in sample:
        ds.count_instances(Path(p), query="animal", device=dev)
    sync()
    timing["body_counter"] = round((time.time() - t0) / len(sample), 4)
    for k, p in enumerate(todo):
        n, found = ds.count_instances(Path(p), query="animal", device=dev)
        boxes[p] = {"n": n, "boxes": [[int(round(c)) for c in d["box"]] for d in found], "from": "counted here"}
    print(f"bodies: {len(pictures) - len(todo)} from earlier counter runs, {len(todo)} counted here; counter "
          f"{timing['body_counter']} s a picture", flush=True)
    (out / "boxes.json").write_text(json.dumps({p: boxes[p] for p in pictures}, indent=0))
    torch.cuda.empty_cache() if dev.type == 'cuda' else None

    # views: the whole picture and up to MAX_CROPS largest bodies, each SIZE x SIZE uint8
    t0 = time.time()
    views, n_views = np.zeros((len(pictures), 1 + MAX_CROPS, SIZE, SIZE, 3), np.uint8), []
    for i, p in enumerate(pictures):
        im = Image.open(p).convert("RGB")
        bx = boxes[p]["boxes"]
        order = sorted(range(len(bx)), key=lambda j: -(bx[j][2] - bx[j][0]) * (bx[j][3] - bx[j][1]))[:MAX_CROPS]
        vs = [pad_square(im)] + [pad_square(crop_body(im, bx, j)[0]) for j in order]
        for j, v in enumerate(vs):
            views[i, j] = np.asarray(v.resize((SIZE, SIZE), Image.BICUBIC))
        n_views.append(len(vs))
        boxes[p]["order"] = order
    timing["cut_views_cpu"] = round((time.time() - t0) / len(pictures), 4)
    (out / "boxes.json").write_text(json.dumps({p: boxes[p] for p in pictures}, indent=0))
    print(f"views cut: {timing['cut_views_cpu']} s a picture on the CPU", flush=True)

    for bb in (a.backbones or BACKBONES):
        image_fn, text_fn, scale = load_backbone(bb, dev)
        embs, warm, t0 = [], min(8, len(pictures) // 2), time.time()
        with torch.no_grad():
            for i in range(len(pictures)):            # one picture at a time, as the dial calls it
                if i == warm:
                    sync()
                    t0 = time.time()
                x = torch.from_numpy(views[i, :n_views[i]]).to(dev).permute(0, 3, 1, 2).float() / 255
                embs.append(image_fn(x).cpu())
            sync()
            per = (time.time() - t0) / (len(pictures) - warm)
            text = {n: text_fn(n).cpu() for n in names} if text_fn else None
        timing[bb] = round(per, 4)
        torch.save({"pictures": pictures, "embeddings": embs, "text": text, "logit_scale": scale},
                   out / f"{bb}.pt")
        print(f"{bb}: {len(pictures)} pictures embedded, {per:.4f} s a picture (whole + crops)", flush=True)
        write_timing(out, timing, dev)
        del image_fn, text_fn
        torch.cuda.empty_cache() if dev.type == 'cuda' else None
    write_timing(out, timing, dev)
    return 0


def write_timing(out: Path, timing: dict, dev) -> None:
    """Merge this run's seconds per picture into timing.json, after each backbone, so a later failure keeps them."""
    import torch
    old = json.loads((out / "timing.json").read_text()) if (out / "timing.json").exists() else {}
    dev_name = torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu"
    (out / "timing.json").write_text(json.dumps({**old, **timing, "device": dev_name}, indent=1))


# ---------------------------------------------------------------- fit (CPU)

def geometry(entry: dict) -> list[float]:
    """Body count and layout: count one-hots, the three largest boxes' area shares, the top two's overlap and gap."""
    n, bx = entry["n"], entry["boxes"]
    order = entry["order"]
    W = H = 1024.0
    areas = [(bx[j][2] - bx[j][0]) * (bx[j][3] - bx[j][1]) / (W * H) for j in order]
    f = [float(n), *(float(n == k) for k in range(4)), float(n >= 4)]
    f += [areas[k] if k < len(areas) else 0.0 for k in range(3)]
    if len(order) >= 2:
        b0, b1 = bx[order[0]], bx[order[1]]
        ix = max(0, min(b0[2], b1[2]) - max(b0[0], b1[0])) * max(0, min(b0[3], b1[3]) - max(b0[1], b1[1]))
        u = (b0[2] - b0[0]) * (b0[3] - b0[1]) + (b1[2] - b1[0]) * (b1[3] - b1[1]) - ix
        c0 = ((b0[0] + b0[2]) / 2, (b0[1] + b0[3]) / 2)
        c1 = ((b1[0] + b1[2]) / 2, (b1[1] + b1[3]) / 2)
        f += [ix / max(u, 1), math.dist(c0, c1) / W]
    else:
        f += [0.0, 0.0]
    return f


def name_shares(E, text: dict, scale: float, a: str, b: str, neither: list[str]):
    """Per view: (larger asked share, smaller asked share, neither share, cos A, cos B), symmetric in A and B; plus
    the best two-body assignment and the weakest asked body."""
    import torch
    import torch.nn.functional as F
    nn_ = [n for n in neither if n not in (a, b)]
    T = torch.stack([text[a], text[b]] + [text[n] for n in nn_])
    V = F.normalize(E, dim=-1)
    cos = V @ T.T
    p = (scale * cos).softmax(-1)
    pa, pb, pn = p[:, 0], p[:, 1], p[:, 2:].sum(-1)
    per = torch.stack([torch.maximum(pa, pb), torch.minimum(pa, pb), pn,
                       torch.maximum(cos[:, 0], cos[:, 1]), torch.minimum(cos[:, 0], cos[:, 1])], -1)
    whole = per[0].tolist() + [float(text[a] @ text[b])]
    crops = []
    for k in range(3):
        crops += per[1 + k].tolist() + [1.0] if 1 + k < len(per) else [0.0] * 6
    cpa, cpb, cpn = pa[1:], pb[1:], pn[1:]
    if len(cpa) >= 2:
        pairm = (cpa[:, None] * cpb[None, :]).fill_diagonal_(0)
        best = float(pairm.max().sqrt())
        weakest = float((cpa + cpb).min())
        n_other = float((cpn > torch.maximum(cpa, cpb)).sum())
        n_a, n_b = float((cpa > 0.5).sum()), float((cpb > 0.5).sum())
    elif len(cpa) == 1:
        best, weakest, n_other, n_a, n_b = 0.0, float(cpa[0] + cpb[0]), float(cpn[0] > max(cpa[0], cpb[0])), \
            float(cpa[0] > 0.5), float(cpb[0] > 0.5)
    else:
        best, weakest, n_other, n_a, n_b = 0.0, 0.0, 0.0, 0.0, 0.0
    return whole + crops + [best, weakest, n_other, max(n_a, n_b), min(n_a, n_b)]


def pooled(E):
    """Whole-picture embedding, then the crops' mean and elementwise max (zeros when no body was found)."""
    import torch
    import torch.nn.functional as F
    V = F.normalize(E, dim=-1)
    if len(V) > 1:
        c = V[1:]
        return torch.cat([V[0], c.mean(0), c.max(0).values]).tolist()
    return torch.cat([V[0], torch.zeros_like(V[0]), torch.zeros_like(V[0])]).tolist()


CANDIDATES = {
    # name: (feature blocks, model). Blocks: "<backbone>:whole" whole embedding only, "<backbone>:pooled" whole plus
    # crop mean and max, "<backbone>:shares" the name-share scalars, "geometry" the body layout
    "clip_l_whole": (["clip_l:whole", "clip_l:shares_whole"], "logistic"),
    "clip_l_crops": (["clip_l:pooled", "clip_l:shares", "geometry"], "logistic"),
    "dinov2_crops": (["dinov2_l:pooled", "geometry"], "logistic"),
    "dinov2_crops_clip_names": (["dinov2_l:pooled", "clip_l:shares", "geometry"], "logistic"),
    "bioclip_crops": (["bioclip:pooled", "bioclip:shares", "geometry"], "logistic"),
    "bigg_crops": (["clip_bigg:pooled", "clip_bigg:shares", "geometry"], "logistic"),
    "shares_only_boosted": (["clip_l:shares", "bioclip:shares", "clip_bigg:shares", "geometry"], "boosted"),
    "all_backbones": (["clip_l:pooled", "dinov2_l:pooled", "clip_bigg:pooled", "clip_l:shares",
                       "clip_bigg:shares", "bioclip:shares", "geometry"], "logistic"),
    # Qwen3-VL-8B's probability of yes to "two clear?" under rules version 4 (paper2_steer_vlm.py); only the threshold
    # is learned, on the training folds
    "qwen3_vl_rules_v4": (["vlm"], "logistic"),
    "all_backbones_plus_vlm": (["clip_l:pooled", "dinov2_l:pooled", "clip_bigg:pooled", "clip_l:shares",
                                "clip_bigg:shares", "bioclip:shares", "geometry", "vlm"], "logistic"),
}


def build_blocks(run: Path, rows: list[dict], needed: set[str]):
    import numpy as np
    import torch
    boxes = json.loads((run / "features" / "boxes.json").read_text())
    neither = json.loads(NEITHER_FILE.read_text())["list"]
    blocks = {}
    if "geometry" in needed:
        blocks["geometry"] = np.array([geometry(boxes[r["picture"]]) for r in rows], np.float32)
    if "vlm" in needed:
        ans = json.loads((run / "vlm" / "answers.json").read_text())
        p = np.clip(np.array([ans[r["item"]] for r in rows], np.float64), 1e-6, 1 - 1e-6)
        blocks["vlm"] = np.stack([p, np.log(p / (1 - p))], 1).astype(np.float32)
    for bb in {n.split(":")[0] for n in needed if ":" in n}:
        d = torch.load(run / "features" / f"{bb}.pt")
        idx = {p: i for i, p in enumerate(d["pictures"])}
        E = [d["embeddings"][idx[r["picture"]]] for r in rows]
        if f"{bb}:whole" in needed:
            import torch.nn.functional as F
            blocks[f"{bb}:whole"] = np.array([F.normalize(e[0], dim=-1).tolist() for e in E], np.float32)
        if f"{bb}:pooled" in needed:
            blocks[f"{bb}:pooled"] = np.array([pooled(e) for e in E], np.float32)
        if f"{bb}:shares" in needed or f"{bb}:shares_whole" in needed:
            s = np.array([name_shares(e, d["text"], d["logit_scale"], r["a"], r["b"], neither)
                          for e, r in zip(E, rows)], np.float32)
            blocks[f"{bb}:shares"] = s
            blocks[f"{bb}:shares_whole"] = s[:, :6]
    return blocks


def make_model(kind: str, c: float, multiclass: bool = False):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if kind == "boosted":                        # c is the learning rate here
        return HistGradientBoostingClassifier(max_depth=3, learning_rate=c, max_iter=200, class_weight="balanced",
                                              random_state=0)
    return make_pipeline(StandardScaler(), LogisticRegression(C=c, class_weight="balanced", max_iter=3000))


def deal_folds(groups: list[str], k: int) -> dict[str, int]:
    """The split rule: largest group first, each to the fold with the fewest pictures so far, ties to the lower fold."""
    size = Counter(groups)
    load, fold = [0] * k, {}
    for g in sorted(size, key=lambda g: (-size[g], g)):
        f = min(range(k), key=lambda i: (load[i], i))
        fold[g] = f
        load[f] += size[g]
    return fold


def best_threshold(y, p) -> float:
    import numpy as np
    grid = np.unique(np.round(p, 4))
    accs = [((p >= t).astype(int) == y).mean() for t in grid]
    return float(grid[int(np.argmax(accs))])


def fit_fold(X, y, groups, model: str):
    """Choose the setting and threshold by inner grouped cross-validation on the training folds, then refit."""
    import numpy as np
    grid = (0.03, 0.1) if model == "boosted" else C_GRID
    inner = deal_folds(list(groups), INNER_FOLDS)
    fi = np.array([inner[g] for g in groups])
    best = None
    for c in grid:
        oof = np.zeros(len(y))
        for f in range(INNER_FOLDS):
            tr, te = fi != f, fi == f
            if te.sum() == 0 or len(set(y[tr])) < 2:
                continue
            oof[te] = make_model(model, c).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        t = best_threshold(y, oof)
        acc = ((oof >= t).astype(int) == y).mean()
        if best is None or acc > best[0]:
            best = (acc, c, t)
    _, c, t = best
    return make_model(model, c).fit(X, y), c, t


def wilson(k: int, n: int) -> list[float]:
    if n == 0:
        return [0.0, 1.0]
    z, p = 1.96, k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 4), round(c + h, 4)]


def summarise(rows, calls) -> dict:
    wrong = sum(int(c) != r["two_clear"] for r, c in zip(rows, calls))
    per_kind = {}
    for k in ALL_KINDS:
        idx = [i for i, r in enumerate(rows) if r["kind"] == k]
        ok = sum(int(calls[i]) == rows[i]["two_clear"] for i in idx)
        per_kind[k] = {"agree": ok, "of": len(idx), "share": round(ok / len(idx), 4) if idx else None}
    per_source = {}
    for s in sorted({r["source"] for r in rows}):
        idx = [i for i, r in enumerate(rows) if r["source"] == s]
        w = sum(int(calls[i]) != rows[i]["two_clear"] for i in idx)
        per_source[s] = {"disagree": w, "of": len(idx), "share": round(w / len(idx), 4)}
    tp = sum(int(c) == 1 and r["two_clear"] == 1 for r, c in zip(rows, calls))
    fp = sum(int(c) == 1 and r["two_clear"] == 0 for r, c in zip(rows, calls))
    return {"disagree": wrong, "of": len(rows), "share": round(wrong / len(rows), 4),
            "interval_95": wilson(wrong, len(rows)), "passes_bar": wrong / len(rows) <= BAR_MAX_DISAGREE,
            "two_clear_precision": round(tp / max(tp + fp, 1), 4), "per_kind": per_kind, "per_source": per_source}


def cmd_fit(a) -> int:
    import numpy as np
    rows = read_table(a.run)
    y = np.array([r["two_clear"] for r in rows])
    groups = np.array([r["group"] for r in rows])
    fold_of = deal_folds(list(groups), N_FOLDS)
    folds = np.array([fold_of[g] for g in groups])
    out = a.run / "fit"
    out.mkdir(parents=True, exist_ok=True)
    have_vlm = (a.run / "vlm" / "answers.json").exists()
    names = a.candidates or [n for n in CANDIDATES if have_vlm or "vlm" not in CANDIDATES[n][0]]
    blocks = build_blocks(a.run, rows, {b for n in names for b in CANDIDATES[n][0]})
    timing = json.loads((a.run / "features" / "timing.json").read_text())
    if have_vlm:
        timing["vlm"] = json.loads((a.run / "vlm" / "timing.json").read_text())["seconds_per_picture"]
    kinds = np.array([r["kind"] for r in rows])
    report = {"split_rule": {"folds": N_FOLDS, "inner_folds": INNER_FOLDS, "fold_of_group": fold_of,
                             "pictures_per_fold": Counter(int(f) for f in folds)},
              "bar": {"max_disagree": BAR_MAX_DISAGREE},
              "base_rate_two_clear": round(float(y.mean()), 4),
              "always_not_two_clear": summarise(rows, [0] * len(rows)), "candidates": {}}
    preds = {}
    for name in names:
        feats, model = CANDIDATES[name]
        X = np.concatenate([blocks[b] for b in feats], axis=1)
        prob, call, chosen = np.zeros(len(y)), np.zeros(len(y), int), []
        mc_call = np.empty(len(y), dtype=object)
        for f in range(N_FOLDS):
            tr, te = folds != f, folds == f
            m, c, t = fit_fold(X[tr], y[tr], groups[tr], model)
            prob[te] = m.predict_proba(X[te])[:, 1]
            call[te] = (prob[te] >= t).astype(int)
            chosen.append({"fold": f, "setting": c, "threshold": round(t, 4)})
            if model == "logistic":                  # the six-kinds-and-other call, same features and setting
                mc = make_model(model, c).fit(X[tr], kinds[tr])
                mc_call[te] = mc.predict(X[te])
        s = summarise(rows, call)
        s.update({"features": feats, "model": model, "dims": int(X.shape[1]), "chosen_per_fold": chosen})
        bbs = {b.split(":")[0] for b in feats if ":" in b}
        uses_bodies = any(b == "geometry" or b.endswith(":pooled") or b.endswith(":shares") for b in feats)
        # the backbone times are for the whole picture and its crops, so a whole-picture judge is overstated
        s["seconds_per_picture"] = round(timing["body_counter"] * uses_bodies + sum(timing[b] for b in bbs)
                                         + timing.get("vlm", 0.0) * ("vlm" in feats), 4)
        if model == "logistic":
            s["seven_way"] = {k: {"agree": int(((mc_call == k) & (kinds == k)).sum()), "of": int((kinds == k).sum())}
                              for k in ALL_KINDS}
            s["seven_way_overall"] = round(float((mc_call == kinds).mean()), 4)
        report["candidates"][name] = s
        preds[name] = prob
        print(f"{name:26s} disagree {s['disagree']:4d}/{s['of']} = {s['share']:.3f} "
              f"(95% {s['interval_95'][0]:.3f}-{s['interval_95'][1]:.3f})  bar {'MET' if s['passes_bar'] else 'missed'}  "
              f"test sheet {s['per_source'].get('test sheet', {}).get('share')}  "
              f"{s['seconds_per_picture']} s/picture", flush=True)
    (out / "report.json").write_text(json.dumps(report, indent=1, default=str))
    with open(out / "predictions.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["item", "picture", "group", "fold", "kind", "two_clear"] + [f"p_{n}" for n in names])
        for i, r in enumerate(rows):
            w.writerow([r["item"], r["picture"], r["group"], int(folds[i]), r["kind"], r["two_clear"]]
                       + [round(float(preds[n][i]), 4) for n in names])
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for n in ("table", "features", "fit"):
        s = sub.add_parser(n)
        s.add_argument("--run", type=Path, required=True)
        if n == "features":
            s.add_argument("--device", default="cuda")
            s.add_argument("--backbones", nargs="*", choices=BACKBONES)
            s.add_argument("--limit", type=int, default=None, help="a smoke run on the first pictures only")
            s.add_argument("--out", type=Path, default=None, help="where the features go (default <run>/features)")
        if n == "fit":
            s.add_argument("--candidates", nargs="*", choices=list(CANDIDATES))
    a = ap.parse_args(argv)
    return {"table": cmd_table, "features": cmd_features, "fit": cmd_fit}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
