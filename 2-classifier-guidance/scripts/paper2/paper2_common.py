"""What the four crop judges share: the picture list, the boxes, the crop, the loop and calls.json.

The judges differ only in how one crop is called (species judge plan, "The judges"). Everything
else lives here so the four cannot drift apart:

  the picture list   a CSV with columns picture, a, b (the picture's path and its two asked
                     species, written bare: "duck", "goose"); `m14-list` below writes one from
                     the joint prompt's blind round
  the boxes          bodies.json from species judge step 0.13 (the body counter on every
                     picture), keyed by the picture path exactly as the list writes it
  the crop           each body's box with a 10% margin on every side, cut from the full-size
                     picture, with every other body's box painted out in the picture's mean
                     colour; a body wholly inside another body's box is left alone, as the old
                     judge does (3-understanding-compositionality/scripts/species_judge.py, 72-82)
  calls.json         each body's crop file, raw scores, call and flag; each picture's kind, the
                     rule that decided it, and its two-clear score
  settings.json      the judge's settings, written before the first picture is called

A judge stores its raw scores per body, so the settings can be chosen on choosing pairs
afterwards by re-deciding from calls.json, with no second pass through the model.

`--dry-run` swaps the model for a stand-in returning fixed scores, and, when no bodies.json is
given, uses two fixed boxes (STAND_IN_BOXES), so the crop, the rules and the files are checked on
the laptop with no torch.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Callable

from PIL import Image, ImageDraw, ImageStat

HERE = Path(__file__).resolve().parent
REPO = Path(os.environ.get("POE_REPO", HERE.parents[2]))
sys.path.insert(0, str(HERE))

from paper2_kinds import CALLS, kind_of  # noqa: E402

# Every large Paper 2 output goes under this one folder on /datasets, one subfolder per run name
# (species judge plan, "Run names and output folders"). PAPER2_ROOT overrides it for a test.
PAPER2_DATASETS = Path(os.environ.get("PAPER2_ROOT", "/datasets/mmolefe/poe_repair_min/paper2"))

# The crop's margin on each side, as a share of the box's own width and height.
CROP_MARGIN = 0.10

# The "neither" names: the 43 animals of the pooled LoRA dial's pair file, fixed before any
# candidate was seen. Each picture drops its own two asked names from this list.
NEITHER_FILE = REPO / "artifacts/notes/progress/leakage/10-co-occurrence-dial/pairs.json"

# The dry run's two bodies on a 1024 by 1024 picture when no bodies.json is given: a left and
# a right box that overlap by 60 pixels, so the paint-out has something to paint.
STAND_IN_BOXES = [[60, 180, 560, 900], [500, 160, 980, 880]]

# The joint prompt's 192 pictures on the laptop, and their blind round's key (pair slug, seed).
M14_PICTURES = REPO / "outputs/loop-6/m14/blind/pictures"
M14_KEY = REPO / "outputs/loop-6/m14/blind/key.json"


# ---------------------------------------------------------------- names

def species_from_slug(slug: str) -> tuple[str, str]:
    """'a_duck__x__a_goose' -> ('duck', 'goose'); 'an_eagle__x__a_fox' -> ('eagle', 'fox')."""
    left, right = slug.split("__x__")

    def bare(s: str) -> str:
        for art in ("an_", "a_"):
            if s.startswith(art):
                s = s[len(art):]
                break
        return s.replace("_", " ")
    return bare(left), bare(right)


def with_article(name: str) -> str:
    return ("an " if name[:1].lower() in "aeiou" else "a ") + name


def neither_names(a: str, b: str, path: Path = NEITHER_FILE) -> list[str]:
    names = json.loads(Path(path).read_text())["list"]
    return [n for n in names if n not in (a, b)]


# ---------------------------------------------------------------- inputs

def read_list(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ("picture", "a", "b"):
            if not r.get(k):
                raise ValueError(f"{path}: row {r} has no {k!r}")
    return rows


def write_m14_list(out: Path, n: int, seed: int = 0) -> list[dict]:
    """Write a picture list of n joint-prompt pictures from the blind round, drawn by seed."""
    key = json.loads(M14_KEY.read_text())
    ids = sorted(key)
    random.Random(seed).shuffle(ids)
    rows = []
    for pid in ids[:n]:
        a, b = species_from_slug(key[pid]["pair"])
        rows.append({"picture": str(M14_PICTURES / f"{pid}.png"), "a": a, "b": b})
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["picture", "a", "b"])
        w.writeheader()
        w.writerows(rows)
    return rows


def _as_box(x) -> list[int]:
    box = x["box"] if isinstance(x, dict) else x
    return [int(round(v)) for v in box]


def load_bodies(path: Path | None) -> dict[str, list[list[int]]] | None:
    """bodies.json -> {picture: [[x0, y0, x1, y1], ...]}.

    Accepts {"pictures": {picture: {"boxes": [...]}}} or {picture: [...]}, each box either a list
    or the body counter's own {"box": [...], "confidence": ...} dict.
    """
    if path is None:
        return None
    raw = json.loads(Path(path).read_text())
    raw = raw.get("pictures", raw)
    out = {}
    for pic, v in raw.items():
        boxes = v.get("boxes", []) if isinstance(v, dict) else v
        out[pic] = [_as_box(x) for x in boxes]
    return out


# ---------------------------------------------------------------- the crop

def crop_body(im: Image.Image, boxes: list[list[int]], i: int,
              margin: float = CROP_MARGIN) -> tuple[Image.Image, list[int]]:
    """Body i's crop with its margin, every other body's box painted out in the mean colour.

    Returns the crop and the crop's own box on the picture (after the margin, clipped to the
    picture). A body whose box lies wholly inside another body's box keeps that other body.
    """
    W, H = im.size
    mean = tuple(int(v) for v in ImageStat.Stat(im).mean[:3])
    x0, y0, x1, y1 = boxes[i]
    mx, my = margin * (x1 - x0), margin * (y1 - y0)
    cx0, cy0 = max(0, int(round(x0 - mx))), max(0, int(round(y0 - my)))
    cx1, cy1 = min(W, int(round(x1 + mx))), min(H, int(round(y1 + my)))
    crop = im.crop((cx0, cy0, cx1, cy1)).convert("RGB")
    draw = ImageDraw.Draw(crop)
    for j, (ox0, oy0, ox1, oy1) in enumerate(boxes):
        if j == i:
            continue
        if ox0 <= x0 and oy0 <= y0 and ox1 >= x1 and oy1 >= y1:
            continue
        ix0, iy0, ix1, iy1 = max(cx0, ox0), max(cy0, oy0), min(cx1, ox1), min(cy1, oy1)
        if ix1 > ix0 and iy1 > iy0:
            # PIL's rectangle includes its far edge, so stop one pixel short of it.
            draw.rectangle((ix0 - cx0, iy0 - cy0, ix1 - cx0 - 1, iy1 - cy0 - 1), fill=mean)
    return crop, [cx0, cy0, cx1, cy1]


# ---------------------------------------------------------------- the loop

class Judge:
    """One way of calling a crop. Subclasses fill in the four methods."""

    name = "judge"

    def __init__(self, args: argparse.Namespace):
        self.args = args

    def settings(self) -> dict:
        raise NotImplementedError

    def start_picture(self) -> None:
        """Called before each picture's first body; resets any stand-in's body counter, so the
        fixed scores go to the first, second and later body of every picture alike."""
        for v in vars(self).values():
            if hasattr(v, "reset"):
                v.reset()

    def covers(self, a: str, b: str) -> bool:
        """False takes the picture out of this judge's count (the ImageNet judge's uncovered
        species); it is kept in calls.json with covered false and no kind."""
        return True

    def raw(self, crop: Image.Image, a: str, b: str) -> dict:
        """The model's raw scores for one crop, as plain floats."""
        raise NotImplementedError

    def decide(self, raw: dict, settings: dict) -> tuple[str, bool]:
        """(call, flag) for one body from its raw scores and the settings."""
        raise NotImplementedError

    def two_clear_score(self, bodies: list[dict], settings: dict) -> float:
        """The picture's two-clear score for within-pair AUC, from its bodies."""
        raise NotImplementedError


def base_parser(description: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--pictures", type=Path, required=True,
                    help="CSV with columns picture, a, b")
    ap.add_argument("--bodies", type=Path, default=None,
                    help="bodies.json from species judge step 0.13; required unless --dry-run")
    ap.add_argument("--out", type=Path, required=True,
                    help=f"output folder; cluster runs use {PAPER2_DATASETS}/<run-name>")
    ap.add_argument("--limit", type=int, default=None, help="call only the first N pictures")
    ap.add_argument("--dry-run", action="store_true",
                    help="stand-in model with fixed scores; stand-in boxes if --bodies is absent")
    ap.add_argument("--device", default="cuda")
    return ap


def redecide(judge: Judge, pictures: list[dict], settings: dict) -> list[dict]:
    """Re-apply a judge's settings to raw scores already in calls.json, without the model."""
    out = []
    for p in pictures:
        if not p.get("covered", True):
            out.append(p)
            continue
        bodies = []
        for body in p["bodies"]:
            call, flag = judge.decide(body["raw"], settings)
            bodies.append({**body, "call": call, "flag": flag})
        kind, rule = kind_of((x["call"], x["flag"]) for x in bodies)
        out.append({**p, "bodies": bodies, "kind": kind, "rule": rule,
                    "two_clear_score": round(judge.two_clear_score(bodies, settings), 6)})
    return out


def run(judge: Judge) -> dict:
    args = judge.args
    rows = read_list(args.pictures)[: args.limit]
    bodies_by_pic = load_bodies(args.bodies)
    if bodies_by_pic is None and not args.dry_run:
        raise SystemExit("--bodies is required unless --dry-run")
    out = Path(args.out)
    (out / "crops").mkdir(parents=True, exist_ok=True)

    settings = judge.settings()
    (out / "settings.json").write_text(json.dumps(
        {"judge": judge.name, "settings": settings, "dry_run": args.dry_run,
         "written": time.strftime("%Y-%m-%dT%H:%M:%S")}, indent=1))

    pictures = []
    t0 = time.time()
    for k, r in enumerate(rows):
        pic, a, b = r["picture"], r["a"], r["b"]
        if bodies_by_pic is None:
            boxes = STAND_IN_BOXES
        elif pic in bodies_by_pic:
            boxes = bodies_by_pic[pic]
        else:
            raise SystemExit(f"{pic} has no entry in {args.bodies}")
        if not judge.covers(a, b):
            pictures.append({"picture": pic, "a": a, "b": b, "covered": False, "n_bodies": len(boxes),
                             "bodies": [], "kind": None, "rule": None, "two_clear_score": None})
            print(f"{k + 1}/{len(rows)} {Path(pic).name} asked {a} and {b}: not covered by this judge",
                  flush=True)
            continue
        im = Image.open(pic).convert("RGB")
        judge.start_picture()
        bodies = []
        for i in range(len(boxes)):
            crop, crop_box = crop_body(im, boxes, i)
            crop_file = f"crops/{k:04d}_{Path(pic).stem}_body{i}.png"
            crop.save(out / crop_file)
            raw = judge.raw(crop, a, b)
            call, flag = judge.decide(raw, settings)
            assert call in CALLS, call
            bodies.append({"box": boxes[i], "crop_box": crop_box, "crop": crop_file,
                           "raw": raw, "call": call, "flag": flag})
        kind, rule = kind_of((x["call"], x["flag"]) for x in bodies)
        pictures.append({"picture": pic, "a": a, "b": b, "covered": True, "n_bodies": len(bodies),
                         "bodies": bodies, "kind": kind, "rule": rule,
                         "two_clear_score": round(judge.two_clear_score(bodies, settings), 6)})
        print(f"{k + 1}/{len(rows)} {Path(pic).name} asked {a} and {b}: {len(bodies)} bodies "
              f"{[x['call'] + ('*' if x['flag'] else '') for x in bodies]} -> {kind} "
              f"(rule {rule}), two-clear {pictures[-1]['two_clear_score']:.3f}", flush=True)

    result = {"judge": judge.name, "settings": settings, "dry_run": args.dry_run,
              "bodies_file": str(args.bodies) if args.bodies else "stand-in boxes",
              "seconds": round(time.time() - t0, 2), "pictures": pictures}
    (out / "calls.json").write_text(json.dumps(result, indent=1))
    covered = sum(p["covered"] for p in pictures)
    print(f"wrote {out / 'calls.json'} ({len(pictures)} pictures, {covered} covered) and "
          f"{sum(len(p['bodies']) for p in pictures)} crops under {out / 'crops'}", flush=True)
    return result


# ---------------------------------------------------------------- helpers for the judges

def softmax(xs: list[float]) -> list[float]:
    import math
    m = max(xs)
    e = [math.exp(x - m) for x in xs]
    s = sum(e)
    return [v / s for v in e]


def decide_by_shares(raw: dict, margin: float, flag_share: float) -> tuple[str, bool]:
    """CLIP's and the ImageNet classifier's rule for one body.

    raw holds p_a, p_b (the two asked species' probabilities) and p_neither_best (the best
    "neither" name's). Called other when the best "neither" beats both asked; mixed when the two
    asked shares, s_a = p_a / (p_a + p_b) and s_b = 1 - s_a, sit within `margin`; else the
    winner, flagged when the loser's share exceeds `flag_share`.
    """
    pa, pb = raw["p_a"], raw["p_b"]
    if raw.get("p_neither_best", 0.0) > max(pa, pb):
        return "other", False
    s_a = pa / (pa + pb) if pa + pb > 0 else 0.5
    s_b = 1.0 - s_a
    if abs(s_a - s_b) < margin:
        return "mixed", False
    loser = min(s_a, s_b)
    return ("A" if s_a > s_b else "B"), loser > flag_share


def share_two_clear(bodies: list[dict]) -> float:
    """The smaller of the two asked species' best-body shares, times one minus the largest
    flagged share (species judge plan, "The judges", CLIP and ImageNet rows).

    A body's share of A is s_a = p_a / (p_a + p_b). The flagged share is a flagged body's losing
    share. A picture with fewer than two bodies scores 0, as the old judge does: rule 4 cannot
    call it two clear, and a score up to 0.5 would rank it above an uncertain two-body picture in
    the within-pair AUC.
    """
    if len(bodies) < 2:
        return 0.0
    s_a = [x["raw"]["p_a"] / max(x["raw"]["p_a"] + x["raw"]["p_b"], 1e-12) for x in bodies]
    best_a, best_b = max(s_a), max(1.0 - s for s in s_a)
    flagged = [min(s, 1.0 - s) for s, x in zip(s_a, bodies) if x["flag"]]
    return min(best_a, best_b) * (1.0 - max(flagged, default=0.0))


def common_cli(argv=None):
    """`python3 paper2_common.py m14-list --n 2 --out list.csv` writes a picture list."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("m14-list", help="a picture list from the joint prompt's blind round")
    m.add_argument("--n", type=int, required=True)
    m.add_argument("--seed", type=int, default=0)
    m.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    if a.cmd == "m14-list":
        rows = write_m14_list(a.out, a.n, a.seed)
        for r in rows:
            print(f"{r['picture']}  asked {r['a']} and {r['b']}")
        print(f"wrote {a.out}")


if __name__ == "__main__":
    common_cli()
