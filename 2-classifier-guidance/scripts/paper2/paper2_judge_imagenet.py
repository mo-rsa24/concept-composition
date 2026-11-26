"""ImageNet classifier on crops: each body named by convnext_large.fb_in22k_ft_in1k through a class table.

Species judge plan, "The judges", ImageNet row; run as step 0.19 (`paper2-judge-imagenet`).

Each species maps to a set of ImageNet-1k classes by the table written in step 0.2 before any
scoring (`IMAGENET_CLASSES` in paper2_bars.py, or a JSON file `{species: [class index, ...]}`
given with --class-table). A species' probability is the sum of its classes' softmax
probabilities. The "neither" names that the table covers compete as in the CLIP judge. A
picture whose asked species has no class is kept in calls.json as not covered and counts for
nothing, so the judge's coverage can be reported. Calls, flags and the two-clear score follow
the CLIP judge's rule (paper2_common.decide_by_shares and share_two_clear).

    python3 paper2_judge_imagenet.py --pictures list.csv --out <dir> --dry-run
    python paper2_judge_imagenet.py --pictures list.csv --bodies bodies.json --out \
        /datasets/mmolefe/poe_repair_min/paper2/paper2-judge-imagenet --margin 0.15 --flag-share 0.25
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from paper2_common import (Judge, base_parser, decide_by_shares, neither_names, run,  # noqa: E402
                           share_two_clear)

TIMM_NAME = "convnext_large.fb_in22k_ft_in1k"
MARGIN_GRID = (0.05, 0.10, 0.15, 0.20, 0.30)
FLAG_SHARE_GRID = (0.15, 0.25, 0.35)


def load_class_table(path: Path | None) -> dict[str, list[int]]:
    if path is not None:
        return {k: [int(i) for i in v] for k, v in json.loads(Path(path).read_text()).items()}
    try:
        from paper2_bars import IMAGENET_CLASSES
    except ImportError as e:
        raise SystemExit("no class table: give --class-table, or write IMAGENET_CLASSES in "
                         "paper2_bars.py (species judge step 0.2)") from e
    return {k: [int(i) for i in v] for k, v in IMAGENET_CLASSES.items()}


class StandIn:
    """Fixed scores, as the CLIP stand-in: first body A, second B, every later one neither."""

    FIXED = ({"p_a": 0.70, "p_b": 0.10, "p_neither_best": 0.05},
             {"p_a": 0.15, "p_b": 0.60, "p_neither_best": 0.05},
             {"p_a": 0.05, "p_b": 0.05, "p_neither_best": 0.70})

    def __init__(self):
        self.k = 0

    def reset(self):
        self.k = 0

    def __call__(self, crop, a, b):
        raw = dict(self.FIXED[min(self.k, 2)])
        self.k += 1
        return raw


class ImageNetNamer:
    def __init__(self, device, table: dict[str, list[int]]):
        import timm
        import torch
        from timm.data import create_transform, resolve_data_config
        self.torch = torch
        self.device = device
        self.table = table
        self.model = timm.create_model(TIMM_NAME, pretrained=True).to(device).eval()
        self.transform = create_transform(**resolve_data_config({}, model=self.model))

    def __call__(self, crop, a, b):
        torch = self.torch
        with torch.no_grad():
            x = self.transform(crop).unsqueeze(0).to(self.device)
            probs = self.model(x).softmax(-1)[0]

        def p(name):
            return float(probs[self.table[name]].sum())
        neither = [n for n in neither_names(a, b) if n in self.table]
        best = max(neither, key=p) if neither else None
        return {"p_a": p(a), "p_b": p(b), "p_neither_best": p(best) if best else 0.0,
                "neither_best": best, "top1": int(probs.argmax())}


class ImageNetJudge(Judge):
    name = "imagenet"

    def __init__(self, args):
        super().__init__(args)
        if args.dry_run:
            self.table = None
            self.namer = StandIn()
        else:
            self.table = load_class_table(args.class_table)
            self.namer = ImageNetNamer(args.device, self.table)

    def settings(self):
        return {"margin": self.args.margin, "flag_share": self.args.flag_share,
                "model": "stand-in" if self.args.dry_run else TIMM_NAME,
                "class_table": "stand-in covers every species" if self.table is None
                else str(self.args.class_table or "paper2_bars.IMAGENET_CLASSES")}

    def covers(self, a, b):
        return self.table is None or (bool(self.table.get(a)) and bool(self.table.get(b)))

    def raw(self, crop, a, b):
        return self.namer(crop, a, b)

    def decide(self, raw, settings):
        return decide_by_shares(raw, settings["margin"], settings["flag_share"])

    def two_clear_score(self, bodies, settings):
        return share_two_clear(bodies)


def main(argv=None):
    ap = base_parser(__doc__.splitlines()[0])
    ap.add_argument("--margin", type=float, choices=MARGIN_GRID, default=0.15)
    ap.add_argument("--flag-share", type=float, choices=FLAG_SHARE_GRID, default=0.25)
    ap.add_argument("--class-table", type=Path, default=None,
                    help="JSON {species: [ImageNet class index, ...]}; default paper2_bars.IMAGENET_CLASSES")
    run(ImageNetJudge(ap.parse_args(argv)))


if __name__ == "__main__":
    main()
