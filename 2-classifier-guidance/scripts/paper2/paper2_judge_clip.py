"""CLIP on crops: each body named by CLIP ViT-L/14 among the two asked species and the "neither" names.

Species judge plan, "The judges", CLIP row; run as step 0.18 (`paper2-judge-clip`).

Each name is written four ways ("a photo of a", "a close-up photo of a", "a drawing of a", "an
engraving of a"); the four text embeddings are normalised and averaged into one per name (CLIP's
usual prompt ensemble), and the crop's probabilities come from a softmax over every name. A body
is called other when the best "neither" name beats both asked names, mixed when the two asked
shares sit within the margin m, else the winner, flagged when the loser's share exceeds p
(paper2_common.decide_by_shares). The two-clear score is the smaller of the two asked species'
best-body shares times one minus the largest flagged share.

    python3 paper2_judge_clip.py --pictures list.csv --out <dir> --dry-run
    python paper2_judge_clip.py --pictures list.csv --bodies bodies.json --out \
        /datasets/mmolefe/poe_repair_min/paper2/paper2-judge-clip --margin 0.15 --flag-share 0.25
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from paper2_common import (Judge, base_parser, decide_by_shares, neither_names, run,  # noqa: E402
                           share_two_clear, with_article)

CLIP_NAME = "openai/clip-vit-large-patch14"
PHRASINGS = ("a photo of {}", "a close-up photo of {}", "a drawing of {}", "an engraving of {}")
MARGIN_GRID = (0.05, 0.10, 0.15, 0.20, 0.30)
FLAG_SHARE_GRID = (0.15, 0.25, 0.35)


class StandIn:
    """Fixed scores: the first body leans to A, the second to B, every later one to neither."""

    FIXED = ({"p_a": 0.80, "p_b": 0.15, "p_neither_best": 0.05},
             {"p_a": 0.20, "p_b": 0.75, "p_neither_best": 0.05},
             {"p_a": 0.10, "p_b": 0.10, "p_neither_best": 0.80})

    def __init__(self):
        self.k = 0

    def reset(self):
        self.k = 0

    def __call__(self, crop, a, b):
        raw = dict(self.FIXED[min(self.k, 2)])
        self.k += 1
        return raw


class ClipNamer:
    def __init__(self, device):
        import torch
        from transformers import CLIPModel, CLIPProcessor
        self.torch = torch
        self.device = device
        self.model = CLIPModel.from_pretrained(CLIP_NAME).to(device).eval()
        self.proc = CLIPProcessor.from_pretrained(CLIP_NAME)
        self.text_cache: dict[str, object] = {}

    def text_embedding(self, name: str):
        if name not in self.text_cache:
            torch = self.torch
            texts = [p.format(with_article(name)) for p in PHRASINGS]
            with torch.no_grad():
                inp = self.proc(text=texts, return_tensors="pt", padding=True).to(self.device)
                e = self.model.get_text_features(**inp)
                e = e / e.norm(dim=-1, keepdim=True)
                e = e.mean(0)
                self.text_cache[name] = e / e.norm()
        return self.text_cache[name]

    def __call__(self, crop, a, b):
        torch = self.torch
        names = [a, b] + neither_names(a, b)
        with torch.no_grad():
            inp = self.proc(images=crop, return_tensors="pt").to(self.device)
            img = self.model.get_image_features(**inp)[0]
            img = img / img.norm()
            txt = torch.stack([self.text_embedding(n) for n in names])
            logits = self.model.logit_scale.exp() * txt @ img
            probs = logits.softmax(-1).tolist()
        best_neither = max(range(2, len(names)), key=lambda i: probs[i])
        return {"p_a": probs[0], "p_b": probs[1], "p_neither_best": probs[best_neither],
                "neither_best": names[best_neither]}


class ClipJudge(Judge):
    name = "clip"

    def __init__(self, args):
        super().__init__(args)
        self.namer = StandIn() if args.dry_run else ClipNamer(args.device)

    def settings(self):
        return {"margin": self.args.margin, "flag_share": self.args.flag_share,
                "model": "stand-in" if self.args.dry_run else CLIP_NAME,
                "phrasings": list(PHRASINGS)}

    def raw(self, crop, a, b):
        return self.namer(crop, a, b)

    def decide(self, raw, settings):
        return decide_by_shares(raw, settings["margin"], settings["flag_share"])

    def two_clear_score(self, bodies, settings):
        return share_two_clear(bodies)


def main(argv=None):
    ap = base_parser(__doc__.splitlines()[0])
    ap.add_argument("--margin", type=float, choices=MARGIN_GRID, default=0.15,
                    help="m: mixed when the two asked shares sit closer than this")
    ap.add_argument("--flag-share", type=float, choices=FLAG_SHARE_GRID, default=0.25,
                    help="p: flagged when the losing asked share exceeds this")
    run(ClipJudge(ap.parse_args(argv)))


if __name__ == "__main__":
    main()
