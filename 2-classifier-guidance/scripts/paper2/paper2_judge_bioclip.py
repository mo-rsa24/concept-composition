"""BioCLIP on crops: each body named by BioCLIP among the two asked species and the "neither" names.

A follow-on judge, after step 0 failed (no judge met the bars on 2026-11-27); the plan's results table names it.
BioCLIP (Stevens et al. 2024, arXiv 2311.18803; `imageomics/bioclip`, loaded by open_clip from HF_HOME offline) is CLIP
trained on the Tree of Life images with taxonomic text. It is run as the CLIP crop judge is (paper2_judge_clip.py): each
name written four ways ("a photo of a", "a close-up photo of a", "a drawing of a", "an engraving of a"), the four text
embeddings normalised and averaged, a softmax over every name; a body is called other when the best "neither" name beats
both asked names, mixed when the two asked shares sit within m, else the winner, flagged when the loser's share exceeds
p (paper2_common.decide_by_shares), over the CLIP judge's own grids. The names are the common names the plan uses, as
for every other judge. The two-clear score is paper2_common.share_two_clear.

    python3 paper2_judge_bioclip.py --pictures list.csv --out <dir> --dry-run
    bash paper2_launch.sh paper2-judge-bioclip co3 6000 paper2_judge_bioclip.py --pictures list.csv \\
        --bodies bodies.json --out /datasets/mmolefe/poe_repair_min/paper2/paper2-judge-bioclip/<part>
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from paper2_common import (Judge, base_parser, decide_by_shares, neither_names, run,  # noqa: E402
                           share_two_clear, with_article)
from paper2_judge_clip import FLAG_SHARE_GRID, MARGIN_GRID, PHRASINGS, StandIn  # noqa: E402

BIOCLIP_NAME = "hf-hub:imageomics/bioclip"


class BioClipNamer:
    def __init__(self, device):
        import open_clip
        import torch
        self.torch, self.device = torch, device
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(BIOCLIP_NAME)
        self.model = self.model.to(device).eval()
        self.tokenizer = open_clip.get_tokenizer(BIOCLIP_NAME)
        self.text_cache: dict[str, object] = {}

    def text_embedding(self, name: str):
        if name not in self.text_cache:
            torch = self.torch
            with torch.no_grad():
                e = self.model.encode_text(self.tokenizer([p.format(with_article(name)) for p in PHRASINGS]).to(self.device))
                e = e / e.norm(dim=-1, keepdim=True)
                e = e.mean(0)
                self.text_cache[name] = e / e.norm()
        return self.text_cache[name]

    def __call__(self, crop, a, b):
        torch = self.torch
        names = [a, b] + neither_names(a, b)
        with torch.no_grad():
            img = self.model.encode_image(self.preprocess(crop).unsqueeze(0).to(self.device))[0]
            img = img / img.norm()
            txt = torch.stack([self.text_embedding(n) for n in names])
            probs = (self.model.logit_scale.exp() * txt @ img).softmax(-1).tolist()
        best = max(range(2, len(names)), key=lambda i: probs[i])
        return {"p_a": probs[0], "p_b": probs[1], "p_neither_best": probs[best], "neither_best": names[best]}


class BioClipJudge(Judge):
    name = "bioclip"

    def __init__(self, args):
        super().__init__(args)
        self.namer = StandIn() if args.dry_run else BioClipNamer(args.device)

    def settings(self):
        return {"margin": self.args.margin, "flag_share": self.args.flag_share,
                "model": "stand-in" if self.args.dry_run else BIOCLIP_NAME, "phrasings": list(PHRASINGS)}

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
    run(BioClipJudge(ap.parse_args(argv)))


if __name__ == "__main__":
    main()
