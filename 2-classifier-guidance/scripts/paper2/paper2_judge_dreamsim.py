"""DreamSim on crops: each body compared with SDXL's own single-species renders of the two asked species.

A follow-on judge, after step 0 failed (no judge met the bars on 2026-11-27); the plan's results table names it ("a crop
compared against SDXL's own single-species renders by DreamSim"). The references are paper2_render_references.py's
renders: plain SDXL, "a photo of a <species>", seeds 9001 to 9008, the first n of each species (n = REFERENCE_PER_SPECIES,
8). DreamSim (Fu et al. 2023, arXiv 2306.09344; the `dreamsim` package, its default ensemble of DINO, CLIP and OpenCLIP
branches) embeds the crop and every reference; a species' similarity is the crop's highest cosine to any of its
references, which is one minus DreamSim's distance to the nearest reference. The raw scores per body are sim_a and
sim_b (nearest), mean_a and mean_b (over the references), so choose can re-decide with no model.

A body is called other when neither similarity reaches the floor f; mixed when the two sit within the margin m; else
the nearer species, flagged as carrying a borrowed part when the gap is still under g. The grids are on a similarity
scale, not a share: two similarities of one crop sit close together, so the CLIP judge's share grid would call almost
everything mixed. They were written before any DreamSim number on these crops was seen, and the setting is chosen on
choosing pairs (paper2_judge_lists.py choose). The two-clear score is the smaller of the two species' best-body leads
(sim_a minus sim_b, and the reverse), floored at 0, and 0 for a picture with fewer than two bodies.

    python3 paper2_judge_dreamsim.py --pictures list.csv --out <dir> --dry-run
    bash paper2_launch.sh paper2-judge-dreamsim co3 8000 paper2_judge_dreamsim.py --pictures list.csv \\
        --bodies bodies.json --out /datasets/mmolefe/poe_repair_min/paper2/paper2-judge-dreamsim/<part>
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from paper2_bars import REFERENCE_PER_SPECIES  # noqa: E402
from paper2_common import PAPER2_DATASETS, Judge, base_parser, run  # noqa: E402

MARGIN_GRID = (0.01, 0.02, 0.04, 0.06, 0.08)
FLAG_GAP_GRID = (0.05, 0.10, 0.15)
FLOOR_GRID = (0.0, 0.4, 0.5)
DEFAULTS = {"margin": 0.04, "flag_gap": 0.10, "floor": 0.4}       # the grids' middle, where the 64-cat check reads
REFERENCES = PAPER2_DATASETS / "paper2-references-sdxl"
DREAMSIM_CACHE = PAPER2_DATASETS / "weights" / "dreamsim"


def decide_by_similarity(raw: dict, margin: float, flag_gap: float, floor: float) -> tuple[str, bool]:
    sa, sb = raw["sim_a"], raw["sim_b"]
    if max(sa, sb) < floor:
        return "other", False
    gap = abs(sa - sb)
    if gap < margin:
        return "mixed", False
    return ("A" if sa > sb else "B"), gap < flag_gap


def lead_two_clear(bodies: list[dict]) -> float:
    if len(bodies) < 2:
        return 0.0
    d = [x["raw"]["sim_a"] - x["raw"]["sim_b"] for x in bodies]
    return max(0.0, min(max(d), max(-v for v in d)))


class StandIn:
    """Fixed similarities: the first body near A, the second near B, every later one near neither."""

    FIXED = ({"sim_a": 0.72, "sim_b": 0.55}, {"sim_a": 0.50, "sim_b": 0.70}, {"sim_a": 0.30, "sim_b": 0.32})

    def __init__(self):
        self.k = 0

    def reset(self):
        self.k = 0

    def __call__(self, crop, a, b):
        raw = dict(self.FIXED[min(self.k, 2)])
        raw.update(mean_a=raw["sim_a"] - 0.05, mean_b=raw["sim_b"] - 0.05)
        self.k += 1
        return raw


class DreamSimMatcher:
    def __init__(self, device, references: Path, n: int):
        os.environ.setdefault("TORCH_HOME", str(DREAMSIM_CACHE / "torch"))   # as paper2_fetch_followon.sh fetched
        import torch
        from dreamsim import dreamsim
        from PIL import Image
        self.torch, self.Image, self.device = torch, Image, device
        self.model, self.preprocess = dreamsim(pretrained=True, device=device, cache_dir=str(DREAMSIM_CACHE))
        self.references, self.n = references, n
        self.ref_cache: dict[str, object] = {}

    def embed(self, im):
        with self.torch.no_grad():
            e = self.model.embed(self.preprocess(im).to(self.device))
        return self.torch.nn.functional.normalize(e, dim=-1)

    def refs(self, species: str):
        if species not in self.ref_cache:
            from paper2_render_references import references
            paths = references(self.references, species, self.n)
            if len(paths) < self.n:
                raise SystemExit(f"{species}: {len(paths)} of {self.n} references under {self.references}; "
                                 f"render them with paper2_render_references.py")
            self.ref_cache[species] = self.torch.cat([self.embed(self.Image.open(p).convert("RGB")) for p in paths])
        return self.ref_cache[species]

    def __call__(self, crop, a, b):
        c = self.embed(crop)
        sims = {s: (self.refs(s) @ c[0]).tolist() for s in (a, b)}
        return {"sim_a": max(sims[a]), "sim_b": max(sims[b]), "mean_a": sum(sims[a]) / len(sims[a]),
                "mean_b": sum(sims[b]) / len(sims[b])}


class DreamSimJudge(Judge):
    name = "dreamsim"

    def __init__(self, args):
        super().__init__(args)
        self.matcher = StandIn() if args.dry_run else DreamSimMatcher(args.device, args.references, args.n_refs)

    def settings(self):
        return {"margin": self.args.margin, "flag_gap": self.args.flag_gap, "floor": self.args.floor,
                "n_refs": self.args.n_refs, "references": str(self.args.references),
                "model": "stand-in" if self.args.dry_run else "dreamsim ensemble (pretrained)"}

    def raw(self, crop, a, b):
        return self.matcher(crop, a, b)

    def decide(self, raw, settings):
        return decide_by_similarity(raw, settings["margin"], settings["flag_gap"], settings["floor"])

    def two_clear_score(self, bodies, settings):
        return lead_two_clear(bodies)


def main(argv=None):
    ap = base_parser(__doc__.splitlines()[0])
    ap.add_argument("--margin", type=float, choices=MARGIN_GRID, default=DEFAULTS["margin"],
                    help="m: mixed when the two nearest-reference similarities sit closer than this")
    ap.add_argument("--flag-gap", type=float, choices=FLAG_GAP_GRID, default=DEFAULTS["flag_gap"],
                    help="g: a called body is flagged when its lead is under this")
    ap.add_argument("--floor", type=float, choices=FLOOR_GRID, default=DEFAULTS["floor"],
                    help="f: other when neither similarity reaches this (0 turns it off)")
    ap.add_argument("--references", type=Path, default=REFERENCES)
    ap.add_argument("--n-refs", type=int, default=REFERENCE_PER_SPECIES)
    run(DreamSimJudge(ap.parse_args(argv)))


if __name__ == "__main__":
    main()
