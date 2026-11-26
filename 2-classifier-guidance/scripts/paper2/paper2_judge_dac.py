"""Detect-and-Compare on crops: each body matched by DINOv2 features against real photographs of each species.

Species judge plan, "The judges", Detect-and-Compare row (MuDI, arXiv 2404.04243, adapted from
personal subjects to species); run as step 0.19b (`paper2-judge-dac`).

Each crop's DINOv2 image feature (`facebook/dinov2-large`, the pooled class token) is compared by
cosine similarity with the reference photographs of each asked species, the first n files in
`<references>/<species>/` in name order, n being 8 or 16. A species' similarity is the mean over
its references. The crop takes the species it matches better, and is flagged as carrying a
borrowed part when the gap between the two similarities is below b, as MuDI's off-diagonal
entries mark a subject that resembles another. With --neither, every "neither" name that has a
reference folder competes too, and the body is called other when one of them beats both asked.
The two-clear score is the smaller of the two asked species' best-body similarities minus the
largest borrowed-part similarity (a flagged body's similarity to the species it lost to), and 0
for a picture with fewer than two bodies.

As planned, with reference photographs of the two asked species only (--neither off), this judge
never calls mixed or other, so it cannot name hybrid or wrong species; each counts as a miss on
that kind.

References for the test are real photographs, never SDXL's own renders.

    python3 paper2_judge_dac.py --pictures list.csv --out <dir> --dry-run
    python paper2_judge_dac.py --pictures list.csv --bodies bodies.json --references <dir> --out \
        /datasets/mmolefe/poe_repair_min/paper2/paper2-judge-dac --borrow-gap 0.2 --n-refs 8
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from paper2_common import Judge, base_parser, neither_names, run  # noqa: E402

DINO_NAME = "facebook/dinov2-large"
BORROW_GAP_GRID = (0.1, 0.2, 0.3)
N_REFS_GRID = (8, 16)
PICTURE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp")


class StandIn:
    """Fixed similarities: the first body matches A, the second B, every later one neither."""

    FIXED = ({"sim_a": 0.62, "sim_b": 0.31, "sim_neither_best": 0.10},
             {"sim_a": 0.35, "sim_b": 0.58, "sim_neither_best": 0.10},
             {"sim_a": 0.12, "sim_b": 0.14, "sim_neither_best": 0.55})

    def __init__(self):
        self.k = 0

    def reset(self):
        self.k = 0

    def __call__(self, crop, a, b):
        raw = dict(self.FIXED[min(self.k, 2)])
        self.k += 1
        return raw


def reference_files(root: Path, species: str, n: int) -> list[Path]:
    folder = Path(root) / species
    files = sorted(p for p in folder.glob("*") if p.suffix.lower() in PICTURE_SUFFIXES) if folder.is_dir() else []
    return files[:n]


class DinoMatcher:
    def __init__(self, device, references: Path, n_refs: int, use_neither: bool):
        import torch
        from transformers import AutoImageProcessor, AutoModel
        self.torch = torch
        self.device = device
        self.references = Path(references)
        self.n_refs = n_refs
        self.use_neither = use_neither
        self.model = AutoModel.from_pretrained(DINO_NAME).to(device).eval()
        self.proc = AutoImageProcessor.from_pretrained(DINO_NAME)
        self.ref_cache: dict[str, object] = {}

    def embed(self, images):
        torch = self.torch
        with torch.no_grad():
            inp = self.proc(images=images, return_tensors="pt").to(self.device)
            e = self.model(**inp).pooler_output
        return e / e.norm(dim=-1, keepdim=True)

    def refs(self, species: str):
        if species not in self.ref_cache:
            from PIL import Image
            files = reference_files(self.references, species, self.n_refs)
            self.ref_cache[species] = (self.embed([Image.open(f).convert("RGB") for f in files])
                                       if len(files) == self.n_refs else None)
        return self.ref_cache[species]

    def sim(self, crop_e, species: str) -> float:
        r = self.refs(species)
        return float((r @ crop_e).mean())

    def __call__(self, crop, a, b):
        for s in (a, b):
            if self.refs(s) is None:
                raise SystemExit(f"need {self.n_refs} reference photographs in {self.references / s}")
        e = self.embed([crop])[0]
        raw = {"sim_a": self.sim(e, a), "sim_b": self.sim(e, b), "sim_neither_best": None,
               "neither_best": None}
        if self.use_neither:
            sims = {n: self.sim(e, n) for n in neither_names(a, b) if self.refs(n) is not None}
            if sims:
                best = max(sims, key=sims.get)
                raw.update(sim_neither_best=sims[best], neither_best=best)
        return raw


class DacJudge(Judge):
    name = "dac"

    def __init__(self, args):
        super().__init__(args)
        if args.dry_run:
            self.matcher = StandIn()
        else:
            if args.references is None:
                raise SystemExit("--references is required unless --dry-run")
            self.matcher = DinoMatcher(args.device, args.references, args.n_refs, args.neither)

    def settings(self):
        return {"borrow_gap": self.args.borrow_gap, "n_refs": self.args.n_refs,
                "neither": self.args.neither,
                "model": "stand-in" if self.args.dry_run else DINO_NAME,
                "references": "stand-in" if self.args.dry_run else str(self.args.references)}

    def raw(self, crop, a, b):
        return self.matcher(crop, a, b)

    def decide(self, raw, settings):
        sa, sb, sn = raw["sim_a"], raw["sim_b"], raw.get("sim_neither_best")
        if settings.get("neither", True) and sn is not None and sn > max(sa, sb):
            return "other", False
        return ("A" if sa >= sb else "B"), abs(sa - sb) < settings["borrow_gap"]

    def two_clear_score(self, bodies, settings):
        # Fewer than two bodies scores 0: rule 4 cannot call such a picture two clear.
        if len(bodies) < 2:
            return 0.0
        best_a = max(x["raw"]["sim_a"] for x in bodies)
        best_b = max(x["raw"]["sim_b"] for x in bodies)
        borrowed = [min(x["raw"]["sim_a"], x["raw"]["sim_b"]) for x in bodies if x["flag"]]
        return min(best_a, best_b) - max(borrowed, default=0.0)


def main(argv=None):
    ap = base_parser(__doc__.splitlines()[0])
    ap.add_argument("--borrow-gap", type=float, choices=BORROW_GAP_GRID, default=0.2,
                    help="b: flagged when the two asked similarities sit closer than this")
    ap.add_argument("--n-refs", type=int, choices=N_REFS_GRID, default=8,
                    help="reference photographs per species")
    ap.add_argument("--references", type=Path, default=None,
                    help="folder of <species>/ subfolders of real photographs")
    ap.add_argument("--neither", action="store_true",
                    help="let 'neither' names with a reference folder compete, so a body can be other")
    run(DacJudge(ap.parse_args(argv)))


if __name__ == "__main__":
    main()
