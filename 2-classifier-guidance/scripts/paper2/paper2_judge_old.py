"""The old judge (the floor) on Paper 2's picture lists and bodies: species_judge.py's namer and verdict, as calls.json.

Species judge steps 0.15 and 0.17 (`paper2-judge-check`, `paper2-judge-old`). The old judge is
3-understanding-compositionality/scripts/species_judge.py, run as committed where it can be: its Namer (CLIP
ViT-L/14 between "a photo of {A}", "a photo of {B}" and "a photo of an animal that is half {A} and half {B}"), its
HYBRID_MARGIN (0.10), its body label (hybrid when the half-and-half text wins or the two names sit within the margin,
else the higher name) and its verdict (one_body, hybrid, composed, two_of_one). Its verdicts become the six kinds by
the plan's table (paper2_bars.OLD_JUDGE_KINDS): composed to two clear, two of one to two of one kind, hybrid to hybrid,
one body to other. It never names borrowed part, wrong species or extra subject.

Where this departs from species_judge.py as committed, and why:
  1. The boxes come from bodies.json (paper2_bodies.py: the same count_instances, the same "animal" query), not from
     species_judge.judge() calling the counter itself, so all five judges in 0.15 read the very same bodies.
  2. The crop is paper2_common.crop_body with no margin, which is species_judge.py's crop (lines 72 to 82: the tight
     box, every other box painted out in the picture's mean colour, a body wholly inside another left alone), except
     that the painted rectangle stops one pixel short of the other box's far edge, where PIL's own rectangle includes it.
  3. The verdict is a copy of species_judge.judge()'s last lines, since judge() also runs the counter;
     tests/test_paper2_judge_old.py checks the copied lines still read the same in species_judge.py.
  4. The output is the shared calls.json: each body's raw probabilities and its label as a call (A, B, or mixed for
     hybrid; never flagged), each picture's mapped kind, and the plan's two-clear score for this judge: 0 below two
     bodies, otherwise the smaller of the two asked species' best-body probabilities.

    python3 paper2_judge_old.py --pictures list.csv --out <dir> --dry-run
    bash paper2_launch.sh paper2-judge-check co3 6000 paper2_judge_old.py --pictures paper2_cats_64.csv \\
        --bodies <paper2-find-bodies>/cats/bodies.json --out /datasets/mmolefe/poe_repair_min/paper2/paper2-judge-check/old
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from paper2_bars import OLD_JUDGE_KINDS  # noqa: E402
from paper2_common import (REPO, STAND_IN_BOXES, base_parser, crop_body, load_bodies, read_list,  # noqa: E402
                           with_article)

OLD_JUDGE = REPO / "3-understanding-compositionality/scripts/species_judge.py"
HYBRID_MARGIN_AS_COMMITTED = 0.10   # species_judge.HYBRID_MARGIN; the real run reads it from the module


def body_label(pa: float, pb: float, ph: float, a: str, b: str, margin: float) -> str:
    """species_judge.py's body label, copied (judge(), the `if ph >= max(pa, pb)` lines)."""
    if ph >= max(pa, pb) or abs(pa - pb) < margin:
        label = "hybrid"
    else:
        label = a if pa > pb else b
    return label


def verdict(n: int, labels: list[str], a: str, b: str) -> str:
    """species_judge.py's verdict, copied (judge(), the `if n < 2` lines)."""
    if n < 2:
        verdict = "hybrid" if labels == ["hybrid"] else "one_body"
    elif "hybrid" in labels:
        verdict = "hybrid"
    elif a in labels and b in labels:
        verdict = "composed"
    else:
        verdict = "two_of_one"
    return verdict


def two_clear_score(bodies: list[dict]) -> float:
    """The plan's score for the old judge: 0 below two bodies, else the smaller of the two best-body probabilities."""
    if len(bodies) < 2:
        return 0.0
    return min(max(x["raw"]["p_a"] for x in bodies), max(x["raw"]["p_b"] for x in bodies))


class StandInNamer:
    FIXED = ((0.90, 0.05, 0.05), (0.06, 0.88, 0.06), (0.40, 0.35, 0.25))

    def __init__(self):
        self.k = 0

    def probs(self, crop, names):
        out = self.FIXED[min(self.k, 2)]
        self.k += 1
        return list(out)


def main(argv=None) -> int:
    ap = base_parser(__doc__.splitlines()[0])
    args = ap.parse_args(argv)
    rows = read_list(args.pictures)[: args.limit]
    bodies_by_pic = load_bodies(args.bodies)
    if bodies_by_pic is None and not args.dry_run:
        raise SystemExit("--bodies is required unless --dry-run")
    out = Path(args.out)
    (out / "crops").mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        namer, margin, model = StandInNamer(), HYBRID_MARGIN_AS_COMMITTED, "stand-in"
    else:
        import torch
        sys.path.insert(0, str(OLD_JUDGE.parent))
        import species_judge
        namer = species_judge.Namer(torch.device(args.device))
        margin, model = species_judge.HYBRID_MARGIN, species_judge.CLIP_NAME
    settings = {"hybrid_margin": margin, "model": model, "as_committed": str(OLD_JUDGE.relative_to(REPO))}
    (out / "settings.json").write_text(json.dumps({"judge": "old", "settings": settings, "dry_run": args.dry_run,
                                                   "written": time.strftime("%Y-%m-%dT%H:%M:%S")}, indent=1))
    pictures = []
    t0 = time.time()
    for k, r in enumerate(rows):
        pic, a, b = r["picture"], r["a"], r["b"]
        boxes = STAND_IN_BOXES if bodies_by_pic is None else bodies_by_pic[pic]
        A, B = with_article(a), with_article(b)      # the old judge's names carry their article: "a cat"
        if isinstance(namer, StandInNamer):
            namer.k = 0
        im = Image.open(pic).convert("RGB")
        bodies, labels = [], []
        for i in range(len(boxes)):
            crop, crop_box = crop_body(im, boxes, i, margin=0.0)
            crop_file = f"crops/{k:04d}_{Path(pic).stem}_body{i}.png"
            crop.save(out / crop_file)
            pa, pb, ph = namer.probs(crop, [A, B])
            label = body_label(pa, pb, ph, A, B, margin)
            labels.append(label)
            call = {A: "A", B: "B", "hybrid": "mixed"}[label]
            bodies.append({"box": boxes[i], "crop_box": crop_box, "crop": crop_file,
                           "raw": {"p_a": round(pa, 4), "p_b": round(pb, 4), "p_hybrid": round(ph, 4), "label": label},
                           "call": call, "flag": False})
        old = verdict(len(boxes), labels, A, B)
        pictures.append({"picture": pic, "a": a, "b": b, "covered": True, "n_bodies": len(boxes), "bodies": bodies,
                         "old_verdict": old, "kind": OLD_JUDGE_KINDS[old], "rule": None,
                         "two_clear_score": round(two_clear_score(bodies), 6)})
        print(f"{k + 1}/{len(rows)} {Path(pic).name} asked {a} and {b}: {len(boxes)} bodies {labels} -> {old} "
              f"({OLD_JUDGE_KINDS[old]})", flush=True)
    result = {"judge": "old", "settings": settings, "dry_run": args.dry_run,
              "bodies_file": str(args.bodies) if args.bodies else "stand-in boxes",
              "seconds": round(time.time() - t0, 2), "pictures": pictures}
    (out / "calls.json").write_text(json.dumps(result, indent=1))
    print(f"wrote {out / 'calls.json'} ({len(pictures)} pictures)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
