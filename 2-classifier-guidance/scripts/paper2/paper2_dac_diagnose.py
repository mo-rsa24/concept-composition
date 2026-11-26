"""Diagnose the Detect-and-Compare judge on a few pictures: the same crop at the judge's margin and at none.

Loop 18's wiring check after step 0.15 (2026-11-26). For each of the first --limit pictures of a list, with its bodies
from bodies.json, prints each body's similarity to the two species' references four ways: the judge's own reading
(paper2_judge_dac.DinoMatcher: DINOv2 large's pooled CLS token, L2-normalised, the mean of the cosines to each
reference) on the judge's crop (10% margin, other bodies painted out) and on the tight box with nothing painted; then
the cosine to the averaged reference on the judge's crop. Also prints the references' cosines among themselves, the
scale MuDI's Detect-and-Compare compares against (S^GT, arXiv 2404.04243, section 4.3). Changes nothing in the judge.

    bash paper2_launch.sh paper2-dac-diagnose co3 6000 paper2_dac_diagnose.py --pictures paper2_cats_64.csv \\
        --bodies <paper2-find-bodies>/cats/bodies.json --references /datasets/mmolefe/poe_repair_min/paper2/references \\
        --limit 4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from paper2_common import crop_body, load_bodies, read_list  # noqa: E402
from paper2_judge_dac import DinoMatcher  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pictures", type=Path, required=True)
    ap.add_argument("--bodies", type=Path, required=True)
    ap.add_argument("--references", type=Path, required=True)
    ap.add_argument("--n-refs", type=int, default=8)
    ap.add_argument("--limit", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args(argv)
    m = DinoMatcher(a.device, a.references, a.n_refs, use_neither=False)
    rows = read_list(a.pictures)[: a.limit]
    bodies = load_bodies(a.bodies)
    species = sorted({r["a"] for r in rows} | {r["b"] for r in rows})
    for s in species:
        r = m.refs(s)
        off = ((r @ r.T).sum() - len(r)) / (len(r) * (len(r) - 1))
        print(f"references, {s} against {s}: mean cosine {float(off):.3f} over {len(r)} photographs")
    if len(species) == 2:
        print(f"references, {species[0]} against {species[1]}: mean cosine {float((m.refs(species[0]) @ m.refs(species[1]).T).mean()):.3f}")
    for r in rows:
        im = Image.open(r["picture"]).convert("RGB")
        boxes = bodies[r["picture"]]
        for i, box in enumerate(boxes):
            judge_crop, _ = crop_body(im, boxes, i)
            tight = im.crop(tuple(box))
            readings = []
            for name, crop in (("judge's crop (10% margin, others painted)", judge_crop), ("tight box, nothing painted", tight)):
                e = m.embed([crop])[0]
                readings.append((name, m.sim(e, r["a"]), m.sim(e, r["b"])))
            e = m.embed([judge_crop])[0]
            proto = lambda s: (m.refs(s).mean(0) / m.refs(s).mean(0).norm())
            readings.append(("judge's crop, cosine to the averaged reference", float(proto(r["a"]) @ e), float(proto(r["b"]) @ e)))
            print(f"{Path(r['picture']).name} body {i} of {len(boxes)}, box {box}, asked {r['a']} against {r['b']}:")
            for name, sa, sb in readings:
                print(f"  {name:48s} {r['a']} {sa:.3f}  {r['b']} {sb:.3f}  gap {sa - sb:+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
