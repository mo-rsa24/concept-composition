#!/usr/bin/env python3
"""How much the red and blue lone samples overlap, and how purple each picture folder draws that overlap. Read-only on pictures.

For each pair seed and count, from the two lone folders:
  overlap_pixels   pixels where the red lone sample's object mask and the blue one's (each pixel more than 0.1 from the mean empty
                   scene on any channel, dilated by 1 pixel: paper1_sample.object_masks) both hold
  touching         the pair touches by the split rule: not paper1_sample.apart (some blue foreground pixel within 1 pixel of red's)
and from each picture folder:
  purple_share     among the picture's foreground pixels inside that overlap, the share that is neither red (R - B > 0.1) nor blue
                   (B - R > 0.1), on 0 to 1; null when the overlap holds no foreground pixel of the picture
Per count and pooled: the medians of each over the chosen seeds and over the touching ones, and the seeds mostly purple (share above
0.5). This is the rule of the diagnosis's overlap_purple.py, over any seeds and any lone folders.

The blue lone folder's blue seed offset (paper1_sample.recorded_blue_offset, 0 for a folder without blue-seed-offset.json) is
recorded beside the numbers. Output: <root>/<run>/overlap-stats.json, one entry per picture folder, merged into an existing file.

  python paper1_overlap_stats.py --run paper1-next-stats --pictures paper1-diag-lower-overlap,paper1-next-region-lower-o100000 \
      --red-dir <lone red run> --blue-dir <lone blue run> --mean-empty <pt> --counts 1,2,3 --split <split.json> --set evaluation --side touching
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

import paper1_sample as ps

MOSTLY_PURPLE = 0.5   # a seed whose purple share is above this is counted as mostly purple


def lone_overlap(red_dir, blue_dir, k, seeds, empty):
    """The overlap of the two lone samples' dilated object masks (n, 64, 64) and whether each pair touches by the apart rule."""
    overlap = (ps.object_masks(red_dir, k, seeds, empty) * ps.object_masks(blue_dir, k, seeds, empty))[:, 0].cpu() > 0
    red = ps.load_pictures(Path(red_dir) / f"count-{k}", seeds)
    blue = ps.load_pictures(Path(blue_dir) / f"count-{k}", seeds)
    return overlap, ~ps.apart(red, blue, empty)


def purple_share(images, empty, overlap):
    """Per picture, the share of its foreground pixels inside the overlap that are neither red nor blue by COLOUR_MARGIN; None
    when there are none. images (n, 64, 64, 3) on 0 to 1, overlap (n, 64, 64) boolean."""
    fg = ps.foreground(images, empty) & overlap
    blend = fg & ((images[..., 0] - images[..., 2]).abs() <= ps.COLOUR_MARGIN)
    n_fg = fg.flatten(1).sum(1)
    return [float(b) / int(f) if int(f) else None for b, f in zip(blend.flatten(1).sum(1), n_fg)]


def summarise(rows):
    med = lambda v: round(float(np.median(v)), 4) if v else None
    touching = [r for r in rows if r["touching"]]
    shares = [r["purple_share"] for r in rows if r["purple_share"] is not None]
    shares_t = [r["purple_share"] for r in touching if r["purple_share"] is not None]
    return {"seeds": len(rows), "touching": len(touching), "touching_share": round(len(touching) / max(1, len(rows)), 4),
            "median_overlap_pixels": med([r["overlap_pixels"] for r in rows]),
            "median_overlap_pixels_touching": med([r["overlap_pixels"] for r in touching]),
            "seeds_with_overlap_foreground": len(shares), "median_purple_share": med(shares),
            "median_purple_share_touching": med(shares_t),
            "seeds_mostly_purple": sum(s > MOSTLY_PURPLE for s in shares)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--run", required=True, help="outputs go to <root>/<run>/overlap-stats.json")
    ap.add_argument("--root", default=str(ps.PAPER1_ROOT))
    ap.add_argument("--pictures", required=True, help="comma-separated picture folders (absolute, or under --root), each holding count-<k>/")
    ap.add_argument("--red-dir", required=True)
    ap.add_argument("--blue-dir", required=True)
    ap.add_argument("--mean-empty", required=True)
    ap.add_argument("--counts", default="1,2,3")
    ap.add_argument("--seeds", help="seed ranges such as 20000-20255; or choose them from a split file below")
    ap.add_argument("--split", help="split.json from paper1_sample.py split")
    ap.add_argument("--set", choices=ps.SEED_SETS)
    ap.add_argument("--side", choices=["touching", "apart"], default="touching")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args(argv)

    folder = ps.run_dir(args)
    empty = ps.load_mean_empty(args.mean_empty)
    counts = ps.parse_counts(args.counts)
    path = folder / "overlap-stats.json"
    result = json.loads(path.read_text()) if path.exists() else {}
    lone = {}
    for k in counts:
        seeds = ps.seeds_for(args, k)
        lone[k] = []
        for batch in ps.batches(seeds, 512):
            overlap, touching = lone_overlap(args.red_dir, args.blue_dir, k, batch, empty)
            lone[k].append((batch, overlap, touching))
    for name in args.pictures.split(","):
        src = Path(name) if Path(name).is_absolute() else Path(args.root) / name
        per_count, pooled = {}, []
        for k in counts:
            rows = []
            for batch, overlap, touching in lone[k]:
                shares = purple_share(ps.load_pictures(src / f"count-{k}", batch), empty, overlap)
                rows += [{"seed": s, "overlap_pixels": int(o.sum()), "touching": bool(t),
                          "purple_share": None if sh is None else round(sh, 4)}
                         for s, o, t, sh in zip(batch, overlap, touching.tolist(), shares)]
            per_count[f"count-{k}"] = {**summarise(rows), "per_seed": rows}
            pooled += rows
        result[name] = {"pictures": str(src), "red_dir": args.red_dir, "blue_dir": args.blue_dir,
                        "blue_seed_offset": ps.recorded_blue_offset(args.blue_dir),
                        "seed_choice": {"seeds": args.seeds, "split": args.split, "set": args.set, "side": args.side, "limit": args.limit},
                        "rule": f"overlap: both lone samples' object masks (foreground > {ps.MASK_THRESHOLD} from the mean empty scene, "
                                f"dilated 1 pixel); touching: not apart (blue foreground within 1 pixel of red); purple share: of the "
                                f"picture's foreground pixels inside the overlap, those with |R - B| <= {ps.COLOUR_MARGIN}",
                        "pooled": summarise(pooled), "per_count": per_count}
        print(json.dumps({name: {"pooled": result[name]["pooled"],
                                 **{c: {key: v[key] for key in ("touching", "median_overlap_pixels", "median_purple_share")}
                                    for c, v in per_count.items()}}}), flush=True)
    path.write_text(json.dumps(result, indent=1) + "\n")


if __name__ == "__main__":
    main()
