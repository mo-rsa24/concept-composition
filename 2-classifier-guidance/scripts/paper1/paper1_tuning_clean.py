#!/usr/bin/env python3
"""Clean touching pairs per seed in picture folders, and the choice between them: the toy's tuning rule on CLEVR.

A seed is clean when both of the project's two pixel reads pass on it: red meets blue (paper1_sample.contact: some red pixel within
1 pixel of some blue pixel, the rule contact.json counts) and it is not mostly purple (paper1_overlap_stats: of the picture's
foreground pixels inside the overlap of the two lone samples' dilated object masks, the share with |R - B| <= 0.1 is at most 0.5;
a seed with no foreground in the overlap is not purple). The squares toy's "clean touching" is the same pair of reads on its squares.

Each folder given with --folders label=path is read at the chosen seeds (normally the first 22 touching tuning seeds per count from
split.json, never evaluation seeds). The folder with the most clean seeds, pooled over counts, is chosen; on a tie the earlier
folder in the list wins, so listing the weights from smallest up gives the toy's tie rule (the smaller weight). Writes
<run>/tuning-clean.json with the counts per folder and per count, every seed's row, and the choice.

Usage (cluster, folders absolute or under /datasets/mmolefe/poe_repair_min/paper1/):
  python paper1_tuning_clean.py --run paper1-methodA-joint2-touch-L4/choose \\
      --folders 0.003=paper1-methodA-joint2-touch-L4-w0.003/tuning-final-rows,0.03=paper1-methodA-joint2-touch-L4-w0.03/tuning-final-rows \\
      --red-dir paper1-first-lone-red --blue-dir paper1-first-lone-blue --mean-empty paper1-first-lone-red/mean-empty.pt \\
      --split paper1-first-lone-red/split.json --set tuning --side touching --counts 1,2,3 --limit 22
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import paper1_sample as ps  # noqa: E402
import paper1_overlap_stats as pos  # noqa: E402


def under(root, path):
    return Path(path) if Path(path).is_absolute() else Path(root) / path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--run", required=True, help="outputs go to <root>/<run>/tuning-clean.json")
    ap.add_argument("--root", default=str(ps.PAPER1_ROOT))
    ap.add_argument("--folders", required=True, help="comma-separated label=folder, each holding count-<k>/<seed:06d>.png")
    ap.add_argument("--red-dir", required=True)
    ap.add_argument("--blue-dir", required=True)
    ap.add_argument("--mean-empty", required=True)
    ap.add_argument("--counts", default="1,2,3")
    ap.add_argument("--seeds")
    ap.add_argument("--split")
    ap.add_argument("--set", choices=ps.SEED_SETS)
    ap.add_argument("--side", choices=["touching", "apart"], default="touching")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args(argv)
    if args.split:
        args.split = str(under(args.root, args.split))
    red_dir, blue_dir = str(under(args.root, args.red_dir)), str(under(args.root, args.blue_dir))

    folder = ps.run_dir(args)
    empty = ps.load_mean_empty(under(args.root, args.mean_empty))
    counts = ps.parse_counts(args.counts)
    seeds = {k: ps.seeds_for(args, k) for k in counts}
    overlap = {k: pos.lone_overlap(red_dir, blue_dir, k, seeds[k], empty)[0] for k in counts}
    out = {"rule": f"clean: red meets blue (some red pixel within 1 pixel of some blue pixel) and not mostly purple (purple share "
                   f"of the overlap's foreground at most {pos.MOSTLY_PURPLE}); the folder with the most clean seeds is chosen, an "
                   "exact tie going to the earlier folder listed",
           "seed_choice": {"seeds": args.seeds, "split": args.split, "set": args.set, "side": args.side, "limit": args.limit},
           "folders": {}}
    order = []
    for item in args.folders.split(","):
        label, _, path = item.partition("=")
        src = under(args.root, path)
        per, rows_all = {}, []
        for k in counts:
            pics = ps.load_pictures(src / f"count-{k}", seeds[k])
            met = ps.contact(pics, empty).tolist()
            shares = pos.purple_share(pics, empty, overlap[k])
            rows = [{"seed": s, "red_meets_blue": bool(m), "purple_share": None if sh is None else round(sh, 4),
                     "mostly_purple": sh is not None and sh > pos.MOSTLY_PURPLE} for s, m, sh in zip(seeds[k], met, shares)]
            for r in rows:
                r["clean"] = r["red_meets_blue"] and not r["mostly_purple"]
            per[f"count-{k}"] = {"seeds": len(rows), "red_meets_blue": sum(r["red_meets_blue"] for r in rows),
                                 "mostly_purple": sum(r["mostly_purple"] for r in rows), "clean": sum(r["clean"] for r in rows),
                                 "per_seed": rows}
            rows_all += rows
        pooled = {"seeds": len(rows_all), "red_meets_blue": sum(r["red_meets_blue"] for r in rows_all),
                  "mostly_purple": sum(r["mostly_purple"] for r in rows_all), "clean": sum(r["clean"] for r in rows_all)}
        pooled["clean_share"] = round(pooled["clean"] / max(1, pooled["seeds"]), 4)
        out["folders"][label] = {"pictures": str(src), "pooled": pooled, "per_count": per}
        order.append(label)
        print(json.dumps({label: pooled, **{c: {x: v[x] for x in ("red_meets_blue", "mostly_purple", "clean")}
                                            for c, v in per.items()}}), flush=True)
    best = max(range(len(order)), key=lambda i: (out["folders"][order[i]]["pooled"]["clean"], -i))
    out["chosen"] = order[best]
    print(json.dumps({"chosen": out["chosen"]}), flush=True)
    (folder / "tuning-clean.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
