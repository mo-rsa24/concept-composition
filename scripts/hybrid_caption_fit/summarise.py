#!/usr/bin/env python
"""Mean and spread of each set's similarities, and the paired product-minus-joint difference.

The quantity the argument turns on is sim_weaker: the smaller of an image's similarity to
"a photo of a cat" and to "a photo of a dog". An image that is a convincing match to both
captions at once cannot have a small one.

Writes summary.json.
"""
from __future__ import annotations

import json
import statistics as st

import paths

FIELDS = ["sim_cat", "sim_dog", "sim_both", "sim_weaker"]
ORDER = ["photo", "product", "joint", "plural"]


def stats(values: list[float]) -> dict:
    return {"n": len(values), "mean": st.mean(values),
            "sd": st.stdev(values) if len(values) > 1 else 0.0,
            "min": min(values), "max": max(values)}


def main() -> None:
    records = json.loads(paths.SIMILARITIES.read_text())["records"]
    by_set: dict[str, list[dict]] = {}
    for rec in records:
        by_set.setdefault(rec["set"], []).append(rec)

    summary = {name: {f: stats([r[f] for r in by_set[name]]) for f in FIELDS}
               for name in ORDER if name in by_set}

    product = {r["seed"]: r for r in by_set["product"]}
    joint = {r["seed"]: r for r in by_set["joint"]}
    shared = sorted(set(product) & set(joint))
    diffs = [product[s]["sim_weaker"] - joint[s]["sim_weaker"] for s in shared]
    summary["paired_product_minus_joint"] = {
        "seeds": shared, "per_seed": dict(zip(map(str, shared), diffs)), **stats(diffs)}

    photo_mean = summary["photo"]["sim_weaker"]["mean"]
    summary["verdict"] = {
        "question": "does the product render's weaker single-animal similarity exceed the photograph's?",
        "product_minus_photo": summary["product"]["sim_weaker"]["mean"] - photo_mean,
        "joint_minus_photo": summary["joint"]["sim_weaker"]["mean"] - photo_mean,
    }

    paths.SUMMARY.write_text(json.dumps(summary, indent=1))

    print(f"{'set':8} {'n':>4}  weaker single-animal similarity   cat      dog      both")
    for name in ORDER:
        if name not in summary:
            continue
        w = summary[name]["sim_weaker"]
        print(f"{name:8} {w['n']:>4}  {w['mean']:.4f} +/- {w['sd']:.4f}"
              f"   [{w['min']:.3f}, {w['max']:.3f}]"
              f"  {summary[name]['sim_cat']['mean']:.4f}  "
              f"{summary[name]['sim_dog']['mean']:.4f}  {summary[name]['sim_both']['mean']:.4f}")
    p = summary["paired_product_minus_joint"]
    print(f"\npaired over {p['n']} shared seeds, product minus joint: "
          f"{p['mean']:+.4f} +/- {p['sd']:.4f}")
    v = summary["verdict"]
    print(f"product minus photo: {v['product_minus_photo']:+.4f}   "
          f"joint minus photo: {v['joint_minus_photo']:+.4f}")
    print("wrote", paths.SUMMARY)


if __name__ == "__main__":
    main()
