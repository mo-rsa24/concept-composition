#!/usr/bin/env python3
"""Loop 12 step Z2: turn the per-picture records into the seven tests and the secondary reads.

Design: artifacts/notes/progress/leakage/02-what-predicts-borrowing/design.md. Bars: z2_bars.py.

  python3 z2_analyse.py <records folder: <pair>/seed-*.json and <pair>/text.json> <W2d part-binding-calls.csv> <out.json>
"""
import csv
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from z2_bars import BORROWED, N_TESTS, WINDOWS, spearman, verdict  # noqa: E402

EYE_CHECK_ABOVE = 10.0      # of 255, against M1's picture
HELD_BACK = "a_giraffe__x__a_tortoise"


def load(folder):
    pairs = {}
    for pair_dir in sorted(Path(folder).iterdir()):
        if not pair_dir.is_dir() or not (pair_dir / "text.json").exists():
            continue
        rows = [json.loads(Path(f).read_text()) for f in sorted(glob.glob(str(pair_dir / "seed-*.json")))]
        pairs[pair_dir.name] = {"text": json.loads((pair_dir / "text.json").read_text()), "rows": rows}
    return pairs


def per_pair(pairs, key, window=None):
    out = {}
    for pair, d in pairs.items():
        if window is None:
            out[pair] = d["text"][key]
        else:
            out[pair] = float(np.mean([r["windows"][window][key] for r in d["rows"]]))
    return out


def main():
    folder, calls_csv, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    pairs = load(folder)
    missing = sorted(set(BORROWED) - set(pairs))
    calls = defaultdict(dict)
    for r in csv.DictReader(open(calls_csv)):
        calls[r["pair"]][int(r["seed"])] = r["final"]

    tests = {"text_joint": per_pair(pairs, "text_joint")}
    for w in WINDOWS:
        tests[f"iou_AB/{w}"] = per_pair(pairs, "iou_AB", w)
        tests[f"dt_cos/{w}"] = per_pair(pairs, "dt_cos", w)
    assert len(tests) == N_TESTS, (len(tests), N_TESTS)

    secondary = {"text_alone": per_pair(pairs, "text_alone"), "text_pooled": per_pair(pairs, "text_pooled")}
    for w in WINDOWS:
        for k in ("soft_AB", "dt_norm_cos", "iou_A_pad", "iou_B_pad", "iou_A_eot", "iou_B_eot"):
            secondary[f"{k}/{w}"] = per_pair(pairs, k, w)

    result = {"pairs": sorted(pairs), "missing_pairs": missing, "n_pictures": sum(len(d["rows"]) for d in pairs.values()),
              "tests": {}, "secondary": {}, "without_hippopotamus": {}, "borrowed_share": {}, "within_pair": {},
              "eye_check": [], "per_pair": {}}
    if not missing:
        for name, values in tests.items():
            result["tests"][name] = {"values": values, **verdict(values)}
        for name, values in secondary.items():
            pairs_ = sorted(BORROWED)
            result["secondary"][name] = {"rho": spearman([values[p] for p in pairs_], [BORROWED[p] for p in pairs_])}
        # the primary outcome without the pair that never shows two subjects
        keep = [p for p in sorted(BORROWED) if p != "a_hippopotamus__x__a_swallow"]
        for name, values in tests.items():
            result["without_hippopotamus"][name] = spearman([values[p] for p in keep], [BORROWED[p] for p in keep])
        # the secondary outcome: borrowed over borrowed plus own_parts, hippopotamus left out
        share = {}
        for p in keep:
            labels = list(calls[p].values())
            two = sum(l in ("borrowed", "own_parts") for l in labels)
            share[p] = labels.count("borrowed") / two if two else None
        for name, values in tests.items():
            ps = [p for p in keep if share[p] is not None]
            result["borrowed_share"][name] = spearman([values[p] for p in ps], [share[p] for p in ps])
        result["borrowed_share_values"] = share
    # within-pair: seeds that borrowed against seeds that kept their own parts, per window
    for p in ("a_bobcat__x__a_quail", "a_seal__x__a_gull", "a_puma__x__a_condor"):
        if p not in pairs:
            continue
        by = defaultdict(lambda: defaultdict(list))
        for r in pairs[p]["rows"]:
            lab = calls[p].get(r["seed"])
            if lab not in ("borrowed", "own_parts"):
                continue
            for w in WINDOWS:
                for k in ("iou_AB", "dt_cos"):
                    by[f"{k}/{w}"][lab].append(r["windows"][w][k])
        result["within_pair"][p] = {k: {lab: [round(x, 3) for x in v] for lab, v in d.items()} for k, d in by.items()}
    for p, d in pairs.items():
        for r in d["rows"]:
            diff = r.get("m1_mean_abs_diff")
            if diff is not None and diff > EYE_CHECK_ABOVE:
                result["eye_check"].append({"pair": p, "seed": r["seed"], "m1_mean_abs_diff": round(diff, 2), "w2d": calls[p].get(r["seed"])})
        result["per_pair"][p] = {"borrowed": BORROWED.get(p), "text_joint": d["text"]["text_joint"],
                                 **{f"{k}/{w}": per_pair({p: d}, k, w)[p] for w in WINDOWS for k in ("iou_AB", "dt_cos")},
                                 "self_iou_min": min(r["steps"][s]["self_iou_A"] for r in d["rows"] for s in r["steps"]),
                                 "seeds": len(d["rows"])}
    result["held_back_from_display"] = HELD_BACK
    out.write_text(json.dumps(result, indent=1) + "\n")
    for name, t in result["tests"].items():
        print(f"{name:16s} rho {t['rho']:+.3f}  p_up {t['p_up']:.4f}  {t['call']}")
    print(f"pictures {result['n_pictures']}, missing pairs {missing}, eye checks {len(result['eye_check'])}")


if __name__ == "__main__":
    main()
