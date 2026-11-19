"""Call every drawn tile of the MNIST share run with the pinned judge and read the verdict off share_bar.py.

Plan 03 of scope 11. Reads data/mnist/share-samples/share-<s>-seed-<t>.npz (pulled from the cluster) and,
for share 0, plan 02's data/mnist/samples/dial-0-seed-<t>.npz (the same training distribution). Writes,
into 4-inclusive-captions/artifacts/results/does-the-share-of-two-digit-pictures-help-the-product-compose-on-mnist/:

  scores.csv     one row per model, row and seed: share, training seed, row, labels, seed, call
  counts.json    the count of each call per model and row, the bar's checks and the verdict

Prints SUPPORTED, NULL, INCONCLUSIVE or PENDING with every share's counts. The bar is share_bar.py,
written before any share model trained. The judge-against-eye check (plan 03, task 4.2) is read from
judge_eye.json in the same folder when it exists; until then the verdict is marked as resting on the
judge alone. Runs on the CPU.

Usage (from the repo root):
    python -I poe_repair/experiments/inclusive_captions/mnist/score_share.py
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import share_bar as bar  # noqa: E402
from judge_checks import pair_call, pinned_judge, single_call  # noqa: E402
from score import ALONE, ROWS, TWO  # noqa: E402

REPO = Path(__file__).resolve().parents[4]
SHARE_SAMPLES = REPO / "data/mnist/share-samples"
DIAL0_SAMPLES = REPO / "data/mnist/samples"
OUT = REPO / "4-inclusive-captions/artifacts/results/does-the-share-of-two-digit-pictures-help-the-product-compose-on-mnist"
SEEDS_T = (1, 2, 3)


def sample_files():
    files = {(0.0, t): DIAL0_SAMPLES / f"dial-0-seed-{t}.npz" for t in SEEDS_T}
    for p in sorted(SHARE_SAMPLES.glob("share-*-seed-*.npz")):
        files[(float(p.stem.split("-")[1]), int(p.stem.split("-")[-1]))] = p
    return {k: v for k, v in files.items() if v.exists()}


def score_all():
    j, net, sha = pinned_judge()
    rows, counts = [], {}
    for (s, ts), path in sorted(sample_files().items()):
        d = np.load(path)
        for row, (kind, asked) in ROWS.items():
            calls = [pair_call(j, net, t, asked) if kind == "pair" else single_call(j, net, t, asked)
                     for t in d[row]]
            counts[(s, ts, row)] = {c: calls.count(c) for c in sorted(set(calls))}
            rows += [{"share": s, "training_seed": ts, "row": row, "labels": str(asked), "seed": i, "call": c}
                     for i, c in enumerate(calls)]
        print(f"scored share {s:g} training {ts} ({path.name})", flush=True)
    return rows, counts, sha


def n(counts, s, ts, row, call):
    return counts.get((s, ts, row), {}).get(call, 0)


def verdict(counts):
    have = {(s, t) for s, t, _ in counts}
    complete = all((s, t) in have for s in bar.SHARES for t in SEEDS_T)
    out = {"models_scored": len(have), "complete": complete, "shares": {}}
    for s in bar.SHARES:
        ts = [t for t in SEEDS_T if (s, t) in have]
        if not ts:
            continue
        floor = bar.PURITY_FLOOR[s]
        seven, eight = [n(counts, s, t, "7 alone", ALONE) for t in ts], [n(counts, s, t, "8 alone", ALONE) for t in ts]
        out["shares"][f"{s:g}"] = {
            "trainings": ts, "purity_floor": floor,
            "counts": len(ts) == 3 and all(c >= floor for c in seven + eight),
            "seven_alone": seven, "eight_alone": eight,
            "seven_alone_two_digits": [n(counts, s, t, "7 alone", "right digit with a second") for t in ts],
            "corrected_7x8_two": [n(counts, s, t, "7 x 8 corrected", TWO) for t in ts],
            "plain_7x8_two": [n(counts, s, t, "7 x 8 plain", TWO) for t in ts],
            "joint_7_and_8_two": [n(counts, s, t, "7 and 8", TWO) for t in ts],
            "corrected_4x7_two": [n(counts, s, t, "4 x 7 corrected", TWO) for t in ts],
        }
    if not complete:
        out["verdict"] = "PENDING"
        return out
    st = out["shares"]
    base_low = all(c <= bar.SHARE0_MAX for c in st["0"]["corrected_7x8_two"])
    judged = {f"{s:g}": st[f"{s:g}"] for s in bar.JUDGED}
    supported = [k for k, v in judged.items() if v["counts"] and base_low
                 and all(c >= bar.SUPPORT_MIN for c in v["corrected_7x8_two"])]
    null = all(np.mean(v["corrected_7x8_two"]) < bar.NULL_MAX_MEAN for v in judged.values() if v["counts"])
    out["verdict"] = "SUPPORTED" if supported else ("NULL" if null else "INCONCLUSIVE")
    out["supported_shares"] = supported
    eye = OUT / "judge_eye.json"
    out["judge_eye_check"] = json.loads(eye.read_text()) if eye.exists() else "pending: plan 03 task 4.2"
    return out


def main():
    torch.set_num_threads(4)
    rows, counts, sha = score_all()
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "scores.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(), w.writerows(rows)
    v = verdict(counts)
    v["judge"] = {"weights_sha256": sha}
    json.dump({"verdict": v, "counts": {f"{s:g}|{t}|{r}": c for (s, t, r), c in counts.items()}},
              open(OUT / "counts.json", "w"), indent=1)
    for s, st in v["shares"].items():
        print(f"share {float(s):g} · trainings {st['trainings']} · floor {st['purity_floor']} · \"7\" alone "
              f"{st['seven_alone']} · \"8\" alone {st['eight_alone']} · 7x8 plain {st['plain_7x8_two']} · "
              f"7x8 corrected {st['corrected_7x8_two']} · \"7 and 8\" {st['joint_7_and_8_two']} · "
              f"4x7 corrected {st['corrected_4x7_two']} (of 256){'' if st['counts'] else ' · does not count'}")
    print(f"{v['verdict']} · {v['models_scored']} of 15 models scored · {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
