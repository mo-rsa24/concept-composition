"""Score the MNIST share run's guidance-1 redraw and read it against share_bar.py's G1_ constants.

Plan 03 of scope 11, task 5. Reads data/mnist/share-samples-g1/share-<s>-seed-<t>.npz (the same 15
checkpoints as the guidance-3 run, redrawn at guidance 1; pulled from the cluster's /datasets and its
/home-mscluster fallback). Writes counts-g1.json and scores-g1.csv beside the guidance-3 files in
4-inclusive-captions/artifacts/results/does-the-share-of-two-digit-pictures-help-the-product-compose-on-mnist/.

The two readings, both written before any guidance-1 picture was drawn (share_bar.py):
  the pairs came from guidance   at every judged share, in all three trainings, "7" alone draws its digit
                                 with a second digit on at most G1_LEAK_MAX[s]
  the product composes beyond    at a judged share, in all three trainings, the plain 7 x 8 product's two
  its labels                     separate right digits are at least G1_COMPOSE_MIN and at least
                                 G1_COMPOSE_FACTOR times the larger of "7" alone's and "8" alone's
                                 "right digit with a second"

Usage (from the repo root):
    python -I poe_repair/experiments/inclusive_captions/mnist/score_share_g1.py
"""
import csv, json, sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import share_bar as bar  # noqa: E402
from judge_checks import pair_call, pinned_judge, single_call  # noqa: E402
from score import ALONE, ROWS, TWO  # noqa: E402

import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[4]
SAMPLES = REPO / "data/mnist/share-samples-g1"
OUT = REPO / "4-inclusive-captions/artifacts/results/does-the-share-of-two-digit-pictures-help-the-product-compose-on-mnist"
SECOND = "right digit with a second"
SEEDS_T = (1, 2, 3)


def main():
    torch.set_num_threads(4)
    j, net, sha = pinned_judge()
    rows, counts = [], {}
    for p in sorted(SAMPLES.glob("share-*-seed-*.npz")):
        s, t = float(p.stem.split("-")[1]), int(p.stem.split("-")[-1])
        d = np.load(p)
        for row, (kind, asked) in ROWS.items():
            calls = [pair_call(j, net, x, asked) if kind == "pair" else single_call(j, net, x, asked) for x in d[row]]
            counts[(s, t, row)] = {c: calls.count(c) for c in sorted(set(calls))}
            rows += [{"share": s, "training_seed": t, "row": row, "seed": i, "call": c} for i, c in enumerate(calls)]
    n = lambda s, t, row, call: counts.get((s, t, row), {}).get(call, 0)
    shares = {}
    for s in sorted({k[0] for k in counts}):
        ts = [t for t in SEEDS_T if (s, t, "7 alone") in counts]
        seven2 = [n(s, t, "7 alone", SECOND) for t in ts]
        eight2 = [n(s, t, "8 alone", SECOND) for t in ts]
        plain = [n(s, t, "7 x 8 plain", TWO) for t in ts]
        shares[f"{s:g}"] = {
            "trainings": ts, "seven_alone_lone": [n(s, t, "7 alone", ALONE) for t in ts],
            "seven_alone_with_a_second": seven2, "eight_alone_with_a_second": eight2,
            "plain_7x8_two": plain, "corrected_7x8_two": [n(s, t, "7 x 8 corrected", TWO) for t in ts],
            "joint_7_and_8_two": [n(s, t, "7 and 8", TWO) for t in ts],
            "corrected_4x7_two": [n(s, t, "4 x 7 corrected", TWO) for t in ts],
            "leak_max": bar.G1_LEAK_MAX.get(s),
            "leak_under_max": None if s not in bar.G1_LEAK_MAX else (len(ts) == 3 and all(v <= bar.G1_LEAK_MAX[s] for v in seven2)),
            "composes_beyond_labels": None if s not in bar.JUDGED else (len(ts) == 3 and all(
                p >= bar.G1_COMPOSE_MIN and p >= bar.G1_COMPOSE_FACTOR * max(a, b) for p, a, b in zip(plain, seven2, eight2))),
        }
    judged = [shares[f"{s:g}"] for s in bar.JUDGED if f"{s:g}" in shares]
    complete = len(judged) == len(bar.JUDGED) and all(len(v["trainings"]) == 3 for v in judged)
    verdict = {
        "complete": complete,
        "the pairs came from guidance": (all(v["leak_under_max"] for v in judged) if complete else "PENDING"),
        "the product composes beyond its labels at": [k for k, v in shares.items() if v["composes_beyond_labels"]],
        "judge_sha256": sha,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "scores-g1.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    json.dump({"verdict": verdict, "shares": shares,
               "counts": {f"{s:g}|{t}|{r}": c for (s, t, r), c in counts.items()}},
              open(OUT / "counts-g1.json", "w"), indent=1)
    for k, v in shares.items():
        print(f"share {k}: 7 alone with a second {v['seven_alone_with_a_second']} (max {v['leak_max']}) · 8 alone with a "
              f"second {v['eight_alone_with_a_second']} · 7x8 plain {v['plain_7x8_two']} · corrected {v['corrected_7x8_two']} "
              f"· 4x7 corrected {v['corrected_4x7_two']} · composes beyond labels {v['composes_beyond_labels']}")
    print(json.dumps(verdict, indent=1))


if __name__ == "__main__":
    main()
