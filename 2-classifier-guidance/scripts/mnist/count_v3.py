"""Loop 1's best judge (the third network, its rule tuned in try 2) as a counter (plan 24, loop 2).

A canvas counts as two clear, right digits when it holds exactly two blobs whose top digits over
classes 0 to 9 are the asked pair, each at least 0.9 sure; the "two digits on one spot" class is
ignored. It missed loop 1's per-sheet bar (worst gap 6 tiles) and counts about 5% low in total, so
it ranks loop 2's rounds on the tuning pairs; held-out counts stay by eye.

Usage: python count_v3.py --judge <judge-v3.pt> canvases <dir with canvases.npz>   -> counts_v3.json
       python count_v3.py --judge <judge-v3.pt> draws <draws.npz> <out.npz>         relabelled draws
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from judge import load
from judge_tune import read_all, call

OV, IDC = 1.01, 0.9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", type=Path, required=True)
    ap.add_argument("mode", choices=["canvases", "draws"])
    ap.add_argument("paths", nargs="+", type=Path)
    a = ap.parse_args()
    net = load(a.judge, "cuda")
    if a.mode == "canvases":
        c = np.load(a.paths[0] / "canvases.npz")
        out = {}
        for k in c.files:
            pair = tuple(int(d) for d in k.split("|")[1])
            out[k] = int(sum(call(r, pair, OV, IDC) for r in read_all(net, c[k])))
            print(k, out[k], flush=True)
        (a.paths[0] / "counts_v3.json").write_text(json.dumps(out, indent=1))
    else:
        d = dict(np.load(a.paths[0]))
        reads = read_all(net, d["x"])
        ok = [call(r, tuple(int(v) for v in p), OV, IDC) for r, p in zip(reads, d["pair"])]
        d["call"] = np.array(["two clear, right digits" if o else "not" for o in ok])
        d["call_v1"] = np.load(a.paths[0])["call"]
        np.savez_compressed(a.paths[1], **d)
        print(json.dumps({"positives_v3": int(sum(ok)), "positives_v1": int((d["call_v1"] == "two clear, right digits").sum()),
                          "of": len(ok)}))


if __name__ == "__main__":
    main()
