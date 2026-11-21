"""Loop 1 of plan 24: tune the judge's decision rule on eye-called drawn canvases, test it once.

For a judge network, every blob of every labelled canvas is read once. The rule then calls a canvas
"two clear, right digits" when it has exactly two blobs, neither read as "two digits on one spot"
with probability at least `ov`, and their top digits (over classes 0 to 9, renormalised) are the
asked pair, each at least `idc` sure. `ov` and `idc` are chosen on the tune sheets only; the chosen
rule is scored once on the held-back sheets against the bar written in the A2b review file.

Usage: python judge_tune.py --weights <pt> [<pt> ...] --try <n>
"""
import argparse, itertools, json, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from judge import load, blobs, split_wide, crop28

OV = [0.5, 0.7, 0.9, 0.97, 0.99, 1.01]
IDC = [0.0, 0.3, 0.5, 0.7, 0.9]


class Ensemble:
    """The judge networks' softmax outputs averaged, returned as log-probabilities."""

    def __init__(self, nets):
        self.nets = nets

    def __call__(self, x):
        return torch.log(torch.stack([torch.softmax(n(x), 1) for n in self.nets]).mean(0))


@torch.no_grad()
def read_all(net, xs):
    out = []
    for c in xs:
        parts = []
        for ys, x in blobs(c):
            parts += split_wide(c, ys, x)
        out.append([torch.softmax(net(crop28(c, ys, x).cuda()), 1)[0].cpu().numpy() for ys, x in parts])
    return out


def call(reads, pair, ov, idc):
    if len(reads) != 2 or any(p[10] >= ov for p in reads):
        return False
    d = [p[:10] / p[:10].sum() for p in reads]
    return sorted(int(q.argmax()) for q in d) == sorted(pair) and all(q.max() >= idc for q in d)


def score(reads, y, pairs, ov, idc):
    agree, diff = [], []
    for s in range(len(y)):
        j = np.array([call(r, pairs[s], ov, idc) for r in reads[s]])
        agree.append(int((j == y[s]).sum())); diff.append(int(j.sum() - y[s].sum()))
    return float(np.mean(agree)), int(max(abs(v) for v in diff)), agree, diff


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", nargs="+", type=Path, required=True)
    ap.add_argument("--try", dest="n", type=int, required=True)
    ap.add_argument("--ensemble", action="store_true", help="average all given networks as one judge")
    a = ap.parse_args()
    L = np.load("data/mnist-canvas/24-mnist-one-model/eye-labels.npz")
    best = None
    judges = [("+".join(str(w) for w in a.weights), Ensemble([load(w, "cuda") for w in a.weights]))] if a.ensemble \
        else [(w, load(w, "cuda")) for w in a.weights]
    for w, net in judges:
        tr = [read_all(net, s) for s in L["tune_x"]]
        for ov, idc in itertools.product(OV, IDC):
            m, mx, _, _ = score(tr, L["tune_y"], L["tune_pair"], ov, idc)
            key = (m, -mx)
            if best is None or key > best[0]:
                best = (key, w, ov, idc)
    (m, negmx), w, ov, idc = best
    net = dict(judges)[w]
    te = [read_all(net, s) for s in L["test_x"]]
    tm, tmx, agree, diff = score(te, L["test_y"], L["test_pair"], ov, idc)
    res = {"try": a.n, "weights": str(w), "ov": ov, "idc": idc, "tune_mean_agreement_of_64": round(m, 2),
           "tune_max_count_diff": -negmx, "test_mean_agreement_of_64": round(tm, 2), "test_max_count_diff": tmx,
           "test_per_sheet": dict(zip(L["test_name"].tolist(), zip(agree, diff))),
           "passes": bool(tm >= 58 and tmx <= 4)}
    Path("data/mnist-canvas/24-mnist-one-model/judge-loop").mkdir(parents=True, exist_ok=True)
    Path(f"data/mnist-canvas/24-mnist-one-model/judge-loop/try{a.n}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
