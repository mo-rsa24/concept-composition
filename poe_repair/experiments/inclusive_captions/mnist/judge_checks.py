"""The shared judge, pinned, and this step's two checks on it.

The judge is the classifier-guidance route's `2-classifier-guidance/scripts/mnist/judge.py` at commit
5bff268f. Its three source files are read from that commit with `git show` into data/mnist/judge/src/,
never from the other session's working tree, and its weights are copied into data/mnist/judge/ with
their sha256 written beside them. Nothing under 2-classifier-guidance/ is edited. Runs on the CPU, so
the drawing on the card is not slowed.

This step's judge differs from the shared one in two ways. Single-label rows read each blob with the
ten digit classes only (`reads_of(..., ten=True)`); product rows keep the eleventh class, "two digits on
one spot". And the 0.90 confidence line that calls a blob fused applies only when the tile holds one blob
(`is_fused`).

Calls. A two-digit tile takes the judge's own ladder, renamed: damaged, one digit, fused, two digits
with a wrong identity, two separate right digits. A single-label tile asked for digit d is called from
the same blob readings: no blob or more than two, or any blob not a confident digit, is damaged; one
blob read d is right digit alone, any other digit is wrong digit; two blobs with a d among them is
right digit with a second, without one is wrong digit.

  check1   500 pasted canvases per call with a known answer, built from MNIST's test digits; each
           call must be right on at least bar.JUDGE_PASTED_MIN of its 500
  check2 <samples .npz> <eye-calls .csv> [row]
           the judge on a row's first 64 seeds (default the plain 7 x 8 product; "7 alone" for
           check 3) against the blind eye calls; must agree on at least bar.JUDGE_EYE_MIN of 64

Usage (from the repo root):
    python -I poe_repair/experiments/inclusive_captions/mnist/judge_checks.py check1
    python -I poe_repair/experiments/inclusive_captions/mnist/judge_checks.py check2 \
        data/mnist/samples/dial-0-seed-1.npz <eye-calls csv>
"""

import argparse
import csv
import hashlib
import importlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bar  # noqa: E402

REPO = Path(__file__).resolve().parents[4]
PIN = "5bff268f"
SRC_DIR = "2-classifier-guidance/scripts/mnist"
THEIR_WEIGHTS = REPO / "2-classifier-guidance/data/mnist-canvas/judge/judge.pt"
HERE = REPO / "data/mnist/judge"
MNIST_RAW = Path("/home/molef/PhD/replications/original-superdiff/data/MNIST/raw")
H, W, D = 32, 64, 28
RENAME = {"two clear, right digits": "two separate right digits",
          "two digits, wrong identity": "two digits with a wrong identity"}
PAIR_CALLS = ["damaged", "one digit", "fused", "two digits with a wrong identity", "two separate right digits"]
SINGLE_CALLS = ["damaged", "wrong digit", "right digit with a second", "right digit alone"]


def pinned_judge():
    src = HERE / "src"
    src.mkdir(parents=True, exist_ok=True)
    for f in ("judge.py", "judge_sheet.py", "canvas_sheet.py"):
        text = subprocess.run(["git", "-C", str(REPO), "show", f"{PIN}:{SRC_DIR}/{f}"], check=True,
                              capture_output=True, text=True).stdout
        (src / f).write_text(text)
    w = HERE / "judge.pt"
    if not w.exists():
        shutil.copy(THEIR_WEIGHTS, w)
    sha = hashlib.sha256(w.read_bytes()).hexdigest()
    (HERE / "pin.json").write_text(json.dumps({"commit": PIN, "weights_from": str(THEIR_WEIGHTS),
                                               "weights_sha256": sha}, indent=1))
    sys.path.insert(0, str(src))
    j = importlib.import_module("judge")
    return j, j.load(w, "cpu"), sha


def reads_of(j, net, canvas, ten=False):
    """Each blob's top class (0-9, or 10 for two digits on one spot unless ten) and its probability."""
    parts = []
    for ys, xs in j.blobs(canvas):
        parts += j.split_wide(canvas, ys, xs)
    out = []
    with torch.no_grad():
        for ys, xs in parts:
            # Single-label rows read each blob with the ten digit classes only (ten=True): the
            # shared judge's eleventh class, "two digits on one spot", was trained on pasted
            # overlays of real digits and fires on the model's own clean single digits (dial 0,
            # seed 1: 24 of 66 "7 alone" and 79 of 123 "4 alone" tiles called damaged were read as
            # that class). Product rows keep it, since it is how a fused glyph is caught there.
            logits = net(j.crop28(canvas, ys, xs))
            p = torch.softmax(logits[:, :10] if ten else logits, 1)[0].numpy()
            out.append((int(p.argmax()), float(p.max())))
    return out


def is_fused(j, r, n_blobs):
    # A blob read as "two digits on one spot" is fused. The 0.90 confidence line applies only to a
    # tile holding one blob, where a fused glyph hides; two separate blobs each count as their top
    # digit. The shared judge applied the line to every blob, which called 43 of 500 clean pasted
    # 7-beside-8 canvases fused (check 1).
    return r[0] == j.TWO_ON_ONE_SPOT or (n_blobs == 1 and r[1] < j.CONFIDENT)


def pair_call(j, net, canvas, asked):
    reads = reads_of(j, net, canvas)
    if not reads or len(reads) > 2:
        return "damaged"
    if any(is_fused(j, r, len(reads)) for r in reads):
        return "fused"
    if len(reads) == 1:
        return "one digit"
    ok = sorted(r[0] for r in reads) == sorted(asked)
    return "two separate right digits" if ok else "two digits with a wrong identity"


def single_call(j, net, canvas, d):
    reads = reads_of(j, net, canvas, ten=True)
    if not reads or len(reads) > 2 or any(is_fused(j, r, len(reads)) for r in reads):
        return "damaged"
    digits = [r[0] for r in reads]
    if len(reads) == 1:
        return "right digit alone" if digits[0] == d else "wrong digit"
    return "right digit with a second" if d in digits else "wrong digit"


class Paster:
    def __init__(self, seed):
        x = np.fromfile(MNIST_RAW / "t10k-images-idx3-ubyte", np.uint8, offset=16).reshape(-1, D, D)
        y = np.fromfile(MNIST_RAW / "t10k-labels-idx1-ubyte", np.uint8, offset=8)
        self.x, self.by, self.rng = x, {d: np.flatnonzero(y == d) for d in range(10)}, np.random.default_rng(seed)

    def digit(self, d):
        return self.x[self.rng.choice(self.by[d])]

    def one(self, d):
        c = np.zeros((H, W), np.uint8)
        r, k = self.rng.integers(0, H - D + 1), self.rng.integers(0, W - D + 1)
        c[r:r + D, k:k + D] = self.digit(d)
        return c

    def two(self, a, b):
        if self.rng.random() < 0.5:
            a, b = b, a
        c = np.zeros((H, W), np.uint8)
        for d, x0 in ((a, 0), (b, 32)):
            r, k = self.rng.integers(0, H - D + 1), x0 + self.rng.integers(0, 5)
            c[r:r + D, k:k + D] = np.maximum(c[r:r + D, k:k + D], self.digit(d))
        return c

    def on_one_spot(self, a, b):
        c = np.zeros((H, W), np.uint8)
        k, r = self.rng.integers(8, W - D - 8), self.rng.integers(0, H - D + 1)
        c[r:r + D, k:k + D] = self.digit(a)
        dx, dy = self.rng.integers(-5, 6), self.rng.integers(-2, 3)
        r2, k2 = min(max(r + dy, 0), H - D), k + dx
        c[r2:r2 + D, k2:k2 + D] = np.maximum(c[r2:r2 + D, k2:k2 + D], self.digit(b))
        return c

    def specks(self):
        c = np.zeros((H, W), np.uint8)
        for _ in range(self.rng.integers(0, 4)):
            r, k = self.rng.integers(0, H - 3), self.rng.integers(0, W - 3)
            c[r:r + 3, k:k + 3] = 255
        return c

    def other(self, avoid):
        return int(self.rng.choice([d for d in range(10) if d not in avoid]))


def check1(n=500):
    j, net, sha = pinned_judge()
    pz = Paster(500)
    pair_known = {
        "damaged": lambda: pz.specks(),
        "one digit": lambda: pz.one(int(pz.rng.choice([7, 8]))),
        "fused": lambda: pz.on_one_spot(7, 8),
        "two digits with a wrong identity": lambda: pz.two(*[(7, 1), (7, 7), (8, 3), (8, 8), (7, 4)][pz.rng.integers(5)]),
        "two separate right digits": lambda: pz.two(7, 8),
    }
    single_known = {
        "damaged": lambda d: pz.specks(),
        "wrong digit": lambda d: pz.one(pz.other([d])),
        "right digit with a second": lambda d: pz.two(d, pz.other([d])),
        "right digit alone": lambda d: pz.one(d),
    }
    res = {"judge_commit": PIN, "weights_sha256": sha, "n_per_call": n, "pair": {}, "single": {}}
    for call, make in pair_known.items():
        got = [pair_call(j, net, make(), (7, 8)) for _ in range(n)]
        res["pair"][call] = {"right": sum(g == call for g in got),
                             "called_as": {c: got.count(c) for c in PAIR_CALLS if got.count(c)}}
    for call, make in single_known.items():
        got = []
        for _ in range(n):
            d = int(pz.rng.choice([7, 8, 4]))
            got.append(single_call(j, net, make(d), d))
        res["single"][call] = {"right": sum(g == call for g in got),
                               "called_as": {c: got.count(c) for c in SINGLE_CALLS if got.count(c)}}
    floor = int(np.ceil(bar.JUDGE_PASTED_MIN * n))
    fails = [f"{k}: {call}" for k in ("pair", "single") for call, v in res[k].items() if v["right"] < floor]
    res["floor"], res["passed"] = floor, not fails
    HERE.mkdir(parents=True, exist_ok=True)
    (HERE / "check1.json").write_text(json.dumps(res, indent=1))
    for k in ("pair", "single"):
        for call, v in res[k].items():
            print(f"{k:6s} {call:34s} {v['right']:3d} of {n} right   called as {v['called_as']}")
    print(f"check 1 {'PASSED' if not fails else 'FAILED'} (floor {floor} of {n})" + ("" if not fails else f": {fails}"))
    return res


def check2(npz, eye_csv, row="7 x 8 plain"):
    j, net, sha = pinned_judge()
    tiles = np.load(npz)[row][:64]
    single = row.endswith(" alone")
    digit = int(row.split()[0]) if single else None
    eye = {int(r["seed"]): r.get("final_call") or r["eye_call"] for r in csv.DictReader(open(eye_csv))}
    # A product row is graded on the verdict's call only: does the tile hold two separate right
    # digits, yes or no. Fused, one digit and the rest all count as "did not compose", and the eye
    # and the judge split fused glyphs from single 8s differently (dial 0.1: 7 of 64 tiles) without
    # that changing any count the verdict reads. A single-label row keeps its full call.
    SEP = "two separate right digits"
    rows, agree, full = [], 0, 0
    for s, t in enumerate(tiles):
        jc = single_call(j, net, t, digit) if single else pair_call(j, net, t, (7, 8))
        ok = jc == eye[s] if single else (jc == SEP) == (eye[s] == SEP)
        agree += ok
        full += jc == eye[s]
        rows.append({"seed": s, "eye_call": eye[s], "judge_call": jc, "agree": ok})
    out = Path(eye_csv).with_name(Path(eye_csv).stem + "-against-judge.csv")
    with open(out, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader(), wr.writerows(rows)
    ok = agree >= bar.JUDGE_EYE_MIN
    print(f"check on {Path(npz).name}, row {row}: judge agrees with the eye calls on {agree} of 64 "
          f"(floor {bar.JUDGE_EYE_MIN}) · {'PASSED' if ok else 'FAILED'} · per tile in {out}"
          + ("" if single else f"\n  graded on two separate right digits, yes or no; on the full call they agree on {full} of 64"))
    for r in rows:
        if not r["agree"]:
            print(f"  seed {r['seed']}: eye {r['eye_call']} · judge {r['judge_call']}")
    return agree


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["check1", "check2"])
    ap.add_argument("args", nargs="*")
    a = ap.parse_args()
    torch.set_num_threads(4)
    check1() if a.cmd == "check1" else check2(*a.args)


if __name__ == "__main__":
    main()
