"""The MNIST stage's training data: 32 x 64 canvases of real MNIST digits, each drawn with the label it
is trained under. Built on the GPU, fresh every batch, so training never sees the same canvas twice.

The design is settled in 4-inclusive-captions/plans/route-inclusive-captions-pilot.md, step 4:

  canvas       32 x 64, digits in 28-pixel boxes, pixel values scaled to [-1, 1]
  one digit    90% of canvases, at a uniformly random place
  two digits   10% of canvases, one per half (columns 0 and 32, plus 0 to 4 pixels of jitter), order
               swapped half the time; half of them hold a 7 and an 8, the other half are spread evenly
               over the other 39 seen pairs
  held out     0 & 3, 1 & 9, 3 & 4, 3 & 6, 4 & 7 never share a canvas (random.Random(0) over the 45
               pairs, the classifier-guidance route's rule)
  labels       blank on 10% of draws; a one-digit canvas takes its digit otherwise; a two-digit canvas
               holding a and b splits the other 90% by weight: "a and b" 1, "a" and "b" each `dial`

Label ids: 0-9 the digits, 10-54 the 45 joint labels in PAIRS order, 55 blank.

Run on its own, it prints the piece-1 counts: canvases per kind and label draws per kind against the
design, for every dial setting.
"""

import itertools
import os
import random
from pathlib import Path

import numpy as np
import torch

H, W, D = 32, 64, 28
MNIST_RAW = Path(os.environ.get("MNIST_RAW", "/home/molef/PhD/replications/original-superdiff/data/MNIST/raw"))
PAIRS = list(itertools.combinations(range(10), 2))
HELD_OUT = sorted(random.Random(0).sample(PAIRS, 5))
SEEN = [p for p in PAIRS if p not in HELD_OUT]
OTHER_SEEN = [p for p in SEEN if p != (7, 8)]
TWO_SHARE, BLANK_RATE, SEVEN_EIGHT_SHARE = 0.10, 0.10, 0.5
BLANK = 55
N_LABELS = 56
DIALS = (0.0, 0.1, 0.3, 0.6, 1.0)

assert HELD_OUT == [(0, 3), (1, 9), (3, 4), (3, 6), (4, 7)], HELD_OUT
assert (7, 8) in SEEN and len(OTHER_SEEN) == 39


def joint_id(a, b):
    return 10 + PAIRS.index((min(a, b), max(a, b)))


def label_name(i):
    if i < 10:
        return str(i)
    if i == BLANK:
        return "blank"
    a, b = PAIRS[i - 10]
    return f"{a} and {b}"


def load_mnist(split="train"):
    pre = "train" if split == "train" else "t10k"
    x = np.fromfile(MNIST_RAW / f"{pre}-images-idx3-ubyte", dtype=np.uint8, offset=16).reshape(-1, D, D)
    y = np.fromfile(MNIST_RAW / f"{pre}-labels-idx1-ubyte", dtype=np.uint8, offset=8)
    return x, y


class CanvasMaker:
    """Draws training batches on `device` from one MNIST split."""

    def __init__(self, dial, device="cuda", split="train", seed=0):
        x, y = load_mnist(split)
        self.dev = device
        self.digits = torch.tensor(x, device=device, dtype=torch.float32) / 127.5 - 1.0   # [-1, 1]
        self.y = torch.tensor(y, device=device, dtype=torch.long)
        self.by = [torch.nonzero(self.y == d).flatten() for d in range(10)]
        self.dial = float(dial)
        self.g = torch.Generator(device=device).manual_seed(seed)
        self.seen_other = torch.tensor(OTHER_SEEN, device=device)
        self.joint_of = torch.full((10, 10), -1, device=device, dtype=torch.long)
        for a, b in PAIRS:
            self.joint_of[a, b] = self.joint_of[b, a] = joint_id(a, b)

    def _rand(self, *shape):
        return torch.rand(*shape, device=self.dev, generator=self.g)

    def _digit_images(self, d):
        """One random MNIST image of each digit in the long tensor d."""
        out = torch.empty(len(d), D, D, device=self.dev)
        for k in range(10):
            m = d == k
            n = int(m.sum())
            if n:
                pool = self.by[k]
                pick = pool[torch.randint(len(pool), (n,), device=self.dev, generator=self.g)]
                out[m] = self.digits[pick]
        return out

    def _paste(self, canvas, imgs, rows, cols):
        """Paste imgs [n,28,28] into canvas [n,H,W] at (rows, cols) by taking the brighter pixel."""
        n = len(imgs)
        ar = torch.arange(D, device=self.dev)
        r = (rows[:, None] + ar)[:, :, None].expand(n, D, D)
        c = (cols[:, None] + ar)[:, None, :].expand(n, D, D)
        b = torch.arange(n, device=self.dev)[:, None, None].expand(n, D, D)
        canvas[b, r, c] = torch.maximum(canvas[b, r, c], imgs)

    def batch(self, n):
        """n canvases [n,1,H,W] in [-1,1], their label ids [n], and what they hold [n,2] (-1 if none)."""
        two = self._rand(n) < TWO_SHARE
        holds = torch.full((n, 2), -1, device=self.dev, dtype=torch.long)
        # one-digit canvases take digits at MNIST's own frequencies
        holds[:, 0] = self.y[torch.randint(len(self.y), (n,), device=self.dev, generator=self.g)]
        nt = int(two.sum())
        if nt:
            seven_eight = self._rand(nt) < SEVEN_EIGHT_SHARE
            other = self.seen_other[torch.randint(39, (nt,), device=self.dev, generator=self.g)]
            pair = torch.where(seven_eight[:, None], torch.tensor([7, 8], device=self.dev), other)
            swap = self._rand(nt) < 0.5
            pair = torch.where(swap[:, None], pair.flip(1), pair)
            holds[two] = pair
        canvas = torch.full((n, H, W), -1.0, device=self.dev)
        one = ~two
        n1 = int(one.sum())
        if n1:
            sub = canvas[one]
            self._paste(sub, self._digit_images(holds[one, 0]),
                        torch.randint(0, H - D + 1, (n1,), device=self.dev, generator=self.g),
                        torch.randint(0, W - D + 1, (n1,), device=self.dev, generator=self.g))
            canvas[one] = sub
        if nt:
            sub = canvas[two]
            for half, x0 in ((0, 0), (1, 32)):
                self._paste(sub, self._digit_images(holds[two, half]),
                            torch.randint(0, H - D + 1, (nt,), device=self.dev, generator=self.g),
                            x0 + torch.randint(0, 5, (nt,), device=self.dev, generator=self.g))
            canvas[two] = sub
        labels = self.labels(holds, two)
        return canvas[:, None], labels, holds

    def labels(self, holds, two):
        n = len(holds)
        lab = holds[:, 0].clone()
        if two.any():
            a, b = holds[two, 0], holds[two, 1]
            w = torch.tensor([1.0, self.dial, self.dial], device=self.dev)
            u = self._rand(int(two.sum())) * w.sum()
            pick = (u[:, None] >= torch.cumsum(w, 0)[None, :-1]).sum(1)      # 0 joint, 1 a, 2 b
            jl = self.joint_of[a, b]
            lab[two] = torch.where(pick == 0, jl, torch.where(pick == 1, a, b))
        blank = self._rand(n) < BLANK_RATE
        lab[blank] = BLANK
        return lab


class ShareCanvasMaker(CanvasMaker):
    """The share run's training data (plan 03): the label first, from one fixed mix, then the canvas.

    Labels: blank BLANK_RATE; the 40 seen joint labels JOINT_RATE ("7 and 8" half of it, the other 39
    evenly); single digits the rest, at MNIST's own digit frequencies. This is dial 0's label mix.
    Canvases: a joint label "a and b" shows a and b; a single digit d shows d beside a second digit with
    probability `share` (8 for 7, 7 for 8, otherwise uniform over d's seen partners) and d alone otherwise;
    a blank draw shows a canvas from dial 0's canvas mix. At share 0 this is dial 0's distribution.
    """

    JOINT_RATE = TWO_SHARE * (1 - BLANK_RATE)

    def __init__(self, share, device="cuda", split="train", seed=0):
        super().__init__(0.0, device, split, seed)
        self.share = float(share)
        partners = {d: sorted({b if a == d else a for a, b in SEEN if d in (a, b)}) for d in range(10)}
        partners[7], partners[8] = [8], [7]
        self.n_part = torch.tensor([len(partners[d]) for d in range(10)], device=device)
        self.part = torch.full((10, 9), -1, device=device, dtype=torch.long)
        for d, ps in partners.items():
            self.part[d, :len(ps)] = torch.tensor(ps, device=device)
        self.partners = partners

    def batch(self, n):
        u = self._rand(n)
        blank = u < BLANK_RATE
        joint = (~blank) & (u < BLANK_RATE + self.JOINT_RATE)
        single = ~(blank | joint)
        holds = torch.full((n, 2), -1, device=self.dev, dtype=torch.long)
        lab = torch.full((n,), BLANK, device=self.dev, dtype=torch.long)
        nj = int(joint.sum())
        if nj:
            seven_eight = self._rand(nj) < SEVEN_EIGHT_SHARE
            other = self.seen_other[torch.randint(len(OTHER_SEEN), (nj,), device=self.dev, generator=self.g)]
            pair = torch.where(seven_eight[:, None], torch.tensor([7, 8], device=self.dev), other)
            holds[joint] = pair
            lab[joint] = self.joint_of[pair[:, 0], pair[:, 1]]
        ns = int(single.sum())
        if ns:
            d = self.y[torch.randint(len(self.y), (ns,), device=self.dev, generator=self.g)]
            with_second = self._rand(ns) < self.share
            k = (self._rand(ns) * self.n_part[d]).long().clamp(max=8)
            second = torch.where(with_second, self.part[d, k], torch.full_like(d, -1))
            holds[single] = torch.stack([d, second], 1)
            lab[single] = d
        nb = int(blank.sum())
        if nb:
            _, _, hb = CanvasMaker.batch(self, nb)
            holds[blank] = hb
        two = holds[:, 1] >= 0
        swap = two & (self._rand(n) < 0.5)
        holds[swap] = holds[swap].flip(1)
        return self._render(holds), lab, holds

    def _render(self, holds):
        n = len(holds)
        two = holds[:, 1] >= 0
        canvas = torch.full((n, H, W), -1.0, device=self.dev)
        one, nt = ~two, int(two.sum())
        n1 = int(one.sum())
        if n1:
            sub = canvas[one]
            self._paste(sub, self._digit_images(holds[one, 0]),
                        torch.randint(0, H - D + 1, (n1,), device=self.dev, generator=self.g),
                        torch.randint(0, W - D + 1, (n1,), device=self.dev, generator=self.g))
            canvas[one] = sub
        if nt:
            sub = canvas[two]
            for half, x0 in ((0, 0), (1, 32)):
                self._paste(sub, self._digit_images(holds[two, half]),
                            torch.randint(0, H - D + 1, (nt,), device=self.dev, generator=self.g),
                            x0 + torch.randint(0, 5, (nt,), device=self.dev, generator=self.g))
            canvas[two] = sub
        return canvas[:, None]


def share_report(shares=(0.0, 0.1, 0.2, 0.4, 0.8), n=400_000, device="cuda"):
    """Plan 03's task 1.2 check: label mix and each label's share of two-digit pictures, per share."""
    for s in shares:
        cm = ShareCanvasMaker(s, device, seed=1)
        labs = torch.zeros(N_LABELS, device=device)
        c = {"7": [0, 0], "8": [0, 0], "4": [0, 0]}
        for _ in range(n // 4000):
            x, lab, holds = cm.batch(4000)
            two = holds[:, 1] >= 0
            labs += torch.bincount(lab, minlength=N_LABELS).float()
            has = lambda v: (holds[:, 0] == v) | (holds[:, 1] == v)
            for name, d, other in (("7", 7, 8), ("8", 8, 7), ("4", 4, None)):
                m = lab == d
                c[name][0] += int(m.sum())
                c[name][1] += int((m & two & (has(other) if other is not None else two)).sum())
            held = torch.zeros_like(two)
            for a, b in HELD_OUT:
                held |= two & (((holds[:, 0] == a) & (holds[:, 1] == b)) | ((holds[:, 0] == b) & (holds[:, 1] == a)))
            assert not held.any(), "a held-out pair reached a canvas"
        tot = labs.sum()
        joint = labs[10:BLANK].sum() / tot
        sh = {k: v[1] / v[0] for k, v in c.items()}
        print(f"share {s:g}: blank {labs[BLANK] / tot:.4f} (0.10) · joint {joint:.4f} (0.09) · single "
              f"{1 - joint - labs[BLANK] / tot:.4f} (0.81) · \"7 and 8\" {labs[joint_id(7, 8)] / tot:.4f} (0.045) · "
              f'"7" with an 8 {sh["7"]:.4f} · "8" with a 7 {sh["8"]:.4f} · "4" with a second digit {sh["4"]:.4f} (each {s:g})')
        for k, v in sh.items():
            assert abs(v - s) < 0.005, f"share {s}: {k} draws {v:.4f} off the design"


def design_shares(dial):
    """The label shares the design implies for a two-digit canvas's non-blank draws."""
    tot = 1 + 2 * dial
    return {"joint": 1 / tot, "single": 2 * dial / tot}


def piece1_report(n=400_000, device="cuda"):
    print(f"held out: {HELD_OUT}; seen pairs: {len(SEEN)}; label ids: {N_LABELS}")
    for dial in DIALS:
        cm = CanvasMaker(dial, device, seed=1)
        kinds = torch.zeros(4, device=device)        # one, two 7&8, two other, -
        labs = torch.zeros(N_LABELS, device=device)
        seven_with_second = seven_total = 0
        for _ in range(n // 4000):
            x, lab, holds = cm.batch(4000)
            two = holds[:, 1] >= 0
            is78 = two & (((holds[:, 0] == 7) & (holds[:, 1] == 8)) | ((holds[:, 0] == 8) & (holds[:, 1] == 7)))
            kinds[0] += (~two).sum(); kinds[1] += is78.sum(); kinds[2] += (two & ~is78).sum()
            labs += torch.bincount(lab, minlength=N_LABELS).float()
            s7 = lab == 7
            seven_total += int(s7.sum()); seven_with_second += int((s7 & two).sum())
            held = torch.zeros_like(two)
            for a, b in HELD_OUT:
                held |= two & (((holds[:, 0] == a) & (holds[:, 1] == b)) | ((holds[:, 0] == b) & (holds[:, 1] == a)))
            assert not held.any(), "a held-out pair reached a canvas"
        k = kinds / kinds.sum()
        j78 = labs[joint_id(7, 8)].item()
        two78_draws = kinds[1].item() * (1 - BLANK_RATE)
        print(f"dial {dial:g}: one digit {k[0]:.4f} (design 0.90) · 7&8 {k[1]:.4f} (0.05) · other pairs "
              f"{k[2]:.4f} (0.05) · blank {labs[BLANK] / labs.sum():.4f} (0.10) · "
              f'"7 and 8" share of 7&8 non-blank draws {j78 / two78_draws:.4f} '
              f"(design {design_shares(dial)['joint']:.4f}) · "
              f'"7" draws with a second digit {seven_with_second / seven_total:.4f}')


if __name__ == "__main__":
    import sys
    share_report() if "--share" in sys.argv else piece1_report()
