"""Can the product draw a layout no training picture holds? Bradley's factorized composition on the toy.

Four exact Gaussian-mixture experts, spread SIGMA0 = 0.1, none of them trained on a both-shapes picture
except the joint caption:
  "a red circle"   the circle at left, centre or right on a plain background
  "a blue square"  the square at left, centre or right on the same background
  background       the plain background alone (the empty prompt)
  joint caption    only circle-left with square-right and square-left with circle-right

The composition is circle x square / background, which is e_circle + e_square - e_background (guidance
1). Its step-0 form is a mixture over pairings (circle at i, square at j) with mean m_i + m_j - m_b,
spread SIGMA0, and weight exp(<m_i - m_b, m_j - m_b> / SIGMA0^2): zero dot product, so equal weight,
for shapes at different positions; a large dot product of either sign for shapes at the same position.

Run on two backgrounds: white, where red and blue are both darker than the background in every channel
(the dot product on overlapping pixels is positive), and grey 0.5, where they differ from it in opposite
directions in two channels (negative).

THE BAR, FIXED BEFORE THE FIRST RUN (exact composition, grey background)
  support   at least 100 of 300 draws land on the four apart layouts no training picture holds, while
            the joint caption puts 0 of 300 there
  null      under 10 of 300 on those four layouts
  otherwise inconclusive
Prediction written with it: white background gives 0 of 300 on unseen layouts (the same-position
pairing dominates); grey gives about 200 of 300 (four of six equally weighted apart layouts).

    /home/molef/micromamba/envs/sdxl-local/bin/python -I toy_factorized.py
"""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import toy_sampler_hybrids as t
import toy_blob_scorer as b

OUT = t.OUT.parents[1] / "results" / "how-much-co-occurrence-the-product-needs"
POS = list(t.XPOS)                          # left, centre, right
TRAINED = {("left", "right"), ("right", "left")}   # (circle position, square position) in the joint caption
RNG = np.random.default_rng(20261104)


def picture(objs, bg):
    y, x = np.mgrid[0:t.H, 0:t.W] + 0.5
    img = np.ones((t.H, t.W, 3)) * np.asarray(bg, float)
    for kind, p in objs:
        cx = t.XPOS[p]
        if kind == "circle":
            img[(x - cx) ** 2 + (y - 8) ** 2 <= 3.6 ** 2] = t.RED
        else:
            img[(abs(x - cx) <= 3.2) & (abs(y - 8) <= 3.2)] = t.BLUE
    return img.ravel()


class Gmix:
    """A Gaussian mixture with its own step-0 spread, diffused exactly."""
    def __init__(self, mu, logw, v0):
        self.mu, self.logw, self.v0 = np.asarray(mu), np.asarray(logw, float), v0

    def score(self, x, k):
        a = t.abar[k]; var = a * self.v0 + 1 - a
        diff = x[:, None, :] - np.sqrt(a) * self.mu[None]
        lp = self.logw[None] - 0.5 * (diff ** 2).sum(-1) / var
        r = np.exp(lp - lp.max(1, keepdims=True)); r /= r.sum(1, keepdims=True)
        return -(r[:, :, None] * diff).sum(1) / var

    def draw(self, n):
        w = np.exp(self.logw - self.logw.max()); w /= w.sum()
        idx = RNG.choice(len(self.mu), size=n, p=w)
        return self.mu[idx] + np.sqrt(self.v0) * RNG.standard_normal((n, self.mu.shape[1])), idx


def experts(bg):
    s2 = t.SIGMA0 ** 2
    circ = Gmix([picture([("circle", p)], bg) for p in POS], np.log(np.ones(3) / 3), s2)
    sq = Gmix([picture([("square", p)], bg) for p in POS], np.log(np.ones(3) / 3), s2)
    back = Gmix([picture([], bg)], [0.0], s2)
    joint = Gmix([picture([("circle", c), ("square", s)], bg) for c, s in sorted(TRAINED)],
                 np.log([0.5, 0.5]), s2)
    pairs = [(c, s) for c in POS for s in POS]
    mb = back.mu[0]
    mu = [circ.mu[POS.index(c)] + sq.mu[POS.index(s)] - mb for c, s in pairs]
    logw = [np.log(1 / 9) + (circ.mu[POS.index(c)] - mb) @ (sq.mu[POS.index(s)] - mb) / s2 for c, s in pairs]
    return circ, sq, back, joint, Gmix(mu, logw, s2), pairs


def reverse(score):
    x = RNG.standard_normal((t.N, t.D))
    for k in range(t.T, 0, -1):
        a_t, a_p = t.abar[k], t.abar[k - 1]
        al = a_t / a_p; be = 1 - al
        eps = -np.sqrt(1 - a_t) * score(x, k)
        x = (x - be / np.sqrt(1 - a_t) * eps) / np.sqrt(al)
        if k > 1:
            x = x + np.sqrt(be * (1 - a_p) / (1 - a_t)) * RNG.standard_normal(x.shape)
    return x


def layouts(x, bg):
    """Read each picture as (circle position, square position), 'overlaid', 'one shape' or 'unreadable'."""
    calls, info = b.read(np.clip(x, 0, 1), bg, smooth=True, purple=False)
    out = []
    for c, d in zip(calls, info):
        if c == "red and blue apart":
            near = lambda v: min(POS, key=lambda p: abs(t.XPOS[p] - v))
            out.append((near(d["red x"]), near(d["blue x"])))
        else:
            out.append({"purple or overlaid": "overlaid or unreadable", "red only": "one shape",
                        "blue only": "one shape", "nothing readable": "overlaid or unreadable"}[c])
    return out


def tally(lays):
    unseen = sum(1 for l in lays if isinstance(l, tuple) and l not in TRAINED)
    seen = sum(1 for l in lays if isinstance(l, tuple) and l in TRAINED)
    other = {k: sum(1 for l in lays if l == k) for k in ("overlaid or unreadable", "one shape")}
    per = {f"circle {c}, square {s}": sum(1 for l in lays if l == (c, s)) for c in POS for s in POS if c != s}
    return {"unseen layouts": unseen, "trained layouts": seen, **other, "per layout": per}


def scorer_check(bg, n=200):
    """The smoothed scorer's hit rate on noisy copies of known pictures on this background."""
    rng = np.random.default_rng(0)
    known = {f"circle {p}": ([("circle", p)], "one shape") for p in POS}
    known.update({f"square {p}": ([("square", p)], "one shape") for p in POS})
    known.update({f"circle {c}, square {s}": ([("circle", c), ("square", s)], (c, s))
                  for c in POS for s in POS if c != s})
    out = {}
    for name, (objs, want) in known.items():
        img = picture(objs, bg)
        out[name] = sum(l == want for l in layouts(img + t.SIGMA0 * rng.standard_normal((n, img.size)), bg)) / n
    over = (picture([("circle", "centre")], bg) + picture([("square", "centre")], bg)) / 2
    out["overlaid centre"] = sum(l == "overlaid or unreadable" for l in
                                 layouts(over + t.SIGMA0 * rng.standard_normal((n, over.size)), bg)) / n
    return out


def main():
    res, keep = {}, {}
    for name, bg in (("white", (1.0, 1.0, 1.0)), ("grey", (0.5, 0.5, 0.5))):
        chk = scorer_check(bg)
        print(f"scorer on {name}: worst {min(chk.values()):.2f} ({min(chk, key=chk.get)})", flush=True)
        res.setdefault("scorer check", {})[name] = chk
    for name, bg in (("white", (1.0, 1.0, 1.0)), ("grey", (0.5, 0.5, 0.5))):
        circ, sq, back, joint, prod, pairs = experts(bg)
        w = np.exp(prod.logw - prod.logw.max()); w /= w.sum()
        xj, _ = joint.draw(t.N)
        xe, idx = prod.draw(t.N)
        x_sum = reverse(lambda x, k: circ.score(x, k) + sq.score(x, k) - back.score(x, k))
        x_true = reverse(prod.score)
        exact_by_index = {"unseen layouts": int(sum(pairs[i][0] != pairs[i][1] and pairs[i] not in TRAINED
                                                    for i in idx)),
                          "same-position pairings": int(sum(pairs[i][0] == pairs[i][1] for i in idx))}
        r = {"pairing weights": {f"circle {c}, square {s}": float(v) for (c, s), v in zip(pairs, w)},
             "joint caption": tally(layouts(xj, bg)),
             "exact composition, read by the scorer": tally(layouts(xe, bg)),
             "exact composition, by the pairing drawn": exact_by_index,
             "reverse diffusion, summed noised scores": tally(layouts(x_sum, bg)),
             "reverse diffusion, true noised-composition score": tally(layouts(x_true, bg))}
        res[name] = r
        keep[name] = (xj[:8], xe[:8], x_sum[:8], x_true[:8])
        print(f"{name}: same-position pairing weight {sum(w[i] for i, p in enumerate(pairs) if p[0] == p[1]):.3f} | "
              f"unseen layouts of 300: joint {r['joint caption']['unseen layouts']}, exact "
              f"{exact_by_index['unseen layouts']} (scorer {r['exact composition, read by the scorer']['unseen layouts']}), "
              f"summed scores {r['reverse diffusion, summed noised scores']['unseen layouts']}, true score "
              f"{r['reverse diffusion, true noised-composition score']['unseen layouts']}", flush=True)
        for k in ("exact composition, read by the scorer", "reverse diffusion, summed noised scores",
                  "reverse diffusion, true noised-composition score"):
            print("   ", k, {kk: v for kk, v in r[k].items() if kk != "per layout"})
    g = res["grey"]["exact composition, by the pairing drawn"]["unseen layouts"]
    j = res["grey"]["joint caption"]["unseen layouts"]
    verdict = "support" if g >= 100 and j == 0 else ("null" if g < 10 else "inconclusive")
    print("VERDICT:", verdict)
    np.savez_compressed(OUT / "toy-factorized-samples.npz",
                        **{f"{n}_{i}": s for n, ss in keep.items() for i, s in enumerate(ss)})
    json.dump({"what": "circle x square / background on the exact toy, white and grey backgrounds, "
                       "against a joint caption trained on two layouts", "verdict": verdict,
               "bar": "support: exact composition on grey puts >= 100 of 300 on the four unseen apart "
                      "layouts and the joint caption 0; null: under 10", "draws": t.N, "results": res},
              open(OUT / "toy-factorized.json", "w"), indent=1)
    print("wrote", OUT / "toy-factorized.json")


if __name__ == "__main__":
    main()
