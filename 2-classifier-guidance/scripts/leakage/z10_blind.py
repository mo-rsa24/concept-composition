#!/usr/bin/env python3
"""Loop 12 step Z10: per-pair grids, and the blind calls batch by batch as each job's pictures land.

Design: artifacts/notes/progress/leakage/10-co-occurrence-dial/design.md, "How pictures are counted" and "The bar".

  grid   <renders dir> <pair slug> <out png>                  four seeds down, the six weights across, weight under each column
  pack   <renders dir> <batch dir> <key dir> <slug> [<slug> ...]   neutral ids for those pairs' pictures; the key kept apart
  count  <batch dir> <key dir> <calls1> <calls2> [<calls3>]   settled calls; one, hybrid, two bodies, two clear per pair and weight
  bar    <key dir> <settled json> [<settled json> ...]         every batch merged: T, H, O per weight over 32 and the design's bar
"""
import csv
import json
import random
import shutil
import sys
from pathlib import Path

WEIGHTS = ("0.00", "0.25", "0.50", "0.75", "1.00", "1.25")
SEEDS = (6001, 6002, 6003, 6004)
RUNGS = ("one", "hybrid", "two_bodies", "two_clear")


def grid(renders, slug, out):
    from PIL import Image, ImageDraw, ImageFont
    f = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
    s = 220
    W = Image.new("RGB", (70 + 6 * (s + 6), 40 + 4 * (s + 6) + 30), "white"); d = ImageDraw.Draw(W)
    d.text((8, 8), slug.replace("__x__", " x ").replace("_", " ") + ": the adapter's weight across, seeds 6001 to 6004 down", fill="black", font=f)
    for r, seed in enumerate(SEEDS):
        d.text((6, 40 + r * (s + 6) + s // 2 - 9), str(seed), fill="black", font=f)
        for c, w in enumerate(WEIGHTS):
            p = Path(renders) / slug / f"seed-{seed}-w{w}.png"
            W.paste(Image.open(p).convert("RGB").resize((s, s)), (70 + c * (s + 6), 40 + r * (s + 6)))
    for c, w in enumerate(WEIGHTS):
        d.text((70 + c * (s + 6) + s // 2 - 50, 40 + 4 * (s + 6) + 4), f"weight {w}", fill="black", font=f)
    W.save(out); print(out)


def pack(renders, batch, keydir, slugs):
    batch, keydir = Path(batch), Path(keydir)
    (batch / "pictures").mkdir(parents=True, exist_ok=True); keydir.mkdir(parents=True, exist_ok=True)
    items = [(sl, seed, w, Path(renders) / sl / f"seed-{seed}-w{w}.png") for sl in slugs for seed in SEEDS for w in WEIGHTS]
    missing = [str(p) for *_, p in items if not p.exists()]
    if missing:
        raise SystemExit(f"{len(missing)} missing, first {missing[0]}")
    rng = random.Random("z10-" + "-".join(slugs)); rng.shuffle(items)
    key, rows = {}, []
    for sl, seed, w, src in items:
        pid = "".join(rng.choice("abcdefghjkmnpqrstuvwxyz23456789") for _ in range(12))
        shutil.copy(src, batch / "pictures" / f"{pid}.png")
        a, b = sl.split("__x__")
        key[pid] = {"pair": sl, "seed": seed, "weight": float(w)}
        rows.append({"id": pid, "prompt": f"{a.replace('_', ' ')} and {b.replace('_', ' ')}"})
    with open(batch / "sheet.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, ["id", "prompt"]); wr.writeheader(); wr.writerows(rows)
    name = batch.name
    (keydir / f"{name}.json").write_text(json.dumps(key, indent=1) + "\n")
    print(f"packed {len(rows)} into {batch}; key {keydir / (name + '.json')}")


def read(p):
    return {r["id"]: r for r in csv.DictReader(open(p))}


def count(batch, keydir, c1, c2, c3=None):
    batch = Path(batch); key = json.loads((Path(keydir) / f"{batch.name}.json").read_text())
    a, b = read(c1), read(c2); third = read(c3) if c3 else {}
    settled, need = {}, []
    for i in key:
        if a[i]["rung"] == b[i]["rung"]:
            settled[i] = a[i]["rung"]
        elif i in third:
            settled[i] = third[i]["rung"]
        else:
            need.append(i)
    if need:
        (batch / "need-third.txt").write_text("\n".join(need) + "\n")
        print(f"{len(need)} need a third call"); return
    table = {}
    for i, k in key.items():
        t = table.setdefault(k["pair"], {w: {r: 0 for r in RUNGS} for w in WEIGHTS})
        t[f"{k['weight']:.2f}"][settled[i]] += 1
    out = {"agreement": sum(a[i]["rung"] == b[i]["rung"] for i in key) / len(key), "settled": {i: {**key[i], "rung": settled[i]} for i in key},
           "per_pair": table}
    (batch / "settled.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({"agreement": out["agreement"], "per_pair": table}))


def bar(keydir, *settled_files):
    from scipy.stats import binom
    rows = {}
    for f in settled_files:
        rows.update(json.loads(Path(f).read_text())["settled"])
    by = {}
    for r in rows.values():
        by[(r["pair"], r["seed"], f"{r['weight']:.2f}")] = r["rung"]
    pairs = sorted({k[0] for k in by})
    cnt = {w: {rg: sum(1 for (p, s, ww), v in by.items() if ww == w and v == rg) for rg in RUNGS} for w in WEIGHTS}
    T = {w: cnt[w]["two_clear"] for w in WEIGHTS}
    delta = T["1.00"] - T["0.00"]
    won = sum(1 for p in pairs for s in SEEDS if by.get((p, s, "1.00")) == "two_clear" and by.get((p, s, "0.00")) != "two_clear")
    lost = sum(1 for p in pairs for s in SEEDS if by.get((p, s, "0.00")) == "two_clear" and by.get((p, s, "1.00")) != "two_clear")
    pval = float(binom.sf(won - 1, won + lost, 0.5)) if won + lost else 1.0
    steps = [T[WEIGHTS[i + 1]] - T[WEIGHTS[i]] for i in range(4)]
    effect = delta >= 6 and pval < 0.05
    if not effect:
        call = "no effect on new pairs"
    elif max(steps) <= delta / 2 and min(steps) >= -2:
        call = "a dial"
    elif max(steps) >= 0.75 * delta:
        call = "a switch"
    else:
        call = "in between"
    res = {"pictures": len(by), "pairs": pairs, "counts_per_weight": cnt, "two_clear": T, "delta_0_to_1": delta, "won": won, "lost": lost,
           "p_one_sided": pval, "steps_0_to_1": steps, "call": call}
    print(json.dumps(res, indent=1)); return res


if __name__ == "__main__":
    cmd = sys.argv[1]
    {"grid": lambda: grid(*sys.argv[2:5]), "pack": lambda: pack(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5:]),
     "count": lambda: count(*sys.argv[2:7]), "bar": lambda: bar(*sys.argv[2:])}[cmd]()
