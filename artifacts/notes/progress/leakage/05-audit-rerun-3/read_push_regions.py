"""Rerun 3 of Loop 12's design audit, read from saved records (board question 97): does each subject's own push pick out its own body?
For each corrector pass's last pass at steps 0 to 5, each animal's push d(A) = eps(A) - eps(empty), length over the 4 channels per cell,
rescaled to 0..1 and cut at 0.4 (Z2's rule). M9's mask rule: a read fails when either mask covers >= 80% of the grid or the two masks
overlap with IoU >= 0.8. On cheetah x cougar each mask is also compared with the box drawn on its own animal (03-one-fix/boxes.json).
Records: the CO3 capture run (Slurm job 94450), seed 6001. Usage: read_push_regions.py <co3-inside folder> <boxes.json>"""
import json, sys, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
R, boxes = sys.argv[1], json.load(open(sys.argv[2]))
def cut(m):
    m = (m - m.min()) / (m.max() - m.min() + 1e-8); return m >= 0.4
def box(b):
    g = np.zeros((128, 128), bool)
    if b: x0, y0, x1, y1 = [int(round(v / 8)) for v in b]; g[y0:y1, x0:x1] = True
    return g
out, fails = {}, 0
fig, ax = plt.subplots(2, 6, figsize=(24, 8.5))
for r, (pair, a, b) in enumerate([("a_cheetah__x__a_cougar", "cheetah", "cougar"), ("an_eagle__x__a_fox", "eagle", "fox")]):
    d = np.load(f"{R}/{pair}/seed-6001/passes.npz"); steps, eps = d["step"], d["eps"].astype(np.float32)
    bx = boxes.get(pair, {}).get("6001")
    out[pair] = {}
    for c, s in enumerate(range(6)):
        i = int(np.where(steps == s)[0][-1]); eu, _, ea, eb = eps[i]
        ma, mb = cut(np.linalg.norm(ea - eu, axis=0)), cut(np.linalg.norm(eb - eu, axis=0))
        iou = float((ma & mb).sum() / max((ma | mb).sum(), 1)); cov = (float(ma.mean()), float(mb.mean()))
        fail = cov[0] >= 0.8 or cov[1] >= 0.8 or iou >= 0.8; fails += fail
        rec = {"cover_a": round(cov[0], 3), "cover_b": round(cov[1], 3), "iou": round(iou, 3), "fails_m9_rule": bool(fail)}
        if bx:
            ba, bb = box(bx["A"]), box(bx["B"])
            rec["share_of_a_mask_in_a_box"] = round(float((ma & ba).sum() / max(ma.sum(), 1)), 3)
            rec["share_of_b_mask_in_b_box"] = round(float((mb & bb).sum() / max(mb.sum(), 1)), 3)
        out[pair][s] = rec
        img = np.zeros((128, 128, 3)); img[..., 2] = ma; img[..., 0] = mb; img[..., 1] = (ma & mb) * 0.8
        ax[r, c].imshow(img); ax[r, c].set_xticks([]); ax[r, c].set_yticks([])
        ax[r, c].set_title(f"{a} x {b}, step {s}\nblue {a} {cov[0]:.0%}, red {b} {cov[1]:.0%}, overlap IoU {iou:.2f}", fontsize=9)
fig.suptitle("Each animal's push (its prediction minus the empty prompt's), per cell, cut at 0.4 of its range; white-green where both masks cover a cell. Seed 6001, CO3 capture run.", fontsize=11)
plt.tight_layout(); plt.savefig("push-regions.png", dpi=75)
json.dump({"reads": 12, "fails": fails, "per_read": out}, open("push-regions.json", "w"), indent=1)
print(json.dumps({"fails": fails}), json.dumps(out["a_cheetah__x__a_cougar"][5]), json.dumps(out["an_eagle__x__a_fox"][5]))
