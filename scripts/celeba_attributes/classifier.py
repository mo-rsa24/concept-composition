"""The scorer: a ResNet-18 from scratch on all of CelebA's train split, blond men included, with two
outputs (male, blond). Batches are drawn with the four groups (blond or not, man or woman) equally
likely, because with blond faces 94% women an unbalanced network reads "woman" as "blond". It is accepted only if, on the test split's 180 blond men, it calls male right
on at least 90% and blond right on at least 90%.

    python classifier.py
"""
import json, sys, time
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torchvision
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ARRAYS, ROOT

DEV = "cuda"
GATE = 0.90


def net():
    m = torchvision.models.resnet18(weights=None, num_classes=2)
    m.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
    m.maxpool = nn.Identity()
    return m


def to_t(imgs):
    return torch.from_numpy(imgs).permute(0, 3, 1, 2).float().div(127.5).sub(1)


@torch.no_grad()
def predict(m, imgs, bs=1024):
    """imgs: uint8 (N, 64, 64, 3) or float in [-1, 1] (N, 3, 64, 64). Returns (male, blond) booleans."""
    out = []
    for i in range(0, len(imgs), bs):
        x = imgs[i:i + bs]
        x = to_t(x) if isinstance(x, np.ndarray) else x
        with torch.autocast("cuda", torch.bfloat16):
            out.append(m(x.to(DEV)).float().cpu())
    p = torch.cat(out) > 0
    return p[:, 0].numpy(), p[:, 1].numpy()


def load():
    m = net().to(DEV)
    m.load_state_dict(torch.load(ROOT / "classifier" / "resnet18.pt", map_location=DEV))
    return m.eval()


def main(epochs=4, bs=256):
    tr = np.load(ARRAYS / "train.npz"); te = np.load(ARRAYS / "test.npz")
    x = torch.from_numpy(tr["images"]); y = torch.tensor(np.stack([tr["male"], tr["blond"]], 1), dtype=torch.float32)
    m = net().to(DEV)
    opt = torch.optim.AdamW(m.parameters(), 1e-3, weight_decay=1e-4)
    steps = epochs * (len(x) // bs); sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=steps)
    t0, s = time.time(), 0
    grp = torch.from_numpy(tr["male"].astype(np.int64) * 2 + tr["blond"].astype(np.int64))
    w = (1.0 / torch.bincount(grp, minlength=4).double())[grp]
    for ep in range(epochs):
        draws = torch.multinomial(w, len(x), replacement=True)
        for i in range(0, len(x) - bs + 1, bs):
            idx = draws[i:i + bs]
            xb = x[idx].permute(0, 3, 1, 2).float().div(127.5).sub(1).to(DEV)
            if torch.rand(()) < 0.5:
                xb = xb.flip(3)
            with torch.autocast("cuda", torch.bfloat16):
                loss = nn.functional.binary_cross_entropy_with_logits(m(xb).float(), y[idx].to(DEV))
            opt.zero_grad(); loss.backward(); opt.step(); sched.step(); s += 1
        print(f"epoch {ep + 1}: loss {loss.item():.4f}, {time.time() - t0:.0f}s", flush=True)
    m.eval()
    pm, pb = predict(m, te["images"])
    tm, tb = te["male"], te["blond"]
    groups = {}
    for name, sel in (("blond men", tm & tb), ("blond women", ~tm & tb), ("dark-haired men", tm & ~tb),
                      ("dark-haired women", ~tm & ~tb)):
        groups[name] = {"n": int(sel.sum()), "male right": float((pm[sel] == tm[sel]).mean()),
                        "blond right": float((pb[sel] == tb[sel]).mean())}
        print(name, groups[name], flush=True)
    bm = groups["blond men"]
    ok = bm["male right"] >= GATE and bm["blond right"] >= GATE
    (ROOT / "classifier").mkdir(exist_ok=True)
    torch.save(m.state_dict(), ROOT / "classifier" / "resnet18.pt")
    json.dump({"gate": f"blond men in test: male right and blond right both >= {GATE}", "passed": ok,
               "groups": groups, "epochs": epochs,
               "sampling": "the four male-by-blond groups equally likely in every batch"}, open(ROOT / "classifier" / "accuracy.json", "w"), indent=1)
    print("GATE", "PASSED" if ok else "FAILED")


if __name__ == "__main__":
    main()
