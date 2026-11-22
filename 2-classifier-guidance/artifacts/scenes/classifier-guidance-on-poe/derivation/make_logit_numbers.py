"""Both checkers' raw outputs (logits) on the four pasted test canvases, asked 4 and 7, at t = 489 (step 25 of 50).

The same canvases and noise as the step 7b architecture sheets (MNIST test digits, rng 11, noise seed 3).
Writes logit_numbers.json beside itself. Run with the sdxl-local python (CPU).
"""
import json, sys
from pathlib import Path
import numpy as np, torch
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO / "scripts/mnist"))
from canvas_sheet import Canvases, D, H, W  # noqa: E402
from canvas_unet import ABAR  # noqa: E402
from checker import STEPS, features, head_feats, load_unet  # noqa: E402
from five_versions import load_head  # noqa: E402

AR = REPO / "data/mnist-canvas/24-mnist-one-model/arch"
MN = Path("/home/molef/PhD/replications/original-superdiff/data/MNIST/raw")
u = load_unet(AR / "models/share-0.01/unet.pt", "cpu")
hp = load_head(AR / "checkers/loop2-r9b-film-onpolicy2/head.pt", "cpu")
hs = load_head(AR / "checkers/separate/head.pt", "cpu")
timg = np.fromfile(MN / "t10k-images-idx3-ubyte", np.uint8, offset=16).reshape(-1, D, D)
tlbl = np.fromfile(MN / "t10k-labels-idx1-ubyte", np.uint8, offset=8)
cv = Canvases(timg, tlbl, np.random.default_rng(11))
ex = {"clean": cv.two(4, 7), "fused": cv.overlaid(4, 7), "single": cv.one(4), "twin": cv.two(4, 4)}
noise = torch.randn(1, 1, H, W, generator=torch.Generator().manual_seed(3))
t = STEPS[25].view(1); a = ABAR[t]
out = {"t": int(t), "asked": [4, 7]}
with torch.no_grad():
    for name, img in ex.items():
        x = a.sqrt() * torch.from_numpy(img).float().div(127.5).sub(1)[None, None] + (1 - a).sqrt() * noise
        z = hs(features(u, x, t), t)[0]
        pair = torch.tensor([[4, 7]])
        lp = hp(head_feats(u, hp, x, t, pair), t, pair)[0, 0]
        out[name] = {"sep_logits": [round(float(v), 3) for v in z], "pair_logit": round(float(lp), 3)}
(HERE / "logit_numbers.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
