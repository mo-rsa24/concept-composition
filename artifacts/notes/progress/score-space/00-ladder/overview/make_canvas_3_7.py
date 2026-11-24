"""The ladder's one real input: a 32 x 64 canvas of a 3 and a 7 apart, built with step V1's own make_canvas.

It also noises the canvas to steps 0 and 15 of the window with the real schedule (numbers.json) and one fixed numpy
noise draw. The draw is real Gaussian noise but is not the sampler's own starting noise. Run from the repository root:
    python3 artifacts/notes/progress/score-space/00-ladder/overview/make_canvas_3_7.py
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[5]
sys.path.insert(0, str(REPO / "2-classifier-guidance" / "scripts" / "abstract_checkers"))
from build_env import load_source, make_canvas, ROOT_SEED  # noqa: E402

MNIST = Path("/home/molef/PhD/replications/original-superdiff/data/MNIST/raw")
N = json.loads((HERE.parent / "numbers" / "numbers.json").read_text())

source = load_source(MNIST, "t10k")
rng = np.random.default_rng(np.random.SeedSequence([ROOT_SEED, 10, 3, 7]))
row = make_canvas("apart", 3, 7, tuple(range(10)), source, rng)
x = row["x"]
Image.fromarray(x).resize((64 * 8, 32 * 8), Image.NEAREST).save(HERE / "canvas-3-7.png")
Image.fromarray(x).save(HERE / "canvas-3-7-native.png")

clean = x.astype(float) / 127.5 - 1
noise = np.random.default_rng(10).standard_normal(clean.shape)
out = {}
for r in (N["sampler"]["window_steps_0_to_15"][0], N["sampler"]["window_steps_0_to_15"][15]):
    xt = r["sqrt_abar"] * clean + r["sqrt_one_minus_abar"] * noise
    img = np.clip((xt + 3) / 6 * 255, 0, 255).astype(np.uint8)
    name = f"canvas-3-7-step{r['step']}-t{r['t']}.png"
    Image.fromarray(img).resize((64 * 8, 32 * 8), Image.NEAREST).save(HERE / name)
    out[name] = {"t": r["t"], "sqrt_abar": r["sqrt_abar"], "display": "x_t mapped from [-3, 3] to grey"}

meta = {"built_by": "build_env.make_canvas('apart', 3, 7, digits 0 to 9, MNIST t10k)", "seed": [ROOT_SEED, 10, 3, 7],
        "holds_left_to_right": [int(v) for v in row["holds"]], "mnist_t10k_index": [int(v) for v in row["src"]],
        "corners": [list(map(int, c)) for c in row["corner"]], "noise": "numpy default_rng(10), illustrative draw",
        "noised": out}
(HERE / "canvas-3-7.json").write_text(json.dumps(meta, indent=1))
print(json.dumps(meta, indent=1))
