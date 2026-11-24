"""Build the real digit canvas the W4 overview shows: one apart canvas of the seen digits 3 and 4, at 32 x 64.

It calls make_canvas from step V1's builder on the MNIST training files, with the random stream V1 gives the training
split's pair (3, 4), so it is drawn the way V1 draws its first apart canvas for that pair. Nothing in V1's dataset is read
or written. Run from anywhere: PYTHONDONTWRITEBYTECODE=1 python3 make_w4_canvas.py
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
sys.path.insert(0, str(REPO / "2-classifier-guidance/scripts/abstract_checkers"))
import build_env as v1  # noqa: E402

MNIST = Path("/home/molef/PhD/replications/original-superdiff/data/MNIST/raw")
source = v1.load_source(MNIST, "train")
rng = np.random.default_rng(np.random.SeedSequence([v1.ROOT_SEED, 0, 3, 4]))
canvas = v1.make_canvas("apart", 3, 4, v1.SEEN, source, rng)
x = canvas["x"]
assert x.shape == (v1.H, v1.W) == (32, 64)
assert sorted(canvas["holds"]) == [3, 4] and all(d in v1.SEEN for d in canvas["holds"])
assert all(int(source["labels"][i]) == d for i, d in zip(canvas["src"], canvas["holds"]))
Image.fromarray(x).save(HERE / "w4-input-canvas.png")
record = {"kind": "apart", "asked": [3, 4], "left_to_right": [int(d) for d in canvas["holds"]],
          "mnist_training_indices": [int(i) for i in canvas["src"]], "corners": [list(map(int, c)) for c in canvas["corner"]],
          "size": [v1.H, v1.W], "mnist_files": str(MNIST), "seed": [v1.ROOT_SEED, 0, 3, 4],
          "mnist_train_images_sha256": v1.sha256_file(MNIST / "train-images-idx3-ubyte")}
(HERE / "w4-input-canvas.json").write_text(json.dumps(record, indent=1))
print(json.dumps(record))
