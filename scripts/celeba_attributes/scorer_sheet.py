"""The test split's blond men, split by what the scorer calls their hair: 24 called blond, 24 called not
blond, so the scorer can be checked against the labels by eye.

    python scorer_sheet.py
"""
import sys
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ARRAYS, ROOT
import classifier

te = np.load(ARRAYS / "test.npz")
sel = np.flatnonzero(te["male"] & te["blond"])
pm, pb = classifier.predict(classifier.load(), te["images"][sel])
rows = []
for want in (True, False):
    ids = sel[pb == want][:24]
    ims = list(te["images"][ids]) + [np.full((64, 64, 3), 255, np.uint8)] * (24 - len(ids))
    rows += [np.concatenate(ims[r * 12:(r + 1) * 12], 1) for r in range(2)]
    print(f"called {'blond' if want else 'not blond'}: {int((pb == want).sum())} of {len(sel)}")
gap = np.full((12, rows[0].shape[1], 3), 255, np.uint8)
g = np.concatenate(rows[:2] + [gap] + rows[2:], 0)
Image.fromarray(g).resize((g.shape[1] * 2, g.shape[0] * 2), Image.NEAREST).save(ROOT / "logs" / "scorer-blond-men.png")
print("wrote", ROOT / "logs" / "scorer-blond-men.png")
