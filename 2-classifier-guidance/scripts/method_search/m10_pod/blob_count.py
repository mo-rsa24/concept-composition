#!/usr/bin/env python3
"""Step M10's quarter check: count the objects of one colour in a 64 x 64 render by colour threshold and connected pieces.

A cheap reader for the futility rule only, never for a bar: the pixel is the colour when its channel leads the other two by MARGIN, pieces
touch by 8-connectivity, and a piece counts when it covers at least MIN_AREA pixels.
  validate <zip> <colour>          share of real training pictures whose count it reads right, per count
  score <samples dir> <colour>     share of samples under count-1, count-2, count-3 whose count it reads as asked
"""
import io, json, sys, zipfile, glob
import numpy as np
from PIL import Image
from scipy.ndimage import label

MARGIN, MIN_AREA = 30, 4


def count(img, colour):
    a = np.asarray(img.convert('RGB')).astype(int)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    m = (r - np.maximum(g, b) > MARGIN) if colour == 'r' else (b - np.maximum(r, g) > MARGIN)
    lab, n = label(m, structure=np.ones((3, 3)))
    return int(sum((lab == i).sum() >= MIN_AREA for i in range(1, n + 1)))


def validate(zpath, colour):
    z = zipfile.ZipFile(zpath); labels = dict(json.loads(z.read('dataset.json'))['labels'])
    by = {}
    for n, l in labels.items():
        by.setdefault(l + 1, []).append(count(Image.open(io.BytesIO(z.read(n))), colour) == l + 1)
    out = {k: round(float(np.mean(v)), 3) for k, v in sorted(by.items())}
    out['all'] = round(float(np.mean(sum(by.values(), []))), 3)
    return out


def score(d, colour):
    by = {k: [count(Image.open(p), colour) == k for p in sorted(glob.glob(f'{d}/count-{k}/*.png'))] for k in (1, 2, 3)}
    out = {k: round(float(np.mean(v)), 3) for k, v in by.items() if v}
    out['all'] = round(float(np.mean(sum(by.values(), []))), 3)
    return out


if __name__ == '__main__':
    print(json.dumps({'validate': validate, 'score': score}[sys.argv[1]](sys.argv[2], sys.argv[3])))
