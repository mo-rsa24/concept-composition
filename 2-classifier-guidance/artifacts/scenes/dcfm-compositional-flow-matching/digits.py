"""Real MNIST test digits, tinted the way DCFM's Colored MNIST shows them (a coloured digit on black, Fig. 5, p. 6).

Reads the raw idx files already on disk; nothing is downloaded. The colours are illustrative: the paper does not print
its ten colours' RGB values.
"""
import gzip
import os

import numpy as np

RAW = "/home/molef/Playground/data/MNIST/raw"
COLOURS = {"red": (0.93, 0.16, 0.13), "green": (0.18, 0.80, 0.25), "blue": (0.25, 0.45, 1.0),
           "yellow": (0.98, 0.88, 0.10), "cyan": (0.10, 0.85, 0.90), "magenta": (0.92, 0.25, 0.85)}


def _read(name):
    path = os.path.join(RAW, name)
    op = open if os.path.exists(path) else gzip.open
    path = path if os.path.exists(path) else path + ".gz"
    with op(path, "rb") as fh:
        return fh.read()


def mnist_test():
    imgs = np.frombuffer(_read("t10k-images-idx3-ubyte"), np.uint8, offset=16).reshape(-1, 28, 28) / 255.0
    labels = np.frombuffer(_read("t10k-labels-idx1-ubyte"), np.uint8, offset=8)
    return imgs, labels


def digit(d, colour, nth=0):
    """The nth test image of digit d, tinted: an (28, 28, 3) array in [0, 1]. Returns (image, test index)."""
    imgs, labels = mnist_test()
    idx = int(np.where(labels == d)[0][nth])
    rgb = np.array(COLOURS[colour])
    return imgs[idx][..., None] * rgb[None, None, :], idx


def noised(img, t, seed=0):
    """x_t = (1 - t) x_0 + t x_1 with x_0 Gaussian noise, shown clipped to [0, 1] (t = 0 noise, t = 1 the digit)."""
    z = np.random.default_rng(seed).standard_normal(img.shape) * 0.35 + 0.5
    return np.clip((1 - t) * z + t * img, 0, 1)
