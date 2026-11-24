"""Read the saved count checker's readout MLP from its .pt file without PyTorch and without unpickling anything.

A .pt file is a zip: data.pkl names each tensor and the storage file holding its bytes, and data/<n> holds those bytes raw.
This script walks data.pkl's opcodes with pickletools.genops, which only lists them and executes nothing, takes each
tensor's storage key, dtype, offset, shape and stride, and reads the bytes with numpy.

Two checks against job 94116's own outputs in ../numbers/real/count-stages.npz, on all 16 saved cells:
  the FiLM layer read here must give the stored gamma and beta, and
  the MLP read here, run on the stored pooled numbers and time features, must give the stored logit.

Get the file first (it is not kept in the repo):
  timeout 300 ssh -o BatchMode=yes mscluster107 "cat /datasets/mmolefe/poe_repair_min/loop-8/v2/count-final/count-head-map.pt" > /tmp/readout-pt-dl/count-head-map.pt
Run:  PYTHONDONTWRITEBYTECODE=1 ~/.claude/venvs/figures/bin/python -I read_readout.py /tmp/readout-pt-dl/count-head-map.pt
Writes readout-mlp.npz (W1 256 x 272, b1 256, w2 256, b2 1), read by make_page_key.py.
"""
import pickletools
import sys
import zipfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
STAGES = HERE.parent / "numbers" / "real" / "count-stages.npz"
WANT = ["film.weight", "film.bias", "mlp.0.weight", "mlp.0.bias", "mlp.2.weight", "mlp.2.bias"]
DTYPES = {"torch FloatStorage"}
TUPLES = {"TUPLE", "TUPLE1", "TUPLE2", "TUPLE3", "EMPTY_TUPLE"}


def tensor_table(pkl):
    """name -> (storage key, dtype, offset, shape, stride), from the opcode stream of a state dict's data.pkl."""
    ops = [(op.name, arg) for op, arg, _ in pickletools.genops(pkl)]
    # every global the file names must be one of these three, so every storage in it is float32
    globals_ = {a for nm, a in ops if nm == "GLOBAL"}
    assert globals_ <= {"collections OrderedDict", "torch._utils _rebuild_tensor_v2", *DTYPES}, globals_
    out, i = {}, 0
    while i < len(ops):
        name, arg = ops[i]
        if name in ("SHORT_BINUNICODE", "BINUNICODE") and arg in WANT:
            j, strs = i + 1, []
            while ops[j][0] != "BINPERSID":            # the persistent id: ('storage', class, key, location, numel)
                if ops[j][0] in ("SHORT_BINUNICODE", "BINUNICODE"):
                    strs.append(ops[j][1])
                j += 1
            key = strs[-2]                             # the last two strings are the storage key and its device
            # after the storage id come: offset, the shape's ints, a tuple op, the stride's ints, a tuple op
            ints, groups = [], []
            for nm, a in ops[j + 1:]:
                if nm.startswith("BININT"):
                    ints.append(a)
                elif nm in TUPLES:
                    groups.append(ints); ints = []
                    if len(groups) == 2:
                        break
            offset, shape, stride = groups[0][0], groups[0][1:], groups[1]
            assert len(shape) == len(stride), (arg, shape, stride)
            out[arg] = (key, np.float32, offset, tuple(shape), tuple(stride))
            i = j
        i += 1
    assert sorted(out) == sorted(WANT), sorted(out)
    return out


def read(path):
    z = zipfile.ZipFile(path)
    root = z.namelist()[0].split("/")[0]
    assert z.read(f"{root}/byteorder") == b"little"
    table = tensor_table(z.read(f"{root}/data.pkl"))
    tensors = {}
    for name, (key, dtype, offset, shape, stride) in table.items():
        raw = np.frombuffer(z.read(f"{root}/data/{key}"), dtype="<f4").astype(dtype)
        n = int(np.prod(shape))
        contiguous = tuple(int(np.prod(shape[k + 1:])) for k in range(len(shape)))
        assert stride == contiguous, (name, stride, contiguous)
        tensors[name] = raw[offset:offset + n].reshape(shape)
    return tensors


def silu(z):
    return z / (1 + np.exp(-z))


def main():
    T = read(sys.argv[1])
    W1, b1, w2, b2 = T["mlp.0.weight"], T["mlp.0.bias"], T["mlp.2.weight"][0], T["mlp.2.bias"]
    assert W1.shape == (256, 272) and w2.shape == (256,) and b2.shape == (1,)
    z = np.load(STAGES, allow_pickle=False)
    cells = sorted({k.split("__")[0] for k in z.files if "__" in k})
    worst_film, worst_logit = 0.0, 0.0
    for c in cells:
        tf, pooled = z[c + "__time_features"].astype(np.float64), z[c + "__pooled"].astype(np.float64)
        gb = T["film.weight"].astype(np.float64) @ tf + T["film.bias"]
        worst_film = max(worst_film, float(np.abs(gb[:128] - z[c + "__film_gamma"]).max()),
                         float(np.abs(gb[128:] - z[c + "__film_beta"]).max()))
        h = silu(W1.astype(np.float64) @ np.concatenate([pooled, tf]) + b1)
        worst_logit = max(worst_logit, abs(float(w2 @ h + b2[0]) - float(z[c + "__logit"])))
    assert len(cells) == 16, cells
    assert worst_film < 1e-4, worst_film
    assert worst_logit < 1e-3, worst_logit
    np.savez_compressed(HERE / "readout-mlp.npz", W1=W1, b1=b1, w2=w2, b2=b2)
    print(f"read {len(T)} tensors; on 16 cells the FiLM layer gives the stored gamma and beta to {worst_film:.1e} "
          f"and the MLP gives the stored logit to {worst_logit:.1e}; wrote readout-mlp.npz")


if __name__ == "__main__":
    main()
