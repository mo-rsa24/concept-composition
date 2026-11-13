"""Decode CelebA's parquet shards into 64 x 64 uint8 arrays with the Male and Blond_Hair labels.

Aligned faces are 178 x 218; the crop is the 148 x 148 box from (15, 40), the usual CelebA face
crop, resized to 64. One file per split: images (N, 64, 64, 3), male (N,), blond (N,), celeb_id (N,).

    python prep.py
"""
import glob, io, sys, time
from multiprocessing import Pool
import numpy as np, pyarrow.parquet as pq
from PIL import Image
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from common import PARQUET, ARRAYS, RES


def shard(path):
    t = pq.read_table(path, columns=["image", "celeb_id", "Male", "Blond_Hair"]).to_pydict()
    imgs = np.empty((len(t["image"]), RES, RES, 3), np.uint8)
    for i, im in enumerate(t["image"]):
        p = Image.open(io.BytesIO(im["bytes"])).convert("RGB").crop((15, 40, 163, 188))
        imgs[i] = np.asarray(p.resize((RES, RES), Image.BICUBIC))
    return imgs, np.asarray(t["Male"], bool), np.asarray(t["Blond_Hair"], bool), np.asarray(t["celeb_id"])


def main():
    ARRAYS.mkdir(parents=True, exist_ok=True)
    for split in ("train", "valid", "test"):
        t0 = time.time()
        files = sorted(glob.glob(str(PARQUET / f"{split}-*.parquet")))
        with Pool(min(16, len(files))) as pool:
            parts = pool.map(shard, files)
        imgs, male, blond, cid = (np.concatenate([p[k] for p in parts]) for k in range(4))
        np.savez(ARRAYS / f"{split}.npz", images=imgs, male=male, blond=blond, celeb_id=cid)
        print(f"{split}: {len(imgs)} faces, {int(male.sum())} male, {int(blond.sum())} blond, "
              f"{int((male & blond).sum())} blond men, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
