"""Copy the Figure 10 images to JPEG under random ids, so the judge never sees the setting.

Reads to_convert.csv (id, image) and writes <out>/<id>.jpg at the image's own size (1024 px),
quality 92. The id-to-setting key stays on the laptop beside to_convert.csv.

Usage (on mscluster, co3 python):
  python review_judge_prepare.py to_convert.csv <out dir>
"""
import csv
import os
import sys

from PIL import Image

src, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
n = 0
for r in csv.DictReader(open(src)):
    dst = os.path.join(out, r["id"] + ".jpg")
    if not os.path.exists(dst):
        Image.open(r["image"]).convert("RGB").save(dst, quality=92)
    n += 1
print(f"{n} images in {out}")
