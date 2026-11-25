"""Z9 before-after: arm 1 (plain product) against another arm, cell by cell on the novel combinations.

Each novel cell shows seed 0 and seed 1 from arm 1 (left pair) and the same seeds from the other arm (right pair), framed green
where the judge calls digit and colour right; under it, the two arms' share right over 100 pictures and the change.
  python before_after.py <arm-1 npz> <arm-1 right npy> <other npz with 'right'> <split json> <label> <out png>
"""
import json
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

a_npz, a_right, b_npz, split_p, label, out = sys.argv[1:]
A = np.load(a_npz); ra = np.load(a_right); B = np.load(b_npz); rb = B["right"]
sp = json.load(open(split_p)); seen = {tuple(k) for k in sp["seen"]}; names = sp["colours"]
assert (A["digits"] == B["digits"]).all() and (A["colours"] == B["colours"]).all()
f = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
fb = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 12)
t, gap = 44, 4
cw = 4 * t + gap
W = Image.new("RGB", (60 + 10 * (cw + 10), 70 + 10 * (t + 30)), "white"); d = ImageDraw.Draw(W)
d.text((10, 6), f"left pair: plain product   right pair: {label}   (seeds 0 and 1; green frame = judge calls it right)", fill="black", font=f)
for c in range(10):
    d.text((60 + c * (cw + 10) + 50, 28), names[c], fill="black", font=f)
first = {(int(x), int(y)): i for i, (x, y) in enumerate(zip(A["digits"], A["colours"])) if i % 100 == 0}
for dg in range(10):
    y = 50 + dg * (t + 30)
    d.text((22, y + t // 2 - 6), str(dg), fill="black", font=f)
    for c in range(10):
        x = 60 + c * (cw + 10)
        if (dg, c) in seen:
            d.rectangle([x, y, x + cw, y + t], fill="#f0f0f0"); continue
        j = first[(dg, c)]
        for k, (imgs, r, off) in enumerate([(A["images"], ra, 0), (A["images"], ra, 1), (B["images"], rb, 0), (B["images"], rb, 1)]):
            xx = x + k * t + (gap if k >= 2 else 0)
            W.paste(Image.fromarray(imgs[j + off].transpose(1, 2, 0)).resize((t, t), Image.NEAREST), (xx, y))
            d.rectangle([xx, y, xx + t - 1, y + t - 1], outline="#2ca02c" if r[j + off] else "#d62728", width=2)
        pa, pb = 100 * ra[j:j + 100].mean(), 100 * rb[j:j + 100].mean()
        col = "#2ca02c" if pb - pa >= 10 else ("#d62728" if pa - pb >= 10 else "black")
        d.text((x + 2, y + t + 2), f"{pa:.0f}% -> {pb:.0f}%", fill=col, font=fb if col != "black" else f)
W.save(out)
print(out, f"arm 1 {100 * ra.mean():.1f}%  {label} {100 * rb.mean():.1f}%")
