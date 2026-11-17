"""Two pictures of the plan 20 image set.

example-one-per-label.png: one image per label at 256 px (top row) and the same image at
64 px shown at 256 with nearest-neighbour upscaling (bottom row), so the 64 px pixels are visible.
coco-both-crops.png: every COCO both-animal crop at 256 px, titled with its file name and the
share of its animal boxes the square kept, so a crop that dropped an animal can be struck off.
"""
import csv
import sys

from PIL import Image, ImageDraw

ROOT = "/datasets/mmolefe/poe_repair_min/outputs/classifier_guidance_photos/images"
out_dir = sys.argv[1]
rows = list(csv.DictReader(open(f"{ROOT}/index.csv")))


def pick(label, source):
    cands = [r for r in rows if r["label"] == label and r["source"] == source]
    cands.sort(key=lambda r: -float(r["box_area_kept"]))
    return cands[0]


chosen = [pick("cat", "coco"), pick("dog", "coco"), pick("both", "coco"), pick("both", "sd35")]
pad, cap = 8, 22
W = len(chosen) * (256 + pad) + pad
sheet = Image.new("RGB", (W, 2 * (256 + cap) + 3 * pad), "white")
d = ImageDraw.Draw(sheet)
for j, r in enumerate(chosen):
    x = pad + j * (256 + pad)
    big = Image.open(f"{ROOT}/256/{r['name']}")
    small = Image.open(f"{ROOT}/64/{r['name']}").resize((256, 256), Image.NEAREST)
    d.text((x, pad), f"{r['name']}  256 px", fill="black")
    sheet.paste(big, (x, pad + cap))
    d.text((x, 2 * pad + 256 + cap), f"{r['name']}  64 px", fill="black")
    sheet.paste(small, (x, 2 * pad + 256 + 2 * cap))
sheet.save(f"{out_dir}/example-one-per-label.png")
print("example:", [r["name"] for r in chosen])

both = [r for r in rows if r["label"] == "both" and r["source"] == "coco"]
cols = 8; tile = 192; n = len(both); nrows = (n + cols - 1) // cols
g = Image.new("RGB", (cols * (tile + pad) + pad, nrows * (tile + cap + pad) + pad), "white")
d = ImageDraw.Draw(g)
for k, r in enumerate(both):
    x = pad + (k % cols) * (tile + pad); y = pad + (k // cols) * (tile + cap + pad)
    d.text((x, y), f"{r['name'][:8]} kept {float(r['box_area_kept']):.2f}", fill="black")
    g.paste(Image.open(f"{ROOT}/256/{r['name']}").resize((tile, tile)), (x, y + cap))
g.save(f"{out_dir}/coco-both-crops.png")
print("coco both crops:", n)
