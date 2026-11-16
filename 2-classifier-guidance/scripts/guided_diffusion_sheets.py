"""Build the reproduce-2105.05233 sheets from the per-seed renders the wrapper wrote.

guide:    one sheet per asked class, scale 1 grid on the left and scale 10 grid on the right,
          each 4 x 4 tile labelled with its seed and the released classifier's top-1 call;
          a green frame where the call is the asked-for class, red where it is not.
identity: seeds 0 to 3, plain sampling above scale 0, each with the first 12 hex of its
          uint8 sha256 and whether the two match.
"""

import argparse
import csv
import os

from PIL import Image, ImageDraw, ImageFont

TILE, PAD, HEAD = 192, 6, 46


def font(size):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
              "/usr/share/fonts/dejavu/DejaVuSans.ttf"):
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def grid(rows, folder, cls, scale, f):
    w = 4 * (TILE + PAD) + PAD
    g = Image.new("RGB", (w, HEAD + 4 * (TILE + 22 + PAD)), "white")
    d = ImageDraw.Draw(g)
    hits = sum(r["hit"] == "1" for r in rows)
    d.text((PAD, 8), f"scale {scale}: {hits} of {len(rows)} called {rows[0]['class_name']}", fill="black", font=font(20))
    for r in rows:
        s = int(r["seed"])
        x, y = PAD + (s % 4) * (TILE + PAD), HEAD + (s // 4) * (TILE + 22 + PAD)
        im = Image.open(os.path.join(folder, f"c{cls}_s{scale}_seed{s}.png")).resize((TILE, TILE))
        g.paste(im, (x, y))
        col = (20, 150, 60) if r["hit"] == "1" else (200, 40, 40)
        d.rectangle([x - 3, y - 3, x + TILE + 2, y + TILE + 2], outline=col, width=3)
        d.text((x, y + TILE + 3), f"seed {s}: {r['top1_name'][:18]} {float(r['top1_p']):.2f}", fill=col, font=font(13))
    return g


def guide(folder, csv_path, out):
    rows = list(csv.DictReader(open(csv_path)))
    for cls in sorted({r["class_asked"] for r in rows}):
        a = [r for r in rows if r["class_asked"] == cls and float(r["scale"]) == 1.0]
        b = [r for r in rows if r["class_asked"] == cls and float(r["scale"]) == 10.0]
        ga, gb = grid(a, folder, cls, "1", font), grid(b, folder, cls, "10", font)
        sheet = Image.new("RGB", (ga.width + gb.width + 20, ga.height + 40), "white")
        ImageDraw.Draw(sheet).text((PAD, 8), f"classifier guidance toward \"{a[0]['class_name']}\" (ImageNet {cls}), released 256 px unconditional model, 250 steps, seeds 0 to 15",
                                   fill="black", font=font(18))
        sheet.paste(ga, (0, 40)); sheet.paste(gb, (ga.width + 20, 40))
        name = a[0]["class_name"].lower().replace(" ", "-")
        p = os.path.join(out, f"sheet-{name}-scale-1-and-10.png")
        sheet.save(p); print(p)


def identity(folder, out):
    rows = list(csv.DictReader(open(os.path.join(folder, "identity_checksums.csv"))))
    by = {(r["seed"], r["run"]): r for r in rows}
    seeds = sorted({r["seed"] for r in rows}, key=int)
    sheet = Image.new("RGB", (PAD + len(seeds) * (TILE + PAD), 40 + 2 * (TILE + 40)), "white")
    d = ImageDraw.Draw(sheet)
    d.text((PAD, 8), "scale 0 (bottom) against plain sampling (top), same seed", fill="black", font=font(18))
    for i, s in enumerate(seeds):
        same = by[(s, "plain")]["sha256_uint8"] == by[(s, "scale0")]["sha256_uint8"]
        for j, run in enumerate(("plain", "scale0")):
            x, y = PAD + i * (TILE + PAD), 40 + j * (TILE + 40)
            sheet.paste(Image.open(os.path.join(folder, f"identity_seed{s}_{run}.png")).resize((TILE, TILE)), (x, y))
            d.text((x, y + TILE + 3), f"seed {s} {run}: {by[(s, run)]['sha256_uint8'][:12]}", fill="black", font=font(12))
            d.text((x, y + TILE + 18), "identical" if same else "DIFFERS", fill=(20, 150, 60) if same else (200, 40, 40), font=font(12))
    p = os.path.join(out, "sheet-identity-scale-0-against-plain.png")
    sheet.save(p); print(p)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["guide", "identity"])
    ap.add_argument("--folder", required=True)
    ap.add_argument("--csv")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    guide(a.folder, a.csv, a.out) if a.mode == "guide" else identity(a.folder, a.out)
