"""Draw the regime outlines onto the source films for artifacts/results/overlap-regimes-on-real-films.

One colour per overlap regime, used across every picture. Each region gets an outline that follows
the lesion and a label written on the image beside that outline, never in a legend. Regions come
from the coordinates recorded per film in
3-understanding-compositionality/artifacts/results/does-poe-match-the-joint-for-each-kind-of-composition/medical-examples/README.md, which were
read by eye from the saved files; where that card records none, the coordinates here were read the
same way from the saved file.
"""
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "artifacts/results/overlap-regimes-on-real-films"
SRC = json.loads((OUT / "sources.json").read_text())

REGIME_COLOUR = {
    "separate": (60, 200, 90),
    "same-region": (255, 165, 0),
    "merged": (70, 170, 255),
    "hidden": (235, 70, 160),
}
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def font_for(width):
    return ImageFont.truetype(FONT, max(12, round(width / 38)))


def draw_label(draw, xy, text, colour, fnt, anchor="ls", bounds=None):
    """Label with a dark halo so it reads over bright lung and dark air alike.

    `bounds` is the image size; the label is pulled back inside it rather than running off the edge.
    """
    x, y = xy
    if bounds is not None:
        w = fnt.getbbox(text)[2] - fnt.getbbox(text)[0]
        h = fnt.getbbox(text)[3] - fnt.getbbox(text)[1]
        left = {"ls": x, "ms": x - w / 2, "rs": x - w}[anchor]
        if left + w > bounds[0] - 4:
            x -= left + w - (bounds[0] - 4)
        if left < 4:
            x += 4 - left
        y = min(max(y, h + 4), bounds[1] - 4)
    for dx in (-2, -1, 0, 1, 2):
        for dy in (-2, -1, 0, 1, 2):
            draw.text((x + dx, y + dy), text, font=fnt, fill=(0, 0, 0), anchor=anchor)
    draw.text((x, y), text, font=fnt, fill=colour, anchor=anchor)


def render(rec, regions):
    img = Image.open(ROOT / rec["source_file"]).convert("RGB")
    draw = ImageDraw.Draw(img)
    colour = REGIME_COLOUR[rec["regime"]]
    fnt = font_for(img.width)
    lw = max(2, round(img.width / 300))
    for reg in regions:
        shape = reg["shape"]
        if shape == "circle":
            cx, cy, r = reg["cx"], reg["cy"], reg["r"]
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=colour, width=lw)
            anchor_pt = (cx + r + 6, cy)
        elif shape == "box":
            x, y, w, h = reg["x"], reg["y"], reg["w"], reg["h"]
            draw.rectangle([x, y, x + w, y + h], outline=colour, width=lw)
            anchor_pt = (x + w + 6, y + 14)
        elif shape == "ellipse":
            draw.ellipse(reg["bbox"], outline=colour, width=lw)
            anchor_pt = (reg["bbox"][2] + 6, (reg["bbox"][1] + reg["bbox"][3]) / 2)
        elif shape == "polygon":
            pts = [tuple(p) for p in reg["points"]]
            draw.line(pts + [pts[0]], fill=colour, width=lw, joint="curve")
            anchor_pt = (max(p[0] for p in pts) + 6, min(p[1] for p in pts) + 14)
        else:
            raise ValueError(shape)
        lx, ly = reg.get("label_at", anchor_pt)
        anchor = reg.get("label_anchor", "ls")
        if lx > img.width * 0.82 and "label_at" not in reg:
            lx, anchor = min(p[0] for p in pts) - 6 if shape == "polygon" else lx - 12, "rs"
        draw.line([anchor_pt, (lx, ly)], fill=colour, width=max(1, lw - 1))
        draw_label(draw, (lx, ly), reg["label"], colour, fnt, anchor, bounds=img.size)
    name = f"{rec['regime']}-{rec['pair']}-{rec['nn']:02d}.png"
    img.save(OUT / name)
    return name


def main(specs):
    by_src = {r["source_file"].split("/")[-1]: r for r in SRC}
    done = []
    for key, (nn, regions) in specs.items():
        rec = by_src[key]
        rec["nn"] = nn
        name = render(rec, regions)
        rec["annotated_file"] = f"artifacts/results/overlap-regimes-on-real-films/{name}"
        rec["regions"] = regions
        rec.pop("nn")
        done.append(name)
    (OUT / "sources.json").write_text(json.dumps(SRC, indent=2))
    for d in done:
        print("wrote", d)


if __name__ == "__main__":
    from annotate_overlap_regimes_specs import SPECS
    main(SPECS)
