#!/usr/bin/env python3
"""Draw the PhD folder as a plain left-to-right tree: how the work splits by kind, then by repo,
then by project. Writes the-phd-layout.svg and a PNG copy beside this script.

    python3 make_the_phd_layout.py
"""
from pathlib import Path

HERE = Path(__file__).resolve().parent

# (label, colour, children). A leaf has no children. Colour marks the kind of work only.
BLUE, GREEN, PURPLE, GREY = "#2563eb", "#16a34a", "#7c3aed", "#6b7280"
TREE = ("PhD", "#111827", [
    ("research portfolios", BLUE, [
        ("concept-composition", BLUE, [
            ("1  LoRA correction", BLUE, []),
            ("2  classifier guidance", BLUE, []),
            ("3  understanding compositionality", BLUE, []),
            ("4  inclusive captions", BLUE, []),
            ("5  text embedding as a third expert", BLUE, []),
        ]),
        ("medical-imaging", BLUE, [
            ("0  PoE fuses a cat and a dog", BLUE, []),
            ("A  two findings in one film", BLUE, []),
            ("B  shortcut through a device", BLUE, []),
            ("C  detector weakened by a second condition", BLUE, []),
        ]),
    ]),
    ("writing", GREEN, [
        ("Paper3", GREEN, []),
        ("Proposal", GREEN, []),
    ]),
    ("replications", PURPLE, [
        ("published-methods-on-our-pairs", PURPLE, []),
        ("latent-diffusion", PURPLE, []),
        ("original-superdiff", PURPLE, []),
        ("projective-composition", PURPLE, []),
        ("reduce-reuse-recycle", PURPLE, []),
    ]),
    ("admin", GREY, [("cluster-bringup", GREY, [])]),
    ("archive", GREY, [("_archive", GREY, [])]),
])

COL_X = [24, 110, 290, 470]          # x of each depth's labels
ROW_H, TOP, FONT = 30, 34, 14
CHAR_W = 7.6                          # approximate width of one character at FONT


def layout(node, depth, rows):
    """Give every node an (x, y): leaves take the next row, parents sit at their children's middle."""
    label, colour, kids = node
    if not kids:
        y = TOP + len(rows) * ROW_H
        rows.append(label)
        return {"label": label, "colour": colour, "x": COL_X[depth], "y": y, "kids": []}
    placed = [layout(k, depth + 1, rows) for k in kids]
    y = (placed[0]["y"] + placed[-1]["y"]) / 2
    return {"label": label, "colour": colour, "x": COL_X[depth], "y": y, "kids": placed}


def draw(n, out):
    w = len(n["label"]) * CHAR_W
    bold = ' font-weight="600"' if n["kids"] else ""
    out.append(f'<text x="{n["x"]}" y="{n["y"] + 5}" fill="{n["colour"]}" font-size="{FONT}"{bold}>{n["label"]}</text>')
    if n["kids"]:
        x0 = n["x"] + w + 8                      # right end of the parent's label
        xm = n["kids"][0]["x"] - 14              # the shared vertical trunk
        ys = [k["y"] for k in n["kids"]]
        out.append(f'<line x1="{x0}" y1="{n["y"]}" x2="{xm}" y2="{n["y"]}" stroke="#9ca3af" stroke-width="1"/>')
        out.append(f'<line x1="{xm}" y1="{min(ys)}" x2="{xm}" y2="{max(ys)}" stroke="#9ca3af" stroke-width="1"/>')
        for k in n["kids"]:
            out.append(f'<line x1="{xm}" y1="{k["y"]}" x2="{k["x"] - 6}" y2="{k["y"]}" stroke="#9ca3af" stroke-width="1"/>')
            draw(k, out)


def main():
    rows = []
    root = layout(TREE, 0, rows)
    width, height = 840, TOP + len(rows) * ROW_H
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
             f'viewBox="0 0 {width} {height}" font-family="Helvetica, Arial, sans-serif">',
             f'<rect width="{width}" height="{height}" fill="white"/>']
    draw(root, parts)
    parts.append("</svg>")
    svg = "\n".join(parts)
    (HERE / "the-phd-layout.svg").write_text(svg)
    import cairosvg
    cairosvg.svg2png(bytestring=svg.encode(), write_to=str(HERE / "the-phd-layout.png"), scale=2)
    print(f"{len(rows)} leaves, {width} x {height}")


if __name__ == "__main__":
    main()
