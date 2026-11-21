"""Draw rung 0: the hld-diagram renderer lays out the boxes, lanes and arrows from the spec, then this
script puts the real pictures on them. A tensor box shows its picture in place of its grid glyph; every
other box with pictures gets a card (title, source tag, tiles with short labels) placed where the spec's
"cards" say, joined to its box by a thin dotted line. Outlines take the ladder's colour key and no shape is
filled with a diagram colour, because the toy's own pictures are red circles and blue squares.

    python3 draw_rung0.py [--out-dir DIR]

Writes the-toy-caption-model-high-level-design-explainer.svg, .html, -light.png and -dark.png.
"""
import base64, html, importlib.util, json, re, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
STEM = "the-toy-caption-model-high-level-design-explainer"
SKILL = Path.home() / ".claude/skills/hld-diagram/scripts/draw_hld.py"
spec_mod = importlib.util.spec_from_file_location("draw_hld", SKILL)
dh = importlib.util.module_from_spec(spec_mod)
spec_mod.loader.exec_module(dh)

OUT = Path(sys.argv[sys.argv.index("--out-dir") + 1]) if "--out-dir" in sys.argv else HERE
spec = json.load(open(HERE / f"{STEM}.json"))
LADDER = json.load(open(HERE / "ladder.json"))["colour_key"]
ROLE = {"inputs": LADDER["inputs"], "hidden": LADDER["hidden units"], "outputs": LADDER["network outputs"],
        "method": next(v for k, v in LADDER.items() if k.startswith("the method")),
        "off": next(v for k, v in LADDER.items() if k.startswith("off")), "plain": "#3d4757"}
ROLE_DARK = {"inputs": "#7fa6ff", "hidden": "#5cc485", "outputs": "#ff7d7d", "method": "#f0a24f", "off": "#9aa7b4", "plain": "#b7c0cf"}

TILE_W, TILE_H, PITCH, PAD, TITLE_H, LINE_H = 88, 44, 116, 11, 34, 13


def b64(path):
    return "data:image/png;base64," + base64.b64encode(Path(path).read_bytes()).decode()


def img_size(path):
    from PIL import Image
    return Image.open(path).size


# 1. A layout spec: the boxes plus one placeholder box per card, so the lanes make room for the cards
lay = json.loads(json.dumps(spec))
card_w = {}
for c in spec["cards"]:
    n = len(c["tiles"])
    c["pitch"] = max(PITCH, max(len(l) for t in c["tiles"] for l in t["label"]) * 6.3 + 14)
    if c["wide"]:
        w = 300
    else:
        w = int(max(PAD * 2 + (n - 1) * c["pitch"] + TILE_W, len(c["title"]) * 6.6 + 2 * PAD + 20))
    lines = max(len(t["label"]) for t in c["tiles"])
    h = TITLE_H + (TILE_H if not c["wide"] else 26) + 6 + LINE_H * lines + PAD
    card_w[c["box"]] = (w, h)
    lay["nodes"].append({"id": "card-" + c["box"], "kind": "tensor", "label": "", "sub": "", "cells": "1x1",
                         "col": c["at"][0], "row": c["at"][1], "w": w, "h": h})
    for lane in lay["lanes"]:
        if c["box"] in lane["members"]:
            lane["members"].append("card-" + c["box"])
svg, n_nodes, n_edges, W, H = dh.render(lay)


# 2. Read every node's drawn box back from the SVG
def groups(svg):
    for m in re.finditer(r'<g class="node" id="n-([^"]+)" style="[^"]*">(.*?)</g>', svg, re.S):
        yield m


def bbox(inner):
    m = re.search(r'<rect class="shape[^"]*" x="(-?[\d.]+)" y="(-?[\d.]+)" width="([\d.]+)" height="([\d.]+)"', inner)
    if m:
        return tuple(float(v) for v in m.groups())
    m = re.search(r'<path class="shape[^"]*" d="([^"]+)"', inner)
    nums = [float(v) for v in re.findall(r"-?[\d.]+", m.group(1))]
    xs, ys = nums[0::2], nums[1::2]
    return min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)


boxes = {}
for m in groups(svg):
    boxes[m.group(1)] = bbox(m.group(2))
out_parts = []


def esc(s):
    return html.escape(str(s), quote=True)


def card_svg(c, x, y, w, h):
    o = [f'<g class="card" id="card-{esc(c["box"])}">',
         f'<rect class="card-frame" x="{x:.1f}" y="{y:.1f}" width="{w}" height="{h}" rx="7"/>',
         f'<text class="card-title" x="{x + PAD:.1f}" y="{y + 17:.1f}">{esc(c["title"])}</text>',
         f'<text class="card-tag" x="{x + PAD:.1f}" y="{y + 29:.1f}">{esc(c["source"])}</text>']
    ty = y + TITLE_H
    for i, t in enumerate(c["tiles"]):
        if c["wide"]:
            iw, ih = w - 2 * PAD, 26
            tx = x + PAD
        else:
            pw, ph = img_size(t["image"])
            ih = TILE_H
            iw = min(TILE_W, ih * pw / ph) if pw / ph < 2 else TILE_W
            ih = iw * ph / pw
            tx = x + PAD + i * c["pitch"] + (TILE_W - iw) / 2
        o.append(f'<image x="{tx:.1f}" y="{ty + (TILE_H - ih) / 2 if not c["wide"] else ty:.1f}" width="{iw:.1f}" height="{ih:.1f}" '
                 f'preserveAspectRatio="none" style="image-rendering:pixelated" href="{b64(t["image"])}"/>')
        o.append(f'<rect class="tile-frame" x="{tx - .5:.1f}" y="{(ty + (TILE_H - ih) / 2 if not c["wide"] else ty) - .5:.1f}" width="{iw + 1:.1f}" height="{ih + 1:.1f}"/>')
        cx = x + PAD + i * c["pitch"] + TILE_W / 2 if not c["wide"] else x + w / 2
        for j, line in enumerate(t["label"]):
            o.append(f'<text class="tile-label{" tile-cat" if j else ""}" x="{cx:.1f}" y="{ty + (TILE_H if not c["wide"] else 26) + 12 + j * LINE_H:.1f}" text-anchor="middle">{esc(line)}</text>')
    o.append("</g>")
    return "\n".join(o)


def connector(a, b):
    """A dotted line from card box a to node box b, between the facing edges."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    if ax + aw < bx or bx + bw < ax:                     # side by side
        y = (max(ay, by) + min(ay + ah, by + bh)) / 2 if max(ay, by) < min(ay + ah, by + bh) else (ay + ah / 2 + by + bh / 2) / 2
        x1, x2 = (ax + aw, bx) if ax < bx else (ax, bx + bw)
        return f'<path class="pic-link" d="M{x1:.1f} {y:.1f} H{x2:.1f}"/>'
    x = (max(ax, bx) + min(ax + aw, bx + bw)) / 2
    y1, y2 = (ay + ah, by) if ay < by else (ay, by + bh)
    return f'<path class="pic-link" d="M{x:.1f} {y1:.1f} V{y2:.1f}"/>'


# 3. Rewrite the node groups: colour roles, pictures in tensor boxes, cards in place of placeholders
links = []


def rewrite(m):
    nid, inner = m.group(1), m.group(2)
    if nid.startswith("card-"):
        c = next(c for c in spec["cards"] if "card-" + c["box"] == nid)
        x, y, w, h = boxes[nid]
        links.append(connector((x, y, w, h), boxes[c["box"]]))
        return card_svg(c, x, y, w, h)
    role = spec["roles"].get(nid, "plain")
    head = f'<g class="node" id="n-{nid}" style="--k:var(--r-{role})">'
    node = next(n for n in spec["nodes"] if n["id"] == nid)
    if node.get("picture"):
        fr = re.search(r'<rect class="tensor-frame[^"]*" x="(-?[\d.]+)" y="(-?[\d.]+)" width="([\d.]+)" height="([\d.]+)"[^>]*/>', inner)
        fx, fy, fw, fh = (float(v) for v in fr.groups())
        inner = re.sub(r'<rect class="cell"[^>]*/>', "", inner)
        pw, ph = img_size(node["picture"])
        iw, ih = fw - 6, fh - 6
        if pw / ph < iw / ih:
            nw = ih * pw / ph
            ix, iy = fx + 3 + (iw - nw) / 2, fy + 3
            iw = nw
        else:
            ix, iy = fx + 3, fy + 3
        img = (f'<image x="{ix:.1f}" y="{iy:.1f}" width="{iw:.1f}" height="{ih:.1f}" preserveAspectRatio="none" '
               f'style="image-rendering:pixelated" href="{b64(node["picture"])}"/>')
        inner = inner.replace(fr.group(0), img + fr.group(0))
        subs = list(re.finditer(r'<text class="sub" x="([\d.]+)" y="([\d.]+)"', inner))
        for i, sm in enumerate(subs[1:], 1):     # the renderer spaces tensor lines 12 px apart, too tight at this font size
            y0 = float(subs[0].group(2))
            inner = inner.replace(sm.group(0), f'<text class="sub" x="{sm.group(1)}" y="{y0 + 16 * i:.1f}"', 1)
    return head + inner + "</g>"


svg = re.sub(r'<g class="node" id="n-([^"]+)" style="[^"]*">(.*?)</g>', rewrite, svg, flags=re.S)
svg = svg.replace("</svg>", "\n".join(links) + "\n</svg>")

# 4. The source key under the subtitle, and the ladder's colours
m = re.search(r'<text class="subtitle" x="([\d.]+)" y="([\d.]+)"', svg)
sx, sy = float(m.group(1)), float(m.group(2))
vb = re.search(r'viewBox="0 0 (\d+) (\d+)"', svg)
VW, VH = int(vb.group(1)), int(vb.group(2)) + 30
key = f'<text class="subtitle key" x="{sx}" y="{VH - 14}">{esc(spec["source_key"])}</text>'
svg = svg.replace(vb.group(0), f'viewBox="0 0 {VW} {VH}"', 1)
svg = re.sub(r'(<svg[^>]*?)width="\d+" height="\d+"', lambda mm: mm.group(1) + f'width="{VW}" height="{VH}"', svg, count=1)
svg = re.sub(r'(<rect class="bg"[^>]*?)height="[\d.]+"', lambda mm: mm.group(1) + f'height="{VH}"', svg, count=1)
svg = svg.replace("</svg>", key + "\n</svg>")
# cards wider than their column push the lanes past x = 0: move everything right by that much
minx = min(float(v) for v in re.findall(r'<rect class="lane" x="(-?[\d.]+)"', svg))
DX = max(0, int(-minx) + 8)
if DX:
    bg = re.search(r'<rect class="bg"[^>]*/>', svg).group(0)
    svg = svg.replace(bg, re.sub(r'width="[\d.]+"', f'width="{VW + DX}"', bg) + f'<g transform="translate({DX},0)">', 1)
    svg = svg.replace("</svg>", "</g>\n</svg>")
    svg = svg.replace(f'viewBox="0 0 {VW} {VH}"', f'viewBox="0 0 {VW + DX} {VH}"', 1)
    svg = re.sub(r'(<svg[^>]*?)width="\d+"', lambda mm: mm.group(1) + f'width="{VW + DX}"', svg, count=1)
light = ";".join(f"--r-{k}:{v}" for k, v in ROLE.items())
dark = ";".join(f"--r-{k}:{v}" for k, v in ROLE_DARK.items())
EXTRA = f"""
svg.hld{{{light}}}
@media (prefers-color-scheme: dark){{:root:not([data-theme="light"]) svg.hld{{{dark}}}}}
[data-theme="dark"] svg.hld{{{dark}}}
svg.hld .shape{{fill:var(--hld-box)!important}}
svg.hld .card-frame{{fill:var(--hld-box);stroke:var(--hld-group-line);stroke-width:1}}
svg.hld .card-title{{fill:var(--hld-ink);font-size:12.5px;font-weight:600}}
svg.hld .card-tag{{fill:var(--hld-muted);font-size:11px;font-style:italic}}
svg.hld .tile-frame{{fill:none;stroke:#b4bccb;stroke-width:1}}
svg.hld .tile-label{{fill:var(--hld-ink);font-size:11px}}
svg.hld .tile-cat{{fill:var(--hld-muted);font-style:italic}}
svg.hld .pic-link{{fill:none;stroke:var(--hld-muted);stroke-width:1.1;stroke-dasharray:2 3}}
svg.hld .key{{font-size:.9em}}
"""
svg = svg.replace("</style>", EXTRA + "</style>", 1)

OUT.mkdir(parents=True, exist_ok=True)
(OUT / f"{STEM}.svg").write_text(svg)
PICS = json.load(open(HERE / "the-toy-caption-model-pictures" / "pictures.json"))["boxes"]
SIDE = {"input": "goes in", "example": "for example", "output": "comes out"}
gal = []
for n in spec["nodes"]:
    entry = PICS.get(n["id"], {})
    items = [(side, it) for side in ("input", "example", "output") for it in entry.get(side, [])]
    name = n["label"] or n["sub"].split(":")[0]
    if not items:
        gal.append(f'<h3>{esc(name)}</h3><p class="none">No picture: the boxes on either side show what goes in and what comes out.</p>')
        continue
    figs = "".join(f'<figure><img src="{b64(it["image"])}" alt="{esc(it.get("alt", it["caption"]))}"><figcaption><b>{SIDE[side]}</b> {esc(it["caption"])}</figcaption></figure>' for side, it in items)
    gal.append(f'<h3>{esc(name)}</h3><div class="gal">{figs}</div>')
page = dh.build_page(spec, svg)
page = page.replace(".wrap{max-width:1180px;", ".wrap{max-width:none;")
page = page.replace("</style>", """.diagram svg{width:100%}
.diagram.full svg{width:auto}
.bar{display:flex;gap:10px;align-items:center;justify-content:space-between;flex-wrap:wrap}
.bar button{font:inherit;font-size:13px;padding:4px 12px;border:1px solid var(--rule);border-radius:8px;background:var(--panel);color:var(--fg);cursor:pointer}
.bar button:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
.gal{display:grid;grid-template-columns:repeat(auto-fill,minmax(190px,1fr));gap:12px}
.gal figure{margin:0}
.gal img{width:160px;image-rendering:pixelated;border:1px solid var(--rule);background:#fff;display:block}
.gal figcaption{font-size:13px;color:var(--muted);margin-top:4px}
details.pics{max-width:none}
.none{color:var(--muted)}
</style>""", 1)
page = page.replace('<div class="diagram">', '<div class="bar"><span class="version">Every picture is real; the tag on each card says where it came from.</span><button type="button" id="fs" aria-pressed="false">Full size</button></div>\n<div class="diagram" id="dg">', 1)
page = page.replace("<details>", f'<details class="pics"><summary>Every picture, box by box, with what it shows</summary>{"".join(gal)}</details>\n<details>', 1)
page = page.replace("</main>", """</main>
<script>
(function(){var b=document.getElementById('fs'),d=document.getElementById('dg');
b.addEventListener('click',function(){var on=d.classList.toggle('full');b.setAttribute('aria-pressed',on);b.textContent=on?'Fit to width':'Full size';});})();
</script>""")
(OUT / f"{STEM}.html").write_text(page)
print(f"wrote {OUT / (STEM + '.svg')}: {len(spec['nodes'])} boxes, {n_edges} arrows, {len(spec['cards'])} cards, {W}x{H}")
dh.render_png(OUT, STEM)
