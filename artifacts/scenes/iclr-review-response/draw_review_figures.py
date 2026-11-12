"""Draw one annotated figure per claim group of the ICLR 56173 reviews.

Each figure has three columns: the reviewer's words (from the OpenReview screenshots), the
submitted paper's weak passage, and the revised paper's answer. The reply we plan to send sits
in a panel below, and labelled arrows join the marked passages. Every excerpt is a crop of a
200 dpi render; every box is found from the words it surrounds, so a figure redraws after a
text change without re-measuring.

Usage, from the repo root:
    python3 artifacts/scenes/iclr-review-response/draw_review_figures.py            # all groups
    python3 artifacts/scenes/iclr-review-response/draw_review_figures.py G7         # one group
    python3 artifacts/scenes/iclr-review-response/draw_review_figures.py --sheet    # contact sheet

The figure specs live in figure_specs.py beside this file.
"""

from __future__ import annotations

import html
import re
import subprocess
import sys
import unicodedata
from functools import lru_cache
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402
from PIL import Image  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
WALK = REPO / "artifacts/drips/present-revision-to-supervisor"
RENDERS = WALK / "renders"  # Reviews.pdf pages go to renders/reviews-300dpi
CROPS = WALK / "crops"
DPI = 200
SOURCE_DPI = {"reviews": 300}   # Reviews.pdf is drawn as vector shapes, so a finer render stays sharp

SOURCES = {
    "submitted": REPO / "56173_Restoring_plurality_in_c.pdf",
    "revised": REPO / "paper/iclr-56173/build/iclr2027_conference.pdf",
    "reviews": REPO / "Reviews.pdf",
}
SOURCE_NAMES = {
    "submitted": "56173_Restoring_plurality_in_c.pdf (submitted)",
    "revised": "paper/iclr-56173/build/iclr2027_conference.pdf (revised)",
    "reviews": "Reviews.pdf",
}

# One colour per role: the reviewer's point, the submitted weakness, the revised answer.
ROLE = {
    "review": {"edge": "#c0392b", "fill": "#f5b7b1", "name": "the reviewer's point"},
    "submitted": {"edge": "#d68910", "fill": "#fad7a0", "name": "the submitted weakness"},
    "revised": {"edge": "#1e8449", "fill": "#abebc6", "name": "the revised answer"},
}
INK = "#1f2430"
GREY = "#6b7280"
PANEL_EDGE = "#9aa0aa"

COL_W = {"review": 2050, "submitted": 1560, "revised": 1560}  # canvas px = 200 dpi px
GUTTER = 400          # strip beside each crop holding the mark labels, on the side the arrows use
GAP = 300             # gap between columns, where the arrows run
MARGIN = 60
TITLE_H = 210
PANEL_TITLE_H = 70
SRC_H = 46
PANEL_PAD = 18
STACK_GAP = 40


# ---------------------------------------------------------------- words with boxes

def _norm(word: str) -> str:
    word = unicodedata.normalize("NFKC", word).lower()
    return re.sub(r"[^a-z0-9]", "", word)


def render_page(source: str, page: int) -> Path:
    """Render one PDF page at 200 dpi, cached under the walk folder."""
    out_dir = RENDERS / (f"{source}-300dpi" if source in SOURCE_DPI else source)
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf = SOURCES[source]
    # pdftoppm zero-pads to the digit count of the page total
    from_existing = sorted(out_dir.glob(f"{source}-*{page}.png"))
    for f in from_existing:
        if int(re.search(r"-(\d+)\.png$", f.name).group(1)) == page:
            if f.stat().st_mtime >= pdf.stat().st_mtime:
                return f
            f.unlink()  # the PDF was rebuilt after this render
    subprocess.run(["pdftoppm", "-r", str(SOURCE_DPI.get(source, DPI)), "-png", "-f", str(page), "-l", str(page),
                    str(pdf), str(out_dir / source)], check=True)
    return render_page(source, page)


@lru_cache(maxsize=None)
def pdf_words(source: str, page: int) -> list[dict]:
    """Words on one page with pixel boxes at 200 dpi, each tagged with its line id."""
    xml = subprocess.run(["pdftotext", "-bbox-layout", "-f", str(page), "-l", str(page),
                          str(SOURCES[source]), "-"], capture_output=True, text=True,
                         check=True).stdout
    s = SOURCE_DPI.get(source, DPI) / 72.0
    words = []
    for li, line in enumerate(re.finditer(r"<line[^>]*>(.*?)</line>", xml, re.S)):
        for w in re.finditer(r'<word xMin="([\d.]+)" yMin="([\d.]+)" xMax="([\d.]+)" '
                             r'yMax="([\d.]+)">(.*?)</word>', line.group(1)):
            x0, y0, x1, y1 = (float(v) * s for v in w.groups()[:4])
            text = html.unescape(w.group(5))
            words.append({"t": text, "n": _norm(text), "box": (x0, y0, x1, y1), "line": li})
    return words


@lru_cache(maxsize=None)
def image_words(path: str) -> list[dict]:
    """Words in a screenshot with pixel boxes, from tesseract."""
    tsv = subprocess.run(["tesseract", path, "-", "tsv"], capture_output=True, text=True,
                         check=True).stdout.splitlines()
    head = tsv[0].split("\t")
    words = []
    for row in tsv[1:]:
        r = dict(zip(head, row.split("\t")))
        if r.get("level") != "5" or not r.get("text", "").strip():
            continue
        x, y, w, h = (int(r[k]) for k in ("left", "top", "width", "height"))
        line = (int(r["block_num"]), int(r["par_num"]), int(r["line_num"]))
        words.append({"t": r["text"], "n": _norm(r["text"]), "box": (x, y, x + w, y + h),
                      "line": line})
    return words


def source_image(src: dict) -> tuple[Image.Image, list[dict], str]:
    """The full page or screenshot, its words, and the line printed under its crop."""
    if src["kind"] == "ocr":
        path = render_page(src["source"], src["page"])
        img = Image.open(path).convert("RGB")
        words = image_words(str(path))
        label = f'{SOURCE_NAMES[src["source"]]}, page {src["page"]} (no text layer; passage located by OCR)'
    elif src["kind"] == "pdf":
        img = Image.open(render_page(src["source"], src["page"])).convert("RGB")
        words = pdf_words(src["source"], src["page"])
        label = f'{SOURCE_NAMES[src["source"]]}, page {src["page"]}'
    else:
        path = WALK / "review-screenshots" / src["file"]
        img = Image.open(path).convert("RGB")
        words = image_words(str(path))
        label = f'OpenReview screenshot review-screenshots/{src["file"]} = Reviews.pdf page {src["reviews_page"]}'
    return img, words, label


def find_phrase(words: list[dict], phrase: str, after_y: float = -1, nth: int = 0) -> list[dict]:
    """The run of words matching the phrase, in reading order. Raises if it is absent."""
    target = [_norm(t) for t in phrase.split() if _norm(t)]
    seq = [w for w in words if w["n"]]
    hits = []
    for i in range(len(seq) - len(target) + 1):
        if all(seq[i + k]["n"] == target[k] for k in range(len(target))):
            if seq[i]["box"][1] >= after_y:
                hits.append(seq[i:i + len(target)])
    if len(hits) <= nth:
        raise LookupError(f"phrase not found: {phrase!r}")
    return hits[nth]


def line_boxes(run: list[dict]) -> list[tuple[float, float, float, float]]:
    """One box per text line the run spans."""
    by_line: dict = {}
    for w in run:
        by_line.setdefault(w["line"], []).append(w["box"])
    out = []
    for boxes in by_line.values():
        out.append((min(b[0] for b in boxes), min(b[1] for b in boxes),
                    max(b[2] for b in boxes), max(b[3] for b in boxes)))
    return sorted(out, key=lambda b: (b[1], b[0]))


# ---------------------------------------------------------------- crops

def _figure_top(img: Image.Image, y_caption: int, fx: tuple[int, int], gap: int) -> int:
    """Walk up from a caption through the figure above it, stopping at the first white band at
    least `gap` px tall once the figure has started. Only the body columns fx are read, so the
    line numbers in the margin do not count as content."""
    import numpy as np
    s = img.width / 1700.0  # fx is given for a 200 dpi page, 1700 px wide
    grey = np.asarray(img.convert("L"))[:, int(fx[0] * s):int(fx[1] * s)]
    ink = (grey < 230).any(axis=1)
    y = y_caption - 1
    while y > 0 and not ink[y]:      # the space between caption and figure
        y -= 1
    top, white = y, 0
    while y > 0:
        if ink[y]:
            top, white = y, 0
        else:
            white += 1
            if white >= gap:
                break
        y -= 1
    return max(0, top - 10)


def build_crop(panel: dict) -> dict:
    """Cut the panel's crop and place its marks in crop coordinates."""
    img, words, src_label = source_image(panel["src"])
    c = panel["crop"]
    pad = c.get("pad", 8)
    if "box" in c:
        x0, y0, x1, y1 = c["box"]
    else:
        top = line_boxes(find_phrase(words, c["from"], nth=c.get("from_nth", 0)))[0]
        bot = line_boxes(find_phrase(words, c["to"], after_y=top[1], nth=c.get("to_nth", 0)))[-1]
        above = [w["box"][3] for w in words if w["box"][3] < top[1] - 2]
        below = [w["box"][1] for w in words if w["box"][1] > bot[3] + 2]
        y0 = (max(above) + top[1]) / 2 if above and top[1] - max(above) < 60 else top[1] - pad
        y1 = (min(below) + bot[3]) / 2 if below and min(below) - bot[3] < 60 else bot[3] + pad
        if c.get("x"):
            x0, x1 = c["x"]
        else:
            inside = [w["box"] for w in words
                      if y0 <= (w["box"][1] + w["box"][3]) / 2 <= y1]
            x0 = min(b[0] for b in inside) - pad
            x1 = max(b[2] for b in inside) + pad
        if c.get("figure_above"):
            y0 = _figure_top(img, int(y0), c.get("fx", (290, 1420)), c.get("gap", 28))
        y0 = c.get("y0", y0)
        y1 = c.get("y1", y1) + c.get("extend_down", 0)
    x0, y0 = max(0, int(x0)), max(0, int(y0))
    x1, y1 = min(img.width, int(x1)), min(img.height, int(y1))
    crop = img.crop((x0, y0, x1, y1))

    marks = []
    for m in panel.get("marks", []):
        if "box" in m:
            boxes = [m["box"]]
        elif "lines_from" in m:
            a = line_boxes(find_phrase(words, m["lines_from"], after_y=y0))[0]
            b = line_boxes(find_phrase(words, m["lines_to"], after_y=a[1]))[-1]
            b = (b[0], b[1], b[2], b[3] + m.get("pad_bottom", 0))
            boxes = [(x0 + 8, a[1], x1 - 8, b[3])] if m.get("full") else \
                    [(min(a[0], b[0]), a[1], max(a[2], b[2]), b[3])]
        else:
            boxes = line_boxes(find_phrase(words, m["phrase"], after_y=y0, nth=m.get("nth", 0)))
        W, H = x1 - x0, y1 - y0
        local = [(max(4, bx0 - x0), max(2, by0 - y0), min(W - 4, bx1 - x0), min(H - 2, by1 - y0))
                 for bx0, by0, bx1, by1 in boxes]
        marks.append({**m, "boxes": local})

    CROPS.mkdir(parents=True, exist_ok=True)
    crop.save(CROPS / f'{panel["crop_name"]}.png')
    return {"img": crop, "marks": marks, "src_label": src_label}


# ---------------------------------------------------------------- reply text

def reply_paragraphs(file: str, sections: list[str]) -> list[tuple[str, str]]:
    """The reply's own paragraphs under the named headings, reviewer quotes left out.

    Returns (kind, text) with kind 'head' for a heading or bold sub-heading and 'para' for prose.
    The text is copied verbatim from the reply file.
    """
    lines = (REPO / file).read_text().splitlines()
    out: list[tuple[str, str]] = []
    for sec in sections:
        heading, _, sub = sec.partition(" > ")
        start = next(i for i, l in enumerate(lines) if l.startswith("#") and l.lstrip("# ").startswith(heading))
        level = len(lines[start]) - len(lines[start].lstrip("#"))
        end = next((i for i in range(start + 1, len(lines))
                    if lines[i].startswith("#") and len(lines[i]) - len(lines[i].lstrip("#")) <= level),
                   len(lines))
        block = lines[start + 1:end]
        if sub:
            s = next(i for i, l in enumerate(block) if l.strip() == f"**{sub}**")
            e = next((i for i in range(s + 1, len(block)) if re.fullmatch(r"\*\*.+\*\*", block[i].strip())),
                     len(block))
            block = block[s + 1:e]
            out.append(("head", f'{lines[start].lstrip("# ")}: {sub}'))
        else:
            out.append(("head", lines[start].lstrip("# ")))
        para: list[str] = []
        for l in block + [""]:
            if l.startswith(">"):
                continue
            if not l.strip():
                if para:
                    out.append(("para", " ".join(para)))
                    para = []
                continue
            if re.fullmatch(r"\*\*.+\*\*", l.strip()):
                out.append(("head", l.strip().strip("*")))
                continue
            para.append(l.strip())
    return out


def _mathsafe(text: str) -> str:
    # matplotlib mathtext: keep $...$, escape bare dollar signs and markdown emphasis
    text = text.replace("**", "")
    text = re.sub(r"`([^`]*)`", r"\1", text)
    return text


def wrap_measured(ax, text: str, width_px: float, size: float, weight="normal") -> list[str]:
    """Greedy wrap by measured width; never breaks inside $...$."""
    tokens = re.findall(r"\$[^$]*\$\S*|\S+", text)
    lines, cur = [], ""
    renderer = ax.figure.canvas.get_renderer()
    for tok in tokens:
        trial = f"{cur} {tok}".strip()
        t = ax.text(0, 0, trial, fontsize=size, weight=weight)
        w = t.get_window_extent(renderer).width
        t.remove()
        if w > width_px and cur:
            lines.append(cur)
            cur = tok
        else:
            cur = trial
    if cur:
        lines.append(cur)
    return lines


# ---------------------------------------------------------------- drawing

def rounded(ax, x, y, w, h, dashed=False, edge=PANEL_EDGE, fill="white", lw=2.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=18",
                                linewidth=lw, edgecolor=edge, facecolor=fill,
                                linestyle=(0, (6, 4)) if dashed else "solid", zorder=1))


def tag(ax, x, y, text, color, size=10.5, ha="left", va="bottom", z=6):
    ax.text(x, y, text, fontsize=size, color=color, ha=ha, va=va, zorder=z,
            bbox=dict(boxstyle="round,pad=0.28", facecolor="white", edgecolor=color, linewidth=1.6))


def draw_figure(spec: dict) -> Path:
    cols = spec["columns"]  # {"review": [panels], "submitted": [...], "revised": [...]}
    built = {}
    col_x = {}
    order = ["review", "submitted", "revised"]
    x = MARGIN
    for role in order:
        col_x[role] = x
        x += COL_W[role] + GAP

    # Measure every panel's height first.
    layout = {}
    col_h = {}
    for role in order:
        y = MARGIN + TITLE_H
        for p in cols.get(role, []):
            inner_w = COL_W[role] - 2 * PANEL_PAD - GUTTER
            if p.get("pending"):
                h_img = p.get("height", 360)
                b = None
            else:
                b = build_crop(p)
                scale = min(1.0, inner_w / b["img"].width)
                h_img = b["img"].height * scale
                b["scale"] = scale
            h = PANEL_TITLE_H + h_img + SRC_H + PANEL_PAD
            layout[p["id"]] = {"x": col_x[role], "y": y, "w": COL_W[role], "h": h, "role": role,
                               "h_img": h_img, "built": b, "panel": p}
            y += h + STACK_GAP
        col_h[role] = y

    body_bottom = max(col_h.values())
    canvas_w = MARGIN * 2 + sum(COL_W.values()) + 2 * GAP

    # Reply panel: measure with a scratch figure.
    reply = reply_paragraphs(spec["reply"]["file"], spec["reply"]["sections"])
    n_text_cols = spec["reply"].get("columns", 2)
    text_col_w = (canvas_w - 2 * MARGIN - 2 * PANEL_PAD - (n_text_cols - 1) * 60) / n_text_cols
    scratch = plt.figure(figsize=(canvas_w / DPI, 4), dpi=DPI)
    sax = scratch.add_axes([0, 0, 1, 1])
    size_p, size_h = 10.5, 10.5
    line_h = size_p * DPI / 72 * 1.5
    blocks = []
    for kind, text in reply:
        text = _mathsafe(text)
        wrapped = wrap_measured(sax, text, text_col_w, size_h if kind == "head" else size_p,
                                weight="bold" if kind == "head" else "normal")
        blocks.append((kind, wrapped, len(wrapped) * line_h + line_h * 0.6))
    plt.close(scratch)
    merged = []
    for b in blocks:
        if merged and merged[-1][0] == "head" and b[0] == "para":
            h0 = merged.pop()[1][0]
            merged.append(("group", [h0, b], h0[2] + b[2]))
        else:
            merged.append((b[0], [b], b[2]))
    blocks = merged
    # Split blocks into text columns, balancing height.
    total = sum(b[2] for b in blocks)
    text_cols, cur, acc = [[]], 0, 0.0
    for b in blocks:
        if acc + b[2] / 2 > total / n_text_cols * (cur + 1) and cur < n_text_cols - 1 and text_cols[cur]:
            cur += 1
            text_cols.append([])
        text_cols[cur].append(b)
        acc += b[2]
    reply_h = PANEL_TITLE_H + max(sum(b[2] for b in c) for c in text_cols) + PANEL_PAD * 2
    reply_y = body_bottom + 40

    footer_lines = spec.get("footer", [])
    canvas_h = reply_y + reply_h + 40 + len(footer_lines) * 40 + MARGIN

    fig = plt.figure(figsize=(canvas_w / DPI, canvas_h / DPI), dpi=DPI)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, canvas_w)
    ax.set_ylim(canvas_h, 0)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    ax.text(canvas_w / 2, MARGIN + 30, spec["title"], fontsize=19, weight="bold", color=INK,
            ha="center", va="center")
    ax.text(canvas_w / 2, MARGIN + 90, spec["subtitle"], fontsize=12, color=GREY,
            ha="center", va="center")

    # Column headers in role colour.
    for role in order:
        ax.text(col_x[role] + 4, MARGIN + TITLE_H - 22, ROLE[role]["name"].upper(), fontsize=10.5,
                weight="bold", color=ROLE[role]["edge"], va="bottom")

    anchors = {}
    for pid, L in layout.items():
        p = L["panel"]
        rounded(ax, L["x"], L["y"], L["w"], L["h"], dashed=p.get("pending", False))
        ax.text(L["x"] + PANEL_PAD, L["y"] + 42, p["title"], fontsize=12, weight="bold",
                color=INK, va="center")
        left_gutter = L["role"] == "revised"
        ix = L["x"] + PANEL_PAD + (GUTTER if left_gutter else 0)
        iy = L["y"] + PANEL_TITLE_H
        if p.get("pending"):
            ax.text(L["x"] + L["w"] / 2, iy + L["h_img"] / 2, p["pending"], fontsize=13,
                    color=GREY, ha="center", va="center", wrap=True)
            ax.text(L["x"] + PANEL_PAD, L["y"] + L["h"] - SRC_H / 2 - 6, p.get("src_note", ""),
                    fontsize=8.5, color=GREY, va="center")
            for m in p.get("marks", []):
                anchors[(pid, m["key"])] = (L["x"], L["x"] + L["w"], iy + L["h_img"] / 2, L["role"])
            continue
        b = L["built"]
        s = b["scale"]
        w_img, h_img = b["img"].width * s, b["img"].height * s
        ax.imshow(b["img"], extent=(ix, ix + w_img, iy + h_img, iy), zorder=2,
                  interpolation="lanczos")
        ax.add_patch(Rectangle((ix, iy), w_img, h_img, fill=False, edgecolor="#d5d8dc",
                               linewidth=1, zorder=3))
        ax.text(ix, L["y"] + L["h"] - SRC_H / 2 - 6, b["src_label"],
                fontsize=8.5, color=GREY, va="center")
        if p.get("unconfirmed"):
            tag(ax, L["x"] + L["w"] - PANEL_PAD, L["y"] + 42, "location unconfirmed", "#7d3c98",
                ha="right", va="center")
        role = L["role"]
        for m in b["marks"]:
            col = ROLE[m.get("role", role)]
            ys = []
            for bx0, by0, bx1, by1 in m["boxes"]:
                X0, Y0 = ix + (bx0 - 5) * s, iy + (by0 - 4) * s
                W, H = (bx1 - bx0 + 10) * s, (by1 - by0 + 8) * s
                ax.add_patch(Rectangle((X0, Y0), W, H, facecolor=col["fill"], alpha=0.35,
                                       edgecolor="none", zorder=3))
                ax.add_patch(Rectangle((X0, Y0), W, H, fill=False, edgecolor=col["edge"],
                                       linewidth=2.4, zorder=4))
                ys.append((Y0, Y0 + H, X0, X0 + W))
            ymid = (min(y[0] for y in ys) + max(y[1] for y in ys)) / 2
            m["_ymid"], m["_ys"], m["_col"] = ymid, ys, col
        placed = []
        for m in sorted(b["marks"], key=lambda m: (round(m["_ymid"] / 12), min(y[2] for y in m["_ys"]))):
            ly = m["_ymid"]
            for prev in placed:
                if abs(ly - prev) < 52:
                    ly = prev + 52
            placed.append(ly)
            col, ys = m["_col"], m["_ys"]
            if left_gutter:
                gx = L["x"] + PANEL_PAD
                box_x = ix
                label_x, ha = gx, "left"
                ax.plot([gx + GUTTER - 30, box_x], [ly, m["_ymid"]], color=col["edge"], lw=1.6, zorder=5)
            else:
                gx = ix + w_img + 20
                box_x = ix + w_img
                label_x, ha = gx + GUTTER - 30, "right"
                ax.plot([box_x, gx + 20], [m["_ymid"], ly], color=col["edge"], lw=1.6, zorder=5)
            if m.get("label"):
                tag(ax, label_x, ly, m["label"], col["edge"], size=9.5, ha=ha, va="center")
            anchors[(pid, m["key"])] = (L["x"], L["x"] + L["w"], ly, m.get("role", role))

    # Arrows between marks, with a square port on each panel edge.
    for a in spec.get("arrows", []):
        sx0, sx1, sy, srole = anchors[(a["from"][0], a["from"][1])]
        tx0, tx1, ty, trole = anchors[(a["to"][0], a["to"][1])]
        color = ROLE[a.get("role", trole)]["edge"]
        start = (sx1, sy)
        end = (tx0, ty)
        for (px, py) in (start, end):
            ax.add_patch(Rectangle((px - 9, py - 16), 18, 32, facecolor=color, edgecolor="none",
                                   zorder=7))
        ax.annotate("", xy=(end[0] - 10, end[1]), xytext=(start[0] + 10, start[1]),
                    arrowprops=dict(arrowstyle="-|>,head_length=0.9,head_width=0.45", color=color,
                                    lw=3.2, shrinkA=0, shrinkB=0), zorder=6)
        if a.get("label"):
            mx, my = (start[0] + end[0]) / 2, (start[1] + end[1]) / 2
            tag(ax, mx, my, a["label"], color, size=10, ha="center", va="center", z=8)

    # Reply panel.
    rx, rw = MARGIN, canvas_w - 2 * MARGIN
    rounded(ax, rx, reply_y, rw, reply_h, edge="#5b6b8c", lw=2.6)
    ax.text(rx + PANEL_PAD, reply_y + 42, spec["reply"]["title"], fontsize=12, weight="bold",
            color=INK, va="center")
    for ci, col_blocks in enumerate(text_cols):
        tx = rx + PANEL_PAD + ci * (text_col_w + 60)
        ty = reply_y + PANEL_TITLE_H + 10
        for _, parts, _h in col_blocks:
            for kind, wrapped, h in parts:
                for ln in wrapped:
                    ax.text(tx, ty, ln, fontsize=size_h if kind == "head" else size_p,
                            weight="bold" if kind == "head" else "normal",
                            color="#24324f" if kind == "head" else INK, va="top")
                    ty += line_h
                ty += line_h * 0.6
    if spec["reply"].get("status"):
        tag(ax, rx + rw - PANEL_PAD, reply_y + 42, spec["reply"]["status"],
            spec["reply"].get("status_color", "#5b6b8c"), ha="right", va="center", size=10.5)

    fy = reply_y + reply_h + 40
    for line in footer_lines:
        ax.text(MARGIN, fy, line, fontsize=9, color=GREY, va="center")
        fy += 40

    HERE.mkdir(parents=True, exist_ok=True)
    stem = HERE / spec["file"]
    fig.savefig(f"{stem}.png", dpi=DPI, facecolor="white")
    fig.savefig(f"{stem}.pdf", dpi=DPI, facecolor="white")
    plt.close(fig)
    return Path(f"{stem}.png")


def contact_sheet(paths: list[Path], out: Path, per_row: int = 3, thumb_w: int = 900) -> None:
    thumbs = []
    for p in paths:
        im = Image.open(p).convert("RGB")
        im.thumbnail((thumb_w, thumb_w * 3))
        thumbs.append((p.stem, im))
    rows = [thumbs[i:i + per_row] for i in range(0, len(thumbs), per_row)]
    row_h = [max(t[1].height for t in r) + 60 for r in rows]
    sheet = Image.new("RGB", (per_row * (thumb_w + 30) + 30, sum(row_h) + 30), "white")
    from PIL import ImageDraw
    d = ImageDraw.Draw(sheet)
    y = 30
    for r, h in zip(rows, row_h):
        x = 30
        for name, im in r:
            d.text((x, y), name, fill=(60, 60, 60))
            sheet.paste(im, (x, y + 24))
            x += thumb_w + 30
        y += h
    sheet.save(out)


def main(argv: list[str]) -> None:
    sys.path.insert(0, str(HERE))
    from figure_specs import SPECS  # noqa: E402

    if "--sheet" in argv:
        pngs = [HERE / f'{s["file"]}.png' for s in SPECS if (HERE / f'{s["file"]}.png').exists()]
        contact_sheet(pngs, HERE / "contact-sheet.png")
        print(HERE / "contact-sheet.png")
        return
    wanted = [a for a in argv if not a.startswith("--")]
    for spec in SPECS:
        if wanted and spec["group"] not in wanted:
            continue
        print(draw_figure(spec))


if __name__ == "__main__":
    main(sys.argv[1:])
