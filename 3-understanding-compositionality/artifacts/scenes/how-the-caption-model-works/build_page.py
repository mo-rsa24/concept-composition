"""Assemble the published page: page-src.html + scene_data.json + page-images/ + verbatim source lines.

    python3 -I build_page.py        writes how-the-caption-model-works.html beside this file
"""
import base64, html, json, re
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CODE = ROOT / "artifacts" / "notes" / "what-poe-loses-when-it-composes"

# key: (file, first line, last line, lines to mark)
SNIPPETS = {
    "corpus": ("toy_caption_threshold.py", 28, 41, [33, 36, 37, 38]),
    "vae": ("toy_caption_dial_ldm.py", 69, 86, [72, 73, 74, 75, 76, 79]),
    "denoiser": ("toy_caption_dial_ldm.py", 89, 99, [92, 93, 98, 99]),
    "label": ("toy_caption_dial_ldm.py", 89, 99, [92, 99]),
    "train": ("toy_caption_dial_ldm.py", 144, 158, [152, 154, 155]),
    "draw": ("toy_caption_dial_ldm.py", 160, 180, [170, 173, 174, 175]),
    "poe": ("toy_caption_dial_ldm.py", 160, 180, [169, 170, 171, 172, 173]),
    "threshold": ("toy_caption_threshold.py", 44, 63, [52, 55, 56]),
    "corrector": ("toy_caption_corrector.py", 37, 66, [59, 60, 61, 62, 63]),
}


def snippet(key):
    f, a, b, mark = SNIPPETS[key]
    lines = (CODE / f).read_text().splitlines()
    rows = []
    for n in range(a, b + 1):
        cls = ' class="hl"' if n in mark else ""
        rows.append(f'<span{cls}><i>{n}</i>{html.escape(lines[n - 1])}</span>')
    return {"file": f"3-understanding-compositionality/artifacts/notes/what-poe-loses-when-it-composes/{f}", "lines": f"{a}-{b}",
            "html": "\n".join(rows)}


data = json.loads((HERE / "scene_data.json").read_text())
imgs = {}
for p in sorted((HERE / "page-images").glob("*.png")):
    imgs[p.name] = "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode()
src = (HERE / "page-src.html").read_text()
RES = ROOT / "artifacts" / "results" / "how-much-co-occurrence-the-product-needs"
full = {}
for f in ["toy-caption-threshold.json", "toy-caption-threshold-600-1200-2000.json"]:
    for k, rows in json.loads((RES / f).read_text())["results"].items():
        full[k] = [r["counts"]["product"] for r in rows]
payload = {"D": data, "IMG": imgs, "CODE": {k: snippet(k) for k in SNIPPETS}, "TC": full}
blob = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
assert src.count("__PAYLOAD__") == 1
out = src.replace("__PAYLOAD__", blob)
(HERE / "how-the-caption-model-works.html").write_text(out)
print("wrote", HERE / "how-the-caption-model-works.html", f"{len(out)/1e6:.2f} MB, {len(imgs)} images")
