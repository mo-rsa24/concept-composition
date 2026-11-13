"""Transpose a paper_tile_figure.py spec: columns become rows and rows become columns.

The old column titles become row labels, and the old row labels become column titles, so a grid of
many rows by few columns turns into a wide grid that fills the text width.

    python3 scripts/transpose_tile_spec.py <spec.json> <out spec.json> <width_in>
"""
import json
import sys

src, out, width = sys.argv[1], sys.argv[2], float(sys.argv[3])
spec = json.load(open(src))
cols = [c for g in spec["groups"] for c in g["columns"]]
rows = spec["rows"]
new = {
    "width_in": width,
    "groups": [{"title": "", "columns": [r["label"] for r in rows]}],
    "rows": [{"label": c.replace("\n", " "), "tiles": [r["tiles"][j] for r in rows]} for j, c in enumerate(cols)],
    "rotate_labels": False,
}
for k in ("pending", "panel", "footnote"):
    if k in spec:
        new[k] = spec[k]
json.dump(new, open(out, "w"), indent=1)
print(f"{len(rows)}x{len(cols)} -> {len(new['rows'])}x{len(rows)}")
