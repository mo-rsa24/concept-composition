#!/usr/bin/env python3
"""Build the correction-timing page as a static folder that opens with no server.

Run on the cluster, where the window sweep's pictures live (/datasets). It rebuilds the
timing manifest from what is on disk, writes a 512 px JPEG copy of every picture the
manifest names, points the manifest at those copies, and renders the inspector's own
timing tab (1-lora-correction/scripts/window_tab.py) into index.html with the manifest
inlined. Only the two lines that loaded pictures through the inspector's Flask route
change: they load the local copies instead.

    PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python
    PYTHONPATH=$REPO $PY 1-lora-correction/artifacts/scenes/when-the-correction-must-arrive/loader/build_scene.py --out /tmp/timing-scene

Then copy <out>/index.html, <out>/manifest.json and <out>/img/ into this scene folder.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from jinja2 import Template
from PIL import Image

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "1-lora-correction/scripts"))
from window_tab import WINDOW_INDEX_HTML  # noqa: E402

ROUTE = '"/interaction_window/img?path=" + encodeURIComponent(cell.image)'
EDGE = 512


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    out = a.out
    (out / "img").mkdir(parents=True, exist_ok=True)

    full = out / "manifest.full.json"
    subprocess.run([sys.executable, str(REPO / "1-lora-correction/scripts/build_window_manifest.py"),
                    "--out", str(full)], check=True)
    m = json.loads(full.read_text())

    n = 0
    for pair, seeds in m["cells"].items():
        for seed, wins in seeds.items():
            for key, cell in wins.items():
                src = cell.get("image")
                if not src or not Path(src).exists():
                    cell["image"] = None
                    continue
                rel = Path("img") / pair / f"seed_{seed}" / f"{key}.jpg"
                dst = out / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                im = Image.open(src).convert("RGB")
                im.thumbnail((EDGE, EDGE))
                im.save(dst, quality=82)
                cell["source_image"] = src
                cell["image"] = str(rel)
                n += 1

    (out / "manifest.json").write_text(json.dumps(m, indent=1))
    curve = m.get("curve") or {}
    html = Template(WINDOW_INDEX_HTML).render(
        width=m.get("width", "—"), stride=m.get("stride", "—"),
        num_steps=m.get("num_steps", 50), fork_step=m.get("fork_step", "—"),
        n_windows=len(m.get("windows") or []), n_on_disk=m.get("n_cells_on_disk", 0),
        n_planned=m.get("n_cells_planned", 0), n_scored=m.get("n_cells_scored", 0),
        scorer=curve.get("scorer"), manifest_json=json.dumps(m),
    )
    assert html.count(ROUTE) == 2, f"expected 2 picture lines, found {html.count(ROUTE)}"
    html = html.replace(ROUTE, "cell.image")
    (out / "index.html").write_text(html)
    print(f"{n} pictures, index.html {len(html) // 1024} KB -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
