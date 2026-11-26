#!/usr/bin/env python3
"""Paper 1's pack: the joint renders squashed to 64 pixels as the head's positives, with their masks, touching flags and checks.

Settings: plans/paper-1-clevr-touching/plans/02-first-result.md, task 7 and "Seeds and scenes" (subset 1 at k = 64 is the pool's first
64, sizes in turn; the other 448 are the head's held-out positives); 03-full-plan.md, step 17 (the squash, composition.txt and the
SHA-256 leakage check); artifacts/notes/guarding-the-first-results/paper-1.md, guardrail 2a (the depth-order count).

Reads a joint render set from clevr_render.py's joint mode: images/<set>_<index>.png at 320 by 240, scenes/<set>_<index>.json (each
object's colour and pixel_coords, whose third value is its depth from the camera) and masks/<set>_<index>.npz (each object's visible
share per pixel at 64 by 64, in the scene file's object order). Squashes each image with clevr_data.py's to_64 (Lanczos), the same
squash the experts were trained on, imported rather than copied.

Writes, under the run folder, for each of the two parts (k<k>-s1 and heldout):
  <part>/count-<c>/<index:06d>.png            the 64-pixel picture; c is each colour's count
  <part>/masks/count-<c>/<index:06d>-red.png  the union of the red objects' masks (an object pixel is one at least half covered)
  <part>/masks/count-<c>/<index:06d>-blue.png
  <part>/touching.json                        {"files": {"count-<c>/<index>.png": touching}, "per_count": {"count-<c>": {...}}}
and at the top: touching_share.json, composition.txt, leakage.txt, depth-order.json and sheet-first-64.png. Exits 1 if any picture
is shared between the two parts (by SHA-256 of the render and of the 64-pixel picture), or if a render is missing.

Usage:
  python paper1_pack.py --run paper1-first-pack --renders paper1-first-render-joint/renders/joint --count 512 --k 64
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "method_search"))
from clevr_data import to_64  # noqa: E402  the experts' own Lanczos squash to 64 pixels

PAPER1_ROOT = Path("/datasets/mmolefe/poe_repair_min/paper1")
MASK_COVER = 0.5              # a 64-pixel mask pixel belongs to an object when the object covers at least half of it
POOL = 512                    # the first result's joint renders: indices 0 to 511 at seed base 8,000,000
K = 64                        # subset 1 at k = 64: the pool's first 64, sizes in turn


def resolve(root, path):
    p = Path(path)
    return p if p.is_absolute() else Path(root) / p


def near(a, b):
    """True when some pixel of a lies within 1 pixel (3 by 3 neighbourhood) of some pixel of b; a and b are (64, 64) booleans."""
    grown = np.zeros_like(a)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            grown |= np.roll(np.roll(np.pad(a, 1), dy, 0), dx, 1)[1:-1, 1:-1]
    return bool((grown & b).any())


def read_scene(renders, name, index):
    stem = "%s_%06d" % (name, index)
    scene = json.loads((renders / "scenes" / f"{stem}.json").read_text())
    masks = np.load(renders / "masks" / f"{stem}.npz")["masks"].astype(np.float32) / 255.0 >= MASK_COVER
    if len(masks) != len(scene["objects"]):
        raise SystemExit(f"{stem}: {len(masks)} masks for {len(scene['objects'])} objects")
    return stem, scene, masks


def depth_order(scene, masks):
    """Guardrail 2a: for each touching red and blue object pair, whether the nearer object (smaller camera depth) is the one whose
    base (lowest mask row) sits lower in the frame. Returns a list of booleans, one per touching pair."""
    out = []
    objs = scene["objects"]
    for i, a in enumerate(objs):
        for j, b in enumerate(objs):
            if a["color"] != "red" or b["color"] != "blue" or not masks[i].any() or not masks[j].any():
                continue
            if not near(masks[i], masks[j]):
                continue
            base_i, base_j = np.nonzero(masks[i])[0].max(), np.nonzero(masks[j])[0].max()
            if base_i == base_j:
                continue
            nearer_i = a["pixel_coords"][2] < b["pixel_coords"][2]
            out.append(bool(nearer_i == (base_i > base_j)))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--root", default=str(PAPER1_ROOT))
    ap.add_argument("--renders", required=True, help="the joint render set's folder, holding images/, scenes/ and masks/")
    ap.add_argument("--count", type=int, default=POOL, help="pool indices 0 to count - 1")
    ap.add_argument("--k", type=int, default=K, help="the subset: the pool's first k")
    args = ap.parse_args(argv)
    folder = resolve(args.root, args.run)
    renders = resolve(args.root, args.renders)
    name = renders.name
    parts = {f"k{args.k}-s1": range(0, args.k), "heldout": range(args.k, args.count)}
    hashes = {part: {} for part in parts}
    stats = {part: {} for part in parts}
    depth = {}
    missing = []
    for part, indices in parts.items():
        flags = {}
        for index in indices:
            image_path = renders / "images" / ("%s_%06d.png" % (name, index))
            if not image_path.exists():
                missing.append(str(image_path))
                continue
            stem, scene, masks = read_scene(renders, name, index)
            colours = [o["color"] for o in scene["objects"]]
            c = colours.count("red")
            if c != colours.count("blue") or c not in (1, 2, 3):
                raise SystemExit(f"{stem}: {colours.count('red')} red and {colours.count('blue')} blue objects")
            small = np.clip(np.round(to_64(image_path) * 255), 0, 255).astype(np.uint8)
            rel = f"count-{c}/{index:06d}.png"
            (folder / part / f"count-{c}").mkdir(parents=True, exist_ok=True)
            (folder / part / "masks" / f"count-{c}").mkdir(parents=True, exist_ok=True)
            Image.fromarray(small, "RGB").save(folder / part / rel)
            red = np.any([m for m, col in zip(masks, colours) if col == "red"], axis=0)
            blue = np.any([m for m, col in zip(masks, colours) if col == "blue"], axis=0)
            Image.fromarray(red.astype(np.uint8) * 255, "L").save(folder / part / "masks" / f"count-{c}" / f"{index:06d}-red.png")
            Image.fromarray(blue.astype(np.uint8) * 255, "L").save(folder / part / "masks" / f"count-{c}" / f"{index:06d}-blue.png")
            flags[rel] = near(red, blue)
            hashes[part][rel] = (hashlib.sha256(image_path.read_bytes()).hexdigest(), hashlib.sha256(small.tobytes()).hexdigest())
            s = stats[part].setdefault(f"count-{c}", {"scenes": 0, "touching": 0, "object_pixels": []})
            s["scenes"] += 1
            s["touching"] += int(flags[rel])
            s["object_pixels"] += [int(m.sum()) for m in masks]
            if part == "heldout":
                depth.setdefault(f"count-{c}", []).extend(depth_order(scene, masks))
        per_count = {k: {"scenes": v["scenes"], "touching": v["touching"], "touching_share": round(v["touching"] / max(1, v["scenes"]), 4)}
                     for k, v in sorted(stats[part].items())}
        (folder / part).mkdir(parents=True, exist_ok=True)
        (folder / part / "touching.json").write_text(json.dumps({"rule": "some red mask pixel within 1 pixel of some blue mask pixel, "
                                                                          f"masks at 64 pixels, object pixel = covered >= {MASK_COVER}",
                                                                  "files": flags, "per_count": per_count}, indent=1) + "\n")
    if missing:
        print(json.dumps({"missing_renders": len(missing), "first": missing[0]}), flush=True)
        raise SystemExit(1)

    share = {part: {k: v["touching_share"] for k, v in json.loads((folder / part / "touching.json").read_text())["per_count"].items()}
             for part in parts}
    pooled = {part: round(sum(v["touching"] for v in stats[part].values()) / max(1, sum(v["scenes"] for v in stats[part].values())), 4)
              for part in parts}
    (folder / "touching_share.json").write_text(json.dumps({"per_count": share, "pooled": pooled}, indent=1) + "\n")

    lines = ["Paper 1 pack: composition of the subset and the held-out scenes", ""]
    for part in parts:
        lines.append(f"{part}: {sum(v['scenes'] for v in stats[part].values())} scenes, touching share pooled {pooled[part]}")
        for k, v in sorted(stats[part].items()):
            px = np.array(v["object_pixels"])
            lines.append(f"  {k}: {v['scenes']} scenes, touching share {round(v['touching'] / max(1, v['scenes']), 4)}, "
                         f"object pixels at 64 (visible, covered >= {MASK_COVER}): median {int(np.median(px))}, "
                         f"10th percentile {int(np.percentile(px, 10))}, 90th percentile {int(np.percentile(px, 90))}")
    (folder / "composition.txt").write_text("\n".join(lines) + "\n")

    shared_render = set(h[0] for h in hashes[f"k{args.k}-s1"].values()) & set(h[0] for h in hashes["heldout"].values())
    shared_small = set(h[1] for h in hashes[f"k{args.k}-s1"].values()) & set(h[1] for h in hashes["heldout"].values())
    (folder / "leakage.txt").write_text(f"subset {len(hashes[f'k{args.k}-s1'])} pictures, held-out {len(hashes['heldout'])}\n"
                                        f"renders shared by SHA-256: {len(shared_render)}\n64-pixel pictures shared by SHA-256: {len(shared_small)}\n")

    depth_out = {"rule": "touching red and blue object pairs in the held-out scene files: the nearer object (smaller camera depth, "
                         "pixel_coords[2]) is the one whose base (lowest mask row at 64 pixels) is lower in the frame; pairs with level "
                         "bases are left out", "per_count": {k: {"pairs": len(v), "nearer_is_lower": sum(v),
                                                                 "share": round(sum(v) / len(v), 4) if v else None}
                                                             for k, v in sorted(depth.items())}}
    (folder / "depth-order.json").write_text(json.dumps(depth_out, indent=1) + "\n")

    first = sorted(hashes[f"k{args.k}-s1"], key=lambda rel: int(Path(rel).stem))[:64]   # render index order
    tiles = [Image.open(folder / f"k{args.k}-s1" / rel).resize((256, 256), Image.NEAREST) for rel in first]
    if tiles:                                         # 4 times size, the asked counts under each (the smoke sheet uses --k 24)
        from PIL import ImageDraw
        sheet = Image.new("RGB", (8 * 258, ((len(tiles) + 7) // 8) * 272), "white")
        draw = ImageDraw.Draw(sheet)
        for i, (t, rel) in enumerate(zip(tiles, first)):
            x, y = (i % 8) * 258, (i // 8) * 272
            sheet.paste(t, (x, y))
            c = rel.split("/")[0].split("-")[1]
            draw.text((x + 2, y + 258), f"{c} red + {c} blue, scene {int(Path(rel).stem)}", fill="black")
        sheet.save(folder / "sheet-first-64.png")

    print(json.dumps({"touching_share": share, "pooled": pooled, "depth_order": {k: v["share"] for k, v in depth_out["per_count"].items()},
                      "shared_renders": len(shared_render), "shared_64": len(shared_small)}), flush=True)
    if shared_render or shared_small:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
