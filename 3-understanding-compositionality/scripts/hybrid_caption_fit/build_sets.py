#!/usr/bin/env python
"""Name every image in the four sets, and prove the local ones open.

The sets:

  photo    the 91 both-animal COCO photographs the inclusive-captions audit accepted. They live
           only on the cluster, so their paths are recorded here and checked there.
  product  the plain product of a cat expert and a dog expert, 7 seeds, 1024 px.
  joint    one sentence naming both animals, no product at all, 32 seeds, cropped from the sheet.
  plural   the plain product prompted "cats" and "dogs", 8 seeds, carried as a fourth point.

Writes sets.json. Nothing is embedded here: this step exists so a missing file is found before
anything is copied to the cluster or run through a model.
"""
from __future__ import annotations

import json
import subprocess

from PIL import Image

import paths


def cluster(command: str) -> str:
    out = subprocess.run(
        [str(paths.REPO / "scripts/cluster.sh"), "sh", command],
        cwd=paths.REPO, capture_output=True, text=True, timeout=180,
    )
    if out.returncode != 0:
        raise SystemExit(f"cluster command failed:\n{out.stderr}")
    return out.stdout


def photographs() -> list[dict]:
    """The 91 accepted photographs, read off the pool manifest on the cluster.

    The manifest's `both` list is already the audited set: the pool script applies EXCLUDED_BOTH
    when it builds it, so the 40 struck off by eye are not in there and no second filter is owed.
    """
    raw = cluster(
        "/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python -c "
        "\"import json;m=json.load(open('%s'));print(json.dumps(m['sets']['both']))\"" % paths.POOL_MANIFEST
    )
    rows = json.loads(raw.strip().splitlines()[-1])
    return [
        {
            "set": "photo",
            "key": f"coco{row['image_id']}",
            "path": row["file"],
            "local": None,
            "caption": row.get("caption"),
        }
        for row in rows
    ]


def renders() -> list[dict]:
    out: list[dict] = []
    for name, folder, pattern in (
        ("product", paths.PRODUCT_TILES, "seed*/poe.png"),
        ("plural", paths.PLURAL_TILES, "seed*/poe.png"),
        ("joint", paths.JOINT_TILES, "seed*.png"),
    ):
        found = sorted(folder.glob(pattern))
        if not found:
            raise SystemExit(f"no {name} images under {folder}")
        for p in found:
            seed = int("".join(c for c in (p.parent.name if name != "joint" else p.stem) if c.isdigit()))
            out.append({
                "set": name,
                "key": f"{name}-seed{seed:02d}",
                "seed": seed,
                "path": str(p.relative_to(paths.REPO)),
                "local": str(p),
            })
    return out


def main() -> None:
    rows = photographs() + renders()

    for row in rows:
        if row["local"]:
            with Image.open(row["local"]) as im:
                row["size"] = list(im.size)

    counts: dict[str, int] = {}
    for row in rows:
        counts[row["set"]] = counts.get(row["set"], 0) + 1

    missing = cluster(
        "/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python -c "
        "\"import json,os;m=json.load(open('%s'));"
        "print(sum(1 for r in m['sets']['both'] if not os.path.exists(r['file'])))\"" % paths.POOL_MANIFEST
    ).strip().splitlines()[-1]

    product_seeds = {r["seed"] for r in rows if r["set"] == "product"}
    joint_seeds = {r["seed"] for r in rows if r["set"] == "joint"}
    matched = sorted(product_seeds & joint_seeds)

    paths.OUT.mkdir(parents=True, exist_ok=True)
    paths.SETS.write_text(json.dumps({
        "captions": paths.CAPTIONS,
        "counts": counts,
        "photographs_missing_on_cluster": int(missing),
        "matched_seeds": matched,
        "images": rows,
    }, indent=1))

    print("counts:", counts)
    print("photographs missing on the cluster:", missing)
    print(f"seeds both render sets cover: {len(matched)} -> {matched}")
    print("wrote", paths.SETS)


if __name__ == "__main__":
    main()
