#!/usr/bin/env python
"""Copy the render images to the cluster and write the manifest the embedding step reads.

The renders live under artifacts/results/, which is gitignored, so they cannot travel with the
commit. They go straight to the cluster's output folder instead. The photographs are already
there and are named in place.

Writes manifest.json beside the images on the cluster: one row per image with its set, its key
and the path to open it by.
"""
from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import paths

NODE = "mscluster84"


def main() -> None:
    sets = json.loads(paths.SETS.read_text())
    rows = sets["images"]

    manifest = []
    staged = Path(tempfile.mkdtemp(prefix="hybrid-caption-fit-"))
    for row in rows:
        if row["local"]:
            name = f"{row['key']}.png"
            (staged / name).write_bytes(Path(row["local"]).read_bytes())
            remote = f"{paths.CLUSTER_IMAGES}/{name}"
        else:
            remote = row["path"]
        manifest.append({"set": row["set"], "key": row["key"], "path": remote,
                         "seed": row.get("seed")})

    (staged / "manifest.json").write_text(json.dumps(
        {"captions": paths.CAPTIONS, "images": manifest}, indent=1))

    subprocess.run(["ssh", NODE, f"mkdir -p {paths.CLUSTER_IMAGES}"], check=True)
    subprocess.run(
        ["rsync", "-az", f"{staged}/", f"{NODE}:{paths.CLUSTER_IMAGES}/"], check=True)
    subprocess.run(
        ["ssh", NODE,
         f"mv {paths.CLUSTER_IMAGES}/manifest.json {paths.CLUSTER_OUT}/manifest.json"], check=True)

    pushed = sum(1 for r in rows if r["local"])
    print(f"pushed {pushed} render images to {NODE}:{paths.CLUSTER_IMAGES}")
    print(f"manifest names {len(manifest)} images at {paths.CLUSTER_OUT}/manifest.json")


if __name__ == "__main__":
    main()
