#!/usr/bin/env python
"""Bring similarities.json back from the cluster into the local artifact folder."""
from __future__ import annotations

import json
import subprocess

import paths

NODE = "mscluster84"


def main() -> None:
    paths.OUT.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["rsync", "-az", f"{NODE}:{paths.CLUSTER_OUT}/similarities.json", str(paths.SIMILARITIES)],
        check=True)
    data = json.loads(paths.SIMILARITIES.read_text())
    counts: dict[str, int] = {}
    for rec in data["records"]:
        counts[rec["set"]] = counts.get(rec["set"], 0) + 1
    print("pulled", paths.SIMILARITIES)
    print("records:", counts, "on", data["device"])


if __name__ == "__main__":
    main()
