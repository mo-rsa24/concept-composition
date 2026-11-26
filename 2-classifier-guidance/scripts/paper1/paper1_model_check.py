#!/usr/bin/env python3
"""Paper 1's model-file check (the setup plan's task 10, the full plan's step 5): find the three frozen models and check their hashes.

Lists every network snapshot and any training-state-*.pt in the red and blue experts' folders, with size and SHA-256, and compares each
snapshot with its line in that folder's pod.sha256; then checks the empty-scene model, the M7 run's final 0.100 snapshot, against its
recorded size and SHA-256 (environment/clevr-and-edm2.md, "The colour experts' model files"). Writes models.txt at the top of the run
folder, which the job script copies to outputs/paper1/<run>/. Needs no torch and no GPU: a CPU job of a few minutes.

Passes (exit 0) when the red and blue experts' 0.100 snapshots and the empty-scene model are found and all three hashes match; any
missing file or mismatch exits 1. Whether training states exist is reported; it decides the full plan's step 53 loader, not this check.

Usage, as the CPU job paper1-model-check on the held node (never run by the setup loop):
  bash 2-classifier-guidance/scripts/paper1/paper1_render.sbatch paper1-model-check \
      python 2-classifier-guidance/scripts/paper1/paper1_model_check.py --run paper1-model-check
"""
import argparse
import hashlib
import sys
from pathlib import Path

PAPER1_ROOT = Path("/datasets/mmolefe/poe_repair_min/paper1")
EXPERT_DIR = Path("/datasets/mmolefe/poe_repair_min/loop-6/m10/full/models")
EXPERTS = {"red": EXPERT_DIR / "r-full", "blue": EXPERT_DIR / "b-full"}
EXPERT_SNAPSHOT = "network-snapshot-0002000-0.100.pkl"     # the 0.100 EMA snapshot the sampler loads (paper1_sample.py RED_NET, BLUE_NET)
EMPTY_NET = Path("/datasets/mmolefe/poe_repair_min/loop-6/m7/full/models/empty/network-snapshot-0002000-0.100.pkl")
EMPTY_BYTES = 248_538_254
EMPTY_SHA256 = "5e7905efd3b723d79507dde0af9d5d63d959738a2e08ab234646101ece8fb272"


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def read_pod_sha256(path):
    """pod.sha256 as {file name: hash}, from sha256sum's '<hash>  <path>' lines (a leading '*' on the path is allowed)."""
    out = {}
    for line in path.read_text().splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) == 2:
            out[Path(parts[1].lstrip("*")).name] = parts[0].lower()
    return out


def check_expert(name, folder, lines):
    """Lines for models.txt and whether this expert's 0.100 snapshot is found and matches."""
    if not folder.is_dir():
        lines.append(f"{name}: folder {folder} MISSING")
        return False
    pod = folder / "pod.sha256"
    recorded = read_pod_sha256(pod) if pod.is_file() else {}
    lines.append(f"{name}: {folder}  (pod.sha256: {'found, ' + str(len(recorded)) + ' lines' if pod.is_file() else 'MISSING'})")
    ok = False
    for snap in sorted(folder.glob("network-snapshot-*.pkl")):
        digest = sha256(snap)
        want = recorded.get(snap.name)
        verdict = "match" if want == digest else ("no line in pod.sha256" if want is None else "MISMATCH")
        lines.append(f"  {snap}  {snap.stat().st_size:,} bytes  sha256 {digest}  {verdict}")
        if snap.name == EXPERT_SNAPSHOT and verdict == "match":
            ok = True
    states = sorted(folder.glob("training-state-*.pt"))
    lines += [f"  {s}  {s.stat().st_size:,} bytes  (training state)" for s in states] or ["  no training-state-*.pt"]
    if not (folder / EXPERT_SNAPSHOT).is_file():
        lines.append(f"  {EXPERT_SNAPSHOT} MISSING")
    return ok


def check_empty(lines):
    if not EMPTY_NET.is_file():
        lines.append(f"empty-scene model: {EMPTY_NET} MISSING")
        return False
    size, digest = EMPTY_NET.stat().st_size, sha256(EMPTY_NET)
    ok = size == EMPTY_BYTES and digest == EMPTY_SHA256
    lines.append(f"empty-scene model: {EMPTY_NET}  {size:,} bytes (recorded {EMPTY_BYTES:,})  sha256 {digest}  "
                 f"{'match' if ok else 'MISMATCH against ' + EMPTY_SHA256}")
    return ok


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--root", default=str(PAPER1_ROOT))
    args = ap.parse_args(argv)
    folder = Path(args.root) / args.run
    folder.mkdir(parents=True, exist_ok=True)
    lines = ["Paper 1 model-file check: the red and blue experts against pod.sha256, the empty-scene model against its recorded hash", ""]
    results = {name: check_expert(name, path, lines) for name, path in EXPERTS.items()}
    results["empty"] = check_empty(lines)
    passes = all(results.values())
    lines += ["", f"red 0.100 snapshot matches: {results['red']}", f"blue 0.100 snapshot matches: {results['blue']}",
              f"empty-scene model matches: {results['empty']}", f"passes: {passes}"]
    (folder / "models.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    sys.exit(0 if passes else 1)


if __name__ == "__main__":
    main()
