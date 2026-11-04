"""Shared paths and the attribute encoding for the CelebA held-out-combination experiment.

Each of the two attributes has three states: 0 = unspecified (dropped), 1 = absent, 2 = present.
  male expert   (male=2, blond=0)        blond expert  (male=0, blond=2)
  empty prompt  (male=0, blond=0)        joint prompt  (male=2, blond=2)
"""
from pathlib import Path

ROOT = Path("/datasets/mmolefe/celeba")
PARQUET = ROOT / "flwrlabs-celeba" / "img_align+identity+attr"
ARRAYS = ROOT / "arrays64"
RUNS = ROOT / "runs"
PY = "/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python"
RES = 64
UNSPEC, ABSENT, PRESENT = 0, 1, 2
CONDITIONS = {"male": (PRESENT, UNSPEC), "blond": (UNSPEC, PRESENT),
              "empty": (UNSPEC, UNSPEC), "joint": (PRESENT, PRESENT)}
