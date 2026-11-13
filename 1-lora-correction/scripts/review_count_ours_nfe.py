"""Count ours' network evaluations per image (protocol rule 6) by running its real renderer on one cell.

Runs 1-lora-correction/scripts/figure_candidates.py unchanged, with every U-Net forward call on the class counted: calls, and rows
(the batch of each call). Prints the totals for the process at exit. The tile goes wherever FIGURE_CANDIDATES_OUT
points, kept apart from the evaluation folders.

    FIGURE_CANDIDATES_OUT=<scratch> python 1-lora-correction/scripts/review_count_ours_nfe.py --column ours --pairs "a seal|a walrus" \
        --seeds 11 --checkpoint <pt> --window 50 --tag count
"""
from __future__ import annotations

import atexit
import os
import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = Path(os.environ.get("POE_REPO", HERE.parent))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

from diffusers import UNet2DConditionModel  # noqa: E402

COUNT = {"calls": 0, "rows": 0}
_inner = UNet2DConditionModel.forward


def counted(self, sample, *args, **kwargs):
    COUNT["calls"] += 1
    COUNT["rows"] += int(sample.shape[0])
    return _inner(self, sample, *args, **kwargs)


UNet2DConditionModel.forward = counted
atexit.register(lambda: print(f"NFE COUNT: {COUNT['rows']} rows in {COUNT['calls']} U-Net calls", flush=True))
sys.argv = [str(HERE / "figure_candidates.py")] + sys.argv[1:]
runpy.run_path(str(HERE / "figure_candidates.py"), run_name="__main__")
