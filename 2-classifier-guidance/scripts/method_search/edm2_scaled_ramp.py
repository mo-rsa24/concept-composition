#!/usr/bin/env python3
"""Run EDM2's train_edm2.py with the preset's learning-rate ramp scaled to this run's length (step M7).

The edm2-img64-xs preset trains on 1,024 x 2^20 images and ramps its learning rate over the first 10 million, 0.93% of training.
EDM2's ramp is a fixed number of images, so a run of 2 million images would never pass 20% of the full rate. This launcher keeps
the preset's fraction and changes no file of EDM2. Run it from the EDM2 folder, with the run's length in M7_NIMG and the usual
train_edm2.py arguments.
"""
import os
import sys

sys.path.insert(0, os.getcwd())

import training.training_loop as loop  # noqa: E402
import train_edm2  # noqa: E402

PRESET_IMAGES = 1024 << 20
PRESET_RAMP_MIMG = 10
RAMP_MIMG = PRESET_RAMP_MIMG * int(os.environ["M7_NIMG"]) / PRESET_IMAGES
original_schedule = loop.learning_rate_schedule


def scaled_schedule(cur_nimg, batch_size, ref_lr=100e-4, ref_batches=70e3, rampup_Mimg=10):
    return original_schedule(cur_nimg, batch_size, ref_lr=ref_lr, ref_batches=ref_batches, rampup_Mimg=RAMP_MIMG)


loop.learning_rate_schedule = scaled_schedule

# Step M10: EDM2 saves a snapshot only when the image count is a multiple of a 1,024-image interval, so quarters of 2,000,896 images
# (1,954 x 1,024) cannot be asked for directly. With M10_SNAPSHOT_AT set to a comma list of image counts, the interval becomes a 1,024
# whose "count % interval" is 0 exactly at those counts, so snapshots land there and nowhere else. Training itself is unchanged.
SNAPSHOT_AT = {int(x) for x in os.environ.get("M10_SNAPSHOT_AT", "").split(",") if x}


class SnapshotAt(int):
    def __rmod__(self, count):
        return 0 if count in SNAPSHOT_AT else 1


if SNAPSHOT_AT:
    original_loop = loop.training_loop

    def loop_with_snapshots(**kwargs):
        kwargs["snapshot_nimg"] = SnapshotAt(1024)
        return original_loop(**kwargs)

    loop.training_loop = loop_with_snapshots

if __name__ == "__main__":
    print(f"learning-rate ramp: {RAMP_MIMG * 1e6:.0f} images; snapshots at {sorted(SNAPSHOT_AT) or 'the interval'}", flush=True)
    train_edm2.cmdline()
