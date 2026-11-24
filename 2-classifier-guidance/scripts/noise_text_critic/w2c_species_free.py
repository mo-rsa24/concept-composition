#!/usr/bin/env python
"""Step W2c, group 4: counting scored with sentences that name no species, on W2's saved image vectors.

The design is noise-text-critic/02c-fair-test/design.md. Pictures, labels, draws, halves, thresholds and the bootstrap are
W2's; only the sentences change. Writes loop-9/w2c/species-free/result.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import w2_read as w2  # noqa: E402
from common import LOOP, latent_clip_text, load_clip, load_latent_clip, text_features  # noqa: E402
from disallowed_subjects import check  # noqa: E402
from w2c_fair import TEMPLATES, phrase  # noqa: E402

# Fixed in 02c-fair-test/design.md, group 4, before any score.
READ_BAR = 0.75
RIGHT = "a photo of two animals"
WRONG = ("a photo of one animal", "a photo of one animal with mixed features")
MODELS = ("noisyclip", "frozen_clip", "latent_clip")


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sentences = [RIGHT, *WRONG]
    for s in sentences:
        check(s)
    loaded = {m: load_clip("noisyclip" if m == "noisyclip" else "frozen", device) for m in MODELS[:2]}
    loaded["latent_clip"] = load_latent_clip(device)

    @torch.no_grad()
    def enc(m, texts):
        if m == "latent_clip":
            return latent_clip_text(*loaded[m], texts, device).float().cpu()
        return text_features(loaded[m][0], loaded[m][1], texts, device).float().cpu()

    text = {}
    for m in MODELS:
        text[m] = enc(m, sentences)
        text[f"{m}_ens"] = torch.stack([F.normalize(enc(m, [t.format(phrase(s)) for t in TEMPLATES]).mean(0), dim=-1)
                                        for s in sentences])
    read = LOOP / "w2/read"
    meta = np.load(read / "scores.npz")
    steps = [int(s) for s in meta["steps"]]
    _, _, tasks, _ = w2.build_items(LOOP / "w2/prep", smoke=False)
    rows = (*MODELS, *(f"{m}_ens" for m in MODELS))
    scores = {}
    for m in MODELS:
        v = torch.from_numpy(np.load(read / f"vectors-{m}.npy").astype(np.float32))   # (steps, draws, items, dim)
        for r in (m, f"{m}_ens"):
            cos = (v @ text[r].T).numpy()                                               # (steps, draws, items, 3)
            scores[r] = (cos[..., 0] - cos[..., 1:].max(-1)).mean(1)
    w2.MODELS, w2.COMPARED = rows, tuple(r for r in rows if r != "frozen_clip")
    res = w2.analyse(scores, steps, tasks["counting"], smoke=False)
    k49 = steps.index(49)
    for r in rows:
        res[r]["step_49_reading_half"] = res[r]["balanced_accuracy_reading_half"][k49]
    reads = [r for r in rows if res[r]["mean_steps_0_to_15"] >= READ_BAR]
    finished = [r for r in rows if res[r]["step_49_reading_half"] >= READ_BAR]
    out = {"sentences": {"right": RIGHT, "wrong": WRONG}, "templates": TEMPLATES, "bar": READ_BAR, "steps": steps,
           "counting": res, "rows_that_read_steps_0_to_15": reads, "rows_that_pass_step_49": finished}
    dest = LOOP / "w2c/species-free"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "result.json").write_text(json.dumps(out, indent=1))
    for r in rows:
        print(f"{r:16s} mean 0-15 {res[r]['mean_steps_0_to_15']:.3f}  step 49 {res[r]['step_49_reading_half']:.3f}", flush=True)
    print("reads:", reads, "passes step 49:", finished, flush=True)
    print("SPECIES DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
