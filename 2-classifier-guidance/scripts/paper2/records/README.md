# Copies of uncommitted records that Paper 2's collector reads

`paper2_collect.py` (species judge step 0.4) runs on the cluster, where git-ignored folders do not exist. These are
byte-for-byte copies, made on 2026-11-26 from the laptop, of the records it needs that were never committed.

| File | Copied from | What it holds |
|---|---|---|
| `m14-key.json` | `outputs/loop-6/m14/blind/key.json` | the joint prompt's 192 blind ids, each with its pair and seed |
| `m14-calls-final.csv` | `outputs/loop-6/m14/blind/calls-final.csv` | each id's settled call (two_clear or two_bodies) and who settled it |
| `m14-calls-first.csv`, `m14-calls-second.csv`, `m14-calls-third.csv` | the same folder | the three reviewers' calls with their notes (the third on the 24 split pictures) |
| `old-judge-labels.json` | `3-understanding-compositionality/artifacts/results/what-sdxl-already-knows/what-the-prompts-draw/can-a-score-see-species/labels.json` | the old species judge's 29 cells: pair, seed, run, step, verdict by eye, confident |

The joint prompt's pictures are at `/datasets/mmolefe/poe_repair_min/paper2/paper2-upload-joint/<id>.png`; the old
judge's tiles at `/datasets/mmolefe/poe_repair_min/outputs/ablation_grid/<slug>/seed<NN>/ours_<run>_s<step:06d>_w50.png`.
