# Execute walk: the positional "and" vector

**Plan:** [today-the-positional-and-vector.md](../../../plans/10-what-poe-loses-when-it-composes/today-the-positional-and-vector.md)
**Mode:** `--request`, depth 0 (every piece needed)
**Branch:** work stays on `exec/does-the-hybrid-fit-both-captions`, the branch the worktree was on; another
session holds `context/world/compose-rate.md` and a scope-06 plan file modified, so nothing is committed
or pushed from this walk and the cluster is driven without a git sync.

## The stop point

**The bar:** `PoE(dog, cat, x_pos)` draws one whole cat and one whole dog on at least **4 of the 7
judgeable seeds**, called from full-resolution tiles, judged by the user from the rebuilt sheet. Set by
the request, matching the mark every earlier third expert was read against. Not changed by this walk.

**The time box:** one sitting, about 1.5 hours. Sitting 1 opened 2026-11-04 ~10:20 local.

## Position

Piece 1 of 5 ticked. Piece 2 running.

## Ticks, with their evidence

| Task | Evidence |
|---|---|
| 1.1 `x_pos` and its columns in `and_vector_sheet.py` | `x_pos` and `poe_x_pos` in `COLUMNS`, appended to `PHRASES` and `SWEEP` through the new `BY_KEY` lookup, plus a `positional` set; `py_compile` exit 0 |
| 1.2 `x_pos` and the build check in `and_vector_norms.py` | `word_pairs` and `positional_x` added, the bar in source as `SCAFFOLD_MAX_DIFF = 0.0`, four series in the figure |
| 1.3 the norm table | largest absolute difference from the joint prompt at the start token, the end token and all 72 pads: **0**, allowed 0. Only positions 1 and 3 differ at all. `artifacts/results/the-positional-and-vector/dog-and-cat-norms.json` and `.png` |

## What the table taught, applied forward

**Position 1 of `x_pos` is exactly zero in both encoders.** A causal mask means position 1 of
"dog and cat" and position 1 of "dog" both read `<start> dog` and produce the identical state, so the
subtraction cancels completely. A position-wise subtraction can only bite from the second content word on.

**So `x_pos` is the joint prompt with its first animal blanked**, everything else the joint prompt's own.
The nearest condition already measured is the joint prompt itself as a third expert, which drew 0 clear
pairs of 7 (idea map run 4). The prediction before the render is a hybrid on most seeds.

## Decisions

| Decision | Chosen | The options there were | Who chose | Why |
|---|---|---|---|---|
| which seeds | 1 to 8, judged on 1 to 7 | 1 to 8, or 9 to 16 | user, `defaults` | every existing sheet uses 1 to 8, so the counts are comparable; the reference image is the 32-seed joint contact sheet and names no 8-seed set |
| which sheets get the column | 1, 3 and 4 | those three, or all eight files | user, `defaults` | only those three carry a third-expert column over seeds |
| the end token and the pooled vector | left as the joint prompt's | left, or subtracted too | user, `defaults` | neither carries a word to align |
| who calls the tiles | Claude at full resolution, user vetoes | Claude, or the user | user, `defaults` | calling a downscaled sheet overcalled clear pairs here once |
| time box | one sitting, ~1.5 h | recommended, or their own | user, `defaults` | |
| where the scripts run | cluster scratch `_scripts/`, no git sync | sync through git, or copy to scratch | Claude | another session holds tracked files modified, so `cluster.sh run` refuses; run 1 set this precedent |
| which node | mscluster108 device 0 | 106 (both cards busy), 109 (drain), 111 (Blackwell, reset history), 108 | Claude | 108's two Quadro RTX 8000 read 1 MiB and 17 MiB used, 0% utilisation |

## What is out of this session

| What | Id | Serves | State |
|---|---|---|---|
| render of `x_pos` and `poe_x_pos`, seeds 1 to 8 | mscluster108 pid 464573, `SET=positional` | task 2.1 | running; log `/datasets/mmolefe/poe_repair_min/outputs/concept_pair_sheet/and_vector/logs/positional.log` |

## Repairs made to the plan

None. The plan was written by this walk.

## Closing pass still owed

- [ ] file the figure and its card under `artifacts/results/the-positional-and-vector/`
- [ ] the `x_pos` row in `supervisor-meeting/README.md` section 1
- [ ] the result back into the idea map's claim 2 and its runs table
- [ ] `/verify-plan` and `/xref-pillar` on the new plan file
- [ ] the retrospective, and its `/drip-retro` line
