# Walk map: route step 4, freeze the evaluation set, the tuning set and the one adapter

Plan: [route step 4](../../../plans/route-answer-the-iclr-reviews.md#4--choose-and-freeze-the-evaluation-set-the-tuning-set-and-the-one-checkpoint), walked with `/drip-execute-plan --request`.
Board: [the paper review board](../../../report/paper-review/board.md). Protocol: [the comparison protocol](../../../report/paper-review/protocol.md).

## Stop point

Bar: the frozen file committed with at least 8 pairs per group (or a smaller count you accept), the adapter named by the protocol's showcase rule, and no adapter result on any evaluation pair.
Time box: 60 minutes of your time; cluster renders run in the background and do not count.
Steps at the open: 5 (pieces 1 to 5, depth full).

| Sitting | Start | End |
|---|---|---|
| 1 | 2026-11-10 19:04 | |

Toy: bison and horse, seed 9.

## Position

Piece 2: launch script `scripts/review_step4_renders.sh` and pair list `scripts/review_step4_census_pairs.txt` written, uncommitted; waiting on your yes to commit, sync and launch on mscluster111 and 112.

## Ticks

| Piece | What | Evidence |
|---|---|---|
| 0 | steps 1 and 2 committed so the cluster checkout carries the protocol and the refill script | commit `084ad92e`, 47 files, no PDF, nothing under `figures/` or `build/`; the repo's only remote is GitHub |
| 1 | 62 candidates outside the 48 training pairs sorted by kind: 30 competing (3 held out), 20 object and place, 10 style, 2 showcase; tent and snowy mountain added to object and place from Figure 5 | pair_prompts*.yaml and pair_pool_v58.yaml read 2026-11-10 |
| 1 | Attend-and-Excite prompt sets fetched from the attachment the repository's README links: animals 66, animals_objects 144, objects 66; none names a training pair | `a.e_prompts.txt`, 10,146 bytes, in scratch |

## Facts found on the way

- `.gitignore` line 43 has `*.pdf` commented out, so PDFs are tracked unless a path rule excludes them; the export block added for `paper/iclr-56173/` excludes its PDFs. `environment/paper.md` says PDFs are gitignored at line 31, which is no longer true.
- The Attend-and-Excite prompt file is not in the repository; the README links it as a GitHub attachment.
- The animals set includes "a cat and a dog" (a showcase pair) and "a dog and a horse" (one of our competing candidates).

## Decisions

| Decision | Options | Who chose | Why |
|---|---|---|---|
| Commit steps 1 and 2 first | commit · launch without | you: commit only if it cannot reach Overleaf, then defaults | the repo's only remote is GitHub; the Overleaf clone is a separate repository nothing here touches |
| Style pairs | leave out · a fourth group | you, by `defaults` | they fit none of PRZE-3b's groups |
| Census scope | all 50 · 30 | you, by `defaults` | |
| Attend-and-Excite prompt sets | R3 only · also 66 animal prompts in the census | you: `a` | census stays at 51 pairs; file saved to `data/prompt-sets/attend-and-excite/a.e_prompts.json` |

## Runs, threads, handoffs

None.
