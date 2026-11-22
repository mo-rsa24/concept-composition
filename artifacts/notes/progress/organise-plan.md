# 🧭 Progress: organising the classifier guidance and inclusive captions artifacts

This plan makes two artifact folders easy to walk and takes the pictures no page uses out of the
repository. Step O1 on the [progress board](board.json) tracks it. Counts were measured on
2026-11-22.

<a href="organise-plan.png"><img src="organise-plan.png" width="520" alt="The 72 flat discriminator folders become six numbered folders, one per report page, plus a pool folder; raw tiles leave for /datasets and a laptop archive"></a>

## Table of contents

- [What changes](#what-changes)
- [The layout](#the-layout)
- [What happens to each file](#what-happens-to-each-file)
- [The stages](#the-stages)
- [Who does what](#who-does-what)
- [Still open](#still-open)

## What changes

| Tree | Files now | Size now | Files left in the repo | Size left in the repo |
|---|---|---|---|---|
| `2-classifier-guidance/artifacts` | 14,258 | 9.12 GB | 1,983 | 0.76 GB |
| `4-inclusive-captions/artifacts` | 673 | 0.08 GB | 666 | 0.08 GB |

Inclusive captions is already small, so its work is ordering, missing cards and 7 surplus copies.

## The layout

The 72 discriminator run folders keep their own names and move under the report page that discusses
them, so a name quoted on a page still finds its folder. The question folders at the top keep their
names, because each is named by 7 to 28 plans, scripts and pages. Their reading order goes on the
generated results index.

```text
2-classifier-guidance/artifacts/results/
  can-a-classifier-supply-the-missing-term/
  does-classifier-guidance-draw-a-dog/
  does-classifier-guidance-work-on-the-released-weights/
  does-a-pair-aware-push-compose-our-photos/
  does-a-small-discriminator-steer-cat-x-dog/
    01-the-first-quick-test/                     4 runs, 0.41 GB now
    02-can-a-checker-read-a-clean-pair/         15 runs, 0.96 GB now
    03-does-an-early-push-separate-them/        17 runs, 0.85 GB now
    04-can-a-second-push-name-each-animal/      11 runs, 0.58 GB now
    05-does-it-carry-to-other-pairs/             8 runs, 0.49 GB now
    06-how-close-it-comes-to-the-joint-prompt/  15 runs, 4.55 GB now
    pools-of-training-pictures/                  2 pools, 0.90 GB now
    comparison-sheets/                          11 sheets, 0.04 GB
  can-two-digits-compose-on-mnist/              three numbered stages, filed
  what-each-line-of-classifier-guidance-computes/
  what-each-line-of-readout-guidance-computes/
  what-each-line-of-our-guided-sampler-computes/

4-inclusive-captions/artifacts/results/         reading order on the index:
  does-relabelling-both-shapes-pictures-help-the-toy-product-compose/
  do-inclusive-labels-help-the-product-compose-on-mnist/
  does-the-share-of-two-digit-pictures-help-the-product-compose-on-mnist/
  does-captioning-both-animal-photos-under-each-name-help-poe-compose/
  how-many-photos-the-inclusive-captions-pilot-can-train-on/
  does-any-caption-say-a-cat-and-a-dog/
```

Each numbered folder gets one card with a row per run: what changed in that run, its result file,
its sheet, and where its tiles went.

## What happens to each file

| What it is | Classifier guidance | Inclusive captions | What happens |
|---|---|---|---|
| Named by a page, card, plan, paper or script | 394 files, 178 MB | 254 files, 38 MB | stays |
| Data, code or notes | 247 files, 28 MB | 75 files, 14 MB | stays |
| One example from each series of tiles | 1,247 files, 452 MB | 38 files, 1 MB | stays |
| Unnamed one-off picture (scene, walk, sheet) | 95 files, 101 MB | 299 files, 28 MB | stays |
| Remaining tiles of a series | 10,591 files, 6.98 GB | none | copied to `/datasets`, then moved to the laptop archive |
| Byte-identical surplus copy nothing names | 964 files, 1.30 GB | 7 files, 0.4 MB | moved to the laptop archive |
| Path names a guarded subject | 720 files, 82 MB | none | moved to the laptop archive, never shown or uploaded |

The cluster copy goes to `/datasets/mmolefe/poe_repair_min/artifacts-archive/<same path as in the
repo>`, so swapping the prefix finds any file. The laptop archive is
`/home/molef/PhD/_archive/concept-composition-offload-2026-11-22/`, with the same inner paths. It sits
outside the repository because the repo's own `_archive/` is tracked in git and 8 GB must never enter
git history. Nothing is deleted by this plan.

"Nothing names it" is a text search over pages, cards, plans, the paper and scripts. A script that
builds a filename at run time is invisible to that search, which is why these files move to an
archive and stay recoverable.

## The stages

Each stage ends in a check, and a failed check stops the next one.

1. **Write the file list.** One row per file: old path, action, new path, size, checksum. Nothing
   moves. The check: rows equal files on disk, and each file has exactly one action.
2. **Copy the tiles to `/datasets`.** Expected about 22 minutes, from a test copy of 14 files
   (15.8 MB) in 2.7 seconds at 5.78 MB/s. The check: every checksum on the cluster matches.
3. **Regroup the discriminator folders.** Move the 72 run folders, repoint the five `rung2` scripts
   that name them, and rewrite page links. The check: no new broken link, every script compiles,
   and `sweep.py --apply-moves` leaves no file lost.
4. **Move the surplus out.** Move the shipped tiles, surplus copies and guarded files to the laptop
   archive, recording every move in `restore-manifest.tsv`. The check: the manifest row count equals
   the files moved, and a restore of one folder reproduces its checksums.
5. **Write the cards.** One per numbered folder, plus the missing ones: 3 result questions, 16
   walks, 2 scenes, 2 decks and 2 idea groupings. Walks and ideas get a keep or archive call on the
   card.
6. **Close.** Rerun the results index, drain both projects, and confirm `sweep.py` prints
   `new 0  changed 0`.

## Who does what

| Who | Work |
|---|---|
| Session 2 | stages 1 to 4 and 6, because they need one owner and one lock |
| Drain agents, ten pages or fewer each | stage 5 cards, one numbered folder per agent |
| Session 3 | applies the old-to-new path table to `plans/`, sent through `requests.md` |
| Session 1 | adds the path table to `RENAMES.md` |

The rules for any deletion, recorded-number change, fixed-folder change or picture overwrite are in
the [decision log](decision-log.md). This plan needs none of the four.

## Still open

- **The owner's approval of this layout.** Stages 2 to 6 wait for it.
- **The 155 web images in `does-any-caption-say-a-cat-and-a-dog/images/`.** They are tracked in git
  and on two remote branches. Moving them does not remove them from history, so they stay until the
  owner decides.
- **The `lora-correction` offload (8.5 GB).** It follows the same stages after these two trees.
