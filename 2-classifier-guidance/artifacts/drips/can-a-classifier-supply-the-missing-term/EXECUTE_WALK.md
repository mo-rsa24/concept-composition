# Walk: can a classifier supply the term the product drops

Walking [the plan](../../../plans/15-can-a-classifier-supply-the-missing-term.md)
in `--request` mode, on branch `exec/can-a-classifier-supply-the-missing-term`.

## Position

Piece 1 of 4, the labelled set. The rule and the photograph side are done; the render side is in
flight on mscluster108 device 0.

## The stop point

**The bar.** The card states the best classifier weight's count out of eight beside plain
product's 0 of 8, the damped product's count and the joint prompt's 8 of 8. Judged by the user
from the weight-sweep sheet.

**The time box.** 2 sittings, about 3 hours of the user's attention, plus about 2 hours of
background GPU.

| Sitting | Start | End |
|---|---|---|
| 1 | 2026-11-04 | open |

## Decisions

| What | Chosen | The options there were | Who chose | Why, in their words |
|---|---|---|---|---|
| The four classifier weights | 0.25, 0.5, 1.0, 2.0 | that, or 0.5 to 4.0 if the gradient is weak | user, by `defaults` | recommended default accepted |
| Where the hybrid negatives come from | every animal-pair product render, minus the eight held-out cat × dog seeds | that, or cat × dog only | user, by `defaults` | recommended default accepted |
| Whether a held-out accuracy curve gates the render step | no gate; balanced accuracy is logged at every probe and drawn at the end | gate on it, or log it only | user | "I don't really need evaluation curves... I am looking to test ideas quickly" |
| How composition gets watched | render through the guided composer every 250 training steps, two seeds at two weights, into W&B | one read at the end, or a probe while training | user | "test the inference of guided classifier during PoE during training so that I can see a change over time" |
| How to run on the cluster | a second checkout as a git worktree at `poe_repair_min-classifier-guided` | that, or `/reconcile-machines` first | user, by `a` | the inclusive-captions branch is another session's live work and is left untouched |

## Repairs made to the plan

None yet.

## What left this session

| Kind | Id | What it serves | State |
|---|---|---|---|
| run | `presence_head_label_set.py` on mscluster108 device 0, log at `/datasets/mmolefe/poe_repair_min/outputs/presence_head/logs/label_set.log` | tasks 1.2 and 1.3, the render side of the labelled set | in flight |

## Ticked this session

Nothing yet. Task 1.1's rule is built and its seven cases ran, but the tick waits on instruction 5,
the eye check on the hybrid sheet, which needs the run above.

## The toy

Seed 9 of `a_cat__x__a_dog`, and three photographs from the pool manifest: one cat-only, one
dog-only, and image 46905, which holds one cat at box fraction 0.445 and two dogs at 0.457.

## Lines for the close

**Piece 1.** The labelling rule is the whole lever: a caption's probability is normalised over
every caption a picture could have, so two captions compete and their product peaks on one animal
that reads as both; two sigmoids share no denominator, so both can read 1 at once, but only if the
labels refuse to call a fused animal both.

## Closing pass still owed

- the figures filed under `2-classifier-guidance/artifacts/results/can-a-classifier-supply-the-missing-term/` with their card
- the review file's verdict ticked by the user
- a finding in `report/`
- `sync-plan-tree` from the scope root
- scope 06's own `## Plans` table, which is missing rows for plans 11, 12 and 14 as well as 15
