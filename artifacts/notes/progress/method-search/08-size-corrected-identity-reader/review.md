# Review: does correcting for each animal's size let the per-place reader see borrowing? (step M12, second reader)

This file holds what the runs in [the second reader's design](design.md) produced. GPU outputs are on the cluster under
`/datasets/mmolefe/poe_repair_min/loop-6/m12/size-corrected/`; the blind calls are on the laptop under
`outputs/loop-6/m12/size-corrected/blind/`, which git ignores.

## Runs

| Run | Job | Commits | Output | Outcome |
|---|---|---|---|---|
| Smoke: one fresh trajectory, puma and condor at seed 6009 | 95006 | `ef5e7c70-15d8bdd` | `size-corrected-smoke/` | passed in 2 minutes 33 seconds on mscluster46; the count checker read 2 animals on every step's clean estimate |
| Fresh CO3 read, 12 tuning pairs at seeds 6009 to 6016, states saved, then the per-place shares and per-step counts | 95007 | `ef5e7c70-15d8bdd` | `co3/`, `perplace/` | done in 1 hour 31 minutes on mscluster46: 96 trajectories, 96 finished pictures |
| Expected shares: each subject alone, plain SDXL, seeds 8001 to 8004 | 95008, then 95077, then 95081 | `ef5e7c70`, `84f3d05a`, `16cabcf2` | `expected/` | 95008 failed on an fp16 latent fed to the fp32 decoder, fixed by decoding with a separate fp32 copy; 95077 failed in 1 second on mscluster48, whose GPU is faulted and is now excluded; 95081 done in 17 minutes on mscluster46, every subject boxed on all 4 seeds |
| Blind calls on the 96 fresh finished pictures, two reviewers | laptop | none | `blind/calls-first.csv`, `-second.csv`, `-third.csv`, `calls-final.csv` | done: the two reviewers agreed on two clear or not for 84 of 96; a third blind call settled the other 12 (3 two clear, 9 not); 41 of 96 two clear in all. The pictures carry neutral ids, and the key (`blind/key.json`) is kept apart from them |

## The size-corrected reader

**It does not see the failure, under the bar as written, so this reader stops.** On the 96 fresh trajectories, against the settled blind
calls (41 two clear, 55 not), the best balanced accuracy is 0.635 at step 25, under the 0.65 floor, and the best area under the curve is
0.691, under the first reader's 0.742. The numbers are in `size-score.json` in the laptop's `outputs/loop-6/m12/size-corrected/`.

| Sampler step | 5 | 10 | 15 | 20 | 25 | 30 |
|---|---:|---:|---:|---:|---:|---:|
| Size-corrected, with the extra-animal rule: balanced accuracy, floor 0.65, bar 0.75 | 0.517 | 0.574 | 0.632 | 0.587 | 0.635 | 0.602 |
| The same: area under the curve | 0.533 | 0.638 | 0.684 | 0.671 | 0.691 | 0.651 |
| Size-corrected, no extra-animal rule: balanced accuracy | 0.517 | 0.577 | 0.623 | 0.559 | 0.599 | 0.575 |
| The first reader, uncorrected, on these fresh trajectories: balanced accuracy | 0.517 | 0.602 | 0.632 | 0.590 | 0.611 | 0.599 |

**The first reader does not repeat on fresh seeds.** Uncorrected, on these 96 trajectories it reaches at best 0.632, against 0.688 on the
tuning trajectories it was first read on. So part of that 0.688 was those seeds.

**The expected shares could not see size.** Plain SDXL draws any one subject to fill the frame, so the box areas came out between 0.39 and
0.94 of the picture for every animal, and every animal pair's expected first share fell between 0.44 and 0.60: hippopotamus and swallow
0.566, where the design needed something near 0.9. Only lighthouse and sailboat moved, to 0.262. The correction therefore changed the
pairs' scores together and changed little between pictures. The extra-animal rule lifted the best balanced accuracy by about 0.01 to 0.04
(it fired on 40 steps in all: 18 on hippopotamus and swallow, 11 on duck and goose, 8 on boar and pheasant).

| Pair | Expected first share | Two clear of 8 | Mean corrected score, step 20 | Steps with more than two animals, of 48 |
|---|---:|---:|---:|---:|
| eagle × fox | 0.517 | 7 | 0.834 | 2 |
| bobcat × quail | 0.527 | 6 | 0.339 | 1 |
| lighthouse × sailboat | 0.262 | 6 | 0.553 | not counted |
| seal × gull | 0.549 | 4 | 0.682 | 0 |
| crow × raven | 0.500 | 3 | 0.850 | 0 |
| puma × condor | 0.508 | 3 | 0.315 | 0 |
| zebra × flamingo | 0.521 | 3 | 0.638 | 0 |
| duck × goose | 0.453 | 2 | 0.296 | 11 |
| boar × pheasant | 0.596 | 0 | 0.596 | 8 |
| cheetah × cougar | 0.441 | 0 | 0.293 | 0 |
| hippopotamus × swallow | 0.566 | 0 | 0.113 | 18 |

Giraffe and tortoise is counted in every number above and left off the table, as on every sheet.

**What follows.** By the design, this reader stops. Two per-place readers have now failed on CO3's states, while the judge reads identity on
the decoded clean estimates from step 15 ([the first reader's review](../07-a-checker-that-reads-identity/review.md)). What M12 tries next is the
owner's call.

## Closed: the checker route for Paper 2

The owner closed the checker route for Paper 2 on 2026-11-25, answering board question 111 with A. Three findings close it.

- **No reader of this kind sees the picture beyond its pair.** A lookup that knows only which animal pair a picture belongs to, and nothing
  about the picture, scores 0.757 balanced accuracy on the fresh seeds ([the reviewer's pass of 2026-11-25](../../reviewer/2026-11-24-review.md)).
  The first per-place reader read 0.688 on the tuning seeds and 0.632 on the fresh ones, and the size-corrected reader 0.635, both below that
  lookup; scored only between pictures of the same pair, the per-place reader is at chance (0.39 to 0.51).
- **The kept checker's push points against the term composition drops.** Its median cosine with r_t = ε(joint) − (ε(A) + ε(B) − ε(empty))
  is −0.105 at step 0 on CO3 and negative at every step from 0 to 15 on plain PoE ([M13's review](../09-gradient-direction-check/review.md)).
- **The trained checkers do not see CO3's failure.** The kept checker and M5's two retrained ones reach at best 0.602, 0.617 and 0.648, under
  the 0.65 floor ([M5's review](../01b-balance-the-checker-data/review.md)).

The 192 blind-labelled pictures (96 tuning, 96 fresh) stay where they are for later use.
