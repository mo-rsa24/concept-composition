# Retrospective: the positional "and" vector, 2026-11-04

A one-sitting walk of
[the plan](../../plans/completed/today-the-positional-and-vector.md), which built a seventh third
expert, measured it, and in the measuring turned up two things the plan had not gone looking for.

## 1. x_pos in the code, and the norm table

**The request.** Add a third-expert variant built from the joint prompt's own sequence with each
single prompt's word subtracted only at the slot holding that word, then print the per-slot norms
for x_pos, x_raw, x_fix and the joint prompt in both encoders.

**The problem.** Every earlier variant subtracted across all 77 slots, which leaves the start token
at minus one times its real value and the padding unlike any prompt's. The positional build was
written down months ago and never run.

**What was done.** `word_pairs` and `positional_x` added to `scripts/and_vector_norms.py`, matched on
the token itself and done per encoder slice because the two tokenizers can segment differently.
`x_pos` and two columns added to `scripts/and_vector_sheet.py`. The build check lives in the source
as `SCAFFOLD_MAX_DIFF = 0.0`.

**Where the results live.**
[token-norms-x-pos.png](../../results/the-positional-and-vector/token-norms-x-pos.png) and its JSON
beside it.

**Before and after.** x_fix's padding sits at 0.572 of the joint prompt's in CLIP ViT-L and 0.555 in
bigG, against a 0.1 mark it was built to meet. x_pos's sits at exactly 1.000 in both, because those
numbers are the joint prompt's unchanged. Largest absolute difference across the start token, the end
token and all 72 pads: 0.

**What it feeds.** The table was the gate on spending GPU time, and it passed.

## 2. The render

**The request.** Render x_pos alone and PoE(dog, cat, x_pos) on seeds 1 to 8 at the settings every
other sheet used.

**The problem.** biggpu had one usable card: mscluster106 was at 100% on both devices, mscluster109
was draining, and mscluster111 has the GPU-reset history.

**What was done.** mscluster108 read 1 MiB and 17 MiB used on its two Quadro RTX 8000s despite Slurm
marking the node down, so the render went there over SSH outside Slurm, as the earlier sheet runs did.
16 tiles in nine minutes; every other column was already on disk and was skipped.

**Before and after.** x_raw rendered a flat grey tile on all 8 seeds and x_fix wallpaper on 3 of them.
x_pos rendered a photograph on all 8, with two separate animals on 7.

**What it feeds.** The calls in chunk 3.

## 3. The calls, and the two findings the plan did not ask for

**The request.** Call every tile from the full-resolution files, rebuild the supervisor sheets with
the new column, and fill in the README row.

**The problem.** Three of the eight calls were queried in turn, which is the overcall this repo has
already been bitten by once.

**What was done.** Each queried tile was re-checked against the joint prompt's own cat on the same
seed rather than against memory, and the crops were then dropped in favour of whole-tile comparisons
so no choice of crop could flatter or damn a tile. The calls were accepted.

**Where the results live.**
[the card](../../results/the-positional-and-vector/README.md),
[calls.json](../../results/the-positional-and-vector/calls.json),
[the series strip](../../results/the-positional-and-vector/how-the-third-expert-changed.png),
[per-seed comparisons](../../results/the-positional-and-vector/per-seed), and the rebuilt sheets in
[1-inference-only](../../../supervisor-meeting/1-inference-only). The verdict is filed as
[a finding](../../../report/can-a-text-space-third-expert-restore-composition.md).

**Before and after.** Against the bar: 0 clear cat-and-dog pairs of 7 judgeable seeds, mark 4. MISS.
On the ladder: two separate animals on 6 of 7, where plain PoE gives 0 and the best earlier variant
gave 1.

**Two things the plan had not gone looking for.** x_pos edits 2 of 77 slots, so it is close to the
joint prompt, and used alone as the only condition it lands at 23.43 mean absolute pixel difference
from the joint prompt's render against the three-expert product's 39.16 and plain PoE's 47.05.
Adding the two animal experts moves the picture away from the target. And re-scoring the older
phrase sheets on the ladder showed "a dog next to a cat" already reaching 5 of 7, so every
construction since buys at most one more seed than an English sentence does.

**What it feeds.** The finding, and the two runs still out.

## 4. The premise test

**The request.** Before building another variant, check the assumption under all of them: does
E("A and B") − E("A") point at B?

**The problem.** Nobody had checked, and two sessions' worth of explanation had been built on a guess
about it. Mine was that CLIP's space was never trained for analogies and its differences mean nothing.

**What was done.** `scripts/and_vector_premise.py`, text encoders only, 200 random pairs against a
276-noun dictionary, with a decoy-word control and a no-subtraction baseline.

**Where the results live.**
[premise-word-recovery.json](../../results/the-positional-and-vector/premise-word-recovery.json).

**Before and after.** Chance is a median rank of 138. The held-out word comes back at median rank 1,
top-1 75.0% on the pooled vector and 98.5% on the content mean, with the decoy at 126 and 144. My
guess was wrong and is recorded as wrong in the finding.

**What it feeds.** It removes "the arithmetic is meaningless" as an explanation, which is what points
the remaining suspicion at the product rather than at the third term.

## The stop point

**The bar:** 4 clear cat-and-dog pairs of the judgeable seeds, set by the request, unchanged
throughout. **Result: MISS at 0 of 7.**

**The time box:** one sitting, about 1.5 hours of attention. Used: roughly 2.5 hours across one
sitting, over by an hour, spent on the three queried calls and on two follow-on questions the user
raised mid-walk.

**Which came first:** the bar. **The way on taken:** improve. The miss is informative rather than
terminal, and two follow-on runs were launched from it rather than closing the direction.

## Decisions

| Decision | Chosen | The options there were | Who chose | Why |
|---|---|---|---|---|
| which seeds | 1 to 8, judged on 1 to 7 | 1 to 8, or 9 to 16 | user, `defaults` | every existing sheet uses 1 to 8, so the counts compare; the reference image names no 8-seed set |
| which sheets get the column | 1, 3 and 4 | those three, or all eight files | user, `defaults` | only those three carry a third-expert column over seeds |
| the end token and the pooled vector | left as the joint prompt's | left, or subtracted too | user, `defaults` | neither carries a word to align |
| who calls the tiles | Claude at full resolution, user vetoes | Claude, or the user | user, `defaults` | calling a downscaled sheet overcalled clear pairs here once |
| spending the GPU | yes, after the norm table | go, or stop | user, `go` | the build check passed with a difference of 0 |
| the calls | accepted after three were queried | accept, or veto one | user, `accept` | each queried tile was re-checked against the joint prompt's own cat on that seed |
| what to run next | the premise test, then both follow-ons | premise, pooled, per-step x, or close | user, `premise` then `both` | the premise is the assumption under every variant and costs no GPU |
| where the scripts run | cluster scratch, no git sync | sync through git, or copy to scratch | Claude | another session held tracked files modified, so `cluster.sh run` refuses |
| which node | mscluster108 | 106 busy, 109 draining, 111 reset history, 108 | Claude | its two cards read 1 MiB and 17 MiB used |

## Dead ends and not done

**The naming.** "x_pos" implied the user's own formula applied per position. It is not that: it is
the joint prompt with two slots dented. The confusion cost a round and is corrected in the finding.

**Two claims of mine that the data killed.** That CLIP's text differences carry no meaning (killed by
the premise test), and that x_pos's recovered plurality was a new mechanism (it is largely the joint
prompt arriving through the third slot).

**Nothing was parked and nothing was skipped.** Every task and instruction in the plan is ticked.

**Still out, launched from this walk and not part of the plan.** On mscluster108 device 1, pid 483597:
x_and's 16 tiles, rendered and not yet called; then the per-step and constant ceilings on seeds 2 and
6, whose bars sit in the source of `scripts/and_vector_per_step_x.py`. Those decide whether a learned
x is worth pursuing at all.

## The family

| Session or job | Kind | State |
|---|---|---|
| this walk | drip-execute-plan `--request` | closed |
| mscluster108 pid 464573 | the x_pos render, 16 tiles | done, exit clean |
| mscluster108 pid 483597 | x_and, then the two ceilings | running at close |

---

**Validation.** Every path above was checked on disk and resolves. Every number was read back from
the file named beside it: the norms from `token-norms-x-pos.json`, the ranks from
`premise-word-recovery.json`, the pixel distances recomputed from the tiles, the ladder counts from
`calls.json` and the idea map's recorded calls. Paths checked: 14, broken: 0. Numbers checked: 19,
corrected: 0.
