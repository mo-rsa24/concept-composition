# Review: does each paper's method beat its simple baseline? (step M1)

This file holds what the runs in [the M1 design](design.md) produced. Both verdicts are **null**. Paper 1's method did not beat
sample-and-paste by more than the noise, and adding the kept push to CO3 did not beat CO3 alone.

## Runs

Outputs are on the cluster under `/datasets/mmolefe/poe_repair_min/loop-6/m1/`, and logs are in
`/datasets/mmolefe/poe_repair_min/loop-6/logs/`. A commit pair is the repository commit, then the replications hub commit.

| Paper | Rung | Slurm job | Commits | Output | State |
|---|---|---|---|---|---|
| 1 | Smoke, overfit, 1% and zero-push check | 93461 | `f0410a14-f2bdb6d` | `paper-1/ladder/` | passed |
| 1 | 1% of the data, 1,000 steps, so the loss is logged | 93583 | `7df3e0b1-f2bdb6d` | `paper-1/ladder/checkers-1pct-1000/` | passed |
| 1 | Comparison: 35,000-sample pool, two checkers, four rows | 93598 | `7df3e0b1-f2bdb6d` | `paper-1/full/rows/` | done, W&B `vekm3sfh` |
| 2 | Concept parse and zero-push check | 93468 | `99ac88a8-f2bdb6d` | `paper-2/controls/check.json` | passed |
| 2 | Smoke and 1%: one pair, one seed, three rows | 93481 | `99ac88a8-f2bdb6d` | `paper-2/smoke/` | passed |
| 2 | Comparison, pair shard 0 of 2 | 93523 | `99ac88a8-f2bdb6d` | `paper-2/full/` | done, 144 pictures |
| 2 | Comparison, pair shard 1 of 2 | 93524 | `99ac88a8-f2bdb6d` | `paper-2/full/` | done, 144 pictures |

**Cluster access on 2026-11-23.** The laptop could not log in from 14:49 until after 16:00. A verbose trace stopped right after the laptop
offered its public key, the point where the server reads the user's home directory, so a stalled `/home-mscluster` is the likely cause
(inferred). Both Paper 2 render jobs finished with exit code 0, with all 288 pictures on disk. The 288 renders used 4.37 GPU-hours.

## Paper 1 controls

**Do both checkers memorise 16 canvases?** ✅ Yes. The pair checker and the separate checker each got 16 of 16 right, against a bar of 15.

**Does the loss fall on 1% of the pool?** ✅ Yes. Over 1,000 updates on 300 training samples, the logged loss went from 0.5793 at update
500 to 0.5035 at update 1,000 for the pair checker, and from 0.3453 to 0.3243 for the separate checker.

**Does a zero push reproduce the method without a push?** ✅ Yes. The largest absolute difference was 0.0, against a bar of 1e-5, in one
process on an RTX 3090.

**Do all four rows run?** ✅ Yes, on digits 4 and 7 at 8 seeds. The counts from that smoke run are too small to compare.

## Paper 2 controls

**Does a zero push reproduce CO3's own loop?** ✅ Yes. The largest absolute difference between final latents was 0.0, against a bar of
1e-5, in one process on a Quadro RTX 8000 at float16. Two plain CO3 runs in that process also differed by 0.0.

**Does the real push change the render?** ✅ Yes. The largest absolute difference in the final latent was 3.79. The push acted on all 16
steps from 0 to 15, and the cap fired on 2 of them.

**Does every tuning prompt give CO3 two concepts?** ✅ Yes, on all 12.

**How long does a picture take?** On the RTX 8000, one seed took 43.2 seconds for CO3 alone, 58.3 with the push and 45.2 with the random
push. That is 146.7 seconds per seed across the three rows, or about 3.9 GPU-hours for all 96 pair and seed combinations, against the
design's budget of 4.8.

## The smoke picture

<a href="smoke-puma-and-condor-seed-6001.png"><img src="smoke-puma-and-condor-seed-6001.png" width="600" alt="Three renders of a puma and a condor at seed 6001: CO3 alone and the random push draw a bird with a puma's face, and the checker push draws a clear puma beside a clear dark bird"></a>

One pair at one seed, read at full size. It shows the code runs and the three rows differ, and it is no evidence for either side of the bar.

## Questions

**Paper 1: does the method beat sample-and-paste?** ❌ Null. The method with the push draws 295 clean digit pairs of 320 and paste draws
287. The lead of 8 is under the bar of 13.54, which is 1.96 times its standard error of 6.91. The zero-push check passed.

<a href="paper-1-four-rows-first-two-seeds.png"><img src="paper-1-four-rows-first-two-seeds.png" width="800" alt="Four rows of digit canvases for five digit pairs at seeds 0 and 1: paste, the method with no push, the method with the checker push, and the method with a random push. A green frame marks a canvas the judge calls clean and a red frame one it does not"></a>

| Row | Clean digit pairs of 320 | 1 and 9 | 3 and 6 | 4 and 7 | 2 and 8 | 0 and 5 |
|---|---:|---:|---:|---:|---:|---:|
| Sample alone and paste | 287 | 57 | 58 | 55 | 58 | 59 |
| Method, no push | 259 | 58 | 53 | 49 | 47 | 52 |
| Method with the checker push | 295 | 59 | 61 | 53 | 60 | 62 |
| Method with a random push | 258 | 56 | 51 | 49 | 50 | 52 |

These matched leads were read after the run and carry no bar of their own. Each margin is 1.96 standard errors.

| Comparison | Lead, of 320 | Margin |
|---|---:|---:|
| Checker push over no push | +36 | 15.61 |
| Checker push over random push | +37 | 15.47 |
| Random push over no push | -1 | 8.75 |
| No push over paste | -28 | 16.57 |

The checker's direction does real work: it lifts the method by 36 pairs where a random push of the same size changes nothing. That lift
brings the method level with paste. Without the push, the side-choosing method is 28 pairs below paste.

**Does the judge agree with the eye?** On 64 canvases shuffled across the four rows and called by eye at full size before the key was
opened, the two agree on 58. The eye called 5 clean that the judge rejected and rejected 1 the judge called clean. The six disagreements
fall in all four rows.

**Paper 2: does CO3 with the push beat CO3 alone?** ❌ Null. CO3 with the kept push makes 7 faithful pairs of 96 and CO3 alone makes 11.
A win needed a lead of at least 5 and more than 10.08, which is 1.96 times the standard error of 5.14. The lead is -4. The random push
makes 12, one more than CO3 alone, so it does not trigger the inconclusive rule. The zero-push check passed.

<a href="paper-2-three-rows-seed-6002.png"><img src="paper-2-three-rows-seed-6002.png" width="800" alt="Three rows of renders at seed 6002 for puma and condor, zebra and flamingo, and seal and gull: CO3 alone, CO3 with the checker push, and CO3 with a random push of the same size"></a>

The sheet shows the first three tuning pairs at seed 6002, each checked at full size. The counts come from the blind calls on all 96
pictures per row, in [the calls](paper-2-blind-calls.csv), with [the scored result](paper-2-blind-result.json) and
[the method key](paper-2-blind-key.json) beside them.

| Row, of 96 pictures | Faithful pair | Two clear subjects, style ignored | Two bodies, not both clear | One subject or a fused hybrid | Not an intact photograph |
|---|---:|---:|---:|---:|---:|
| CO3 alone | 11 | 32 | 64 | 0 | 73 |
| CO3 with the checker push | 7 | 30 | 66 | 0 | 80 |
| CO3 with a random push | 12 | 35 | 61 | 0 | 72 |

These readings came after the run and carry no bar of their own.

- **Drawn style drives most of the damage count.** The prompts do not ask for a photograph, CO3 often draws an illustration, and the
  rubric counts a drawing as damaged. Reviewers' notes name a drawn style on 53, 41 and 53 of the 96 pictures in the three rows.
- **The null holds with style ignored.** On two clear subjects, the push is 2 below CO3 alone, against a margin of 11.25.
- **CO3 already separates the bodies here.** None of the 288 pictures is one subject or a fused hybrid. What remains is borrowed parts,
  the wrong species or an extra subject.
- **Few pairs produce any faithful picture.** Eight of the 12 pairs make 0 in all three rows. The other four, as CO3, push and random, are
  lighthouse and sailboat (6, 3, 6), bobcat and quail (3, 0, 3), seal and gull (2, 1, 3), and eagle and fox (0, 3, 0).
- **Reviewers differ in strictness.** The six reviewers passed 7, 0, 6, 4, 4 and 9 of their 48 pictures. Each batch mixes all three rows,
  so this adds noise to the comparison and does not favour a row.

The kept checker's two presence outputs never trained on these subjects, so this result covers the kept push as it stands.
