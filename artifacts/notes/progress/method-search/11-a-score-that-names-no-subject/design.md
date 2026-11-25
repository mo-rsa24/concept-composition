# Can a score that names no subject rank CO3's pictures within a pair? (step M15)

## What this asks

M14, read as the owner intended, found the joint prompt the wrong target where it fails, at the first step
([M14's review](../10-is-the-joint-the-wrong-target/review.md)). Both papers then need a target measured from the picture, one that names no
subject. Loop 15's second experiment ([limits and what to test](../../directions/04-limits-and-what-to-test.md#three-experiments-ranked))
asks whether SDXL already holds such a measure early in sampling: how much the two nouns' attention overlaps, and how many instances each
noun has. On cat and dog the overlap already separates pictures that came out as two animals apart from those that did not
([the attention figure](../../../../../2-classifier-guidance/artifacts/results/does-a-small-discriminator-steer-cat-x-dog/03-does-an-early-push-separate-them/attention.png):
the joint prompt's median overlap 0.225 to 0.394 against the product's 0.506 to 0.814). The owner reopened this step on 2026-11-25.

## Words this uses

**Cross-attention map.** For one noun in the joint prompt ("puma" in "a puma and a condor"), how much each of the 32 by 32 latent cells
attends to that word, averaged over heads and over SDXL's 32 by 32 attention layers, at one sampling step.

**Overlap.** The intersection over union of the two nouns' top-10% cells, the measure the cat-and-dog figure plots. 0 when the two nouns look
at different places, 1 when they look at the same ones.

**Instance count.** How many separate things each noun covers, read from SDXL's self-attention in the way Make It Count and ISAC do. The
self-attention of the same 32 by 32 layers, averaged over heads and layers, gives each cell a row of 1,024 weights; k-means with k = 6 (seed
0) groups the cells by those rows; a group belongs to a noun when more than half its cells are in that noun's top-10% cells; touching groups
of the same noun merge (4-connectivity); the count is what remains.

**The score.** S = −overlap − |count of the first noun − 1| − |count of the second noun − 1|. Higher is more like one of each, apart. Nothing
in it names a subject or is fitted to data: k, the top-10% cut and the layers are fixed here before any run.

**Within-pair area under the curve.** The reviewer's rule ([the reviewer's pass](../../reviewer/2026-11-24-review.md)): take every two pictures
of the same animal pair where one is called two clear and the other is not, and count how often the score ranks the two-clear one higher;
ties count half. 0.5 is chance, and a measure that knows only the pair scores exactly 0.5, which is why it is used here.

## The run

**Group.** Runs that try ideas. The result says whether a subject-free target exists to aim at, and makes no claim in either paper.

**Data.** CO3's saved states on the 12 tuning pairs at seeds 6001 to 6016, 192 trajectories, at steps 5, 10 and 15: the tuning half under
`loop-6/m5/co3-read-states-shard0/` and `shard1/`, the fresh half under `loop-6/m12/size-corrected/co3/`. Each state is passed once through
SDXL under the joint prompt, with attention processors that keep the maps. The labels are the blind calls on CO3's finished pictures (M1's on
seeds 6001 to 6008, the settled calls on 6009 to 6016): 73 two clear of 192. Nine pairs carry both labels, giving 415 within-pair couples.

**Pairs held out.** Nothing in the score is fitted, so every pair is scored by a rule that never saw it; the within-pair rule then compares
only pictures of the same pair.

**Shuffled control.** At each step the labels are shuffled within each pair 2,000 times (seed 0) and the within-pair area recomputed, giving
the score's chance band on these exact couples.

## The bar, written on 2026-11-25 before the run

| Result | What it takes | What follows |
|---|---|---|
| A subject-free score ranks the pictures | a within-pair area of at least 0.70 at some step 5, 10 or 15, and above the shuffled control's 99th percentile at that step | it is the target M14 points to, and the scale for the co-occurrence and count dials (Loop 15) |
| It does not | at or below the shuffled control's 95th percentile at every step | the where-map result stays a digits-only result (Loop 15) |
| Inconclusive | anything between | reported, and the owner decides |

Reported beside the bar, deciding nothing: the overlap alone and the instance counts alone, each scored the same way; the judge on the
decoded clean estimate, which reached a within-pair area of 0.567, 0.610 and 0.744 at steps 5, 10 and 15 (the reviewer's pass); and the
overlap's median on two-clear pictures against the rest, per step, as the cat-and-dog figure shows it.

**What would surprise.** The overlap ranking pictures inside a pair better than the judge does at step 5. That would mean the attention maps
settle which body is which before the picture shows it.

## Cost and what it buys

**Cost:** about 0.5 GPU-hours: 576 states, one SDXL pass each with attention kept, and a k-means of 1,024 rows per state. No training, no
API tokens. **Buys:** whether SDXL holds a measure of "two separate subjects" that ranks pictures inside a pair at steps 5 to 15, which no
reader has done so far.

## What this cannot say

The top-10% cut and k = 6 are one choice each; a score that fails here could pass with other values, and trying them after seeing this result
would be a new test. Attention maps at 32 by 32 are coarse, so two small animals close together can share cells. Nine of the 12 pairs carry
both labels; boar and pheasant, cheetah and cougar, and hippopotamus and swallow never draw two clear and teach nothing within a pair.
