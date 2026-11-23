# Review: does giving each subject its own region fix identity? (step M9)

This file holds what the runs in [the M9 design](design.md) produced. With bootstrapping on, rung 1 reaches 8 clear pairs of 8 on right boxes, and its written verdict is "regions without names". All 8 of those pictures are damaged. Rung 2 stopped at its check on the masks, with 11 of 24 reads failing against a limit of 6, and rendered no picture.

## Runs

Outputs are on the cluster under `/datasets/mmolefe/poe_repair_min/loop-6/m9/`, and logs are in `/datasets/mmolefe/poe_repair_min/loop-6/logs/`.

| Rung | Slurm job | Commits | Output | State |
|---|---|---|---|---|
| Rung 1 smoke, seed 9 | 93719 | `275c54f4-15d8bdd` | none | stopped by the GPU guard, because another process held the card on mscluster107 |
| Rung 1 smoke, seed 9 | 93724 | `275c54f4-15d8bdd` | `smoke/` | passed |
| Rung 1, all 8 seeds | 93794 | `275c54f4-15d8bdd` | `full/` | finished, 23 minutes 50 seconds on one RTX 3090 |
| Bootstrapping smoke, seed 9 | 93852 | `321f458b-15d8bdd` | `smoke/` | finished, 1 hour 59 minutes for 4 renders during a cluster stall |
| Bootstrapping, all 8 seeds | 93906 | `100772b7-15d8bdd` | `full/` | finished, 34 minutes 16 seconds |
| Rung 2 zero-binding check | 93935 | `42022b80-15d8bdd` | `rung2/controls/` | passed, and showed the first box reader returning the whole grid for both subjects |
| Rung 2 check on the masks, Otsu reader | 93936 | `d5aeb4b6-15d8bdd` | `rung2/mask-read/` | failed its limit, 11 of 24 reads |

## Rung 1 controls on the smoke seed

**Does the joint render match the picture its boxes came from?** ✅ Yes. The mean absolute difference on seed 9 is 0.0 of 255, against a limit of 3.

**Does the regional composer reduce to the plain product?** ✅ Yes. Masks of 0.5 at guidance 7.5 and the plain product at guidance 3.75 differ
by 1.53 of 255 in mean absolute pixel value, against a limit of 3.

## Rung 1 count without bootstrapping

All 48 pictures were called blind at full size by two reviewers, who agreed on 47. A third blind call settled the other one, a joint render on
seed 19, as two bodies with borrowed parts. The calls and key are in `rung-1-calls-first.csv`, `rung-1-calls-second.csv`, `rung-1-calls-third.csv`
and `rung-1-key.json`, and the counts are in `rung-1-result.json`.

<a href="rung-1-six-rows-eight-seeds.png"><img src="rung-1-six-rows-eight-seeds.png" width="420" alt="Eight seeds by six conditions: the joint prompt draws a cat beside a dog, and every region row draws one animal or a fused cat-dog"></a>

| Row | Clear pairs of 8 | Pictures with two separate animals, of 8 |
|---|---|---|
| Joint prompt | 7 | 8 |
| Plain product | 0 | 0 |
| Right boxes | 0 | 0 |
| Random boxes | 0 | 0 |
| Swapped prompts | 0 | 0 |
| Same prompt twice | 0 | 0 |

**What this count can say.** Fusing the two predictions by region from step 0 did not separate the animals on any of the 8 seeds, even with
the layout taken from a render that succeeded. Right boxes drew 6 fused cat-dogs and 2 single animals. The bar calls 0 of 8 a null, and the
oracle-layout plan holds a null until bootstrapping has been run on the same layouts.

## Rung 1 count with bootstrapping

For the first 15 of 50 steps each region was denoised over a constant colour, and the latents were fused. The 32 region pictures were
shuffled with the 8 joint renders and called blind at full size. The two reviewers agreed on `two_clear` for 38 of 40, and a third blind call
settled the 3 pictures they split on for `two_clear` or for the count of animals. The calls and key are in the `rung-1-bootstrapped-*` files, and
the counts are in `rung-1-bootstrapped-result.json`.

<a href="rung-1-bootstrapped-five-rows-eight-seeds.png"><img src="rung-1-bootstrapped-five-rows-eight-seeds.png" width="420" alt="Eight seeds by five conditions: with bootstrapping the region rows draw two separate animals as cut-outs on flat grey, and the same-prompt row draws two cats"></a>

| Row | Clear pairs of 8 | Two separate animals, of 8 | Marked undamaged by both reviewers, of 8 |
|---|---|---|---|
| Joint prompt | 6 | 8 | 5 |
| Right boxes | 8 | 8 | 0 |
| Random boxes | 4 | 4 | 0 |
| Swapped prompts | 7 | 8 | 1 |
| Same prompt twice | 0 | 8 | 1 |

**Verdict under the written bar: regions without names.** Right boxes reach 8, which clears the bar of 4, and random boxes reach 4, which is
more than 1 below. Swapped prompts reach 7 and the same prompt reaches 8 pictures with two animals, so both are within 1 of right boxes.
Rung 2 runs under this verdict.

**What the pictures add to that label.** The same-prompt row drew two cats and no dog on all 8 seeds, and the swapped row drew a cat and a dog.
So the prompt a region carries does decide its species. What the two controls show is that any two bootstrapped regions make two objects,
wherever the boxes sit relative to the joint render's layout.

**Why this is not yet a usable method.** Bootstrapping is what separated the animals, since the same fusion without it reached 0 of 8. It also
leaves cut-outs on flat grey, and neither reviewer passed any right-box picture as undamaged, against 5 of 8 for the joint prompt. The joint
renders were 7 clear pairs in the first count and 6 in this one, so a difference of 1 between rows is within how much the calls move.

## Rung 2: the check on the masks

**The zero-binding control passes.** With the regions computed and discarded, the final latent matches plain CO3 with a largest absolute
difference of 0.0, against a limit of 1e-5, on an RTX 3090 at fp16. That process's plain CO3 picture differs from M1's by 2.38 of 255 in mean
absolute pixel value, inside the project's limit of 3. The numbers are in `rung-2-zero-binding-check.json`.

**The masks fail their limit.** A read fails if either box covers at least 80% of the grid, or if the two boxes overlap with an intersection over
union of at least 0.8. On the 12 tuning pairs at seeds 6001 and 6002, 11 of 24 reads fail, against a limit of 6: 7 on overlap and 4 on box size.
The median overlap is 0.53. The written rule stops rung 2 here, so no rung 2 picture was rendered, and the comparison with CO3 alone on faithful
pairs of 96 has not been made. The reads are in `rung-2-mask-reads.json`.

<a href="rung-2-step-10-word-attention-boxes.png"><img src="rung-2-step-10-word-attention-boxes.png" width="520" alt="Step-10 word attention for 22 reads: most show two separate bodies, and in the failing reads the second word lights up both bodies or the background, so its box lands on the wrong body or the whole frame"></a>

The figure shows each word's attention at step 10, red for the first subject and blue for the second, dark where both are high, with the box read
from each word. Giraffe and tortoise is counted and left off the figure.

**What the attention maps show.** Most maps already hold two separate bodies at step 10. The reader fails for two visible reasons. In reads such as
bobcat and quail at seed 6001, the second word lights up both bodies, and its largest part is the first subject's body. In reads such as puma and
condor, the second word's map keeps a background floor, so its box spreads across the frame. This says the box reader could not supply regions. It
says nothing yet about whether regions would fix identity on CO3.

## GPU time

Rung 1 used 3.23 GPU-hours against a budget of about 1: 0.27 for the smoke test, 0.40 for the eight seeds, 1.99 for a bootstrapping smoke test
that ran through a cluster stall, and 0.57 for the eight bootstrapped seeds. Rung 2's two checks used 0.30 GPU-hours, so M9 stands at 3.53 of the
6.3 added for it. Rendering rung 2 would add about 4.8.
