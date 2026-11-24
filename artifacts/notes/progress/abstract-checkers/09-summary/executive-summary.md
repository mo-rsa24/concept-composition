# Loop 8 in one page: can a checker trained on one set of concepts make the model draw two things it never trained on?

**On digits, yes for counting. Not yet tested against any published rival, and not yet on SDXL.** A small checker that reads only *where* each asked digit acts learned "two digits sitting apart" from digits 0 to 4. On digits 5 to 9, which it never saw, its push turns the plain product's 0 clean pairs of 320 into 43. The same checker cannot tell *which* digits are present at the noise levels it works at, and that is the part SDXL's strongest rival already gets wrong.

<a href="evidence-at-a-glance.png"><img src="evidence-at-a-glance.png" width="900" alt="Left: plain-product canvases for digit pairs 5 and 6 to 5 and 9, each a single fused glyph; below, the same seeds with the count checker's push, many now two separate digits, a few framed green as clean. Right: clean pairs of 320, plain 0, random push 1, best of 4 seeds 0, count checker push 43 with an error bar of 1.96 standard errors."></a>

## What was found

| Question | Answer | Evidence | Conclusive? |
|---|---|---|---|
| Can a checker learn "two digits apart" on digits it never saw? | **Yes** | balanced accuracy 0.791 on digits 5 to 9 against 0.782 on digits 0 to 4, pass mark 0.75 (0.5 is chance) | **yes**, by a bar written before the run |
| Can it learn *which* digits are present? | **No**, at these noise levels | 0.555 on digits it never saw, pass mark 0.75 | **yes, a miss** |
| Does its push make two separate, right digits? | **Yes** | 43 clean pairs of 320 against 0 plain; random push 1; best of 4 seeds 0; lead 5.7 standard errors | **yes**; a rerun on another machine gave 41 |
| Does each checker control only its own property? | **Mostly, for 2 of 3** | removing count drops count 38% to 10%; removing order drops order 88% to 48%; each also moves one other score about 7 points | **no**: null by the written rule |
| Does any of this work on SDXL, against the rivals? | **Unknown** | step V4 has not run | **not tested** |

The explanations, figures and run records are in the stage reviews: [the test bed](../04-environment/review.md), [learning on unseen digits](../05-does-it-learn/review.md), [composing unseen digits](../06-does-it-compose/review.md) and [removing one checker](../08-remove-one-checker/review.md).

## Against the rivals

**On digits, we beat the simple baselines.** The plain product, a random push of the same size and best-of-4 selection all stay at 0 or 1 clean pairs of 320, where the checker's push reaches 43. These are the baselines a reviewer asks for first, and the gain comes from the gradient during sampling, because picking the best of four samples by the checker's own score finds nothing.

**Against published rivals, there is no comparison yet.** Every Paper 2 target is on SDXL or CLEVR, and Loop 8 has run neither:

| Benchmark | Rival | Rival's printed score | Ours |
|---|---|---|---|
| T2I-CompBench Color | CoMat | 0.7827 | not run |
| T2I-CompBench Shape | ToMe | 0.6051 | not run |
| T2I-CompBench Texture | ToMe | 0.6894 | not run |
| T2I-CompBench Complex | Rectify Then Diffuse | 0.4661 ImageReward | not run |
| GenEval, two objects | FKC | 0.78 | not run |
| CLEVR, objects composed | Bradley's local model | 5 objects | not run |

A qualitative grid of the kind "the rival fuses, ours draws two" exists only on digits (the figure above). It does not yet exist against any rival.

## How it lines up with the other loops

<a href="where-loop-8-stands.png"><img src="where-loop-8-stands.png" width="560" alt="Loop 8's steps in order with their verdicts, V4 not run; Loops 9 and 10 point to V2 as agreeing that identity is unreadable early and that SDXL's noise level carries more signal; Loop 6 points to V4 as the risk that counting is what the rival already does."></a>

**Loops 9 and 10 agree with Loop 8.** Loop 9 found that no sentence score reads counting, prompt match or left-right order at the earliest 16 steps (best 0.56 against a bar of 0.75). Loop 10's score path reads 0.700 at SDXL's noise level and 0.513 at the toy's earliest steps. Three different readers find the same thing: at the noisiest steps, layout is readable and identity is not.

**Loop 6 marks the main risk.** On SDXL, the kept push added to CO3 gave 7 faithful pairs of 96 against CO3's 11, and CO3 already draws two separate bodies in all 96 of its tuning pictures. Its failures are wrong species, borrowed parts and extra subjects, which are identity errors. Loop 8's working checker supplies count, which CO3 already has, and its identity checker is the one that fails.

## Is the loop getting there?

| Step | State |
|---|---|
| V1 test bed | done |
| V2 a checker learns on unseen digits | done, with the count checker |
| V3 its push composes unseen digits | done, support |
| V5 remove one checker at a time | done, null by the written rule |
| V4 the best checker on SDXL's unseen objects | waiting on Loop 6's step D3 |

Four of five steps are done, using about 7.3 of the loop's 30 GPU-hours. The digit question is answered. The question that matters for the paper, whether this beats a rival on SDXL, starts with V4.

## What would change the picture

**Identity read later in sampling.** Loop 9 found NoisyCLIP reads prompt match from about step 30 of 50. An identity checker pushing in a later window, beside the count checker in the early one, is the obvious next design to test. It has no step yet.

**V4 judged against CO3, not only plain PoE.** The brief already asks V4 to report both, so a count checker that passes against plain PoE and adds nothing to CO3 will show as such.

**A like-for-like CLEVR test.** Loop 6's CLEVR Stage 2 uses models told *which* objects, not *where*, the same setting as ours. Loop 8 has asked Loop 6 to add a count-checker row and a Du-sampler row there.

## What this summary cannot say

Every result here is on 32 by 64 pixel digit canvases with one diffusion model. Pushed renders are not bit-reproducible across machines (43 against 41 on the same seeds), so per-pair counts carry more noise than the bootstrap errors show. Nothing here supports a claim about SDXL or about any published rival.
