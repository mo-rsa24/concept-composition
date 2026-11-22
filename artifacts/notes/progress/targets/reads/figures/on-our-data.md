# Our own runs of published methods: how they fail on our pairs

**Every number on this page comes from a run this project did itself.** None of them is a published
result, so none may be quoted as what a paper reports. The papers' own figures are recorded
separately. Six of the 19 rivals have a run on our pairs, and the project's own records disagree on
whether one of the six, ToMe, ran at all.

Paths in plain text start at the repository root. A path starting `replications/` sits under
`/home/molef/PhD/`.

## Words this uses

**The ladder**

The four calls the replications hub makes on a picture, by eye, for the two things the prompt named:
one thing (`single`), fused hybrid (`hybrid`), two bodies or heads (`two-bodies`) and two clear
things (`two-clear`). The [ladder's definition](../../../../../../../replications/published-methods-on-our-pairs/report/composition-papers-on-our-pairs/how-a-tile-is-called.md)
says what each looks like. Only FKC and the plain product have ladder calls. Every other row quotes
the words its finding used, and no rung has been assigned on its behalf.

**Our pairs**

Concept pairs this project rendered, such as cat × dog and the 53 held-out pairs. The 276
Attend-and-Excite prompts are a public benchmark, so their scores sit in their own table.

**BLIP-VQA**

A score from 0 to 1: the product of a model's yes-probabilities, one for each thing the prompt names.
A value written as 0.7757 ± 0.0512 is the mean and the half-width of its 95% interval.

**Instance count**

How many animals the GroundingDINO detector boxes in a picture. It counts two cats as two, so it
cannot say which two animals are present.

## 1. Rivals run on our pairs

| Rival | Pairs | How it fails on the ladder | Count as recorded | Void list | File |
|---|---|---|---|---|---|
| FKC | cat × dog, seeds 9 to 16 | Fused hybrid on all 8 seeds, called by eye on the ladder | **Our run:** hybrid 8 of 8; two bodies or two clear 0 of 8; two clear 0 of 8. Recorded verdict: MISS | No entry | [replications/published-methods-on-our-pairs/report/composition-papers-on-our-pairs/paper-01-feynman-kac-correctors.md](../../../../../../../replications/published-methods-on-our-pairs/report/composition-papers-on-our-pairs/paper-01-feynman-kac-correctors.md)<br>[replications/published-methods-on-our-pairs/artifacts/results/does-a-published-method-separate-cat-and-dog/feynman-kac-correctors/calls.csv](../../../../../../../replications/published-methods-on-our-pairs/artifacts/results/does-a-published-method-separate-cat-and-dog/feynman-kac-correctors/calls.csv) |
| SuperDiff | cat × dog, seeds 9 to 12; butterfly × meadow; dog and rain, camel and rain, seeds 1, 2, 4, 9 and 10; lion × meerkat, seed 9 | No ladder call. At its default 200 steps the finding says "every seed comes out as one blended animal" | **Our run,** cat × dog seed 9, instance count: 1 (blend) at 200 steps at both clamp settings; 2 (compose) at 50 steps clamped; 1 at 50 steps unclamped, "where the eye reads compose".<br>**Our run,** cat × dog seeds 9 to 12 at κ 0.5 with nothing added: one animal on every seed, by eye.<br>**Our run,** butterfly × meadow, by eye: blur at 200 steps, distinct butterfly at 50, at both clamp settings.<br>**Our run,** camel and rain, 5 seeds, wetness call: broken 3, half 1, cannot tell 1.<br>**Our run,** dog and rain, and the one lion × meerkat render: no call recorded | No entry on these counts. Two entries (rank 16 at step 10k and λ 0.5; rank 32 at step 20k) sit on the finding about our adapter trained on SuperDiff's residual. Those counts measure our adapter and are left out | [1-lora-correction/report/is-the-gap-the-samplers-or-the-models/does-superdiff-compose-at-its-own-defaults.md](../../../../../../1-lora-correction/report/is-the-gap-the-samplers-or-the-models/does-superdiff-compose-at-its-own-defaults.md)<br>[3-understanding-compositionality/report/a-thing-in-the-rain/does-a-camel-in-the-rain-come-out-wet.md](../../../../../../3-understanding-compositionality/report/a-thing-in-the-rain/does-a-camel-in-the-rain-come-out-wet.md)<br>[3-understanding-compositionality/report/a-thing-in-the-rain/does-a-dog-in-the-rain-come-out-wet.md](../../../../../../3-understanding-compositionality/report/a-thing-in-the-rain/does-a-dog-in-the-rain-come-out-wet.md)<br>[1-lora-correction/report/paper-review/prze-10-compute.md](../../../../../../1-lora-correction/report/paper-review/prze-10-compute.md) |
| CO3 | cat × dog, seeds 6001 to 6128; the 53 held-out pairs, seeds 9 to 16; four hand-picked pairs, seed 9 or 10 | No ladder call. On the 128 seeds the loop records "two cats or three animals (24) and three silhouette or hybrid pictures" as its misses. On cat × dog seed 42, "CO3 draws two cats" | **Our run,** cat × dog, of 128: faithful pairs 89 and not an intact photograph 9 (scorer); clean or near-clean 101 and clean 97 (eye).<br>**Our run,** 53 held-out pairs, BLIP-VQA: all 0.7757 ± 0.0512; hard, 3 pairs, 0.5937 ± 0.2742; easy but unseen, 29, 0.7716 ± 0.0640; not competing, 21, 0.8074 ± 0.0815.<br>**Our run,** four hand-picked pairs: both things drawn on all four. The finding says no rate belongs to this grid | **Yes, for the 53-pair numbers:** the entry "Every method uses the same seed for each prompt, with 50 steps at guidance 7.5". The CO3 runner defaults to CFG++ at guidance 0.8, and the table labels CO3 DDIM at 7.5. The kept-push entry sits on the 128-seed page and names the kept push only | [2-classifier-guidance/report/classifier-guided-poe/how-close-the-kept-push-comes-to-the-joint-prompt.md](../../../../../../2-classifier-guidance/report/classifier-guided-poe/how-close-the-kept-push-comes-to-the-joint-prompt.md)<br>[2-classifier-guidance/artifacts/drips/rung2-pair-aware-discriminator/checker-loop.md](../../../../../../2-classifier-guidance/artifacts/drips/rung2-pair-aware-discriminator/checker-loop.md)<br>[1-lora-correction/report/against-other-methods/how-does-every-sdxl-method-compare-on-the-held-out-pairs.md](../../../../../../1-lora-correction/report/against-other-methods/how-does-every-sdxl-method-compare-on-the-held-out-pairs.md)<br>[1-lora-correction/report/measuring-composition/what-the-main-study-grids-show.md](../../../../../../1-lora-correction/report/measuring-composition/what-the-main-study-grids-show.md)<br>[1-lora-correction/report/against-other-methods/how-do-four-methods-compare-on-four-hand-picked-pairs.md](../../../../../../1-lora-correction/report/against-other-methods/how-do-four-methods-compare-on-four-hand-picked-pairs.md) |
| ToMe (**whether it ran is disputed**, see section 4) | Figure 5 of the revision manuscript prints a ToMe column on elephant and turtle, cat and lion, tent and mountain, lighthouse and sea | No ladder call, and no eye call recorded on these pairs | **Our run:** no count on our pairs. Its only scores are on the benchmark prompts in the table below | No entry names ToMe. Its finding records the same DDIM and 7.5 label on ToMe's row, and its run settings are outside the saved score file | [1-lora-correction/report/against-other-methods/how-does-the-adapter-score-on-the-attend-and-excite-prompts.md](../../../../../../1-lora-correction/report/against-other-methods/how-does-the-adapter-score-on-the-attend-and-excite-prompts.md)<br>[1-lora-correction/artifacts/results/the-adapter-on-cat-and-dog/how-far-one-adapter-reaches/does-one-adapter-compose-four-pairs/figure5-print-six-columns.tex](../../../../../../1-lora-correction/artifacts/results/the-adapter-on-cat-and-dog/how-far-one-adapter-reaches/does-one-adapter-compose-four-pairs/figure5-print-six-columns.tex) |
| Liu (recorded as the plain product) | cat × dog, seeds 9 to 16 and 6001 to 6128; the 53 held-out pairs, seeds 9 to 16 | Fused hybrid on 7 of 8 seeds and two bodies or heads on seed 16 ("two heads on one cat body"), called by eye on the ladder | **Our run,** seeds 9 to 16: two bodies or two clear 1 of 8; two clear 0 of 8 by the ladder; compose rate 0 of 8 by instance count (see section 4).<br>**Our run,** of 128: faithful pairs 0 and not an intact photograph 72 (scorer); clean or near-clean 0 and clean 0 (eye, marked `stated`).<br>**Our run,** 53 held-out pairs, BLIP-VQA: 0.5187 ± 0.0789 | No entry on these counts | [replications/published-methods-on-our-pairs/artifacts/results/does-a-published-method-separate-cat-and-dog/baseline/calls.csv](../../../../../../../replications/published-methods-on-our-pairs/artifacts/results/does-a-published-method-separate-cat-and-dog/baseline/calls.csv)<br>[1-lora-correction/report/is-the-gap-the-samplers-or-the-models/does-a-corrector-alone-produce-two-animals.md](../../../../../../1-lora-correction/report/is-the-gap-the-samplers-or-the-models/does-a-corrector-alone-produce-two-animals.md)<br>[2-classifier-guidance/report/classifier-guided-poe/how-close-the-kept-push-comes-to-the-joint-prompt.md](../../../../../../2-classifier-guidance/report/classifier-guided-poe/how-close-the-kept-push-comes-to-the-joint-prompt.md)<br>[1-lora-correction/report/against-other-methods/how-does-every-sdxl-method-compare-on-the-held-out-pairs.md](../../../../../../1-lora-correction/report/against-other-methods/how-does-every-sdxl-method-compare-on-the-held-out-pairs.md) |
| Du (recorded as the product with a Langevin corrector) | cat × dog, seeds 9 to 16; the four Figure 5 pairs at seed 10; the 3 hard held-out pairs, seeds 9 to 16 | No ladder call. The finding says "every one of the forty renders from a sliding ten-step corrector window is one animal too". At seed 10 the eye read is one dog on cat and dog, and an "elephant with penguin markings" on elephant and penguin | **Our run,** cat × dog seeds 9 to 16, compose rate by instance count: 0 of 8 (0.00).<br>**Our run,** cat × dog seeds 9 to 12: 0 of 4 in each of the nine ten-step windows and with the corrector on all 50 steps.<br>**Our run,** butterfly × meadow, the control pair: 8 of 8.<br>**Our run,** 3 hard held-out pairs, BLIP-VQA: 0.3774 ± 0.3634 | No entry on these counts. The sharpness entry names this finding's sweep script, and this row quotes no sharpness score | [1-lora-correction/report/is-the-gap-the-samplers-or-the-models/does-a-corrector-alone-produce-two-animals.md](../../../../../../1-lora-correction/report/is-the-gap-the-samplers-or-the-models/does-a-corrector-alone-produce-two-animals.md)<br>[1-lora-correction/report/the-adapter-on-cat-and-dog/does-one-adapter-compose-four-pairs.md](../../../../../../1-lora-correction/report/the-adapter-on-cat-and-dog/does-one-adapter-compose-four-pairs.md)<br>[1-lora-correction/report/against-other-methods/how-does-every-sdxl-method-compare-on-the-held-out-pairs.md](../../../../../../1-lora-correction/report/against-other-methods/how-does-every-sdxl-method-compare-on-the-held-out-pairs.md) |

**How the Liu and Du rows were tied to their papers**

No finding calls these runs a reproduction of either paper. The Liu row rests on the [layout sheet](../../../../../../1-lora-correction/report/paper-review/layout-sheet.md), whose Figure 5 column reads "PoE (Composable Diffusion)", and on the revision manuscript citing Liu for the product rule. The Du row rests on that manuscript's Figure 5 caption (line 681), which cites Du for the Langevin corrector. The [review protocol](../../../../../../1-lora-correction/report/paper-review/protocol.md) says the corrector has no published default on SDXL, so its settings (c = 3, k = 20) are this project's.

**Our runs on the public Attend-and-Excite prompts**

These 276 prompts are outside our pair sets. The first three number columns are mean BLIP-VQA over seeds 9 to 12. The last is the share of the 264 animal images where the detector counted two or more animals. Every value is **our run**, copied from the [Attend-and-Excite finding](../../../../../../1-lora-correction/report/against-other-methods/how-does-the-adapter-score-on-the-attend-and-excite-prompts.md). The CO3 row carries the void-list entry named above.

| Rival | Animals, 66 prompts | Animals with objects, 144 | Objects, 66 | Two or more animals counted |
|---|---|---|---|---|
| CO3 | 0.7437 | 0.8688 | 0.4185 | 0.9394 |
| ToMe | 0.8059 | 0.8778 | 0.7020 | 0.9621 |
| Liu (plain product) | 0.2251 | 0.5453 | 0.4524 | 0.0189 |
| Du (product + Langevin) | 0.2830 | not scored | not scored | 0.0492 |

## 2. Rivals with no run on our pairs

Thirteen rivals: Bradley-Projective, Bradley-Local, CoInD, MultiDiffusion, PiCo, FreeFuse, DomainExpansion, Sculpting, Catastrophic, TestTime, AttendExcite, FlipConcept and CFGRevisited.

Three of them come close enough to be mistaken for a run:

**AttendExcite**

Its 276 prompts are used as a benchmark, and no record shows its method run on them or on our pairs. Paper 11 on the [hub's board](../../../../../../../replications/published-methods-on-our-pairs/report/composition-papers-on-our-pairs/board.md) is `todo`, the SD 1.5 run in the [review's score tasks](../../../../../../1-lora-correction/report/paper-review/prze-1-scores.md) is unticked, and `poe_repair/experiments/mechanism_study/aae_intervene.py` is headed "not yet run".

**MultiDiffusion**

The [layout finding](../../../../../../3-understanding-compositionality/report/what-sdxl-already-knows/is-the-layout-the-missing-piece.md) rendered one preview from this project's own regional sampler. Its null outcome waits on "MultiDiffusion's own code" being run.

**Bradley-Projective**

The [replication's SDXL script](../../../../../../../replications/projective-composition/runbook/computing-the-sdxl-bridge-numbers.md) reads one cached plain-product trajectory of cat × dog and "samples nothing".

## 3. Hub methods outside the 19 rivals

The hub lists 14 papers. FKC and Attend-and-Excite are rivals, and the other 12 are below, with venues as the [hub README](../../../../../../../replications/published-methods-on-our-pairs/README.md) states them and arXiv ids as its [paper table](../../../../../../../replications/published-methods-on-our-pairs/report/composition-papers-on-our-pairs/paper-table.md) links them. **None of the 12 has a run on our pairs.** The only code of theirs that has run is the authors' 2D toy from Lee et al. Whether any should be a rival is left open.

| Method | Venue or arXiv id as stated | What it does, from the hub's description |
|---|---|---|
| Sjöberg et al., MCMC-Correction of Score-Based Diffusion Models for Model Composition | Entropy 2026 | Runs a corrector between reverse steps on the summed scores, with a Metropolis-Hastings accept step |
| Lee et al., On the Collapse of Generative Paths (ACE) | ICML 2026 | Gives the Feynman-Kac corrector a time-varying exponent on each expert |
| Zhang et al., Product of Experts for Visual Generation | arXiv 2506.08894 | Runs annealed importance sampling over the experts' scores |
| Huang et al., From Global to Factor-Wise Expert Composition in Discrete Diffusion Models | UAI 2026 | Weights each factor's expert separately, in discrete diffusion models |
| Bansal et al., Universal Guidance for Diffusion Models | ICLR 2024, arXiv 2302.07121 | Adds a guide's gradient, taken through the denoised estimate, to the score |
| Yu et al., Conflict-Aware Additive Guidance for Flow Models under Compositional Rewards | ICML 2026 | Combines reward gradients for flow models by projecting one off another |
| Choi et al., Concept Removal Guidance | ICML 2026 | Uses guidance to remove a named concept |
| Alesiani et al., Logical Guidance for the Exact Composition of Diffusion Models | arXiv 2602.05549 | Adds a correction for broken conditional independence to the summed score |
| Yu and Gao, Improving Compositional Generation with Diffusion Models Using Lift Scores | ICML 2025 | Uses a lift score to pick the best of several draws |
| Wang and Lin, Training-free Conditional Object Composition | TAAI 2026, arXiv 2611.00804 | Applies a condition to a two-object prompt. The hub marks the mechanism as inferred until the paper is read |
| Mondal et al., CountLoop | TMLR 2026, arXiv 2508.16644 | Runs a planner and critic loop to reach a target object count |
| Jeong et al., When Do Diffusion Models Learn to Generate Multiple Objects? (MOSAIC) | ICML 2026 | Tests whether training data causes the failure, with a data recipe called MOSAIC |

## 4. Where the sources disagree

Both sides stay as recorded, and this page does not choose between them.

| What | One source | The other source |
|---|---|---|
| Whether ToMe ran | `1-lora-correction/report/paper-review/protocol.md`, rule 2: ToMe "is not run", because its code "did not import within the 30 minutes". `1-lora-correction/report/paper-review/history.md`, 2026-11-11: "ToMe dropped from SDXL" | `1-lora-correction/artifacts/results/against-other-methods/the-head-to-head-tables/how-the-adapter-scores-against-other-methods/main_table.csv` holds ToMe rows, and `tables_sidecar.json` beside it records 264, 576 and 264 images scored. `writing/restoring-plurality-in-compositional-diffusion/iclr2027_conference_revision.tex`, lines 681 and 798, lists ToMe as a compared method |
| The plain product on cat × dog, seeds 9 to 16 | `replications/published-methods-on-our-pairs/artifacts/results/does-a-published-method-separate-cat-and-dog/baseline/calls.csv`: seed 16 is `two-bodies`, so 1 of 8 by eye | `1-lora-correction/report/is-the-gap-the-samplers-or-the-models/does-a-corrector-alone-produce-two-animals.md`: 0 of 8 by instance count |
| SuperDiff at 50 steps on cat × dog | `1-lora-correction/report/is-the-gap-the-samplers-or-the-models/does-superdiff-compose-at-its-own-defaults.md`, the claim: "composes both test pairs at 50 steps". `report/00-INDEX.md` gives the count as 2 (compose) | The same finding's figure caption: "a dog-shaped silhouette filled with cat figures at 50". Its count is 1 at 50 steps unclamped |
| CO3's sampler and guidance | `main_table.csv` and `held_out_ablation.csv`, in the folder named above: DDIM at guidance 7.5 | The per-image score file: DDIM with the CO3 corrector, CFG++, guidance 0.8, as `1-lora-correction/report/against-other-methods/how-does-the-adapter-score-on-the-attend-and-excite-prompts.md` records. This is the void-list entry on the CO3 row |
| The product with Langevin at seed 10 | `1-lora-correction/artifacts/results/against-other-methods/samplers-and-correctors/does-a-langevin-corrector-fix-the-product/README.md`: "one dog, one elephant, a mountain with no tent" | `1-lora-correction/report/the-adapter-on-cat-and-dog/does-one-adapter-compose-four-pairs.md`: "elephant with penguin markings", and a "snowy peak above a dark dome" that "may be the tent" |
| The plain product on lighthouse and stormy sea, seed 9 | `1-lora-correction/artifacts/results/against-other-methods/the-head-to-head-tables/how-four-methods-compare-on-four-pairs/README.md`: "the plain product fuses or drops one" across the four cells | `1-lora-correction/report/against-other-methods/how-do-four-methods-compare-on-four-hand-picked-pairs.md`: the tile shows a lighthouse and a stormy sea |

## 5. What could not be traced

- **SuperDiff's detector counts.** The finding lifted them from its review file and marks them `inferred`. The scorer output was not reopened.
- **ToMe's run.** The saved score file holds no ToMe rows, so its image list and settings could not be checked.
- **CO3's 128-seed counts.** Its `scores.jsonl` and `eye-calls.csv` are not on this laptop, so these numbers are copied from the finding and the loop.
- **The plain product's eye counts on 128 seeds.** The finding marks both zeros `stated`.
- **Liu and Du as the source of these runs.** The link is a column label and a citation, and no page checks either run against the paper's algorithm.
- **The "no run" list.** It comes from the report pages, the hub and the code on this laptop. Cluster output folders were not listed.
- **The hub's method descriptions.** The paper table marks its smallest settings and port verdicts as inferred until each port's walk confirms them.
- **Feynman-Kac steering (Singhal et al., arXiv 2501.06848).** The [steering finding](../../../../../../1-lora-correction/report/is-the-gap-the-samplers-or-the-models/does-steering-on-the-scorer-find-a-composing-proposal.md) names Singhal et al., a different paper from FKC (Skreta et al.), so its counts are left out of the FKC row.
- **CO3 on the seven-pair, ten-seed grids.** [That finding](../../../../../../1-lora-correction/report/the-adapter-on-cat-and-dog/does-each-adapter-hold-across-ten-seeds.md) holds a CO3 column with no per-tile calls, so it adds no count here.
