# The comparison protocol for the review response

Every new number and image in the revision of ICLR 2027 submission 56173 obeys the rules below. They are fixed before any evaluation job starts, so no result can be called an unfair comparison afterwards, and no rule may change after a result has been seen. A change before that point is made here and dated in [the history](history.md).

Status: **agreed 2026-11-10.** The adapter is named in rule 1.

## The rules

| # | Rule | Defends against |
|---|---|---|
| 1 | One adapter for every adapter number and image, **chosen once by the showcase rule below** and frozen before any held-out rate is read, applied over all 50 sampling steps | PrzE weakness 2; Figure 5 mixes three adapters |
| 2 | Every method on one model sees the same starting noise per pair and seed, 50 DDIM steps, the same guidance weight and resolution | PrzE weakness 1; UHz6 on fairness to CO3 |
| 3 | Baselines run at their published defaults, plus one tuning grid of the same size per method, tuned only on the tuning set | PrzE weakness 2 |
| 4 | No pair the adapter trained on, no pair any tuning touched, and no pair used to choose the adapter is ever counted in a held-out rate | PrzE question 2; UHz6 on held-out pairs |
| 5 | The images shown are picked by a rule fixed here, never by looking | UHz6 on cherry-picking |
| 6 | Network evaluations per image are a column in every table | PrzE on cost; the Langevin corrector's price |
| 7 | A comparison that differs on more than one axis is marked mixed | UHz6 on CO3's own noise |
| 8 | Every number ships with a qualitative grid from the same run | PrzE weakness 1 |
| 9 | Every result row carries the axis values that produced it | being unable to attribute a number later |
| 10 | We run every method ourselves and never copy a published number into our rows | PrzE question 1 (CO3's table) |

## What each rule means in practice

**1. The adapter, and the showcase rule that picks it.** The adapter is picked once, on the two showcase pairs (cat and dog, elephant and penguin), and is then used for every adapter number and image in the paper. Picking on those pairs is stated in the paper, and those two pairs never count toward a held-out rate (rule 4).

The candidates are fixed here and none is added later. Each was held out from cat and dog and from elephant and penguin at training time:

| Candidate | Steps |
|---|---|
| v58-03-self (W&B `qtk5gi37`) | 25,000 · 27,500 · 30,000 · 50,000 |
| v58-05-self-contrast | 15,000 |
| v57-11b-x0-self-decay | 50,000 |
| v58-01-plain | 47,500 |
| v58-02-contrast | 27,500 |
| v58f-05-self-contrast | 25,000 |

**Named 2026-11-10: v58-03-self at step 30,000, shared seed 10.** It is clear on both showcase pairs at seeds 10, 11, 12, 13 and 16, five seeds against four for the next two candidates. The choice lives in `artifacts/_shared/cross_pair_pool_configs/adapter_of_record.yaml`, which every later script reads; the calls behind it are `report/paper-review/showcase-calls.csv`, and `python3 scripts/review_showcase_pick.py` reranks and rewrites the choice when a call or an override changes.

The procedure:

1. Every candidate renders cat and dog and elephant and penguin at seeds 9 to 16 through the product, with the adapter applied over all 50 steps, under rule 2's settings. That makes 9 candidates × 2 pairs × 8 seeds, or 144 renders.
2. Each render is called from its full-resolution tile as one of: two clear animals, the named species; two bodies or heads, fused or wrong species; a single animal or hybrid; destroyed. Claude makes the calls and you can overrule any of them.
3. The winner has the most seeds on which both pairs are two clear animals. A tie goes to the most clear renders out of 16, then to the candidate with the most optimiser steps.
4. The shared seed is the lowest seed on which the winner gives two clear animals for both pairs.
5. The winner, its calls and the shared seed are written into this page before any held-out job starts.

The winner's training recipe becomes the one the paper describes (PRZE-5). At sampling the adapter is applied over all 50 steps (`adapter_steps: 50`), as in every submitted figure. Restricting the window is PRZE-11's question and is reported there as its own column.

These adapters are not candidates and appear in no figure or table of the main paper:

| Adapter | Why |
|---|---|
| `refl-kl-K0` to `K3` (W&B `t4wiwy5p`, `u73uvqyg`, `yj7l40h5`, `ybjd4trt`) and `refl-03-self-contrast-15k` | reward fine-tunes from v58-05 step 15,000 that trained directly on cat and dog and on elephant and penguin, seeds 1 to 8 |
| any checkpoint not in the candidate table | not fixed before the choice |

**2. The same starting noise.** On SDXL every method runs at 1024 by 1024, 50 DDIM steps, guidance weight 7.5, fp16. The initial latent for a seed is the cached seed's latent, which is the same for every pair (cache noise is per seed and shared across pairs), and every method receives that tensor. A method whose official code cannot accept a given latent without modification keeps its own noise from the same integer seed, and its column is marked mixed under rule 7. The CO3-protocol table (rule 10) runs at that paper's own settings, 50 DDIM steps and guidance 5.0, for every method in it. On SDXL, ToMe runs from its official code, unmodified. On SD 1.5 every method runs at 512 by 512, with Attend-and-Excite, SynGen, Divide and Bind, InitNO and CONFORM from their official code, unmodified, at their defaults. FK steering runs as a noise-search control in the CO3-protocol table only.

**3. Tuning.** Each baseline is run at its published defaults. On top of that it gets one grid of at most six settings over its own main knob (CO3's corrector strength, the Langevin step size and step count, Feynman-Kac's particle count and temperature, SuperDiff's mixing rule, ToMe's merge settings), listed per method in step 4 before any tuning run. The setting with the best tuning-set score is frozen, and only that setting is evaluated. Both the default row and the tuned row are reported. The adapter gets no extra tuning.

**4. The pair sets.** These lists are disjoint, and step 4 writes them out pair by pair:

| Set | What is in it | Used for |
|---|---|---|
| training | the 48 pairs of `pair_pool_v58.yaml`, seeds 1 to 8 and 17 to 58 | nothing in evaluation |
| tuning | lion and meerkat, camel and pelican, donkey and duck, chicken and rabbit (training pairs, already used by the species-fidelity session), seeds 9 to 16 | rule 3's grid; never reported as a result |
| showcase | cat and dog, elephant and penguin, seeds 9 to 16 (held out at training) | choosing the adapter (rule 1) and Figure 5; never in a held-out rate |
| held out at training | bison and horse, piano and candle, skateboard and helmet (the rest of the five in v58-03-self's own `pair_pool.json`) | evaluation |
| unseen, outside any list | tent and snowy mountain, lighthouse and stormy sea, and the PRZE-3b groups | evaluation, labelled as never declared held out |
| benchmark | the 276 Attend-and-Excite prompts (`data/prompt-sets/attend-and-excite/a.e_prompts.json`): animals 66, animals with objects 144, objects 66; none names a training pair | the main table, R1 |

The main table, R1, scores every method on the 276 benchmark prompts at seeds 9 to 12. Each prompt is split into its two concepts at " and " or " with " for the product and the adapter ("a horse with a glasses" gives "a horse" and "glasses"), and the joint prompt is the prompt as written. "a cat and a dog" is a showcase pair, so it is scored and marked. The 55 held-out pairs of the census, sorted into hard, easy but unseen and not competing, form the ablation, R3, at seeds 9 to 16. No run trained on seeds 9 to 16. The 45 benchmark prompts that name a frog or a mouse run under the decision recorded in `artifacts/_shared/cross_pair_pool_configs/pair_ledger.md`.

**5. The images shown.** Figure 5 keeps its four submitted rows (cat and dog, elephant and penguin, tent and snowy mountain, lighthouse and stormy sea), all at rule 1's shared seed, re-rendered under rules 1 and 2, and its caption says the adapter and seed were chosen on the first two rows. A new figure shows seeds 9 and 10 for every pair in its group, in the list order step 4 writes. All eight evaluation seeds appear in an appendix grid.

**6. Network evaluations per image.** This is the count of U-Net forward passes to make one image, counted by a counter on the U-Net call during the run, never computed by hand. The Langevin corrector costs about 20 times a product image at 20 steps (an estimate until counted), so the correctors run on a labelled subset, and the product and the adapter are reported on that same subset beside them.

**7. Mixed comparisons.** A table cell or figure column that differs from its comparison on more than one axis (noise, model, sampler, subset, steps) carries a dagger, and the caption names the axes. CO3 with its own noise, SD 1.5 beside SDXL, and the corrector subset are the known cases.

**8. A grid with every number.** Each table has a qualitative grid from the same run: rows are pairs and seeds by rule 5, columns are the compared methods in the table's column order. The grid and the table read the same result file.

**9. Axis values on every row.** A result row holds `model`, `method`, `adapter`, `sampler`, `steps`, `guidance`, `resolution`, `pair`, `seed`, `group`, `setting` and `nfe` beside its scores. `python3 scripts/refresh_paper_slots.py --check` fails a slot whose result file lacks any of these columns.

**10. Our runs only.** The CO3 paper (arXiv 2509.25940) does not state its prompts per set, seeds, resolution or metric code. R1 scores the same 276 prompts with the same metrics, so its numbers can stand beside CO3's Table 1; CO3's numbers appear only as a grey cited row under ours, never in our rows, and the caption says the two differ in seeds and settings.

## One row worked through: elephant and penguin, seed 10

The submitted Figure 5 row against the same row under this protocol.

| Axis | Submitted Figure 5 | Under this protocol | Rule |
|---|---|---|---|
| adapter | v58-05-self-contrast, step 15,000, `rawA` weights, picked for this row alone | the showcase rule's winner, the same for all four rows | 1 |
| role of the pair | called held out | showcase: used to pick the adapter, said in the caption, left out of held-out rates | 1, 4 |
| seed trained on | no (seed 10) | no (the shared seed is one of 9 to 16) | 4 |
| seed shown | 10, picked by hand; the other rows use 9 | the shared seed, the same for all four rows | 1, 5 |
| starting noise, joint and product and ours | shared, the seed's cache latent | shared, the seed's cache latent | 2 |
| starting noise, CO3 | its own | the seed's latent if its code takes one, otherwise its own and marked mixed | 2, 7 |
| steps, guidance, resolution | 50 DDIM, 7.5, 1024 | the same | 2 |
| network evaluations per image | not reported | a column | 6 |
| numbers beside it | none | the winner's calls on all 16 showcase renders, in the appendix | 8 |

Under this protocol the tile changes in four places: the adapter, the seed, the noise CO3 draws, and the caption, which now says how the adapter and seed were picked. The tile record behind the submitted image is `/datasets/mmolefe/poe_repair_min/outputs/ablation_grid/an_elephant__x__a_penguin/seed10/ours_v58-05-self-contrast_s015000_rawA_w50.json`.
