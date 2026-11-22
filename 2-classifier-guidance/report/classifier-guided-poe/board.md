# Classifier-guided PoE: what the route promises

Every outcome [the route](../../plans/route-classifier-guided-poe-from-toy-to-animals.md) promises,
whether it is delivered, and where the proof is. Per `~/.claude/OUTCOME_BOARD_FORMAT.md`.

| Row | What must be true | Status | What showed it | Step |
|---|---|---|---|---|
| G0 | classifier guidance works on the authors' released 256 px weights: scale 0 identical, [scale 10 draws the asked-for class](does-classifier-guidance-work-on-the-released-weights.md#scale-10-draws-the-asked-for-class) | done | [the card](../../artifacts/results/does-classifier-guidance-work-on-the-released-weights/README.md): scale 0 identical on 4 of 4 seeds; at scale 10 the classifier calls 16, 15 and 15 of 16 for corgi, tabby, golden retriever, against 2, 2 and 5 at scale 1; by eye, tile by tile, 9, 11 and 10 at scale 10 (route instruction 13) | 4 |
| R0 | the product sampler pushes unrescaled with an upper cap, and the canary still matches | done | the product sampler pushes unrescaled, capped at 0.1 of its prediction (commit `445cef9f`); at strength 0 plain, no head and untrained head share one md5 on cat × dog seed 9 (job 89341). Plan 17's sampler stays uncapped, as the paper's reproduction | 5 |
| [A1](a1-exact-toy-classifier-push.md) | exact toy: the pair-aware discriminator's push composes and the separate-label push does not | done | at pair weight 0.01, seed 7: the pair-aware push gives 172 of 300 crisp at w = 1 and 300 at w = 2 against 0 unguided; the separate-label push at most 10; on the toy the product's target composes and the push repairs the sampler | 7 |
| A2 | toy caption model: the same guidance composes learned experts | open | not on the current route | |
| [A2b](24-mnist-one-model.md) | MNIST digits on a 32 × 64 canvas, two-digit share at 0, 0.01 and 0.1, five pairs held out: the pair-aware push gives two clear, right digits where plain product fuses, on seen and held-out pairs, and shows what checker accuracy rung 2 needs | done | beside rung 2, gating nothing | 7b |
| [A2d](32-mnist-two-experts.md) | MNIST, a 0–4 expert and a 5–9 expert trained apart, composed over a shared base: the step 7b pushes still give two clear, right digits | done | [the card](../../artifacts/results/can-two-digits-compose-on-mnist/02-two-models-trained-apart/README.md): no. Experts that draw on either side give 4 of 320 with both pushes (bar 224), because the product is destroyed; fixed-side experts compose on 139 of 320 and both pushes lift them to 168 | 7b follow-on |
| [Plan 33](can-two-separately-trained-digit-models-compose-while-sampling.md) | MNIST, the same 0 to 4 and 5 to 9 models trained apart, composed while sampling: at least 1,152 of 1,280 clean pairs and at least 45 of 64 on every pair, on 20 pairs kept out of tuning, counted by eye | open | by the judge 1,200 of 1,280 with each model on a known half, weakest pair 56 of 64; by eye 1,201 to 1,208, waiting for the owner; on one shared canvas with no halves at most 121 of 320, against a bar of 288 | 7b follow-on |
| A2r | our cat and dog photos, released 256 px unconditional model frozen, classifiers trained on our photos: the pair-aware push beats the separate-label push | todo | | 8 |
| A2c | our cat and dog photos, a 64 px model we train with a null label, with and without the both-animal photos: the same | todo | | 8 |
| [A3q](a3q-cat-x-dog-quick-discriminator-test.md) | SDXL cat × dog, judged on held-out seeds 9 to 12: a pair-aware discriminator trained on 10 to 15 renders checked by eye from seeds 1 to 4, with the product's own renders from those seeds as negatives, composes where plain product and best-of-N at equal compute do not | issue | rung 2: draft reading 7 (too weak to read); the discriminator failed its held-out check with hard negatives, so the next rung needs a stronger one | 9, 10 |
| A3 | SDXL cat × dog, same seeds for training and testing: plan 15's pair-aware head composes where the species-only pushes do not | open | off the ladder: rung 2 is A3q; re-routed from step 11 with the head ablations | 11 |
| K1 | animal and animal: rung 2's recipe on 3 pairs from the guard-clean Attend-and-Excite set (cat and dog excluded), seeds 9 to 12, composes where plain product does not | todo | rung 3: a cat and a bear, a bird and a bear, a bird and a turtle | 10b, 10c |
| K2 | animal and object: the same, on 3 of the 90 "and" prompts with the colour words stripped | todo | rung 3: a cat and a backpack, a monkey and a suitcase, a lion and a clock | 10b, 10c |
| K3 | object and object: the same, on 3 of the 66 prompts with the colour words stripped | todo | rung 3: a suitcase and a balloon, a bow and a bowl, a bowl and a bench | 10b, 10c |
| A4 | cat × dog on held-out seeds 9 to 16, beside best-of-N at equal compute | todo | rung 4a: rung 2's discriminator on seeds 9 to 16 | 10d, 10e |
| B1 | at A4's setting, adding a presence head helps or is dropped | open | | |
| B2 | at A4's setting, adding an instance-count head helps or is dropped | open | | |
| B3 | at A4's setting, adding a spatial mask head helps or is dropped | open | | |
| B4 | at A4's setting, adding a hybrid and quality head helps or is dropped | open | | |
| B5 | at A4's setting, the weak-strong-fade schedule helps or is dropped | open | | |
| B6 | at A4's setting, gradient agreement helps or is dropped | open | | |
| B7 | at A4's setting, per-head early stopping helps or is dropped | open | | |
| C1 | the best configuration trained over the v58 pool, held-out seeds, beside the rank-32 LoRA | todo | rung 4b: one discriminator over the v58 pool | 10d, 10e |
| C2 | the best configuration on held-out pairs, beside the rank-32 LoRA | todo | rung 4c: 4b's discriminator on the 53 held-out pairs | 10d, 10e |
| P1 | the paper's comparison table and sheet, and its paragraph | open | re-routed from step 11 once rung 4 closes | 11 |
| X1 | you can explain the result without notes | todo | each rung's retrospective, walked in step 11 | 11 |

## The scale-up ladder

Each rung is the proof of concept for the next and changes one thing. Every rung's compute plan
(GPU-hours and wall clock from measured rates, node and card with a fallback, the seeds and pairs it
buys) is written and ticked on its row before it launches.

| Rung | What changes from the rung before | Rows | Goes on when |
|---|---|---|---|
| 1 | the exact toy, where the true correction is known | A1 | A1's bar: pair-aware at least 150 of 300 crisp pairs at some s, separate-label at most 15 at every s |
| 2 | SDXL, cat × dog only, 4 held-out seeds | A3q | the first of A3q's seven readings: two animals on held-out seeds, pictures intact |
| 3 | the pair kind: animal and animal, animal and object, object and object | K1, K2, K3 | at least one kind passes; a kind that misses keeps its row and its diagnosis |
| 4a | more seeds: cat × dog on all 8 held-out seeds, 9 to 16 | A4 | A4's bar |
| 4b | more pairs: the v58 pool, 182 cells over 48 pairs, held-out seeds | C1 | C1's bar |
| 4c | unseen pairs: the 53 held-out pairs, the 23 sharing no concept with training reported apart | C2 | C2's bar |

Beside the ladder, gating nothing: our photos (A2r, A2c) as the controlled-data check, and the head
ablations (B1 to B7) at A4's setting.

## Expected outcome per row

What you should see if each row works, in plain words. The number each line needs comes from the
route's step 2, which frames the hypothesis; until then the line says "number from step 2".

| Row | You should see | Number |
|---|---|---|
| G0 | at scale 0 the same picture as unguided on 4 of 4 seeds; at scale 10 the released classifier calls at least 14 of 16 samples corgi, tabby or golden retriever | 4 of 4 identical, 14 of 16 per class |
| R0 | a render at push strength 0 through the repaired sampler that matches the plain product byte for byte | identical checksums |
| A1 | at pair weight 0.01, the pair-aware push lands draws as two separate shapes where unguided reverse diffusion gives 0 of 300, and the separate-label push stays near 0; the push points the same way as the true correction in the early steps; overlaid shapes do not rise | pair-aware at least 150 of 300 crisp pairs at some s; separate-label at most 15 of 300 at every s |
| A2 | in the trained caption model, at the low dial settings where plain PoE draws few pairs, guided draws are mostly separate pairs, and each single caption still draws its own shape | from step 2 |
| A2b | on 7 & 8 at the share where plain product fuses, the pair-aware push gives two clear, right digits on more of 64 seeds than plain product, the separate-label push, a size-matched random push or best-of-N; the same on the five held-out pairs; the compose count rises with checker accuracy | plain at most 8 of 64; pair-aware at least 32 of 64 at some s; random and separate-label within 6 of plain; best-of-N at least 8 below; held-out pairs at least 160 of 320 |
| A2r | on seeds 1 to 16 at 256 px, the pair-aware push gives two separate animals on more seeds than the two separate-label pushes alone | at least 4 more of 16 in one arm |
| A2c | on seeds 1 to 16 at 64 px, the same, in the model trained without both-animal photos and in the one trained with them | at least 4 more of 16 in one arm |
| A3 | off the ladder; on cat × dog, on the seeds it trained on, some seeds show two separate animals with the push where the plain product shows none, and the pictures are not wrecked; a yes says only that the push can compose at all | from step 2 |
| A3q | on held-out seeds 9 to 12, more seeds show two separate animals at full resolution with the push than with plain product or with best-of-N at the same compute, and the pictures are not wrecked | guided at least 2 of 4 on seeds 9 to 12, more than best-of-N; the other six readings on A3q's page |
| K1 | on 12 held-out tiles (3 pairs × seeds 9 to 12), more tiles with two separate animals with the push than with plain product or best-of-N, pictures not wrecked | at least 4 more of 12 than plain product; "nothing to fix" when plain is already at least 9 of 12 |
| K2 | the same on 3 animal and object pairs, two separate things in frame | the same counts |
| K3 | the same on 3 object pairs | the same counts |
| A4 | on held-out seeds 9 to 16, more seeds show two animals with the push than with best-of-N at the same compute, with picture damage no worse than the plain product's | on seeds 13 to 16 (s was chosen on 9 to 12): guided at least 2 of 4, more than best-of-N (proposed; step 10d confirms) |
| B1 | adding the presence head raises A4's count or keeps it with cleaner pictures; pushed alone, it makes both species appear | from step 2 |
| B2 | adding the instance-count head does the same; pushed alone, it makes a second body appear | from step 2 |
| B3 | adding the spatial mask head does the same; pushed alone, it pulls the two bodies apart | from step 2 |
| B4 | adding the hybrid and quality head does the same; pushed alone, it splits a fused animal instead of changing its fur | from step 2 |
| B5 | the weak-early, strong-middle, fading schedule gives the same or a higher count with less damage than a constant push | from step 2 |
| B6 | checking that the heads' pushes agree gives fewer broken pictures at the same count | from step 2 |
| B7 | stopping each head once it is satisfied gives fewer overshoots (a third animal, a smeared extra) at the same count | from step 2 |
| C1 | trained over the v58 pool and tested on held-out seeds, the guided product's two-animal count is close to the rank-32 LoRA's and above best-of-N at equal compute | guided two-things rate on held-out seeds within 10 percentage points of the rank-32 LoRA's and above best-of-N (proposed; step 10d confirms) |
| C2 | the same on held-out pairs, reported separately for the 23 pairs that share no concept with training | the same as C1, reported apart for the 23 pairs sharing no concept (proposed; step 10d confirms) |
| P1 | one table and one sheet with plain product, best-of-N, guided product, joint prompt and LoRA on held-out pairs, and a paragraph saying whether the push can stand in for the LoRA | none, it reports the others |
| X1 | at each rung's close you predict each result before seeing it, and explain it without notes | every prediction made before its reveal |

## Outcomes per row

Ticked with a link and a `verified`, `stated` or `inferred` mark as each is delivered.

**G0**

- [x] 💻 code ran: the released 256 px model and classifier sampled at scale 0, 1 and 10 · [the wrapper](../../scripts/guided_diffusion_reproduce.py), Slurm jobs 89232 and 89235, nohup run on mscluster107, upstream commit `22e0df8` `verified`
- [x] 🖼️ screenshot: the corgi, tabby and golden retriever sheets · `sheet-pembroke-welsh-corgi-scale-1-and-10.png` is not shown because it failed the subject check `sheet-tabby-cat-scale-1-and-10.png` is not shown because it failed the subject check `sheet-golden-retriever-scale-1-and-10.png` is not shown because it failed the subject check `verified`
- [x] ✅ check passed: scale 0 identical on 4 of 4; at least 14 of 16 per class at scale 10 · [the identity checksums](../../artifacts/results/does-classifier-guidance-work-on-the-released-weights/identity/identity_checksums.csv), [the per-sample calls](../../artifacts/results/does-classifier-guidance-work-on-the-released-weights/classifier_calls.csv): 16, 15, 15 of 16 `verified`

**R0**

- [x] 📁 folder created: plans 15, 17 and 18 and the classifier code in this repo · [the README](../../README.md) `verified`
- [x] 💻 code ran: `run_poe_classifier_guided` pushes strength times sqrt(1 - alpha_bar_t) times the gradient, capped at `PUSH_CAP_FRACTION` 0.1 of the product prediction's norm, logging the raw gradient, the push before and after the cap, and whether it fired · commit `445cef9f`, the canary on commit `d1b8b61f` `verified`
- [x] ✅ check passed: strength 0 byte-identical to the plain product with the repaired sampler · md5 `c291b259647d30a450e87b234423fe99` for plain, strength 0 and strength 0 with an untrained head, 0 pixels differing, job 89341 on mscluster108 · `/datasets/mmolefe/poe_repair_min/outputs/presence_head/canary/canary.json` `verified`

**A1**

- [ ] 📝 compute plan written before launch: GPU-hours and wall clock from measured rates, node, card and fallback card, seeds and pairs it buys
- [ ] 💻 code ran: discriminator trained on exact product draws, which are the pairs, against the summed-score sampler's draws
- [ ] 🖼️ screenshot: cosine and norm ratio with the true correction, per reverse-diffusion step
- [ ] ✅ check passed: crisp pairs of 300, guided against unguided, against the rung's bar

**A2**

- [ ] 🖼️ screenshot: caption-model draws, plain product against guided, per dial setting
- [ ] ✅ check passed: crisp pairs of 300 against the rung's bar

**A2b**

- [ ] 📝 the bar copied into a review file in `plans/review/` before any model trains
- [ ] 📝 training time measured on the laptop's RTX 4070 before the three models launch
- [ ] 🖼️ screenshot: plain product and the support check, as a sheet, per share
- [ ] ✅ check passed: the judge agrees with the user's eye calls on at least 58 of 64 tiles
- [ ] 🖼️ screenshot: the five-version sheet on 7 & 8 and on the five held-out pairs
- [ ] 📈 figure: compose count against checker accuracy, with 0.655 and 0.771 marked, and the cap sweep
- [ ] ✅ check passed: the reading printed as pass or miss against the bar

**A2r, A2c**

- [ ] 💻 code ran: both 64 px trainings, and classifiers trained on our photos for both arms
- [ ] 🖼️ screenshot: one sheet per arm, plain product, separate-label push, pair-aware push
- [ ] ✅ check passed: the arm's count against its bar, called from full-resolution tiles

**A3q, K1, K2, K3, A4, C1, C2**

- [ ] 📝 compute plan written before launch: GPU-hours and wall clock from measured rates, node, card and fallback card, seeds and pairs it buys

**A3, A4, B1 to B7, C1, C2, A3q, K1, K2, K3**

- [ ] 🖼️ screenshot: the sheet with plain product, best-of-N at equal compute, guided product, joint
      prompt (and the rank-32 LoRA on C1 and C2)
- [ ] ✅ check passed: two separate animals called by eye at full resolution, and by an evaluator
      the classifier was not trained on, against the rung's bar
- [ ] 📝 markdown created: one finding in `report/`

**P1**

- [ ] 📝 markdown created: the paper's table and its paragraph

**X1**

- [ ] ✅ check passed: the retrospective walked, each prediction made before its result was shown

## Dropped

None.

## Cross-references

- The mention of **the scale-up ladder** in [the rung 2 review file, a small pair-aware discriminator on cat × dog](../../plans/review/a3q-cat-x-dog-quick-discriminator-test.md).
