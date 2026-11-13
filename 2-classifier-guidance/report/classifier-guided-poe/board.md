# Classifier-guided PoE: what the route promises

Every outcome [the route](../../plans/route-classifier-guided-poe-from-toy-to-animals.md) promises,
whether it is delivered, and where the proof is. Per `~/.claude/OUTCOME_BOARD_FORMAT.md`.

| Row | What must be true | Status | What showed it | Step |
|---|---|---|---|---|
| R0 | every sampler pushes unrescaled with an upper cap, and the canary still matches | todo | plans 17 and 18 are in this repo's `plans/` | 4, 5 |
| A1 | exact toy: the discriminator's guidance matches the true correction and composes | todo | | 4, 6 |
| A2 | toy caption model: the same guidance composes learned experts | open | | |
| A2b | MNIST digits on a canvas, both-share dial kept: guidance composes where plain PoE does not | open | | |
| A2c | animal photos at 64×64, pasted pairs, dial kept: the same | open | | |
| A3 | SDXL cat × dog, same seeds for training and testing: the push can compose at all | open | | |
| A4 | cat × dog on held-out seeds 9 to 16, beside best-of-N at equal compute | open | | |
| B1 | at A4's setting, adding a presence head helps or is dropped | open | | |
| B2 | at A4's setting, adding an instance-count head helps or is dropped | open | | |
| B3 | at A4's setting, adding a spatial mask head helps or is dropped | open | | |
| B4 | at A4's setting, adding a hybrid and quality head helps or is dropped | open | | |
| B5 | at A4's setting, the weak-strong-fade schedule helps or is dropped | open | | |
| B6 | at A4's setting, gradient agreement helps or is dropped | open | | |
| B7 | at A4's setting, per-head early stopping helps or is dropped | open | | |
| C1 | the best configuration trained over the v58 pool, held-out seeds, beside the rank-32 LoRA | open | | |
| C2 | the best configuration on held-out pairs, beside the rank-32 LoRA | open | | |
| P1 | the paper's comparison table and sheet, and its paragraph | open | | |
| X1 | you can explain the result without notes | todo | A1's part delivered by step 7; later rungs by their own retrospectives | 7 |

## Expected outcome per row

What you should see if each row works, in plain words. The number each line needs comes from the
route's step 2, which frames the hypothesis; until then the line says "number from step 2".

| Row | You should see | Number |
|---|---|---|
| R0 | a render at push strength 0 through the repaired sampler that matches the plain product byte for byte | identical checksums |
| A1 | at pair weight 0.01, guided draws mostly land as two separate shapes where unguided reverse diffusion gives 0 of 300; the push points the same way as the true correction in the early steps; overlaid shapes do not rise | crisp pairs of 300 and the cosine floor, from step 2 |
| A2 | in the trained caption model, at the low dial settings where plain PoE draws few pairs, guided draws are mostly separate pairs, and each single caption still draws its own shape | from step 2 |
| A2b | two separate digits on the canvas where plain PoE draws one overlaid digit, at the low dial settings, with each digit still readable | from step 2 |
| A2c | two separate animals at 64 px where plain PoE draws one fused animal, at the low dial settings; the minimal-pair read shows the classifier scores two animals and not the paste seam | from step 2 |
| A3 | on cat × dog, on the seeds it trained on, some seeds show two separate animals with the push where the plain product shows none, and the pictures are not wrecked; a yes says only that the push can compose at all | from step 2 |
| A4 | on held-out seeds 9 to 16, more seeds show two animals with the push than with best-of-N at the same compute, with picture damage no worse than the plain product's | from step 2 |
| B1 | adding the presence head raises A4's count or keeps it with cleaner pictures; pushed alone, it makes both species appear | from step 2 |
| B2 | adding the instance-count head does the same; pushed alone, it makes a second body appear | from step 2 |
| B3 | adding the spatial mask head does the same; pushed alone, it pulls the two bodies apart | from step 2 |
| B4 | adding the hybrid and quality head does the same; pushed alone, it splits a fused animal instead of changing its fur | from step 2 |
| B5 | the weak-early, strong-middle, fading schedule gives the same or a higher count with less damage than a constant push | from step 2 |
| B6 | checking that the heads' pushes agree gives fewer broken pictures at the same count | from step 2 |
| B7 | stopping each head once it is satisfied gives fewer overshoots (a third animal, a smeared extra) at the same count | from step 2 |
| C1 | trained over the v58 pool and tested on held-out seeds, the guided product's two-animal count is close to the rank-32 LoRA's and above best-of-N at equal compute | from step 2 |
| C2 | the same on held-out pairs, reported separately for the 23 pairs that share no concept with training | from step 2 |
| P1 | one table and one sheet with plain product, best-of-N, guided product, joint prompt and LoRA on held-out pairs, and a paragraph saying whether the push can stand in for the LoRA | none, it reports the others |
| X1 | at each rung's close you predict each result before seeing it, and explain it without notes | every prediction made before its reveal |

## Outcomes per row

Ticked with a link and a `verified`, `stated` or `inferred` mark as each is delivered.

**R0**

- [x] 📁 folder created: plans 15, 17 and 18 and the classifier code in this repo · [the README](../../README.md) `verified`
- [ ] ✅ check passed: strength 0 byte-identical to the plain product with the repaired sampler

**A1**

- [ ] 💻 code ran: discriminator trained on exact pair draws against exact product draws
- [ ] 🖼️ screenshot: cosine and norm ratio with the true correction, per reverse-diffusion step
- [ ] ✅ check passed: crisp pairs of 300, guided against unguided, against the rung's bar

**A2**

- [ ] 🖼️ screenshot: caption-model draws, plain product against guided, per dial setting
- [ ] ✅ check passed: crisp pairs of 300 against the rung's bar

**A2b, A2c**

- [ ] 🖼️ screenshot: plain product against guided, per dial setting, with the joint model
- [ ] ✅ check passed: separate pairs of 300 against the rung's bar, and D1 to D4 read

**A3, A4, B1 to B7, C1, C2**

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
