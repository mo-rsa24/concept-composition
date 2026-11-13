# Prompt: route step 8, prove the Langevin corrector and render the six-column figure

Paste into a fresh session in `/home/molef/PhD/poe_repair_min`. It replaces the prompt block under step 8 of `plans/route-answer-the-iclr-reviews.md`.

```text
/drip-execute-plan --request --depth objective "Prove our Langevin corrector against the Reduce,
Reuse, Recycle paper on two Gaussians, add the joint-prompt and unadjusted HMC drivers, choose the
settings on the tuning pairs, and render the six-column comparison figure"
Model: Opus, medium effort. Box: 120 minutes of your time. Filing: on, because the tuning contact
sheets and the figure are what picture-cadence and record-cadence should file.
Controls: piece 1 prove (the exact stationary variance against the chain), piece 2 debug (the
driver at zero corrector steps), piece 3 example (one tuning cell with its image), piece 5 draw
(the figure's layout before the render).
Side routes: piece 1 /drip-execute-plan --reproduce 2302.11552 only if the Gaussian check misses
and the paper's own toy is needed as a second reference.

Background
Reviewer PrzE asked for an MCMC corrector baseline. Our Langevin corrector
(poe_repair/methods/_poe_langevin.py) uses the paper's unadjusted Langevin update with a different
step-size rule and a different order of corrector and step. This step proves the update on a
toy with a known answer, builds the joint-prompt and unadjusted HMC drivers, and renders the
figure. The repo is /home/molef/PhD/poe_repair_min.

Objectives
1. Write a numpy check of the update on two Gaussian experts and print the chain's mean and
   variance against the exact stationary values, for four step sizes.
2. Measure the order difference, corrector first against step first, on the same toy.
3. Build the joint-prompt Langevin driver and the unadjusted HMC driver, each byte-identical to the
   plain sampler at zero corrector steps.
4. Choose c and k on the tuning pairs by a numeric rule written before the run, saving an image per
   tuning cell.
5. Render the six-column figure (slot R6): rows are Figure 5's four pairs (cat and dog, elephant
   and penguin, tent and snowy mountain, lighthouse and stormy sea) at the shared seed 10, columns
   joint, joint with Langevin, PoE, PoE with Langevin, PoE with unadjusted HMC, CO3, all from one
   starting noise per row. Draw it with 1-lora-correction/scripts/paper_tile_figure.py at 5.5 in wide, so its labels
   print at 8 pt (layout-sheet.md, "Before a slot is called finished").

What to read
- poe_repair/methods/_poe_langevin.py: the corrector and its precision rules.
- composition/reduce_reuse_recycle/anneal_samplers.py: the authors' samplers, which differ from
  their paper in 28 places (the audit below); implement from the paper's equations.
- The code-against-paper audit at
  /home/molef/goal-setting/learning/deep-learning/diffusion-models/02-sampler-correctors-for-composition/drips/read-reduce-reuse-recycle/routes/code-vs-paper.md
- 1-lora-correction/plans/06-is-the-gap-the-samplers-or-the-models/plans/tools/02-the-corrector-and-the-step-size-it-runs-at.md
  and
  1-lora-correction/report/is-the-gap-the-samplers-or-the-models/does-a-langevin-corrector-remove-part-of-the-correction.md
- 1-lora-correction/report/paper-review/protocol.md, held-out-set.md, layout-sheet.md: rules, the tuning pairs and the
  figure's slot.
- 1-lora-correction/scripts/corrector_residual_curve.py and 1-lora-correction/scripts/figure_candidates.py: existing drivers and the
  grid.

What is already known
- Our update matches the paper's unadjusted Langevin step; the order, the step-size schedule and
  the constant differ. Adjusted MALA and HMC need an energy that SDXL does not output.
- On a Gaussian with variance V, the update with step delta has stationary variance
  V / (1 - delta / (2 V)); the product of N(-1, 0.5) and N(1, 0.8) is N(-0.2308, 0.3077) with
  variance as the second argument (audit numbers).
- Single-seed residual ratios scatter 15 to 70 percent; a latent-norm guard is blind to destroyed
  samples; c = 3 with k = 20 was the working point; long chains drift from photo to illustration.
- The corrector costs about 21 times a PoE image at k = 20 over all 50 levels; the 0 to 10 variant
  is a labelled deviation, never the headline row.
- cat and dog and elephant and penguin stay out of every tuning set. The tuning pairs are lion and
  meerkat, camel and pelican, donkey and duck, chicken and rabbit at seeds 9 to 16, with a grid of
  at most six settings over the step size and step count (protocol rules 3 and 4).
- The adapter of record is v58-03-self at step 30,000, shared seed 10 (adapter_of_record.yaml).
- The same tuned Langevin setting fills the PoE with Langevin row of the main table R1 (a labelled
  subset, rule 6) and its column in Figure 5 (slot R2).

When something is missing
- If the Gaussian check misses the exact values, stop, show the numbers, and ask the user before
  changing the update.
- If CO3 cannot take an external starting noise, it keeps its own noise from the same integer seed
  and its column is marked mixed with a dagger (protocol rules 2 and 7); say so in the sidecar.
- If the tuning pairs are not frozen yet, stop and say step 4 is owed.

Rules
- Show the check's pass or miss line before building the drivers.
- The tuning rule is written to the protocol before any tuning cell runs.
- Add this step's GPU jobs (tuning grid, the R6 renders, the Langevin rows of R1 and Figure 5) to
  1-lora-correction/report/paper-review/prze-10-compute.md as a later wave, and launch them only after the user's yes
  to each command, by the shared-device path.

What to hand back
- 1-lora-correction/scripts/check_langevin_two_gaussians.py and its printed table.
- The two drivers, the tuning contact sheets and the chosen c and k with the rule that chose them.
- The six-column figure in its slot, with its sidecar file listing seeds, noise and settings.
- Rows added to "What we know so far" in plans/route-answer-the-iclr-reviews.md.
- Proof: python3 1-lora-correction/scripts/check_langevin_two_gaussians.py prints PASS for all four step sizes, and
  the board redraws with --step 8 ticking the PRZE-2 lines.

What you will see
Along the way: the check's table, then the order measurement, then a tuning contact sheet, then
the figure draft, each followed by one question. At the end: what was asked, what was done with
paths, PASS or MISS, what is open, and at most one question. PASS means the Gaussian check passes
at all four step sizes, the drivers match the plain sampler at zero steps, and the figure is built
from the frozen settings. You judge from the table and the figure.
```

- `added:` the figure's rows, columns and lettering (`layout-sheet.md` R6) · `added:` the tuning pairs and grid size (`protocol.md` rules 3 and 4) · `added:` the Langevin row of R1 and column of R2 this step also fills · `added:` the adapter of record
- `dropped:` asking the user what to do if CO3 cannot take our noise (why: protocol rules 2 and 7 already decide it)
