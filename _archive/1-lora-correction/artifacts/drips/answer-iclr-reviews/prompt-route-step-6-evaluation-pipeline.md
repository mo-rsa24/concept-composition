# Prompt: route step 6, draw the evaluation pipeline

Paste into a fresh session in `/home/molef/PhD/concept-composition`. It replaces the prompt block under step 6 of `plans/route-answer-the-iclr-reviews.md`, whose method and machine facts predate step 4's protocol changes.

```text
/drip-hld-diagram --from plans/route-answer-the-iclr-reviews.md the evaluation pipeline from a
method to an image to a score to a table to a slot in the paper
Model: Sonnet, medium effort. Box: 45 minutes of your time. Filing: tied.
Controls: round 1 options (other ways to cut the pipeline into pieces), piece 4 why (why each scorer
sits where it does), piece 7 plumb (the shared-device launch path).
Side routes: none; hld-diagram's own modes cover the hardware level.

Background
Submission 56173 to ICLR 2027 is being revised to answer three reviews; it needs a results section
with scores, baselines and held-out rates. The pipeline that makes them has many parts, and several
unfair comparisons would be easy to introduce by accident. This step draws the pipeline once so
every later step and the paper's evaluation paragraph agree on it. The repo is
/home/molef/PhD/concept-composition.

Objectives
1. Check hld-diagram/references/registry.json for an existing page of this system, and say what is
   found.
2. Cut the pipeline into 4 to 8 named pieces and settle and draw one per round.
3. Add the hardware level for the machines and the filesystems.
4. Mark on each piece where an unfair comparison could enter, and the protocol rule that guards it.
5. Publish the page, save the SVG and its spec under artifacts/scenes/, and write the card.

What to read
- plans/route-answer-the-iclr-reviews.md: the chain, the table "What we know so far", every step.
- 1-lora-correction/report/paper-review/protocol.md (rules 1 to 10), layout-sheet.md (slots R1 to R14, each with its
  result file), held-out-set.md, and prze-10-compute.md once step 5 has left it.
- artifacts/_shared/cross_pair_pool_configs/adapter_of_record.yaml: the one adapter.
- 1-lora-correction/report/paper-review/board.md and the pages for the rows each piece serves.
- environment/overview.md, environment/hpc/nodes.md, environment/storage.md, environment/paper.md.
- composition/focus/eval_metrics.py: the scorers that already exist here.
- 1-lora-correction/scripts/refresh_paper_slots.py: how a result file becomes a table or figure in a slot.
- hld-diagram/references/registry.json: existing pages, to avoid a second page of the same system.

What is already known
- No page of this pipeline exists in the registry (18 pages, checked 2026-11-10).
- Methods on SDXL (main table R1 and ablation R3): joint prompt, PoE, CO3, ToMe, SuperDiff, PoE with
  a Langevin corrector (a labelled subset), and the frozen adapter v58-03-self at step 30,000
  (layout-sheet.md R1; adapter_of_record.yaml). PoE with unadjusted HMC appears only in the sampler
  figure R6. Feynman-Kac steering runs only as a noise-search control in the CO3-protocol table
  (protocol rule 2).
- Methods on SD 1.5 (R5): PoE, PoE with the measured correction, Attend-and-Excite, SynGen, Divide
  and Bind, InitNO and CONFORM, from their official code, unmodified, at their defaults.
- Adjusted MALA and HMC cannot run on SDXL, which outputs a noise prediction and no energy.
- Inputs: the 276 Attend-and-Excite prompts at seeds 9 to 12 (R1), the 53 held-out pairs at seeds
  9 to 16 (R3), GenEval and T2I-CompBench on their own prompts (R14), the tuning pairs (lion and
  meerkat, camel and pelican, donkey and duck, chicken and rabbit), the showcase pairs (cat and dog,
  elephant and penguin, never in a rate).
- The scorers: GroundingDINO instance count (animals only; it cannot tell species, so a species
  check is done by eye), CLIP score, BLIP-VQA, ImageReward, GenEval's and T2I-CompBench's official
  evaluators, and network evaluations per image counted on the U-Net call (protocol rule 6).
- Outputs: result CSVs carrying protocol rule 9's columns, read by 1-lora-correction/scripts/refresh_paper_slots.py
  into the slots of paper/iclr-56173/, which is built with Tectonic on the cluster and uploaded to
  Overleaf by hand.
- Machines: the laptop writes code and the paper; mscluster107 is the user's biggpu job 83423 with
  two RTX 8000, ending around midday Thursday 12 November; mscluster112 has one RTX PRO 6000
  Blackwell; outputs go to /datasets/mmolefe/poe_repair_min/ only.
- Fairness rules from the protocol: the same starting latent per seed for every method (rule 2),
  tuning only on the tuning pairs (rule 3), one frozen adapter (rule 1), a qualitative grid beside
  every table from the same result file (rule 8), our runs only, with CO3's printed numbers as a grey
  cited row (rule 10).

When something is missing
- If prze-10-compute.md does not exist yet, draw the pieces that do not depend on it, mark the
  others as planned, and say which are planned.
- If the hardware level needs a fact not in environment/, ask the user, who can run the command.

Rules
- One piece per round. Wait for the user's yes before drawing the next.
- Draw only what exists or is planned in the route. Mark each piece built, planned or open.
- Do not publish until the user has seen the whole picture.

What to hand back
- The published page URL, the saved SVG and spec under artifacts/scenes/, and the card entry.
- A line added to 1-lora-correction/report/paper-review/prze-9-slots.md linking the page.
- Proof: the registry lists the new page, and the page opens with every piece present.

What you will see
Along the way: the registry check, then the piece list, then each piece drawn with its fairness
mark, each followed by one question. At the end: what was asked, what was made with paths and the
URL, PASS or MISS, what is open, and at most one question. PASS means the page is published and
shows every piece with its mark. You judge by opening the page.
```

- `added:` ToMe and SuperDiff on SDXL, SynGen, Divide and Bind and InitNO on SD 1.5, Feynman-Kac moved to the CO3-protocol table (from `protocol.md` rule 2 and `layout-sheet.md`) · `added:` the inputs, the slot refill path and the adapter of record · `added:` the 107 allocation's end time (`squeue`, 2026-11-11)
- `dropped:` "seven pieces" fixed in advance (why: the objective already asks for 4 to 8, and the piece count is settled in round 1)
