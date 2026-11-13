# Prompt: route step 11, the three-concept test

Paste into a fresh session in `/home/molef/PhD/poe_repair_min`. It replaces the prompt block under step 11 of `plans/route-answer-the-iclr-reviews.md`.

```text
/drip-execute-plan --request --depth objective "Test the frozen adapter inside a three-concept
product against plain three-concept PoE and the joint prompt, with the triples and the bar written
before any image exists"
Model: Sonnet, medium effort. Box: 30 minutes of your time. Filing: on.
Controls: piece 1 example (one triple's four branches), piece 3 test (that the adapter was active in
all four branches), piece 4 draw (the grid's layout before the render).
Side routes: none; no installed mode fits better than the walk itself.

Background
Reviewer PrzE's second question asks how the method performs when composing more than two concepts.
The adapter was trained on pairs, takes two concept prompts and the null prompt, and has no input
slot for a third. This step tests it zero-shot inside a three-concept product, with a bar written
before the run. The repo is /home/molef/PhD/poe_repair_min.

Objectives
1. Write the triple-selection rule and the list of about 10 triples and 3 tuning triples, and commit
   them before any image exists.
2. Write the bar into the protocol: the signal and no-signal intervals.
3. Tune the guidance weight for plain 3-PoE and the adapter equally on the tuning triples only.
4. Run joint, plain 3-PoE and adapter 3-PoE on 10 triples at seeds 9 to 16 from one starting noise
   per triple and seed, and report the rates and the interval.
5. Fill slot R11 (appendix C.6) with the grid, failures included: four triples at the shared seed
   10, columns joint prompt, plain product, ours, drawn with 1-lora-correction/scripts/paper_tile_figure.py at 5.5 in
   wide and within the slot's 3.0 in height, labels 8 pt (layout-sheet.md R11).

What to read
- 1-lora-correction/report/paper-review/protocol.md, held-out-set.md and prze-4-three-concepts.md.
- poe_repair/methods/_sampling.py: the repo's PoE rule and where the adapter is attached.
- The submitted paper, Appendix B.1: the adapter's three branches (equation 13).
- artifacts/_shared/cross_pair_pool_configs/pair_pool_v58.yaml: the 48 training pairs, whose
  concepts the triples must avoid.
- 1-lora-correction/scripts/joint_poe_adapter_grid.py and 1-lora-correction/scripts/figure_candidates.py: existing grid builders.

What is already known
- The adapter of record is v58-03-self at step 30,000, shared seed 10
  (artifacts/_shared/cross_pair_pool_configs/adapter_of_record.yaml): a rank-32 module on the
  cross-attention layers, with no input slot for a third concept, trained on pairs only.
- The paper sets both guidance weights to 7.5; a three-term product has no published weight.
- Our instance count (GroundingDINO "animal", count of at least 3 for a triple) counts animals and
  cannot tell species; a contact sheet is read by label, and object triples are called by eye.
- SDXL's joint prompt fails more often as concepts are added, so triple targets would be weak.
- Training a three-concept adapter is out of scope here; it opens only on the user's yes.

When something is missing
- If fewer than 8 triples avoid the training concepts, report the count and ask the user whether to
  accept fewer or allow a triple that reuses one training concept, labelled.
- If the frozen checkpoint cannot be attached to a four-branch call, stop and show the error.
- If the frozen file from step 4 is missing, stop and say step 4 is owed.

Rules
- The triples, rule and bar are committed before the first image is generated.
- Tune on tuning triples only. Never look at the 10 triples to choose a weight.
- Add this step's GPU job to 1-lora-correction/report/paper-review/prze-10-compute.md and launch it after the user's
  yes.

What to hand back
- 1-lora-correction/report/paper-review/three-concept-set.md with the rule, the triples and the commit hash.
- Images under /datasets/mmolefe/poe_repair_min/outputs/paper_review/three_concepts/, a contact
  sheet with a card entry, and the rates with intervals.
- The figure in its slot, and rows added to "What we know so far" in the route file.
- Proof: the script prints SIGNAL or NO SIGNAL with the interval, and the board redraws with
  --step 11 ticking the PRZE-4 lines.

What you will see
Along the way: the triple list and the bar, then the tuning result, then the contact sheet and the
rates, each followed by one question. At the end: what was asked, what was done with paths, SIGNAL
or
NO SIGNAL against the written bar, what is open, and at most one question. You judge from the
contact sheet and the interval.
```

- `added:` the adapter of record (`adapter_of_record.yaml`) · `added:` slot R11's rows, columns, height and lettering (`layout-sheet.md`) · `added:` the count threshold for a triple and the by-eye call for objects
- `dropped:` nothing
