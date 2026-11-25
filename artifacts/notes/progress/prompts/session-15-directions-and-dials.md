# Session 15: directions that generalise, and dials for abstract concepts

This runs as a background session, in the same permission mode as the other loops. To start it by
hand from `/home/molef/PhD/concept-composition`:

```text
claude --bg --permission-mode auto --model claude-opus-5-5 --effort high --name "🧭 PhD Loop 15: Directions And Dials" "/loop Follow artifacts/notes/progress/prompts/session-15-directions-and-dials.md to its end. Read artifacts/notes/progress/prompts/shared-rules.md on every tick."
```

## Background

The project composes concepts in pretrained diffusion models. Paper 1 composes experts trained apart.
Paper 2 corrects how one frozen text-to-image model draws two subjects. Both rest on one equation: a
product of experts drops an interaction term, and the exact joint score is the product plus the
gradient of how much more likely two concepts are together than apart.

Several attempts to supply that term while sampling have failed or stalled. A checker's push could fix
parts on two bodies already drawn but could not split one hybrid body. Three readers of "which animal is
where" scored below simply knowing which pair it was. A classifier trained to get the answer right need
not have a gradient that points anywhere useful. On a pair nobody saw together, the data says only
"never seen together", so the direction must come from an assumed structure. A toy showed that a
low-rank structure reaches such a pair and that no structure cannot. The joint itself is not ground
truth either: the joint prompt loses a subject through overlapping attention.

The owner's goal links the two papers. It is to represent and guide pretrained models by the
**strength** of abstract concepts, as dials. For example, a strength s from 1 to 10 that makes two
subjects more or less co-occurring. Or a count read from attention maps, so that "dog" times 2 gives two
dogs and "cat" times 3 gives three cats. Or a strength for how firmly an attribute binds to its object.
This session reads what is known about learning such directions, and where the limits are, so the next
experiments are chosen from the literature rather than by trial.

Your owner code on the board is `s15`, and your steps are H1 to H4.

## Objectives

1. **Step H1:** write `artifacts/notes/progress/directions/01-directions-that-generalise.md`, on
   what makes a learned direction useful for steering and when it carries to new subjects. Cover robust
   classifier guidance and perceptually aligned gradients, concept sliders, semantic directions inside
   the U-Net (the "h-space" work), and any work that learns generic scene relations (occlusion,
   lighting, separateness, counting) as directions.
2. **Step H2:** write `artifacts/notes/progress/directions/02-shared-representations.md`, on where
   text, latents and scores share structure that a dial could act on. Cover prompt weighting and
   embedding arithmetic, attention-map control (reweighting, counting, binding), score arithmetic
   (composition, negation, logical guidance), symbolic and neurosymbolic composition, and the sampler
   mathematics of composition (stochastic differential equations, sequential Monte Carlo and
   Feynman-Kac correctors, energy-based composition, sampler error against model error).
3. **Step H3:** write `artifacts/notes/progress/directions/03-dials.md`, on concept strength as a
   continuous control. For each of the owner's three dials (co-occurrence strength, count, attribute
   binding strength), give what exists, whether the outcome rises steadily with the strength (a
   strength-against-outcome curve with its source), and where it breaks.
4. **Step H4:** write `artifacts/notes/progress/directions/04-limits-and-what-to-test.md`. Say what is
   learnable in principle, what the project's approaches cannot do, and what changes when the joint
   itself is wrong. Rank three experiments that link Paper 1 and Paper 2, each with its first small
   test, its cost and what its result would decide. File one board question listing them.
5. Send each finished step to the manager with one line in `artifacts/notes/progress/requests.md`
   addressed to `s13`.

## What to read

- `artifacts/notes/progress/prompts/shared-rules.md`: the rules, read on every tick.
- `artifacts/notes/progress/manager/ratio-classifier-idea.md`: the equation and what it does not get
  for free.
- `artifacts/notes/progress/reviewer/2026-11-24-review.md`, the sections "The ratio classifier,
  2026-11-25", "The fourth arm, 2026-11-25" and "The last identity reader, 2026-11-25": prior art
  already found (CFGComposer, Logical Guidance, Lift Scores, Readout Guidance, Discriminator Guidance,
  Reduce Reuse Recycle) and why unseen pairs are not learnable from data alone.
- `artifacts/results/can-a-classifier-reach-an-unseen-pair/README.md`: the toy, with its numbers.
- `artifacts/notes/progress/leakage/08-first-correction-test/review.md`: the checker push on plain PoE,
  the joint prompt and CO3.
- `artifacts/notes/progress/method-search/07-a-checker-that-reads-identity/review.md` and
  `08-size-corrected-identity-reader/review.md`: the readers that scored below knowing the pair.
- `artifacts/notes/progress/gaps/01-is-paper-2-still-open.md` to `04-exploiting-dcfm.md`, and the
  manager's `who-counts-borrowed-parts-and-extra-subjects.md`.
- `artifacts/notes/progress/targets/dcfm-compositional-flow-matching-from-separate-silos.md` and
  `3-understanding-compositionality/artifacts/notes/coind-conditional-independence-loss/README.md`.
- `context/methods/01-the-product-of-experts/` and `context/methods/07-the-score-path/`: the project's
  own pages on the product and the score path.
- The reading tables already written, cited and never reread: `score-space/01-reading/reading-table.md`,
  `noise-text-critic/01-reading/reading-table.md`, `abstract-checkers/01-reading/reading-table.md` and
  `leakage/01-reading/mechanism-table.md`, all under `artifacts/notes/progress/`.
- `report/00-INDEX.md` and `context/00-INDEX.md` before stating what this project found or what a term
  means.
- The owner's reading profile at
  `/home/molef/.claude/projects/-home-molef-PhD-concept-composition/memory/paper-scout-profile.md`.

## What is already known

- The exact joint score equals the product's score plus the gradient of
  log p(A, B | x) / (p(A | x) p(B | x)) at the same noise level (the reviewer's section on the ratio
  classifier, from CFGComposer Eq. 9).
- On a pair no data holder saw, that ratio is zero, so a classifier faithful to the data pushes away
  from the pair (the same section, and the toy: share 0.000 within 2 spreads).
- A low-rank interaction head trained on seen cells reaches a held-out cell (0.844 against the true
  joint's 0.924), and fails when the true interaction is full-rank (0.120) (the toy's card).
- The checker push made no plain-PoE picture two clear (0 of 12) and spoiled 4 of 8 draws CO3 already
  drew right (Z8's review).
- Knowing only which pair a picture shows predicts two-clear at 0.757 on fresh seeds, above every
  identity reader tried, 0.632 to 0.688 (the reviewer's last identity reader section).
- The project's LoRA adapter, trained densely on the dropped term, drew two animals in 58 of 64 pictures
  of pairs it never trained on (`1-lora-correction/report/does-the-fix-reach-unseen-pairs/does-one-pooled-adapter-compose-on-pairs-it-never-trained-on.md`).
  30 of those 53 held-out pairs share an animal with a training pair.
- On digits, a push for the abstract "two digits" concept lifted composition by 36 pairs where a random
  push of the same size changed nothing (`report/00-INDEX.md`, the digit method against paste).
- These titles were suggested from memory and have not been checked against the project's notes:
  Kawar, Ganz and Elad on robust classifier guidance; Gandikota and colleagues on concept sliders; Kwon
  and colleagues on semantic directions in the U-Net's middle layer. Confirm each before citing it.

## When something is missing

- **A paper is behind a paywall or will not download.** Try its arXiv version and the authors' page
  first. If neither exists, record it as unread and carry on.
- **A claim needs a project number that no report page holds.** Ask `s13` with one line in
  `requests.md`, and write the claim as open until the answer comes.
- **A ranking in H4 depends on a choice only the owner can make**, such as which model the dials target.
  File it with `python3 scripts/progress_board.py ask "<question>" --holds H4 --default "<your recommendation>" --by s15`
  and finish the rest of H4.

## Rules

- Run no experiments and no GPU work. This session reads and writes notes only.
- Use the `paper-scout` skill to find and rank papers for each step.
- Write only to `artifacts/notes/progress/directions/`. Never edit another loop's files.
- Confirm every arXiv id against its title before citing it, and give a page for every number.
- Mark every claim as read from the paper, read from a summary, or inferred.
- Cite the reading tables listed above instead of rereading those papers.
- Spend no Claude API tokens from the owner's key.
- Send one short message to `s13` when each step lands, with its main table.
- Commit only the step's named files, holding the lock as `shared-rules.md` says.

## What to hand back

- `artifacts/notes/progress/directions/01-directions-that-generalise.md`.
- `artifacts/notes/progress/directions/02-shared-representations.md`.
- `artifacts/notes/progress/directions/03-dials.md`, with one row per dial.
- `artifacts/notes/progress/directions/04-limits-and-what-to-test.md`, with three ranked experiments and
  the board question that lists them.
- One line per finished step in `artifacts/notes/progress/requests.md`, addressed to `s13`.
- The proof for each step is its file existing with every row sourced, checked with
  `python3 scripts/progress_board.py show` after setting the step to done.

## What you will see

Along the way, each step's main table arrives as a message when it lands, with the number of papers
behind it. Each step should take a few hours.

At the end, the closing report in chat says what was asked in one sentence. It lists each file by
path. It states the outcome: which kinds of direction carry to new subjects and why, whether each of
the owner's three dials has a published strength-against-outcome curve, what is learnable in principle,
and the three ranked experiments. It names what is still unread or uncertain, and asks at most one
question.

The owner judges this from H3's dials and H4's three experiments, and the loop is done when H1 to H4
are green.
