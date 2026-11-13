# 🧭 Route: from the literature report to a scoped problem

Navigation: 📋 [TOC](#table-of-contents) | [Quick context](#quick-context-where-you-are) ➡️

| Field | Value |
|---|---|
| Position | step 0 of 8, nothing started |
| Goal | know what the problem is, so the experiments run on a pair chosen on purpose |
| Objective | all five decisions answered in writing, each with the number that decided it |
| Outcome | a scoped problem: the claim, the dataset with its label source per regime, the chosen pairs with counts, and the experiment table |
| Hands to | [the compose-and-score route](route-compose-x-rays-and-score-them.md), whose pair assumption this re-aims |

## Table of contents

- [Quick context: where you are](#quick-context-where-you-are)
- [What we know so far](#what-we-know-so-far)
- [What you will achieve](#what-you-will-achieve)
- [The route at a glance](#the-route-at-a-glance)
- [The two lanes](#the-two-lanes)
- [Schedule](#schedule)
- [Tasks](#tasks)
- [Instructions](#instructions)
- [Quizzes and feedback](#quizzes-and-feedback)
- [Later](#later)
- [Why this chain](#why-this-chain)
- [References](#references)

## Quick context: where you are

Navigation: ⬅️ [Top](#-route-from-the-literature-report-to-a-scoped-problem) | 📋 [TOC](#table-of-contents) | [Next](#what-we-know-so-far) ➡️

A literature search over sixteen disease pairs came back today, in
[the return folder](../artifacts/ideas/silicotb-conversation/routes/05-return/). Five decisions were
written before it ran, in [the decision document](../artifacts/ideas/silicotb-conversation/from-the-literature-to-an-experiment.md),
and none of them is answered. This route answers them, in order, and the answers re-aim
[the compose-and-score route](route-compose-x-rays-and-score-them.md), whose step 0 was never run and
whose chosen pair the co-occurrence counts have since unsettled.

The claim walk this all came out of sits at claim 1.9 in
[the claim ledger](../artifacts/ideas/silicotb-conversation/IDEA_MAP.md), 17 of 108 claims marked.
This route does not continue that walk; it uses what the walk settled.

## What we know so far

Navigation: ⬅️ [Quick context](#quick-context-where-you-are) | 📋 [TOC](#table-of-contents) | [Next](#what-you-will-achieve) ➡️

Every row is a fact some step below depends on. Nothing here is re-derived in a step.

| Fact | The number | Where it came from |
|---|---|---|
| The merged regime has no verified example on any modality | 7 candidates, all inferred; 0 radiograph captions | `routes/05-return/candidate-figures.md`, its search finding |
| A fifth state exists: both present, neither visible | 4 of 54 normal-radiograph patients had two infections | `routes/05-return/where-the-four-regimes-fail.md`, section 1 |
| One disease can change where the other appears | occluded segments spared from oedema, 1 case | `routes/05-return/where-the-four-regimes-fail.md`, section 2 |
| HIV with tuberculosis is a disease plus a modifier, not a pair | normal films 52/323 at CD4 ≤150, 26/1,085 HIV-negative | `routes/05-return/pairs-and-their-regimes.md`, section 2 |
| Nobody has published the four regimes, or three of them | closest is 2 of 4, in the CPFE statement's Table 10 | `routes/05-return/the-words-the-field-uses.md` |
| No textbook chapter teaches reading a film with two diseases | 7 volumes checked, 0 such chapters | `routes/05-return/the-textbook-shelf.md` |
| The pair with the published evidence is fibrosis with emphysema | combination-specific lesion in 16/22, vs 0/8 and 0/17 | `routes/05-return/candidate-figures.md`, closing note |
| That pair is too rare in our data to use | 36 of 112,120 films; neither class has boxes | run 7, `IDEA_MAP.md` |
| The pair with the data is effusion with atelectasis | 3,275 films in ChestX-ray14; 90 in VinDr, 78 after the drop | run 7 |
| And it is cause and consequence, not two diseases | 68 of 78 separate, median best overlap 0.000 | run 7 |
| Two regimes are measurable off boxes, two are not | 1,951/1,972 separate; 201/489 same region at a floor of 0.25 | runs 5 and 6 |
| VinDr has no tuberculosis and no silicosis class | 14 classes, neither among them | run 7 |
| Films per class needed to separate two methods | about 100; 25 against 25 gives an interval of ±0.14 | Hanley and McNeil, cited in the decision document |
| The baseline number we have been citing is suspect | 0.226 to 0.726 cannot come from a monotone recalibration | this route, step 6 |

## What you will achieve

Navigation: ⬅️ [What we know so far](#what-we-know-so-far) | 📋 [TOC](#table-of-contents) | [Next](#the-route-at-a-glance) ➡️

**Stage 1, the returns read (own).** You can say, without the files open, which regimes the
literature documents and which are ours, and name the one pair whose evidence is quantitative.
Worked by hand: map each of the sixteen pairs onto the four regimes on paper, then check against the
report's table. Quiz: nine questions on the regimes, the fifth state and the pairs.

**Stage 2, the claim and the scope decided (own).** You can state in one sentence what the paper
claims and which of the three things it does, with the count that forced each.

**Stage 3, the pipeline drawn (gloss).** A published page showing films to scored classifier, with
every undecided part dotted and linked to the fact that would decide it.

**Stage 4, the films seen (own).** You have opened films of the two candidate pairs and written what
you saw, before the pair is chosen. Worked by hand: call ten films' regime yourself, then compare
with the box query's call.

**Stage 5, the dataset and pairs decided (own).** The chosen pairs with counts, and the written
sentence saying which regimes this dataset cannot label.

**Stage 6, one external number in hand (build).** Either the recalibration result reproduced, or the
statement that the number was misread, with what the paper actually measured.

**Stage 7, the experiment table (build).** One row per run, each with the axis it varies, the
comparison it belongs to, and the three numbers that would support, null or leave it inconclusive.

**Stage 8, the argued figure (own).** The regime counts beside the films they came from, in a form a
supervisor can read without you in the room.

**Outcome.** A scoped problem you can defend: `plans/route-compose-x-rays-and-score-them.md` re-aimed
with its pairs and dataset filled in, and the experiment table it will walk.

## The route at a glance

Navigation: ⬅️ [What you will achieve](#what-you-will-achieve) | 📋 [TOC](#table-of-contents) | [Next](#the-two-lanes) ➡️

```
you:  read the returns ──→ decide 1 and 2 ──→ see the films ──→ decide 3 and 4 ──→ experiment table
        (steps 1a, 1b)        (step 2)          (step 5)           (step 6)          (step 7)
AI:   ‖ draw the pipeline (step 3) ‖ inventory the data (step 4) ‖ reproduce the number (step 6b) ‖
close: the argued figure (step 8) ──→ register the pages ──→ retrospective and quiz
```

Spliced from flow 1 (idea to provable claim), flow 20 (the experiment spine, its front half only) and
flow 3 (experiment output to figures, its `evidence-ladder` and `pair-figure` branches). The front of
flow 20 is skipped: no hypothesis is framed here, because what the hypothesis is about is what this
route decides.

Alternatives, bare:

- `/drip-idea --prompt` on this route's own sentence first, then re-route. Better when you think the
  framing is wrong rather than the decisions.
- `init-master-plan` then `populate-plans` on a new scope. Better if the decisions turn out to need
  three weeks rather than two days.

## The two lanes

Navigation: ⬅️ [The route at a glance](#the-route-at-a-glance) | 📋 [TOC](#table-of-contents) | [Next](#schedule) ➡️

| Lane | What it does | Steps |
|---|---|---|
| Yours | reads, decides, looks at films, calls regimes by eye, writes the sentences | 1a, 1b, 2, 5, 6, 7, 8 |
| The AI's | draws the pipeline, inventories and renders the data, runs the reproduce | 3, 4, 6b |

They meet twice. At step 5 your by-eye regime calls on ten films are compared with the box query's
calls, and a disagreement is a finding about the measure. At step 7 the experiment table has to
account for every row of the pipeline drawing from step 3, so a box nobody is testing is either cut
from the drawing or added to the table.

## Schedule

Navigation: ⬅️ [The two lanes](#the-two-lanes) | 📋 [TOC](#table-of-contents) | [Next](#tasks) ➡️

Today is the decisions. The reproduce and the figure are a second round, because the two together are
another full day and shrinking either would make the answer not worth having.

| Block | Run | You, while it runs | Checkpoint |
|---|---|---|---|
| 1, about 2h40 | steps 1a and 1b, with step 3 dispatched at the open | step 3's drawing arrives mid-block | all six returns read; the pipeline page published |
| 2, about 1h20 | step 2, with step 4 dispatched at the open | 📐 CFG as base plus omega times a difference, 20 min | decisions 1 and 2 written, with their numbers |
| 3, about 1h | step 5, on the rendered grids from step 4 | — | ten films called by eye and compared with the query |
| 4, about 1h20 | step 6 | — | decisions 3 and 4 written |
| 5, about 2h | step 7 | — | the experiment table, every row with its three numbers |
| 6, about 30 min | the close: register, retrospective, quiz | — | pages linked; `drip-retro` run |
| second round, about 8h20 | steps 6b and 8 | — | one external number, and the argued figure |

Totals: completion about 11h of your attention (estimate, from pieces times 30 minutes plus half
again for tangents); budget about 17h at 2× for learning; AI about 2h, nearly all of it beside your
work. Today's half is about 8h25.

**If the day runs short**, stop after block 4. Decisions 1 to 4 are the ones that change what gets
built; the experiment table can be written against them in a second sitting without re-reading
anything.

## Tasks

For Claude to execute. Ask Claude to do these.

Navigation: ⬅️ [Schedule](#schedule) | 📋 [TOC](#table-of-contents) | [Next](#instructions) ➡️

### 1a. 📖 Read the four returns that carry the regime evidence

- [ ] **Walk the four evidence files as one idea**
  - you ~45 min · AI ~15 min · Sonnet, medium
  - objective: every one of the four regimes has a written verdict saying whether the literature
    documents it, and for which pair
  - outcome: `artifacts/drips/what-two-diseases-do-on-one-film/COMBINE_MAP.md`
  - serves: stage 1 of the Achieve list, and the evidence decision 1 rests on

  ```text
  /drip-read --combine --lean
  artifacts/ideas/silicotb-conversation/routes/05-return/pairs-and-their-regimes.md
  artifacts/ideas/silicotb-conversation/routes/05-return/candidate-figures.md
  artifacts/ideas/silicotb-conversation/routes/05-return/one-pair-many-regimes.md
  artifacts/ideas/silicotb-conversation/routes/05-return/where-the-four-regimes-fail.md

  The idea to read across all four: how two chest diseases sit relative to each other in one
  chest, and which of those arrangements anyone has actually published a picture of.

  Starts from: these four files are one literature search, run today, over sixteen disease
  pairs. They are not papers; they are a report that cites papers, and every claim in them
  carries a section or figure reference. Read them as four views of one question, not in any
  file's own order. The four arrangements they are written against are set out in
  artifacts/ideas/silicotb-conversation/routes/05-deep-research-the-disease-pairs.prompt.md,
  which also lists the ten sources we already held before the search ran.

  Settled facts, carried whole from artifacts/ideas/silicotb-conversation/IDEA_MAP.md and the
  return files:
  - separate places, measured on our own films: 1,951 of 1,972 films, 98%, and the share does
    not move at any box-overlap floor
  - same region, measured: 201 of 489 films, 41%, at an overlap floor of 0.25, median 0.405
  - the merged arrangement and the one where a disease is present but unreadable cannot be read
    off annotation boxes at all
  - the report found 7 candidates for the merged arrangement and verified none, and found no
    chest radiograph caption establishing it
  - the report proposes a fifth state, both diseases confirmed and neither visible, from 4 of 54
    normal-radiograph patients carrying two simultaneous infections
  - the pair with quantitative evidence is pulmonary fibrosis with emphysema: a lesion specific
    to the combination in 16 of 22 patients, against 0 of 8 with fibrosis alone and 0 of 17 with
    emphysema alone

  Outputs: artifacts/drips/what-two-diseases-do-on-one-film/COMBINE_MAP.md, with one comparison
  table per arrangement: the pairs documenting it, the pairs where it was inferred rather than
  documented, and the pairs where it is absent.

  Target: all four arrangements have a verdict, and the fifth state has a written decision on
  whether it is a fifth arrangement or an exclusion from the whole framework's scope.

  Progress: the map's position table shows arrangements settled out of four.

  Proof: read the map's four verdicts back against the report's own self-audit rows. A verdict
  the self-audit contradicts is a misreading, not a finding.

  Limits: these files are a report, so a claim in them is only as good as the citation beside
  it. Where a claim carries no citation, mark it as the report's own inference. Patient data
  lives only on the cluster and no step here touches it.

  Hand-off: the map is what step 2 reads to decide what the paper claims.

  Chain: a scoped problem · this step delivers the regime evidence decisions 1 and 4 rest on
  Learning path:
    composition, rung 1 of 4: 📐 Girsanov: the path KL is a control cost,
      goal-setting/learning/deep-learning/diffusion-models/04-schrodinger-bridges/plans/
      02-girsanov-path-kl-is-a-control-cost.md, 0 of 10 ticked
    (rungs 2 to 4 are listed under step 7, which is where composition is actually needed)
  Controls:
    the arrangements piece: prove, on whether the merged arrangement exists at all
    the fifth-state piece: break, before accepting it as a fifth rather than an exclusion
    the pairs piece: example, on the one pair with a quantitative comparison
    at the close: test, for stage 1's quiz
  Side routes:
    the fifth-state piece: /picture-speak --take-apart "the four arrangements and what each one
      requires to be visible", when the definitions start colliding
  ```

  - gate: the map's four verdicts exist before step 1b opens
  - moves: Achieve stage 1
  - leaves: `COMBINE_MAP.md` for steps 2 and 6
  - added: the six settled facts with their numbers (from `IDEA_MAP.md` runs 5 to 7 and the return
    files), the output path, and the proof against the self-audit rows
  - dropped: the two returns about vocabulary and textbooks (why: they carry no regime evidence, so
    they are step 1b, where the question is different)

### 1b. 📖 Read the two returns that say whether this is anybody else's framework

- [ ] **Walk the vocabulary and textbook returns**
  - you ~35 min · AI ~10 min · Sonnet, medium
  - objective: a written answer on whether the four arrangements are ours alone, and what the field
    calls each one instead
  - outcome: the same `COMBINE_MAP.md`, extended with a naming table
  - serves: stage 1, and the part of decision 1 about whether we own the axis

  ```text
  /drip-read --combine --lean
  artifacts/ideas/silicotb-conversation/routes/05-return/the-words-the-field-uses.md
  artifacts/ideas/silicotb-conversation/routes/05-return/the-textbook-shelf.md

  The idea to read across both: whether anyone outside this project has a name for two diseases
  in one chest, and whether anyone teaches how to read such a film.

  Starts from: artifacts/drips/what-two-diseases-do-on-one-film/COMBINE_MAP.md, written by the
  previous step, which holds the verdict per arrangement. Extend that map rather than starting a
  new one.

  Settled facts:
  - no source names all four arrangements, and none names three
  - the closest is the official statement on combined pulmonary fibrosis and emphysema, which
    distinguishes separated from intermingled distributions, matching two of ours without their
    requirement that both diseases stay readable
  - seven textbook volumes were checked with confirmed chapters and pages, and none holds a
    chapter on reading a film carrying two diseases
  - our own words comorbidity and superimposed disease are in use elsewhere with wider meanings,
    and compound disease is not a radiology term

  Outputs: a naming table appended to COMBINE_MAP.md: one row per arrangement, the terms the
  field uses for it, the specialty that uses each, and one source per term.

  Target: every arrangement and every one of our five vocabulary words either mapped onto an
  existing term with its source, or written as unmapped.

  Progress: the naming table's unmapped count.

  Proof: for each mapped term, the cited source has to use it for the arrangement we mapped it
  to, not merely mention it. Check three at random by opening the citation.

  Limits: a term that means something else in standard use is mapped as a clash, never as a
  match. Do not invent a name for an unmapped arrangement in this step; that belongs to the
  paper.

  Hand-off: the naming table is what decision 1 uses to answer whether we own the axis.

  Chain: a scoped problem · this step delivers the naming half of decision 1
  Controls:
    the naming piece: known, on whether a term exists that the search missed
    the textbook piece: prove, on the claim that no such chapter exists
  Side routes:
    the naming piece: /drip-idea --prompt, if the naming turns out to undermine the framework
      rather than just leaving it unnamed
  ```

  - gate: the naming table exists before step 2 opens
  - moves: Achieve stage 1
  - leaves: the extended `COMBINE_MAP.md`
  - added: the four settled facts and the output shape
  - dropped: nothing

### 3. 🚀 Draw the pipeline, with every undecided part dotted

- [ ] **Draw films to scored classifier, as a design**
  - you ~15 min · AI ~25 min, in the background · Sonnet, medium
  - objective: one page where every box that depends on an unmade decision is drawn dotted and
    linked to the fact that would decide it
  - outcome: a published page and its SVG under `artifacts/scenes/`
  - serves: stage 3, and it is what step 7's table has to account for row by row

  ```text
  /hld-diagram --design "the pipeline from chest films we hold to a scorecard that says whether
  a composed image of two diseases is any use: which dataset, where the regime label comes from,
  which composition operator, and what the classifier is scored on"

  Starts from: nothing of this is built. Three pages already exist for this repo in the HLD
  register and none is this pipeline, so this is a new draw rather than an edit. Read, before
  drawing: artifacts/ideas/silicotb-conversation/from-the-literature-to-an-experiment.md for the
  five decisions and their options, and plans/route-compose-x-rays-and-score-them.md for the
  chain that would run once they are decided.

  Settled facts that fix some boxes solid:
  - the films are VinDr-CXR, 15,002 radiographs at /datasets/mmolefe/vinbigdata/, 161 GB, every
    training film read by exactly three radiologists with boxes carrying the reader's identity,
    and NIH ChestX-ray14, 112,120 films with 14 labels and 984 boxes over 880 films
  - two of the four arrangements are measurable off those boxes and two are not
  - the label source is boxes only, by decision; reports and a radiologist's reading are out of
    scope for this route

  Outputs: a published private page, its SVG saved under
  artifacts/scenes/the-two-disease-pipeline/, and the register entry.

  Target: every box is solid with its source named, or dotted with the decision number it waits
  on, one to five. No box is drawn without one or the other.

  Progress: the count of dotted boxes, which should fall as the decisions close.

  Proof: read the dotted boxes against the five decisions. A decision with no dotted box means
  the drawing missed it, and a dotted box with no decision means the drawing invented a choice
  nobody is making.

  Limits: the pipeline runs on a cluster, so draw the hardware level under the software one.
  Patient data lives only on the cluster and the drawing names paths without reading films.

  Hand-off: the dotted boxes are the checklist steps 2, 6 and 7 close.
  ```

  - gate: none; it runs beside step 1
  - moves: Achieve stage 3
  - leaves: the page URL and SVG path for step 7
  - added: the dataset facts with sizes and paths, the dotted-box rule, the hardware level
  - dropped: the MIMIC-CXR and radiologist options (why: decided out of scope, so drawing them
    would show choices nobody is making)
  - after it publishes, its own closing step: write the page's section in the scene library, then
    `scene-cards.py --apply`, `link-concepts.py --refresh-marks --apply` and
    `build-artifact-index.py --apply`, and check the register's `Linked from` column no longer
    reads **none**

### 4. 🚀 Inventory the two candidate pairs' films, then render them

- [ ] **Count and render, read-only**
  - you ~20 min · AI ~30 min, in the background · Sonnet, medium
  - objective: a sample grid and an extremes grid for each candidate pair, so step 5 has films to
    look at rather than counts to trust
  - outcome: `01-inventory.md` and `03-visual.md` with their figures
  - serves: stage 4, and decision 4 is made after it rather than before

  ```text
  /data-inventory /datasets/mmolefe/vinbigdata/train

  Then, once 01-inventory.md exists:
  /visualize-data-samples /datasets/mmolefe/vinbigdata/train

  Starts from: the films are DICOM, MONOCHROME1, so the raw array displays inverted and has to be
  flipped before anything is drawn. Box coordinates in train.csv are in original pixel space, so
  any resize applies to the boxes too. Every count and every render runs through
  scripts/cluster.sh sh "<command>" from /home/molef/PhD/poe_repair_min-dog-and-rain with
  CLUSTER_DIR=/home-mscluster/mmolefe/Playground/PhD/poe_repair_min-dog-and-rain, and the
  finished figures come back only with scripts/cluster.sh pull.

  Settled facts: 15,002 DICOM files, 161 GB, 14 finding classes, three readers per training film.
  The two pairs the grids are for: pleural effusion with atelectasis, 90 films carrying both and
  78 after dropping the 12 where one reader drew one box and attached both labels; and lung
  opacity with pulmonary fibrosis, 489 films after its own drop of 177.

  Outputs: data/vinbigdata/eda/01-inventory.md and 03-visual.md, and under
  artifacts/results/the-two-candidate-pairs/ a sample grid of 12 films per pair with both
  findings' boxes drawn, and an extremes grid showing the largest and smallest overlap in each
  pair.

  Target: 24 films rendered per pair, boxes drawn in the original pixel space, each labelled with
  its image id and which reader drew which box.

  Progress: the figure count under artifacts/results/the-two-candidate-pairs/.

  Proof: pick three rendered films and check the drawn boxes against train.csv by hand. A box in
  the wrong place means the rescaling is wrong and every grid is useless.

  Limits: read-only on the dataset. No film is committed; artifacts/results/ is gitignored, so
  the card is what makes a picture findable again. Do not download model weights.

  Hand-off: the grids are what you look at in step 5.
  ```

  - gate: step 1 has opened; this runs beside it
  - moves: Achieve stage 4
  - leaves: the grids for step 5
  - added: the MONOCHROME1 and box-rescaling facts, the cluster command shape, the film counts per
    pair, the by-hand box check
  - dropped: ChestX-ray14's films (why: its boxes cover 8 classes and 880 films, and neither
    candidate pair's regime split can be read from them)

### 6b. 🚀 Find out what the recalibration paper actually measured, then reproduce it

- [ ] **Reproduce it, or establish the number was misread**
  - you ~6h · AI ~45 min · Opus, high
  - objective: either the reported jump reproduced on films we hold, or a written statement of what
    the paper measured and why the number cannot be what we have been citing
  - outcome: a plan file, a results card, and `EXECUTE_WALK.md`
  - serves: stage 6, and it decides whether the baseline row of step 7's table is real
  - **second round, not today**

  ```text
  /drip-execute-plan --reproduce https://arxiv.org/abs/2609.07922

  The target: the reported rise in classifier performance from recalibrating for prevalence on
  films whose disease mix does not match the training set, which we have been citing as 0.226
  rising to 0.726.

  Starts from: that number cannot be what we have been citing. The area under the ROC curve is
  rank-based, so a recalibration that preserves the order of the scores cannot change it at all,
  and 0.226 is worse than guessing. So the first round of this walk reads the paper to establish
  what was recalibrated and what was measured, before anything is run. If it turns out the metric
  is not the area under the ROC curve, or the recalibration is not order-preserving, say so and
  reproduce what the paper actually did. If the number is a misreading on our side, that is the
  result and the walk closes on it.

  Settled facts:
  - we hold VinDr-CXR, 15,002 films with three readers and boxes, and NIH ChestX-ray14, 112,120
    films with 14 labels
  - about 100 films per class are needed before two methods differing by less than 0.1 can be
    told apart; 25 against 25 gives an interval of about ±0.14 at a true value of 0.75
  - this number is the baseline every generative method in this project has to beat, so whether
    it is real decides whether the comparison exists

  Outputs: plans/reproduce-the-prevalence-recalibration.md, its review file, and
  artifacts/results/what-recalibrating-for-prevalence-actually-does/ with the card and the
  numbers.

  Target: each reproduced number within 30% of the paper's, with the curve rising and falling
  where theirs does; or a written statement of what the paper measured, with the passage quoted.

  Progress: the plan's ticked tasks, and the results card's filled rows.

  Proof: a control that must not move. Score the same classifier on an unshifted test set before
  and after recalibration; if the metric moves there, the implementation is wrong rather than the
  paper.

  Limits: every run goes through scripts/cluster.sh. Ask before any model weights are downloaded
  onto a node. Patient data stays on the cluster.

  Hand-off: the card fills the baseline row of the experiment table in step 7.

  Chain: a scoped problem · this step delivers the baseline the experiment table compares against
  Learning path:
    AUROC: no journey teaches it, expect a gap line. The first rung to write is what the area
      under the ROC curve is invariant to, since that is exactly what this step turns on.
  Controls:
    the reading piece: prove, on whether the metric can move under an order-preserving map
    the implementation piece: build, since you type the scoring yourself
    the control piece: break, on the unshifted test set that must not move
  Side routes:
    the metric piece: /plumb "the area under the ROC curve, computed two ways on the same
      scores", to see the invariance rather than argue it
  ```

  - gate: steps 1 to 7 are done, or at least decision 2 is, since what the baseline is for depends
    on what the paper claims
  - moves: Achieve stage 6
  - leaves: the results card for step 7's baseline row
  - added: the invariance objection, the control that must not move, the dataset facts, the
    tolerance
  - dropped: the FCFG reproduction (why: it scores on MIMIC-CXR, which this route put out of
    scope, so it cannot carry a bar here)

### 7. 🔧 Write the experiment table

- [ ] **One row per run, each with its three numbers**
  - you ~2h · AI ~20 min · Opus, high
  - objective: every candidate experiment written as a row whose comparison differs in exactly one
    axis, with the number that would support it, the number that would be a null, and the band
    between
  - outcome: a plan file carrying the table
  - serves: stage 7, and it is decision 5 answered

  ```text
  /drip-execute-plan --request "write the experiment table for the two-disease composition work:
  one row per run, each with the single axis it varies, the comparison it belongs to, and the
  three numbers that would support the claim, be a null, or leave it inconclusive"

  Starts from: artifacts/ideas/silicotb-conversation/from-the-literature-to-an-experiment.md,
  decision 5, which lists five candidates and the rule for ordering them. The pipeline drawing
  from step 3 is the checklist: every box in it is either tested by a row of this table or cut
  from the drawing. Decisions 1 to 4, written in the same document by then, fix the claim, the
  scope, the dataset and the pairs, so no row may assume a pair the decisions did not choose.

  Settled facts:
  - about 100 films per class are needed before two methods differing by less than 0.1 can be
    separated, so a row whose cell counts fall below that says what it cannot conclude
  - the share of real both-disease films the generator may see, stepped 100, 25, 5, 1 and 0
    percent, is the axis composition can win on, because composition only beats training on the
    pair directly where the pair is too rare to fit
  - editing a film to add the second disease never needs both-disease films, so it has no such
    axis and cannot be compared along it
  - two of the four arrangements cannot be labelled from boxes, so no row may score a model on
    the merged arrangement or on the one where a disease is present but unreadable
  - the cheapest comparison that could kill the premise goes first: if the plain product of two
    experts already composes two findings correctly where they sit in separate places, the
    correction has nothing to do there

  Outputs: plans/the-two-disease-experiment-table.md, with the table and one paragraph per row
  saying what the row is for, and the five decisions' answers copied into
  artifacts/ideas/silicotb-conversation/from-the-literature-to-an-experiment.md.

  Target: every row has exactly one axis differing from its comparison, and three numbers. A row
  that cannot state its three numbers is not ready and is written into a later list instead.

  Progress: rows with three numbers, out of rows written.

  Proof: read each row's comparison and name the second thing that differs. If you can name one,
  the comparison is mixed and the row says so rather than claiming cause and effect.

  Limits: no run is launched from this step. Patient data stays on the cluster. A row naming a
  pair the decisions did not choose is a defect, not an extension.

  Hand-off: the table is what /drip-idea --hypothesis turns into a falsification criterion, one
  part per round, and what plans/route-compose-x-rays-and-score-them.md is re-aimed against.

  Chain: a scoped problem · this step delivers decision 5 and the table
  Learning path:
    composition, rung 2 of 4: 📐 Doob's h-transform: conditioning a process on its end,
      04-schrodinger-bridges/plans/04-doobs-h-transform.md, 0 of 7 ticked
    composition, rung 3 of 4: 📐 Diffusion as a one-sided bridge,
      04-schrodinger-bridges/plans/05-diffusion-as-a-one-sided-bridge.md, 0 of 5 ticked,
      with CME296 lecture 2, 1:25:14 to 1:30:32
    composition, rung 4 of 4: 🔨 Flow matching on two moons,
      04-schrodinger-bridges/plans/06-flow-matching-on-two-moons.md, 0 of 6 ticked,
      with CME296 lecture 3, 46:16 to 47:58
    guidance, rung 1 of 1: 📐 Classifier-free guidance as base plus omega times a difference,
      05-poe-composition-diffusion/plans/02-cfg-as-base-plus-omega-difference.md, 0 of 3 ticked,
      with CME296 lecture 4, 1:27:42 to 1:34:34
  Controls:
    the ordering piece: prove, on which comparison could kill the premise most cheaply
    the editing piece: break, on whether it belongs in the table at all
    the counts piece: example, on one row's cell counts worked through
    at the close: test, for stage 7's quiz
  Side routes:
    the operators piece: /picture-speak --consequence "the product of two experts with the
      correction term switched off", on the one row where a term can be removed
  ```

  - gate: decisions 1 to 4 are written before this opens
  - moves: Achieve stage 7
  - leaves: the table for `/drip-idea --hypothesis` and for the compose-and-score route
  - added: the five settled facts with their numbers, the one-axis proof, the ban on rows scoring
    unlabellable arrangements
  - dropped: the runs themselves (why: this step writes the table, and running it is the
    compose-and-score route's work)

### 8. 🔧 Put the counts beside the films they came from

- [ ] **Stage the scoping result as claims in order**
  - you ~1h40 · AI ~40 min · Sonnet, medium
  - objective: each claim of the scoping result shown as a chart beside the real films it rests on
  - outcome: paired step figures and their card under `artifacts/results/`
  - serves: stage 8
  - **second round, not today**

  ```text
  /evidence-ladder "two chest diseases in one chest sit in four arrangements; two of them can be
  read off annotation boxes and two cannot; the pair with the published evidence is too rare in
  our data to use, and the pair with the data is cause and consequence rather than two diseases"

  Starts from: artifacts/drips/what-two-diseases-do-on-one-film/COMBINE_MAP.md for the literature
  verdicts, artifacts/ideas/silicotb-conversation/IDEA_MAP.md runs 5 to 7 for the measured counts,
  and artifacts/results/the-two-candidate-pairs/ for the rendered films from step 4.

  Settled facts, each of which is one rung: separate places 1,951 of 1,972 films at 98%; same
  region 201 of 489 at 41% with median overlap 0.405; pulmonary fibrosis with emphysema 36 of
  112,120 films; pleural effusion with atelectasis 3,275 films in one dataset and 78 in the
  other, 68 of those 78 separate with median overlap 0.000.

  Outputs: artifacts/results/how-two-diseases-sit-in-one-chest/, each rung as a chart and the
  real films beside it, with the card naming what each number means and where it came from.

  Target: four rungs, each with a chart and at least two real films, and every number on a chart
  carrying its unit and its denominator.

  Progress: the rung count under that folder.

  Proof: each chart's number has to be recomputable from the file the card names. Pick one rung
  and recompute it.

  Limits: artifacts/results/ is gitignored, so the card is the record. Films come back only with
  scripts/cluster.sh pull, and none is committed.

  Hand-off: the ladder is what a supervisor reads, and what /pair-figure extends if one rung
  needs its own figure.
  ```

  - gate: steps 1, 4 and 6 are done
  - moves: Achieve stage 8
  - leaves: the ladder's card
  - added: the four rungs with their numbers and denominators, the recompute proof
  - dropped: nothing
  - if one rung needs a figure of its own: `/pair-figure the 68 of 78 films where pleural effusion
    and atelectasis do not overlap, against the clinical account that one causes the other`

### 9. 🔄 Close out

- [ ] **Register the pages, then explain it back**
  - you ~30 min
  - objective: every page built here is reachable from something a reader meets, and you can state
    the five decisions without the files open
  - outcome: the register's links, and a retrospective

  - Write the step 3 page's section in the scene library, then run `scene-cards.py --apply`,
    `link-concepts.py --refresh-marks --apply` and `build-artifact-index.py --apply`. The check is
    the register's `Linked from` column reading something other than **none**.
  - Dispatch the `xref-cadence` agent over the files this route wrote: the combine map, the decision
    document, the experiment table, the ladder's card.
  - `/drip-retro` on this route, which the last walk's close prints.
  - `/quiz-me` over the retrospective, nine questions, three per stage closed.

## Instructions

For you to follow manually. Do these yourself.

Navigation: ⬅️ [Tasks](#tasks) | 📋 [TOC](#table-of-contents) | [Next](#quizzes-and-feedback) ➡️

### 2. 📈 Decide what the paper claims, and what it does

- [ ] **Write decision 1**
  - Open [the decision document](../artifacts/ideas/silicotb-conversation/from-the-literature-to-an-experiment.md)
    at decision 1, and the combine map from step 1 beside it.
  - Count the filled cells in the report's pairs table, keeping figure-supported cells separate from
    prose-inferred ones. The four rows of decision 1's table each say what the claim becomes.
  - Write the claim as one paragraph, in the document, under decision 1. Say which row of the table
    the count put you in.
  - ✅ Done when the paragraph names a count and a row.

- [ ] **Write decision 2**
  - Three options: synthesise for a pair nobody has and score a classifier on real films of it;
    synthesise along the share-of-real-films axis; or measure what the product drops and synthesise
    nothing.
  - What decides it is whether any pair has about a hundred real both-disease films that could be
    held out without destroying the training set. Run 7's counts are in the document.
  - ✅ Done when the chosen option is one sentence with the count that made it possible or not.
  - ▶ Next: step 5, after the grids from step 4 arrive.

### 5. 📊 Call ten films' arrangement by eye, before the pair is chosen

- [ ] **Look at the grids**
  - Open the sample grids from step 4 under `artifacts/results/the-two-candidate-pairs/`, pulled back
    with `scripts/cluster.sh pull`.
  - For ten films of pleural effusion with atelectasis, write down which of the four arrangements you
    think it is, from the film alone, before looking at the box query's call.
  - Then compare with the query. The query calls 68 of 78 separate places.
  - ✅ Done when ten calls are written and the disagreement count is recorded. **A disagreement is a
    finding about the measure, not a mistake of yours.** More than three reversals in ten means the
    box query is measuring boxes rather than arrangements, which is the check claim 1.10 left
    outstanding.

### 6. 📈 Decide the dataset and the pairs

- [ ] **Write decision 3**
  - Boxes only, by your decision. So write the sentence saying which arrangements this dataset
    cannot label and what the paper therefore cannot claim.
  - ✅ Done when that sentence is in the document.

- [ ] **Write decision 4**
  - Four criteria are in the document. The forced choice is also there: no pair carries both the
    literature and the data.
  - Your by-eye calls from step 5 are evidence here, not just the counts.
  - ✅ Done when the chosen pairs are listed with their counts and, for each, the sentence carrying
    it to silicotuberculosis.
  - ▶ Next: step 7.

## Quizzes and feedback

Navigation: ⬅️ [Instructions](#instructions) | 📋 [TOC](#table-of-contents) | [Next](#later) ➡️

| When | What | How |
|---|---|---|
| after step 1b | nine questions on the arrangements, the fifth state, and the pairs | the walk's `test` control, which dispatches `quiz-cadence` |
| after step 7 | nine questions on the axes, the one-axis rule and the three numbers | the walk's `test` control |
| after step 9 | the retrospective's own questions, each asking your prediction first | `/quiz-me` over the retrospective |

Run `quiz-prep-pulse` on each quiz before it is built, so no question asks for something nothing in
your material teaches. The area under the ROC curve has no journey, so its questions will have no
preparation until that gap is filled.

**The confidence bar before you write a decision:** every question that the decision depends on
answered right. Below it, the quiz's misses become the next reading's first rungs.

## Later

Navigation: ⬅️ [Quizzes and feedback](#quizzes-and-feedback) | 📋 [TOC](#table-of-contents) | [Next](#why-this-chain) ➡️

- **MIMIC-CXR credentialing.** The only label source that can carry the merged and hidden
  arrangements. Out of scope by decision, and the sentence decision 3 writes is what records that.
- **A radiologist reading our films.** The only way to get all four arrangements on films we hold.
- **The fifth state as a fifth arrangement.** Step 1a decides whether it is one or an exclusion from
  the framework's scope; either way, pursuing it is its own question.
- **The exclusion-and-redistribution failure.** One disease changing where the other can appear.
  It breaks the separate-places arrangement and has one case behind it.
- **The area under the ROC curve has no journey.** Write its first rung: what the measure is
  invariant to. Step 6b turns on exactly that, so it is owed before the second round.
- **Reproducing FCFG.** Out of reach while MIMIC-CXR is out of scope.
- **The Codex job `20261013-082911-405afd`** still waits for its check step, unrelated to this route.

## Why this chain

<details>
<summary>Why the decisions come before any run, and why the reproduce is last</summary>

The compose-and-score route was written before the counts were taken. It names a pair, a dataset and
an operator, and run 7 unsettled the first two. Running it now would produce a confident scorecard
for a pair chosen by accident, which is the failure the experiment conventions exist to prevent.

The reproduce is last among today's work and first in the second round because its value changed. It
was going to establish the baseline; it is now a check on whether the baseline we have been citing
exists. That is worth more, and it is worth less urgently, since no row of the experiment table needs
the number before the table is written.

Looking at the films sits between decisions 2 and 3 rather than at the start, because what to look
for depends on what the paper claims. Before decision 1 the grids would be a browse; after it they
are a check on the one measure the whole scoping rests on.

</details>

## References

- [the five decisions](../artifacts/ideas/silicotb-conversation/from-the-literature-to-an-experiment.md)
- [the six returns](../artifacts/ideas/silicotb-conversation/routes/05-return/)
- [the search prompt they came from](../artifacts/ideas/silicotb-conversation/routes/05-deep-research-the-disease-pairs.prompt.md)
- [the claim ledger, with runs 5 to 7](../artifacts/ideas/silicotb-conversation/IDEA_MAP.md)
- [the chain this re-aims](route-compose-x-rays-and-score-them.md)
- [the composition vocabulary](../artifacts/ideas/poe-overlap-dial/composition-terms.md)
- [the earlier combined reading on what composition means](../artifacts/drips/combine-kinds-of-composition/COMBINE_MAP.md)
