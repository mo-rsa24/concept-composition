# Session 14: is Paper 2's problem still open, and what will reviewers ask of Paper 1

This runs as a background session, in the same permission mode as the other loops. To start it by
hand from `/home/molef/PhD/concept-composition`:

```text
claude --bg --permission-mode auto --model claude-opus-5-5 --effort high --name "🔭 PhD Loop 14: Gaps And Reviewer Questions" "/loop Follow artifacts/notes/progress/prompts/session-14-gaps-and-reviewer-questions.md to its end. Read artifacts/notes/progress/prompts/shared-rules.md on every tick."
```

## Background

The project is writing two papers. Paper 1 composes diffusion experts that were trained apart and
never saw their concepts together. Paper 2 corrects how a single text-to-image model draws two
subjects, with the model left frozen. Both need to stand on their own as contributions, and the
owner wants every direction kept or dropped on evidence rather than habit.

Two questions have no answer in the project's records. For Paper 2: drawing two subjects may already
be largely solved on newer models such as SD3 and FLUX, which would leave little for a correction to
add. For Paper 1: the only measure run so far, counting objects on CLEVR, cannot tell the method from
pasting separately drawn objects together, and reviewers will ask about interactions that pasting
cannot produce. This session reads the literature with the `paper-scout` skill to answer both, and
lists other open problems in composing diffusion models in case Paper 2 changes direction.

Your owner code on the board is `s14`, and your steps are G1 to G3.

## Objectives

1. **Step G1:** decide whether two-subject drawing is still an open problem worth Paper 2, and write
   it in `artifacts/notes/progress/gaps/01-is-paper-2-still-open.md`.
2. In G1, find the best published two-subject and attribute-binding numbers on SD3, FLUX and
   SDXL (GenEval two-object, GenEval colour attribution, T2I-CompBench colour and shape), each with
   its paper, page, sample count and whether the model was trained or left frozen.
3. In G1, write the missing reading notes for CO3 (arXiv 2509.25940) and ToMe (arXiv 2411.07132)
   in `gaps/notes/<paper-slug>.md`, and say whether either already claims Paper 2's contribution.
4. In G1, end with one recommendation: continue Paper 2 as designed, narrow it to a setting where
   the gap is still large, or change direction, with the evidence for it.
5. **Step G2:** list the questions a reviewer will ask of Paper 1 in
   `artifacts/notes/progress/gaps/02-reviewer-questions-paper-1.md`, one row per question, with the
   papers that bear on it and the measure that would answer it.
6. G2 must cover the owner's question about how separately trained experts interact: whether two
   composed objects under one light source cast consistent shadows, including one object's shadow
   on the other, whether a nearer object correctly hides a farther one, and, on chest X-rays,
   whether one finding changes how another appears.
7. **Step G3:** rank other open problems in composing diffusion models in
   `artifacts/notes/progress/gaps/03-open-problems.md`, each with the papers that leave it open,
   why it is open, what a first small experiment would be, and its rough cost.
8. Send each finished step to the right reader: G1 to `s13`, G2 to `s4` for Paper 1's evaluation
   plan, and G3 to `s13`.

## What to read

- `artifacts/notes/progress/prompts/shared-rules.md`: the rules, read on every tick.
- `plans/route-two-papers.md`: what each paper is for and which datasets each runs.
- `artifacts/notes/progress/problem-statements.md`: the two papers' problem statements.
- `artifacts/notes/progress/manager/paper2-four-way-design.md`: Paper 2's stated contribution and
  the test that decides it.
- `artifacts/notes/progress/reviewer/2026-11-24-review.md`, last section, "Paper 2's four-way test,
  2026-11-25": the reviewer's novelty check and its gaps.
- `artifacts/notes/progress/benchmark-spec.md` and `field-table.md`: every rival's number, split and
  sample count already read.
- `2-classifier-guidance/report/the-method-search/does-sample-and-paste-already-solve-small-clevr.md`:
  why counting objects cannot separate Paper 1 from pasting, and what that page says is untested.
- `1-lora-correction/report/against-other-methods/how-does-the-adapter-score-on-the-standard-benchmarks.md`:
  this project's own runs of the joint prompt, CO3 and PoE on GenEval and T2I-CompBench.
- `context/methods/01-the-product-of-experts/03-the-miniature.md`: the interaction term that a
  product of experts leaves out, which is where effects needing both objects would live.
- The reading tables already written, cited and never reread: `score-space/01-reading/reading-table.md`,
  `noise-text-critic/01-reading/reading-table.md`, `abstract-checkers/01-reading/reading-table.md`,
  `leakage/01-reading/mechanism-table.md`, all under `artifacts/notes/progress/`, and the folder
  `artifacts/notes/progress/field-scouts/`.
- The owner's reading profile at
  `/home/molef/.claude/projects/-home-molef-PhD-concept-composition/memory/paper-scout-profile.md`,
  which `paper-scout` reads for field, mathematics level and papers already judged.
- `report/00-INDEX.md` and `context/00-INDEX.md` before stating what this project found or what a
  term means.

## What is already known

- On this project's own runs, CO3 is level with the joint prompt on standard benchmarks: GenEval two
  objects 0.750 against 0.763, T2I-CompBench colour 0.599 against 0.601 (the standard benchmarks page).
- Those runs used 2 seeds and 200 images, and CO3's sampler settings are on the void list, so they are
  this project's measurements and not reproductions of CO3's paper (`prompts/session-4-rivals-and-targets.md`).
- On cat and dog, CO3 draws both clearly on 89 of 128 seeds and the joint prompt on 57, by the
  automatic scorer (`report/00-INDEX.md`, the kept push comparison).
- FKC reports 0.78 on GenEval two-object on SDXL, and ToMe reports 0.6051 on T2I-CompBench shape on
  SDXL (`benchmark-spec.md`, rows for Paper 2).
- No paper the project has read claims that one learned checker lifts several samplers, and CO3 and
  ToMe had no reading notes when the reviewer checked (the reviewer's 2026-11-25 section).
- On CLEVR, pasting separately drawn objects puts every object in place in 1,024 of 1,024 images,
  which predicts pasting reaches the target of 7 objects (the paste finding).
- The paste finding names two untested cases: crowded scenes where objects overlap or shade each
  other, and the lighter seams around pasted objects.
- On digits, Paper 1's method drew 295 clean pairs of 320 and pasting drew 287, a null by its bar
  (`report/00-INDEX.md`).
- Bradley et al.'s printed CLEVR numbers come from training about 16 times longer than this project's
  runs, so they are context and not a target to beat (`benchmark-spec.md`, lines 55 to 57).
- Paper 1's route lists CLEVR, the CoInD datasets, the CIFAR-10 disjoint split and chest X-ray as its
  datasets, and no chest X-ray design exists yet (`plans/route-two-papers.md`, line 80).
- The owner's working hypothesis for X-rays: both experts agree on anatomy, so a product likely keeps
  ribs, spine and lungs, while each expert treats the other's finding as unlikely. Tuberculosis with
  silicosis, and effusion with cardiomegaly, are the two pairs the owner named. Test this against the
  literature; it is a hypothesis, not a finding.

## When something is missing

- **A paper is behind a paywall or will not download.** Try its arXiv version and the authors'
  page first. If neither exists, record it as unread in the step's file and carry on.
- **A newer model has no published two-object number.** Say so in G1's table as unknown. Do not
  estimate it and do not run the model.
- **A claim needs a project number that no report page holds.** Ask `s13` through one line in
  `requests.md`, and write the claim as open until the answer comes.
- **The recommendation in G1 depends on a choice only the owner can make**, such as which model
  generation Paper 2 targets. File it with `python3 scripts/progress_board.py ask "<question>" --holds G1 --default "<your recommendation>" --by s14` and finish the rest of G1.

## Rules

- Run no experiments and no GPU work. This session reads and writes notes only.
- Write only to `artifacts/notes/progress/gaps/`. Never edit another loop's files.
- Confirm every arXiv id before citing it, and give a page for every number.
- Cite the reading tables listed above instead of rereading those papers.
- Mark every claim as read from the paper, read from a summary, or inferred.
- Spend no Claude API tokens from the owner's key.
- Send one short message when each step lands, with its table.
- Commit only the step's named files, holding the lock as `shared-rules.md` says.

## What to hand back

- `artifacts/notes/progress/gaps/01-is-paper-2-still-open.md`, with the model and benchmark table,
  the CO3 and ToMe notes under `gaps/notes/`, and one recommendation.
- `artifacts/notes/progress/gaps/02-reviewer-questions-paper-1.md`, with one row per question.
- `artifacts/notes/progress/gaps/03-open-problems.md`, ranked, with a first experiment and cost each.
- One line per finished step in `artifacts/notes/progress/requests.md`: G1 and G3 to `s13`, G2 to `s4`.
- The proof for each step is its file existing with every row sourced, checked with
  `python3 scripts/progress_board.py show` after setting the step to done.

## What you will see

Along the way, each step's table arrives as a message when it lands, with the paper count behind it.
G1 should take a few hours and G2 and G3 about as long each.

At the end, the closing report in chat says what was asked in one sentence. It lists each file by
path. It states the outcome: whether Paper 2's problem is still open on SD3 and FLUX, with the best
numbers found, how many reviewer questions G2 lists and how many have a measure, and the top three
open problems. It names what is still unread or uncertain, and asks at most one question.

The owner judges this from G1's recommendation and G2's questions, and the loop is done when G1 to G3
are green.
