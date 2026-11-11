# Prompt: route step 13, revise the paper

Paste into a fresh session in `/home/molef/PhD/poe_repair_min`. It replaces the prompt block under step 13 of `plans/route-answer-the-iclr-reviews.md`, adding what the presentation pass of 2026-11-11 left for this step.

```text
/drip-write the evaluation section and the limitations paragraph of the revision for the ICLR
reviews, after the AI applies the mechanical fixes and appendix tables as one diff for one yes
Model: Opus, medium effort. Box: 120 minutes of your time. Filing: tied.
Controls: each sentence example (the table row it describes), the evaluation section prove (every
number in the prose matches its table), the limitations paragraph why (why each limit is there).
Side routes: after the draft, /drip-proofread <the revised sections> (a full proofread pass over the
edited text).

Background
Submission 56173 got three reviews at ICLR 2027. Reviewer PrzE asked for scores on standard
benchmarks, more baselines, the pair lists and more than two concepts. The earlier steps made the
protocol, the held-out set, the scores, the corrector figure, the SD 1.5 set and the three-concept
test. This step puts them in the paper. paper/iclr-56173/ is the working copy and the user uploads it to Overleaf.
The repo is /home/molef/PhD/poe_repair_min and the manuscript map is paper/iclr-56173/DRAFT_MAP.md.

Objectives
1. Confirm the page limit and the working copy (paper/iclr-56173/) from the call, the template and
   the step 1 comparison.
2. Apply the mechanical fixes and appendix tables as one diff, and show it for one yes.
3. Draft the evaluation section, one sentence at a time with the user steering: one compact table
   and one figure in the main text, the rest in the appendix.
4. Draft the limitations paragraph, including the closed gates with their numbers.
5. Compile, count pages, run the refresh script's check, and write a plain list of changes per
   reviewer point.

What to read
- paper/iclr-56173/DRAFT_MAP.md and the working copy's tex: where each section stands.
- report/paper-review/board.md, protocol.md, layout-sheet.md, held-out-set.md and
  three-concept-set.md.
- The results files and tables from steps 8 to 11, in their slots.
- Reviews.pdf: the exact wording of every point being answered.
- The manuscript voice and number rules in the memory notes, and environment/paper.md for the build.
- 56173_Restoring_plurality_in_c.pdf: the submitted text, Sections 3 to 5 and Appendices A and B.

What is already known
- Section 4 writes "as shown in Figure~\ref{eq:corrected-sampler}", which points at an equation and
  prints its number (10); it should be Figure~\ref{fig:adapter-training}, the training diagram
  (Figure 7 in the working copy). Section 5 has a broken sentence (reviewers UHz6 and k4nd).
- The adapter of record is v58-03-self at step 30,000, which is 600 epochs of 50 steps. The tex
  sets \adepochs to 1,000 and \adsteps to 50,000 (lines 42 and 43 of iclr2027_conference.tex);
  change those two macros, and every mention follows. \adgpu and \adhours still print "[GPU]" and
  "[N]": fill them from environment/hpc/throughput.md or the training run, or cut the sentence.
- Figure 5 (slot R2) has seven columns: joint, PoE, CO3, ToMe, SuperDiff, PoE with Langevin, ours.
  Its caption still describes four; it must name all seven, credit PoE as Composable Diffusion
  (Liu et al., 2022), and say what each dagger marks (CO3 on its own noise if its code cannot take
  ours; Langevin on a labelled subset).
- The rank and steps ablation is in appendix C.5 and cited from Section 4.1 as "Figure 12 shows";
  name the appendix there ("Figure 12, Appendix C.5").
- The limitations figure (Figure 6) uses pictures cropped from the step-50,000 checkpoint, and its
  note says "Adapter trained for 1,000 epochs". Render its pictures again from the adapter of record
  (scripts/figure7_limitations.py --stacked draws it) and change the note to match.
- Figure 1 (final_figures/molefe_fig1.png) and the training diagram
  (training-the-compositional-adapter.jpg) fail the lettering audit (3.1 and 3.6 pt); the user
  re-letters them with runbook/making-the-papers-figures/relettering-an-image-with-baked-in-text.md,
  then switch their includes to the PDFs and rerun python3 scripts/figure_text_audit.py.
- hyperref draws green boxes round links; set \hypersetup{colorlinks=true, linkcolor=..., citecolor=...}
  in the ICLR style's colours.
- Layout rules already in the tex: main-text figures [t], at most two a page, a fifth of each page
  text, each section's floats defined at its start, a \FloatBarrier before the Discussion; the
  appendix's review material in six subsections (C.1 to C.6), each with a lead-in paragraph. After
  the results fill the slots, do one tidy pass: build, measure empty page feet
  (pdftoppm and a pixel scan), and adjust \Needspace values to the real heights.
- The training loss is only in an appendix diagram and must be an equation in Section 4 (k4nd).
- Figure 5's caption says CO3 used different starting noise while Section 4.1 says the same (k4nd).
- Figure 1 (final_figures/molefe_fig1.png) has its text baked into the pixels, printing at 3 to
  4 pt against the 8 pt rule, in two faces (serif in panel a, sans in panel b), with panel letters
  "a)" and "b)" where the redrawn figures use "(a)"; panel b's pictures print about 0.3 in wide.
  The fix is the re-lettering recipe in runbook/making-the-papers-figures/.
- Figure 7 (final_figures/molefe_fig2.png) has its text baked in at 1.3 to 1.8 pt in a sans face;
  its note "Adapter trained for 1,000 epochs" is step 50,000 by Appendix A's 50 steps per epoch,
  so its renders come from a later checkpoint than the adapter of record (v58-03-self, step 30,000,
  protocol rule 1) and are rendered again from it; the red zoom boxes in panel b are coloured frames, and the top-right box runs off the
  picture's edge; it prints at 0.9 of the text width, leaving white space at the sides.
- The paper builds with Tectonic, not pdfLaTeX (environment/paper.md).
- Closed gates are reported with the number that closed them. Mixed comparisons are marked mixed.
- Draft to supervisors Thursday 12 or Friday 13 November; OpenReview closes Wednesday 18.

When something is missing
- If paper/iclr-56173/ is missing or older than the Overleaf project, ask the user for a fresh
  export zip from Overleaf and unpack it there.
- If a result is missing for a slot, leave the placeholder and list the slot.
- If the paper exceeds the page limit, move the least central material to the appendix and show the
  user the list before cutting.

Rules
- Never ask for, print or store an Overleaf token. The user clones and pushes.
- Prose changes go through the user one sentence at a time; mechanical changes through one diff.
- Do not change a number except by refilling its slot from its result file.
- Do not claim more than the written bar allows.

What to hand back
- The revised tex in the working copy, committed locally, with the compiled PDF.
- paper/iclr-56173/CHANGES_FOR_SUPERVISORS.md: one line per edit, grouped by reviewer point.
- Rows added to "What we know so far" in plans/route-answer-the-iclr-reviews.md.
- Proof: the compile log shows no errors and the page count; the refresh script's check prints PASS;
  the board redraws with --step 13 ticking the PRZE-5 lines this step owns.

What you will see
Along the way: the page limit, then the mechanical diff, then the evaluation section a sentence at
a time, then the limitations paragraph, each followed by one question. At the end: what was asked,
what was done with paths, PASS or MISS against compiling inside the page limit, what is open, and at
most one question. You judge by opening the compiled PDF.
```

- `added:` the true cause of the "Figure 10" reference · `added:` the step-count macros and the two unfilled ones · `added:` Figure 5's caption, the appendix pointer, the limitations figure, the two figures to re-letter, link colours, and the layout rules with the tidy pass (all from the presentation retro, `artifacts/drips/answer-iclr-reviews/2026-11-11-presentation-retro.md`)
- `dropped:` "it is Figure 8" (why: the numbering changed; the reference is fixed by its label, not a number)
