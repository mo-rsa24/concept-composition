# Prompt: route step 14, draft the replies

Paste into a fresh session in `/home/molef/PhD/concept-composition`. It replaces the prompt block under step 14 of `plans/route-answer-the-iclr-reviews.md`.

```text
/drip-write the reply to Reviewer PrzE for submission 56173, point by point, after the AI builds the
reply files and the number check
Model: Opus, medium effort. Box: 90 minutes of your time. Filing: tied.
Controls: each part example (the table row or figure it cites), the opening why (why the reviewer's
strengths are named first), the closing prove (every number matches the revised paper).
Side routes: after the draft, /drip-proofread <the reply files> (a proofread pass over the replies).

Background
Reviewer PrzE rated submission 56173 a 1 with confidence 4 and listed two strengths, two weaknesses
and two questions. The paper has been revised (step 13). This step writes the reply, point by
point, in a markdown source that builds a LaTeX file for the Overleaf replies project and a plain
text file for OpenReview. The repo is /home/molef/PhD/concept-composition.

Objectives
1. Read the OpenReview forum page for the comment length limit and format, and say what it is.
2. Build paper/iclr-56173/replies/prze.md, scripts/build_reply_project.py and scripts/check_reply_numbers.py.
3. Draft the opening, weakness 1 with question 1, weakness 2, question 2 and the closing, one part
   at
   a time with the user steering.
4. Draft the short common comment to all reviewers: what changed, with the section and figure list.
5. Run the number check, build the Overleaf zip and the plain-text files, and print character
   counts.

What to read
- paper/iclr-56173/Reviews.pdf: Reviewer PrzE's exact words, and the other two reviews for overlapping points.
- paper/iclr-56173/CHANGES_FOR_SUPERVISORS.md and the revised paper's tables: the facts the reply cites.
- 1-lora-correction/report/paper-review/board.md, protocol.md and the results tables from steps 8 to 11.
- The OpenReview forum page for submission 56173 (the user's browser, or a pasted copy).

What is already known
- Reviewer PrzE's strengths: the early-stage ablation (Figures 3 and 4) and a computable quantity
  between the joint prompt and PoE. Weaknesses: no quantitative benchmarks, narrow baselines, the 48
  pairs not listed. Questions: evaluation scores, and results for more than two concepts.
- Figure 5 used three adapters in the submission; in the revision it uses one, v58-03-self at step
  30,000 (shared seed 10), and has seven columns: joint, PoE, CO3, ToMe, SuperDiff, PoE with
  Langevin, ours.
- The revision renumbered figures. PrzE's "Figures 3 and 4" are the submission's numbers; cite the
  revised numbers and map each one the reviewer used (take the mapping from
  paper/iclr-56173/CHANGES_FOR_SUPERVISORS.md, or build it from the submitted PDF
  56173_Restoring_plurality_in_c.pdf and the revised build).
- Appendix C.2 lists the 53 held-out pairs and the two showcase pairs. The table of the 48 training
  pairs (board line PRZE-3a) is not in the working copy as of 2026-11-11; check that step 13 added
  it before the reply says the pairs are listed.
- The main table is the 276 Attend-and-Excite prompts at seeds 9 to 12, the ablation the 53 held-out
  pairs at seeds 9 to 16, and GenEval and T2I-CompBench appear as their own table (protocol.md
  rule 4; layout-sheet.md R1, R3, R14).
- OpenReview closes Wednesday 18 November.
- Adjusted MALA and HMC cannot run on SDXL, which outputs no energy; the unadjusted versions can.
- Any baseline that beats the adapter in a table is reported as it came out.
- The user posts on OpenReview and creates the Overleaf replies project; Claude builds the files.

When something is missing
- If a number in a draft has no match in the paper or results file, remove it or ask the user for
  the source. Never estimate.
- If the character limit is unknown, assume 5,000 and ask the user to confirm on the forum page.
- If a result the reply needs does not exist yet, write the sentence as a placeholder marked TODO.

Rules
- Name the reviewer's own words first, in one clause, before answering each point.
- Say what changed and where, with the section, table or figure number.
- Claim nothing beyond the written bar. No defensive phrasing.
- Plain text for OpenReview: no LaTeX, ordinary punctuation, no dashes between clauses.
- Never ask for or store any Overleaf or OpenReview credential.

What to hand back
- paper/iclr-56173/replies/prze.md and paper/iclr-56173/replies/common.md, the built LaTeX project zip, and one plain-text file per
  comment with its character count.
- The number check's output with every number matched.
- Rows added to "What we know so far" in plans/route-answer-the-iclr-reviews.md.
- Proof: python3 scripts/check_reply_numbers.py prints PASS with the count of numbers matched, and
  the board redraws with --step 14 ticking the PRZE-6 lines.

What you will see
Along the way: the forum's limit, then each part of the reply drafted with the table it cites, each
followed by one question. At the end: what was asked, what was done with paths, PASS or MISS on the
number check, what is open, and at most one question. You judge by reading the plain-text files.
```

- `added:` the adapter of record and Figure 5's seven columns · `added:` the renumbering and where the mapping comes from · `added:` the open PRZE-3a training-pair table · `added:` the evaluation inputs by table, and the closing date
- `dropped:` nothing
