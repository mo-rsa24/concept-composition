# Prompt: answer Reviewers k4nd and UHz6, fixing the paper where we can

Paste the code block into a fresh session in `/home/molef/PhD/concept-composition`. It runs beside route steps 5 to 9 and needs no GPU.

```text
/drip-execute-plan --request "Walk Reviewer k4nd's review of ICLR 2027 submission 56173 point by
point, quoting each point, then UHz6's four questions; fix in the paper what can be fixed now,
and draft a reply to what can only be answered"
Model: Opus, medium effort. Box: 90 minutes of your time. Filing: on.

Background
Submission 56173 got three reviews. Reviewer PrzE is handled by the route in
plans/route-answer-the-iclr-reviews.md. This session takes Reviewer k4nd (rating 1, confidence 3)
and Reviewer UHz6's questions (rating 2, confidence 4) while the quantitative jobs of route
steps 5 to 9 run elsewhere. The working copy of the paper is paper/iclr-56173/iclr2027_conference.tex;
the submitted version is 56173_Restoring_plurality_in_c.pdf. paper/iclr-56173/Reviews.pdf is a scan with no text
layer: read it as images (k4nd is pages 4 and 5, UHz6 page 3).

How to walk it
One reviewer point per message. Quote the reviewer's words first, then what the submitted paper
actually says (quoted, with its line or equation number), then whether the point holds on the
merits, then what can be fixed in the paper now and what can only be answered. End each message
with the draft reply line and one question for the user. Wait for the user before the next point.

Order
1. k4nd W4 and W5 and UHz6's presentation weakness: the text fixes (list below). Make the purely
   corrective ones as one diff for one yes; show any new sentence before it goes in.
2. k4nd W1 (does the residual measure plurality), W2 (novelty), W3 (no aggregate evaluation).
3. k4nd's four questions, then UHz6's four questions:
   - Could the authors list the 48 training pairs and the held-out pairs, and clarify whether
     cat/dog is in the training set?
   - What is the success rate of the adapter compared with PoE and CO3 over all held-out pairs and
     multiple seeds?
   - Given the finding in Sec. 3, does restricting the adapter to the early steps change the
     failure modes in Fig. 7?
   - Does the adapter degrade PoE on pairs that do not compete for the same region, such as
     foreground/background pairs like lighthouse and sea?

Text fixes already located (search by the quoted text; line numbers move)
- "as shown in Figure~\ref{eq:corrected-sampler}" in Section 4: it points at an equation label, so
  it prints "Figure 10". It means the combining rule (an equation); the training diagram is
  \ref{fig:adapter-training}.
- "unable to create images with plurality, inside create hybrid objects" in the Discussion:
  "inside" should be "and instead".
- Figure 5's caption says "CO3 uses different starting noise" while Section 4.1 says "all examples
  are generated from the same initial seeds". Protocol rule 2 (1-lora-correction/report/paper-review/protocol.md):
  every method gets the same seed; CO3's released code draws its own noise from that integer seed,
  so its column is marked with a dagger. Make both sentences say this.
- Section 4 says each network evaluation "returns the predicted correction defined in
  Equation~\ref{eq:interaction-term}", while Appendix B says the adapted predictions a_psi, b_psi,
  u_psi are composed by the PoE rule and only the composition is matched to the joint-prompt
  prediction. Appendix B is right; make Section 4 say the same.
- The training loss appears only in the Appendix A diagram. Add it to Section 4 as an equation.
  Take its exact form from the run that made the adapter of record:
  /datasets/mmolefe/poe_repair_min/outputs/showcase/cohort3/v58-03-self/config.json on the cluster
  (read with scripts/cluster.sh sh "cat ..."): the loss space (x0 or noise), any orthogonality
  weight, and the steps it trains on. 1-lora-correction/report/paper-review/prze-3a-pair-lists.md lists these too.
- \adepochs and \adsteps (near the top of the tex) say 1,000 epochs and 50,000 steps; the adapter
  of record is step 30,000, which is 600 epochs of 50 steps
  (artifacts/_shared/cross_pair_pool_configs/adapter_of_record.yaml).
- k4nd W4 also calls the "learning an abstract concept" sentences repetitive; propose cuts.

Settled facts (do not re-derive; cite them)
- W1: r_t (Equation 9) is the whole joint-minus-product difference, and lambda = 1 recovers the
  joint prompt by construction (paper line 216). Figure 2's controls show a pair's own r_t is
  needed (another pair's, another seed's, or a reordered one fails); Figures 3 and 4 show the
  first 10 of 50 steps suffice. That makes r_t specific and early, not plurality alone. Candidate
  fix: rename it (e.g. "joint-prompt correction") and keep plurality as the property it restores.
  Optional evidence: at lambda 0.6 to 0.65 in Figure 2 the second animal appears while layout and
  style still look like the product's render; a distance-to-joint-render number on those pictures
  would support "a small part of r_t restores plurality" without proving isolation.
- Training targets: every v57 and v58 training run, the adapter of record v58-03-self included,
  has render_teacher=False (33 configs under /datasets/mmolefe/poe_repair_min/outputs/showcase/);
  each target is SDXL's joint-prompt noise prediction (eps_j_raw in each cached state). No run
  trained on a CO3 picture. The submitted Section 4 says a pair is retained "if either image
  contains both concepts" (joint prompt or CO3); the verdict record
  (1-lora-correction/artifacts/results/the-adapter-on-cat-and-dog/how-it-trains/which-joint-prompt-targets-can-the-adapter-learn-from/verdicts.json, cluster
  checkout) holds joint-prompt calls only, and no record shows a pair kept by CO3 alone. The user
  remembers a mixed CO3-and-joint pool but cannot name a pair. Present this to the user and let
  them choose the wording; this answers k4nd's Q3 and UHz6's training-filter weakness.
- Held-out pairs: the 53 evaluation pairs (1-lora-correction/report/paper-review/held-out-set.md) are unseen as
  pairs only. 30 share a concept with a training pair and 6 share both. The paper's "Neither concept
  in a held-out pair appears in any training pair" holds only for the submission's four held-out
  pairs (cat and dog, elephant and penguin, piano and candle, skateboard and helmet). Cat and dog
  is held out and was never trained on (pair_pool_v58.yaml).
- Lists: appendix A now has a training-pairs paragraph (48 pairs: 28 two animals, 3 an animal and
  an object, 17 two objects; 182 cells over 23 seeds; seeds 9 to 16 never trained) and Figure 7,
  nine example targets. Appendix C.2 lists the 53 held-out pairs with their groups. A joint-beside-
  CO3 figure is commented out in appendix A until the filter wording is settled.
- Results not yet available: success rates over held-out pairs and seeds (Table 2, slot R3), the
  early-steps failure counts (slot R9), and the not-competing group (21 pairs, R3) all come from
  route steps 5, 9 and 12. Answer those questions with what will be shown and where, marked TODO
  with the slot name; never estimate a number.

What to read
- paper/iclr-56173/Reviews.pdf pages 3 to 5; 56173_Restoring_plurality_in_c.pdf (pdftotext -layout gives line
  numbers); the working copy's tex.
- 1-lora-correction/report/paper-review/protocol.md, held-out-set.md, layout-sheet.md, board.md.
- 1-lora-correction/artifacts/drips/answering-the-reviewers/answer-iclr-reviews/prompt-route-step-13-revise-the-paper.md and
  prompt-route-step-14-draft-the-replies.md: the paper and reply work this feeds; do not do their
  results-dependent parts here.
- The memory notes on training targets, held-out pairs and manuscript voice.

Outputs
- The text fixes in paper/iclr-56173/iclr2027_conference.tex, built with
  python3 1-lora-correction/scripts/refresh_paper_slots.py --compile (23 or 24 pages, no undefined references).
- paper/iclr-56173/replies/k4nd.md and paper/iclr-56173/replies/uhz6.md: one section per point, the reviewer's words quoted, the reply
  beneath, each claim citing a section, figure, table or slot.
- A line per change in paper/iclr-56173/CHANGES_FOR_SUPERVISORS.md, grouped by reviewer point.

Done when
Every k4nd weakness and question and every UHz6 question has a reply section; every fixable text
issue above is fixed or has the user's no recorded; the paper compiles; pdftotext of the build
shows no "Figure 10" pointing at the training diagram and no "inside create".

Rules
- No GPU job and no cluster render. Reading cluster files with scripts/cluster.sh sh is fine.
- New sentences go to the user before they go in; purely corrective edits go as one diff.
- Commit only the files this session changed, by name. Never git commit -a: the working tree holds
  the user's uncommitted trainer edits under poe_repair/experiments/.
- Push nothing to Overleaf. Edit the paper only under paper/iclr-56173/.
- Claim nothing the record does not show; no defensive phrasing in replies.

What you will see
One reviewer point per message: the quote, the paper's own words, the verdict on the merits, the
fix or the reply, then one question. At the end: the fixes made with the commit, the reply files,
what is TODO and which slot fills it.
```

- `added:` the located text fixes with search strings, the training-target and held-out facts, the W1 analysis, and the slots the results come from (all from this session's checks on 2026-11-11)
- `dropped:` reading k4nd from scratch (why: it is summarised above, and the session still reads the scan to quote it)
