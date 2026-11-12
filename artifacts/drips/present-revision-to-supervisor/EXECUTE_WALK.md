# Walk map: present the ICLR 56173 revision to the supervisor

Plan: `plans/present-revision-to-supervisor.md` · mode `--request` · branch `exec/correction-loss-01`

## Stop point

- Bar: every claim group has a figure built from real crops; the bundle builds with Tectonic on the cluster at 20 pages, main text ending on page 10, 0 undefined references; the supervisor PDF reads claim, weakness, change, reply for every claim.
- Time box: 2 sittings, about 4 hours of Molefe's time.
- steps at the open: 9 (depth full)
- Sitting 1 start: 2026-11-12

## Toy

Group G7, the training loss and what the adapter returns (k4nd W4a, W4b, Q4), carried through every piece.

## Decisions

| Decision | Options | Who chose | Why |
|---|---|---|---|
| Supervisor document format | PDF · web page · slides | Molefe (defaults) | PDF, plus each figure as its own PNG and PDF |
| Where the revision sits in Overleaf | `revision/` folder · project root | Molefe (defaults) | `revision/`, nothing of the original replaced; upload by drag and drop |
| New main file name | `iclr2027_conference_revision.tex` · own | Molefe (defaults) | default |
| Commit footer on figures | keep · drop | Molefe (defaults) | kept |
| Review crop source | Reviews.pdf renders · OpenReview screenshots | Molefe | screenshots, annotated directly; Reviews.pdf for page numbers |

## Ticks

- Task 1, claim table: 35 claims, 12 groups (commit 8ce63cce). Instruction 4.1: "done. Happy with the claim table".
- Task 2.1 built, waiting on stop 2: G7 figure, review crops from Reviews.pdf at 300 dpi located by OCR (screenshots kept as the record).
- Paper and reply edits asked for at stop 2 (Molefe's word): Equation 12 unweighted with no weighting sentence; inference is the usual PoE step with the adapter on; the strength-λ equation removed; after Equation 12, "the joint prompt's prediction is the product's prediction plus r_t ... switching the adapter on adds it" (commits ead813ab, c980aa5a). Rebuild: 20 pages, main text ends p. 10, 0 undefined references.
- Added at Molefe's request: task 3.3, review.tex grouping the claims with quotes, what was done and the reply, done towards the end.
- Open for the close: the k4nd W4 misquote (Figure 7 for Figure 8) and PrzE Q1's "Tables 1 to 4"; Molefe has not said whether to fix them.

## Out of session

| id | kind | serves | state |
|---|---|---|---|
