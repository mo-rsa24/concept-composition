# Changes for supervisors: Reviewers k4nd and UHz6

One line per change to the paper, grouped by the reviewer point it answers. Commits are on
`exec/correction-loss-01`; each holds that change and its reply only. The replies themselves are in
`replies/k4nd-and-uhz6.md`.

## k4nd W4: method details

- Section 4 states the training loss as Equation 12, with its per-step weight (`8ef00a17`).
- The training diagram (old Figure 8) is hidden until its notation matches Equation 12 (`2e9af75c`).
- Section 4 says each adapted evaluation returns a noise prediction, and inference uses the adapted composition directly; the "Figure 10" reference is gone (`88f02a45`).
- The abstract-concept sentences are cut from Sections 3 and 4 and the Discussion (`06a9cf98`).
- Section 4.1 opens with the evaluation protocol (`508d2647`).

## k4nd Q3 and UHz6's training filter

- Section 4 describes the training set, with CO3 as a second view during screening only (`14a457e3`).
- Figure 8 is now the joint prompt beside CO3 on five candidate pairs, one green pick per seed (`c2178737`, `fcaff3b3`).

## UHz6 presentation

- "inside create hybrid objects" reads "and instead create hybrid objects" in the Discussion.

## Corrections no reviewer raised

These go in no reply.

- The adapter trains the query, key and value projections of the self-attention and cross-attention layers, 35.8 M parameters; the submission said cross-attention only and 19.8 M (`b871c89e`).
- Appendix A names the checkpoint used: step 30,000 (600 epochs) of a 50,000-step cosine schedule; the submission said 50,000 steps and 1,000 epochs.

## Open

- Slot R11, three concepts: rendering as job 84832; four kinds of triple by joint prompt, plain product and ours at seed 10.
