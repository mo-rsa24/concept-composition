# Paper review board: Reviewer PrzE, ICLR 2027 submission 56173

What Reviewer PrzE's points ask for, whether each is delivered, and where the proof is. Reviewers UHz6 and
k4nd ask for overlapping things (aggregate success rates, the pair list, more baselines, the broken figure
reference) and get their own rows when their round starts, except PRZE-11, which takes UHz6's early-steps
question now because it uses the same compute. PRZE-3b also answers UHz6's success-rate question. Rows PRZE-8
to PRZE-10 are the working rules and plumbing the other rows depend on.

| Row | What must be true | Status | What showed it | Step |
|---|---|---|---|---|
| [PRZE-1](prze-1-scores.md) | the adapter and every SDXL method have scores on the same pairs and seeds, and an SD 1.5 set runs the official Attend-and-Excite and CONFORM code | todo |  | 3, 7, 9, 10, 12 |
| [PRZE-2](prze-2-baselines.md) | the Langevin corrector is proved on two Gaussians, and a six-column figure compares joint, PoE and the correctors on the same starting noise | todo |  | 8 |
| [PRZE-3a](prze-3a-pair-lists.md) | the training and held-out pairs are listed exactly, match the checkpoint the paper used, and say which test each pair passed | in progress | adapter of record v58-03-self step 50,000; its 48 training pairs equal the pool file's, its held-out list has a fifth pair the file dropped; Figure 5 mixes three runs ([the pair lists page](prze-3a-pair-lists.md#the-adapter-behind-the-paper)) | 1, 4 |
| [PRZE-3b](prze-3b-held-out-set.md) | a held-out set of three groups at seeds 9 to 16 exists, with rates and uncertainty | in progress | [the frozen set](held-out-set.md), commit `089abd6e`: hard 3, easy 29, not competing 21; rates wait on step 9 | 4, 9 |
| [PRZE-4](prze-4-three-concepts.md) | a three-concept test with a bar written first exists, and the paper states the limit | todo |  | 11 |
| [PRZE-5](prze-5-paper-revised.md) | the paper is revised from the right source and your supervisors can open it in Overleaf | in progress | the source is the live Overleaf project, exported 2026-11-10 and matched to the PDF; both local copies are older ([the PRZE-5 page](prze-5-paper-revised.md)) | 1, 13, 15 |
| [PRZE-6](prze-6-replies.md) | a second Overleaf project holds a respectful reply to every point | todo |  | 14 |
| [PRZE-7](prze-7-comment-posted.md) | the comment is posted on OpenReview under submission 56173 | todo |  | 14, then you post (F) |
| [PRZE-8](prze-8-protocol.md) | the comparison protocol is written and agreed before any run | in progress | [the protocol](protocol.md) agreed; v58-03-self 30k named by the showcase rule, seed 10, in [the frozen set](held-out-set.md) | 2, 4 |
| [PRZE-9](prze-9-slots.md) | every new figure and table has a fixed size and layout, a placeholder in the paper, and a script that refills it when its result file changes | in progress | 13 slots on [the layout sheet](layout-sheet.md), placeholders compile in `paper/iclr-56173/`, `scripts/refresh_paper_slots.py --check` PASS; R1 too wide at 8 pt | 2, 6, 9, 13 |
| [PRZE-10](prze-10-compute.md) | the cluster nodes are checked, every job has a node and a priority, and jobs are queued in that order | in progress | all five biggpu nodes probed on 2026-11-10 with `nvidia-smi -L` and a torch matmul: 107, 108, 111, 112 usable; 110 healthy but held by another user's job until about 2026-11-12 ([node table](prze-10-compute.md#the-nodes-probed-2026-11-10)) | 1, 5 |
| [PRZE-11](prze-11-early-training.md) | UHz6's question is answered: restricting the adapter to the early steps, at inference first and by training second, is measured against the failure types in Figure 7 | todo |  | 1, 5, 12 |

## Dropped

Nothing yet.
