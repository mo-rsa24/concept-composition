# PRZE-5: the paper revised and shared

## Outcomes

- [x] ✅ check passed: the source to edit is settled. ✓ verified 2026-11-10: the live Overleaf project, exported as `_ICLR2027__LoRA_Plurality.zip` (62 files, `iclr2027_conference.tex` 89,856 bytes), is the submitted source: its 14 section and subsection titles equal the PDF's, and every PDF sentence the matcher flagged was found in it by exact grep. Both local copies are older and must not be edited: `paper/overleaf-iclr/` (the [Overleaf clone](../../runbook/syncing-the-paper-with-overleaf.md#where-the-paper-lives), last pulled 2026-09-26 08:17 UTC, with uncommitted edits) lacks the final restructure, and `paper/iclr/` (2026-09-21) is an earlier draft. Detail in [the ground check](../../artifacts/drips/answer-iclr-reviews/ground-check.md#3-the-local-tex-against-the-submitted-pdf)
- [ ] ✏️ markdown updated: the adapter is described as the run trained it: LoRA on self- and cross-attention query, key and value, trained on the ten noisiest of 50 steps (the PDF says cross-attention only on pages 1 and 7, and "every denoising step" on page 1)
- [ ] 🔗 page published: the paper's Overleaf project holds the revision, and your supervisors have access (you share it; Claude cannot)
- [ ] ✏️ markdown updated: the reference to "Figure 10" in Section 4 now points to Figure 8, and the broken sentence in Section 5 is fixed
- [ ] ✏️ markdown updated: the new results section and appendix are in, each figure carrying its plain explanation
- [ ] ✅ check passed: the paper compiles under the ICLR template and stays inside the page limit; new material beyond one compact table and one figure sits in the appendix
- [ ] 🔗 page published: you upload the revised PDF on OpenReview (it needs your login) and the comment says which sections changed
