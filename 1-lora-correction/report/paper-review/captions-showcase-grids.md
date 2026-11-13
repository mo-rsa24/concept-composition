# Captions for the appendix render grids

Draft captions and the sentences that point to them, for [route step 13](../../plans/route-answer-the-iclr-reviews.md#13--revise-the-paper) to place in the paper. Every grid starts each row from the same noise for all methods except where marked, and each caption says how its rows were chosen ([protocol rule 5](protocol.md#what-each-rule-means-in-practice)). The figures live in `artifacts/results/`; the BLIP-VQA values behind the choices are in `scores/scores_long.csv` and stay out of the figures.

## Where our method does best (two figures)

Files: `how-the-adapter-scores-against-other-methods/best-cases-held-out.pdf` and `best-cases-benchmark.pdf`.

> **Figure A.** Where our method does best. Rows are the two pairs in each [held-out group](held-out-set.md#the-held-out-groups) (left: hard, easy but unseen, not competing) and each benchmark set (right: animals, animals with objects, objects) on which our method scored highest under an automatic two-object check, each at its best seed. None of these pairs was trained on. Columns start from the same noise.

Pointing sentence for the main text: *Appendix Figure A shows the held-out and benchmark pairs on which the adapter does best; on these it draws both named things where the [plain product](../../context/world/poe-composition.md#what-poe-composition-is) drops or fuses one.*

## Where our method fails (two figures)

Files: `how-the-adapter-scores-against-other-methods/worst-cases-held-out.pdf` and `worst-cases-benchmark.pdf`.

> **Figure B.** Where our method fails. Rows are the two pairs per held-out group and benchmark set on which our method scored lowest under the same check, each at its worst seed. Our method drops one of the two things or draws the wrong species (two cows for "a cow and a horse"); CO3 keeps both on most of these rows.

Pointing sentence for the main text: *Appendix Figure B shows where the adapter fails: on these pairs it loses one object or substitutes a species, and CO3 often keeps both.*

## The render grids behind the tables

> **R4.** Every SDXL method on the held-out pairs. For each held-out group, the pair and seed where our method scored highest under the two-object check and the one where it scored lowest. † PoE + Langevin runs on a labelled subset.

> **R12.** Every SDXL method on the benchmark prompts. For each benchmark set, the prompt and seed where our method scored highest and the one where it scored lowest. † PoE + Langevin runs on the animals set only.

> **R10.** The adapter's window. R4's rows with the adapter off (the plain product), applied on steps 0 to 10, on steps 0 to 25 and on all 50 steps.

## Still to settle

- Network evaluations per image, counted with a counter on the U-Net (protocol rule 6): ours 300 rows in 100 calls as rendered (`scripts/review_count_ours_nfe.py`, seal and walrus at seed 11, log `logs/count_ours_nfe.log`), CO3 260 rows in 90 calls. The renderer appears to evaluate the plain product and the adapter each step and mix them by λ, so at λ = 1 half its rows are weighted by zero; the method alone would need 150. That is inferred, not checked in the code. Settle it before any sentence compares cost, and before the reply to UHz6, whose review credits the adapter with needing no extra evaluations.
- PoE + Langevin's column in Figures A and B fills as `langevin_extra/` and the subset finish; a cell it never reaches stays grey and needs the dagger.
