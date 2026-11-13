# For route step 13: a draft results paragraph for Tables 1 and 2

From the walk answering Reviewers k4nd and UHz6 (2026-11-11). A draft, not in the tex. It reads only
the rows that exist on 2026-11-11 (joint prompt, PoE, ours) in `paper/iclr-56173/slots/R1.tex` and
`R3.tex`. Every sentence that needs CO3 or PoE + Langevin is a TODO, and so is the verdict against
the bar, which comes from the paired-differences script, not from these means. It carries no
numbers in the prose; the tables hold them.

Proposed place: Section 4.1, after the evaluation paragraph and before "Figure 5 shows".

> Table~\ref{tab:review-R1} reports composition and image quality on the Attend-and-Excite prompts.
> On the animal prompts, where the product most often fuses the two animals, the adapter raises
> both the share of images with two animals (Comp.) and BLIP-VQA well above the plain product, and on
> the object prompts its BLIP-VQA is the highest of the three rows. ImageReward rises over the product
> on all three sets but stays below the joint prompt, while CLIP score rises on the animal prompts and
> is flat or slightly lower on the other two. [TODO: CO3's and PoE + Langevin's rows, and the verdict
> against the bar of Section~\ref{sec:results} from the paired differences.]
> Table~\ref{tab:review-R3} reports the 53 held-out pairs by group. On the 29 easy-but-unseen pairs
> the adapter lifts the product's BLIP-VQA, and the two intervals do not overlap; on the 21
> not-competing pairs it neither helps nor harms; the three hard pairs are too few for the intervals to
> separate the methods. [TODO: CO3's and PoE + Langevin's rows.]

Checked against the rows (R1, per set, joint / PoE / ours): animals Comp. 0.89 / 0.02 / 0.97, BLIP
0.71 / 0.23 / 0.50; objects BLIP 0.48 / 0.45 / 0.55; ImageReward animals 0.89 / -1.31 / 0.16,
animals + objects 1.60 / -0.25 / 0.42, objects 0.77 / -0.22 / 0.19; CLIP animals 27.0 / 21.6 /
24.6, animals + objects 30.0 / 27.2 / 27.1, objects 31.1 / 29.5 / 28.7. R3 BLIP: easy but unseen
0.74 ± 0.08 / 0.38 ± 0.09 / 0.58 ± 0.09; not competing 0.85 ± 0.06 / 0.74 ± 0.10 / 0.73 ± 0.08;
hard 0.46 ± 0.40 / 0.33 ± 0.39 / 0.45 ± 0.42.

CO3's renders finished at 12:08 (job 84590: R1 1,104 of 1,104, R3 424 of 424); no CO3 score file
exists yet under `/datasets/mmolefe/poe_repair_min/outputs/paper_review/scores/`.
