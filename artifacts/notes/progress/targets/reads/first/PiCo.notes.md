# PiCo first read notes

PDF: arXiv 2505.03203v1, 16 pages. Page numbers are PDF pages.

## Numbers that could not be read

None. Table V (p10) prints a dash for FID on the four Seed Selection rows, so those four cells hold no number and have no row.

## Curves that are plot-only

None. Figures 1 to 18 show images, masks and seed labels, and print no result value.

## Protocol facts a comparison depends on

**Model.** PiCo runs on SD v2.1 with 50 DDIM steps (p4). The defaults are T_s = 5, r_s = 10, delta = 2, T_c = 25 and gamma = 15 (p4).

**Baselines.** The paper compares against SD v2.1, Magnet, Attend-and-Excite, BoxDiff and GORS with official code and default hyperparameters (p4). It does not name the backbone of the four baselines other than SD v2.1, so their `model` cell is empty.

**Prompts and seeds.** The evaluation uses 100 prompts from CC-500 and 400 from T2I-Compbench, with 10 seeds per prompt, for 5,000 images (p4).

**Human evaluation.** Annotators judge 500 sampled two-object images (p4). Both means two concepts appear, Single one, None zero (p4). The tables omit the percent sign, and the text gives these rates in percent (p5, p9).

**Objective metrics.** CLIPScore, VQAScore and DINO-Score use 4,000 images from 400 T2I-Compbench prompts (p4, p7). CLIP-I and FID use MS-COCO (p4). FID in Table V uses 5k image-text pairs from the MS-COCO val set (p9). Table I does not state its FID image count, so its `samples` cell is empty.

**Selected noise.** The Selected columns of Table I give every baseline the noises PiCo chose (p5), so those baseline numbers include PiCo's noise selection.

**Not stated.** Image resolution for the evaluation, the guidance scale and the number of annotators are not stated. The only resolution printed is the 512x512 canvas in the GPT-4o prompt that makes BoxDiff's boxes (p7). Tables III, V and VI do not restate their prompt subset or sample count.

## Hardware and time

The paper names no GPU. Noise selection runs 5 noises in parallel "due to equipment limitations" (p9). Runtime per prompt is 0.84 s at T_s = 1, 2.03 s at T_s = 5 and 6.41 s at T_s = 20 (Table III, p5). PiCo needs no training (p1).

## Contradictions inside the paper

**None rate at r_s = 10.** The text on p9 says the None rate falls from 29.1% to 13.4%. Table V on p10 prints 18.4 for r_s = 10, and Table III on p5 also prints 18.4 for the default setting. Both values are in the CSV: `PiCo-X-001` holds 13.4.

**Stable Diffusion Both change in Table I (p5).** The table prints +6.8. Its own columns give 34.7 minus 27.5 = 7.2. The other 17 printed changes in the table match their columns.

**BoxDiff alignment votes in Table II (p5).** Win 45.8, Loss 16.7 and Tie 38.2 sum to 100.7. The other nine vote triples sum to 100.0.

**Table VI rates (p10).** Both/Single plus None sums to 101.0 at percentile 50, 101.0 at percentile 100 and 101.8 at delta 1. The other three rows sum to 100.0.

**The default PiCo setting has three different human scores.** Table I prints Both 41.8 and None 11.8 (p5). Table III and the r_s = 10 row of Table V print Both 36.2 and None 18.4 (p5, p10). The T_c = 25 and gamma = 15 rows of Table V print Both 37.8 and None 19.7 (p10). The paper does not explain the difference.

**Table VI "Ours" matches the Stable Diffusion row of Table I.** Its None 19.8 and DINO 0.659 (p10) equal the Selected values for Stable Diffusion in Table I, while PiCo's Selected values there are 11.8 and 0.730 (p5).

**One hyperparameter has two symbols.** p3, p4 and Table VI call it delta. The discussion of Table VI on p9 and p10 calls it sigma.
