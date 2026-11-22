# CO3 first read: notes

PDF: arXiv 2509.25940v2, 25 pages. Page numbers are PDF pages.

## Numbers that could not be read

None. Every printed table value was legible at 300 dpi.

## Plot-only

Figure 2 (p5) plots ImageReward against timestep for the CO3 resampler and the CO3 corrector. It prints no values, so it has no rows.

## Protocol facts a comparison depends on

**Sampler.** The main experiments use SDXL with 50 DDIM steps and guidance scale 5.0 (p6). CO3 runs in the first 20% of denoising steps, and the first three steps are resampling (p6).

**Table 1 settings.** T_c=10 correction timesteps, T_r=3 resampling steps, P=5 corrector iterations, beta=0.8, with w_0 at 1.0 for the resampler and 2.0 for the corrector (p17).

**Not stated.** The paper gives no generated image count per prompt, no seed count and no resolution for any table.

**Benchmarks.** Attend-Excite Animal-Animal, Animal-Object and Object-Object (p6, Table 1). T2ICompbench Complex and RareBench Concat, Relations and Complex (p6, Table 2).

**Human evaluation.** 11 participants rate images from 0 to 5, normalized to 0 to 1, with method names hidden and order shuffled, following R2F (p17).

**Baselines.** Attend-Excite, Divide-Bind, InitNO, SynGen and Composable Diffusion are SD1.5 methods. Tweediemix, Magnet, R2F and ToMe are SDXL methods. Composable Diffusion and SynGen were also adapted to SDXL (p17). Tweediemix runs without its LoRA customization stage (p7).

**Model left empty on purpose.** Table 1's Comp-diff row (p7) and Table 5's SynGen row (p17) do not say whether they used SD1.5 or the SDXL adaptation.

**Tables 6 to 13 name no benchmark.** Their columns are Animals, Animals Objects and Objects, which are the Attend-Excite categories, so the `dataset` text says so. These tables print "SD" in Tables 6 and 7, and the text on p17 calls the main run SDXL.

**Avg is undefined.** The paper never defines the Avg column in Tables 3, 6, 7 and 8. Most printed values equal the mean of the six scores in the row.

## Hardware and cost

All experiments are stated to run on a single NVIDIA A6000 GPU with 48 GB (p6). Table 8's runtime and memory were measured on an NVIDIA A100 (p18).

## Contradictions inside the paper

**Table 8's SDXL row carries 10-step scores.** The row labelled "SDXL (50 steps)" (p18) prints 0.4906, 1.1503, 0.1131, 0.6428, 0.8378, 0.4752, 0.6183. Those seven values are exactly Table 7's 10-step "SD w/ DDIM" row. Table 7's 50-step row is 0.7820, 1.5574, 0.6789, 0.6950, 0.8658, 0.4926, 0.8452. The 7.20 s and 8.5 GB in the same row may therefore belong to either step count.

**The same SDXL 50-step run differs across tables.** ImageReward A-O is 1.5574 in Tables 1, 6 and 7 and 1.5474 in Table 3. BLIP-VQA A-A is 0.6950 in Tables 1, 6 and 7 and 0.6951 in Table 3. BLIP-VQA A-O is 0.8654 in Tables 1 and 3 and 0.8658 in Tables 6 and 7. BLIP-VQA O-O is 0.4926 in Tables 1, 3 and 7 and 0.4925 in Table 6. Avg is 0.8435 in Table 3 and 0.8452 in Tables 6 and 7.

**The same CO3 50-step run differs across tables.** Tables 1 and 3 print ImageReward 1.2342 and 1.6744. Tables 6, 7 and 8 print 1.2341 and 1.6743. Table 8 prints BLIP-VQA 0.7440, 0.8877 and 0.5145, against 0.7441, 0.8878 and 0.5146 elsewhere. Avg is 1.0118 in Table 3, 1.0112 in Tables 6 and 7, and 1.0120 in Table 8.

**ToMe differs between Tables 1 and 8.** ImageReward A-A is 0.3895 against 0.3894, ImageReward O-O is 1.0118 against 1.0117, and BLIP-VQA A-O is 0.8808 against 0.8807.

**CO3 at 20 DDIM steps.** Avg is 0.8671 in Table 7 and 0.8679 in Table 8, with the same six scores. Their mean is 0.86795.

**Avg that does not match its row.** Table 3's "+ Resampling" row prints 0.9437, and the mean of its six scores is 0.94073.

**Sweep tables disagree by one digit.** With beta=0.3, num_ts=7, BLIP-VQA Animals_Objects is 0.8810 in Table 9 and 0.8809 in Table 12. With num_ts=6, beta=1.1, BLIP-VQA Objects is 0.4812 in Table 10 and 0.4811 in Table 12.

**Highlights that mark the wrong cell.** In Table 1, ImageReward A-O marks ToMe's 1.5736 as second best, and Magnet's 1.5895 is higher. DINOv2 A-A marks Comp-diff's 0.2497 as second best, and Tweediemix's 0.2519 is higher. In Table 3, BLIP-VQA A-O marks 0.8794 as second best, and SDXL+Corrector's 0.8801 is higher. In Table 4, BLIP-VQA A-O marks PixArt's 0.861 as best, and PixArt + CO3's 0.864 is higher.

**Bold that marks the wrong cell.** Table 10 bolds 1.7117 for ImageReward Animals_Objects, and num_ts=4 has 1.7136. Table 12 (num_ts=6) bolds 0.8808 for BLIP-VQA Animals_Objects, and beta=0.9 has 0.8809. Table 13 (num_ts=6) bolds 0.8842, and beta=0.7 has 0.8851.

**Claim against Table 2.** Page 8 says CO3 "consistently outperforms or matches the best-performing baselines across all categories and metrics" and "matches or exceeds R2F on RareBench". In Table 2, RareBench Relations ImageReward is 0.2588 for CO3 against 0.3620 for R2F and 0.3607 for SDXL. RareBench Complex human evaluation is 0.61166 for CO3 against 0.6283 for SDXL.

**Lambda is used for two scales.** Page 6 sets guidance scale lambda to 5.0. Page 19 defines lambda as scaling the composed score for CFG, and Tables 9 to 13 hold it at 0.8 or 0.9.
