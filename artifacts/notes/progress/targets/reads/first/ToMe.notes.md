# ToMe first read: notes

PDF: arXiv 2411.07132v1, 27 pages. Page numbers are PDF pages.

## Numbers that could not be read

None. Every printed table and figure value was legible at 300 or 400 dpi.

## Plot-only and single-example numbers

Figure 3(b) (p5) is a PCA scatter with no printed point values, so it has no rows.

Figure 9(a) (p17) prints detector confidences on two single example images (sunglasses 84.6, hat 83.4). Figure 10 (p18) prints two example GPT-4o scores (1.00 and 0.75). These illustrate the metric on one image each and report no result, so they have no rows.

Figures 2, 9(b), 9(c) and 15 print their values, and those are in the CSV with kind `plot`.

## Protocol facts a comparison depends on

**Attribute binding.** T2I-CompBench color, texture and shape, with 300 validation prompts per subset, scored by BLIP-VQA and by ImageReward as the human-preference score (p7).

**Object binding.** The authors' GPT-4o Benchmark has 50 prompts of the form "a [objectA] with a [itemA] and a [objectB] with a [itemB]", scored by GPT-4o (p7). The rubric has nine levels from 0 to 100 (p17, p18). Table 1 prints these scores on a 0-to-1 scale.

**Sampler.** The base model is SDXL, and the composite token is updated in the first 20% of denoising steps (p7). Table 3 shows the 50-step row equals Table 1's ToMe row, and no other page states the step count for Table 1.

**Not stated.** The paper gives no generated image count per prompt, no seed count and no resolution for Tables 1, 2 and 4.

**Quoted scores.** Appendix C.2 (p16) says the SDXL versions of Ranni, ELLA and CoMat are not open-sourced, so their BLIP-VQA scores are taken from their own papers. SynGen was adapted to SDXL by the authors. The `protocol` text marks the quoted rows.

**DetScore and entropy.** Figures 2 and 9 are computed on 100 generated images (p5, p16, p17), with MMDetection and GLIP for detection (p16).

**User study.** Figure 15 has 20 participants rating into four levels (p20, p21). The paper does not state the prompts, the image count, or which base model SynGen, ELLA and Ranni used, so `model` is empty for those rows.

## Hardware and cost

All experiments ran on an NVIDIA-A40 GPU (p16). Table 3 reports time per method on a single A40 (p17). The paper does not state whether the time is per image or per batch.

## Contradictions inside the paper

**SynGen on SD1.5.** BLIP-VQA Shape is 0.4661 in Table 1 (p8) and 0.4666 in Table 4 (p21).

**Times for models the paper says it could not run.** Table 3 (p17) lists times for Ranni (SDXL) and ELLA (SDXL) and says all were measured on one A40. Appendix C.2 (p16) says those SDXL versions are not open-sourced.

**ELLA on SDXL, training mark.** Table 1 marks it as requiring training (p8). Table 4 marks it as not requiring training (p21).

**Highlight that marks the wrong cell.** Table 1's human-preference Shape column marks SynGen (SDXL) 0.4016 as second best, and SDXL's 0.4029 is higher (p8).
