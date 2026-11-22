# RPG first read: notes

PDF: arXiv 2401.11708v3, 17 pages, ICML 2024. Page numbers are PDF pages.

## Numbers that could not be read

None. All 66 cells of Table 1 (p9) were legible at 400 dpi. The paper has one table.

## Plot-only numbers

No figure plots a result, so there are no plot-only curves. The running text states no measured result that Table 1 lacks. The base ratios printed in Fig 4 (p5) and Fig 16 (p13) are settings, so they have no rows.

## Table 1: scorer, prompts, seeds and images per prompt

**Scorer.** Not stated. The caption says "Evaluation results on T2I-CompBench" (p9), and the text says the benchmark is used to assess text-image alignment (p10). The metric is recorded in the caption's words. No scorer model is named anywhere for Table 1.

**Prompts.** Not stated. The paper gives no prompt count and no split for any subset.

**Seeds.** No seed count is stated. Equation 4 calls `s` "a fixed random seed" and gives no value (p6).

**Images per prompt.** Not stated.

**Other settings not stated for Table 1.** Sampler, step count, guidance scale, resolution and the base ratio. Fig 4 draws sampling steps up to 40 and a base ratio of 0.2 for one example (p5), and neither is said to apply to Table 1.

## Rows quoted from other papers

The caption says "The baseline data is quoted from Chen et al. (2023a)" (p9). Chen et al. (2023a) is the PixArt-α paper (reference list, p14). The caption names no rows, so the ten rows other than RPG are all recorded as possibly quoted. Only the RPG row is presented as this paper's own measurement.

The ConPreDiff row cites Yang et al. (2023b). That reference lists "Yang, L." first and "Bin, C." last (p16), which match the names of RPG's authors Ling Yang and Bin Cui. Whether that row appears in Chen et al. (2023a) cannot be checked from this PDF.

## Base model

RPG runs on SDXL as "the base diffusion backbone" (p9). Figs 10 and 11 (p10) show other language models and other backbones, with images only and no scores.

## What RPG needs beyond the image model

**A language model.** GPT-4 is the recaptioner and chain-of-thought planner in the experiments (p9). Llama 2, Vicuna and MiniGPT-4 are shown as alternatives in Fig 10 (p10).

**Hand-written prompting material.** The authors design a task-aware template and in-context examples for few-shot prompting (p6, p9).

**A region plan.** The language model splits the image into non-overlapping rectangular regions and assigns one recaptioned subprompt to each (p5, p6). The user supplies no layout, because the language model writes it.

**A base ratio per prompt.** The authors set it higher when entities share a class and lower when they differ (p9).

**For editing only.** A multimodal language model such as GPT-4 or Gemini Pro (p7), a contour mask from Kirillov et al. (2023), and diffusion inpainting (p8).

## The plain SDXL row

Table 1 prints SDXL at 0.6369 on Color, 0.5408 on Shape and 0.5637 on Texture (p9). These match the expected 0.6369, 0.5408 and 0.5637 exactly. The row's other scores are 0.2032 on Spatial, 0.3110 on Non-Spatial and 0.4091 on Complex.

CoMat's Table 1 (CoMat p7, in `reads/first/CoMat.csv`) prints SDXL at 0.5879, 0.4687 and 0.5299 on the same three subsets, so the two papers' SDXL baselines differ.

## Hardware and running time

Neither is stated.

## Where the paper contradicts itself

**Numeric accuracy has no score.** The text defines three scenarios (attribute binding, numeric accuracy, complex relationship) and says Table 1 shows RPG "significantly superior" in "all three" (p9, p10). Table 1 has no numeric accuracy column. Its columns are Color, Shape, Texture, Spatial, Non-Spatial and Complex.

**Two named comparators are missing from the table.** The main results compare against DALL-E 3, SDXL and LMD+ (p9). Table 1 has no DALL-E 3 row and no LMD+ row, so those two comparisons are pictures only (Fig 8, p8).

**The SDXL row cites the wrong paper.** Table 1 cites SDXL as Betker et al., 2023 (p9), the reference the paper uses for DALL-E 3. Everywhere else SDXL is Podell et al., 2023 (p1, p2, p7, p8, p9, p12).

**The base prompt is defined one way and shown another.** Page 6 sets the base prompt equal to the full user prompt. Fig 3 (p3) uses a shorter base prompt, "A beautiful girl standing in her bright room". Fig 9 (p9) has a user prompt asking for five floors and a base prompt saying three floors.

**Fig 9 mislabels a region.** Its six-region example lists "Region 4" twice and no "Region 5" (p9).
