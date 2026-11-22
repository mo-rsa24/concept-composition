# AttendExcite first read: notes

PDF: arXiv 2301.13826v2, 24 pages. Page numbers are PDF pages.

## Numbers that could not be read

None. Every printed table and figure value was legible at 300 or 400 dpi.

## Plot-only

Figure 8 (p7) and Figure 14 (p14) draw the absolute CLIP similarities as bars and the Upper Bound as a dashed line, with no printed values. Those heights have no rows. The percentages printed above the bars are in the CSV with kind `plot`.

## Running-text numbers

The CSV holds the text-only results: the lowest-performing prompt's vote shares (p8) and the runtimes (p10). The runtimes are stated as approximate.

Summary claims that restate a table or figure have no row of their own: "at least 4.7%" (p8), "at least 7%" (p7), "at least 5%" and "at least 3%" (p14), and "3.8%" and "4.4%" (p14).

## Protocol facts a comparison depends on

**Model and sampler.** Stable Diffusion v1.4 with guidance scale 7.5 (p10). T=50 denoising steps, with latent updates stopping at step 25 (p4).

**Prompts.** 66 Animal-Animal, 144 Animal-Object and 66 Object-Object prompts built from 12 animals, 12 objects and 11 colors (p6, p10). Each prompt gets 64 images from 64 seeds shared across methods (p6).

**Evaluators.** CLIP ViT-B/16 with 80 prompt templates, and BLIP captions from the MS-COCO fine-tuned model (p10). Seeds that returned a black NSFW image are left out of the averages (p10).

**Upper Bound.** It comes from 50 collected images per subject (p7), and its value is not printed.

**User study.** 65 respondents, 10 prompts per subset, 4 seeds (p8).

**Prompt-to-Prompt comparison.** It uses only 20 randomly selected prompts per subset (p13, p14), so Attend-and-Excite's Table 4 values differ from its Table 1 values. This is a different prompt set and is no contradiction.

**Complex prompts.** Table 5 uses 40 prompts with 64 images each (p14).

**Signs.** Plus signs in Table 6 are dropped from `value`, as the column rule requires. The positive value there means Stable Diffusion scored higher.

## Hardware and cost

Runtime was measured on an A100 GPU: about 5.6 s per image for Stable Diffusion, 9.7 s for Attend-and-Excite without iterative refinement and 15.4 s with it. With float16 these fall to about 6.6 s and 11.8 s (p10).

## Contradictions inside the paper

**The percentage's denominator.** Page 7 describes the percentages as the relative decrease "compared to Attend-and-Excite". The printed values use the baseline as the denominator. For Composable Diffusion in Table 1, (0.692 - 0.806) / 0.692 is -16.47%, as printed, and dividing by 0.806 gives -14.14%.

**"At least 3%" against Table 4.** Page 14 says Attend-and-Excite beats Prompt-to-Prompt by at least 3% in all three subsets. Table 4 prints -2.99% for Animal-Object.

**One percentage outside rounding.** Table 4 prints -4.29% for Animal-Animal. From 0.771 and 0.803, allowing for rounding of both values, the largest possible magnitude is 4.28% (p14).
