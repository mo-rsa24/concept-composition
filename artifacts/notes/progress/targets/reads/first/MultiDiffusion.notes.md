# MultiDiffusion: first-read notes

PDF: arXiv 2302.08113v1, 16 pages. Page numbers are PDF pages.

## Numbers that could not be read

None. In Table 1 (p5) a dashed rule crosses the top of the SI row, so that row was read again from a 300 dpi crop: 45.5 ± 14.5, 0.26, 5.76.

## Plot-only curves

None. The figures are image grids and diagrams, with no plotted curves.

## Protocol facts a comparison depends on

- All experiments use Stable Diffusion v2 through the HuggingFace implementation, with a 64x64x4 latent and 512x512 output (p5).
- Panoramas are 512x4608, nine times the training width (p5).
- Panorama FID compares one random 512x512 crop from each of N = 2000 panoramas against N = 2000 images sampled from the reference model, for one prompt (p6).
- The evaluation is repeated for 8 text prompts, and Table 1 reports the mean and standard deviation over them (p6). The ± in Table 1 is therefore spread across prompts.
- The `Stable Diffusion` row is an independent set of samples from the reference model (p6).
- The region evaluation uses a COCO validation subset filtered to 2 to 4 foreground objects, excluding people and masks under 5% of the image, giving 1K samples (p7).
- IoU is measured with an off-the-shelf segmentation model against ground-truth masks. The `COCO dataset` row is the IoU on the original images, reported as an upper bound (p7).
- Bootstrapping stops at T_init = 800, stated as 20% of the generation process (p5).
- Seeds, guidance weight and step count for the table numbers are not stated.

## Hardware and training time

The paper states no hardware, runtime or training time. The method uses a pretrained model "without any further training or finetuning" (p1).

## Contradictions inside the paper

- Page 6 says the method "outperforms the existing baselines in all metrics". Table 1 (p5) prints CLIP-score 0.27 for both the method and BLD, and bolds both.
