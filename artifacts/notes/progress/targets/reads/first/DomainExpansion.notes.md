# DomainExpansion: first-read notes

PDF: arXiv 2605.23275v1, 17 pages. Page numbers are PDF pages.

## Numbers that could not be read

None. The dashes in the Table 1 Size column for Concat and MultiDiffusion are printed as dashes and have no row.

## Plot-only curves

Figure B.4 (p15) plots FID against sample count for N = 96 and N = 128. It prints no values, so it has no rows.

## How rows are attributed

Table 2 labels Du's method `RRR` (p4), so those rows carry `paper` = `Du`. MultiDiffusion rows carry `paper` = `MultiDiffusion`. Both were re-implemented and measured by this paper, on its own base models.

## Protocol facts a comparison depends on

- Music: Slakh2100, 1,500 training tracks of four stems (p9). Base track length l is 12 seconds, so 4l is 48 seconds and 10l is 120 seconds (p3).
- FAD uses 128 generated tracks against middle cuts of training tracks, resampled from 22,050 Hz to 16,000 Hz (p10).
- Each music method reports its saved checkpoint with the smallest FAD (p10). The checkpoint is chosen on the reported metric.
- CLEVR: 30,000 images downsampled from 128x128 to 64x64 (p11). The coordinator sees up to 2 conditionings in training and is tested up to 5 (p4).
- CLEVR accuracy uses 256 random position sets per number of conditionings, with coordinates in 0.3 to 0.7 and at least 0.15 apart (p12). Du's paper states no such limits, so the two CLEVR accuracy tables may not share a protocol.
- CLEVR sampling uses 100 timesteps and guidance weight 20, or 4 for RRR (p12).
- Satellite: 7,300 samples from a 20 by 20 km square, cropped to 512x512 and downsampled to 128x128 (p12). In Table 3, N is the image side and not a sample count.
- Satellite FID is CleanFID between 3,072 generated samples and all 7,300 images (p15). The reference set is the whole dataset, and no held-out split is stated.

## Hardware and training time

- All training runs on a single A100 GPU (p9).
- Music base model: 405M parameters (p3). Music coordinators: 10 epochs at batch size 4 (p10).
- CLEVR base model: 42M parameters, 1,050 epochs at batch size 32 (p4, p11). CLEVR coordinator: 38M parameters (p4).
- Satellite base model: 270M parameters, 200 epochs (p4, p13). Satellite coordinator: 26M parameters (p4, p13).
- Wall-clock training time is not stated.

## Contradictions inside the paper

- Music base model epochs: the Table 1 caption (p3) says 80, and p10 says 300.
- CLEVR coordinator epochs: p4 says 20, and p12 says 40.
- Satellite coordinator epochs: p4 says 20, and p15 says 100.
- Heun sampler for CLEVR: p12 gives S_churn = 20.0 and, in the next sentence, S_churn = 0.
- Page 4 says the coordinator "outperforms the baselines significantly". In Table 2, at 1 and 2 conditionings, RRR with Heun (98.0, 94.5) is above DDE (96.5, 94.1). DDE leads only at 3, 4 and 5.
- The Table 2 caption spells the dataset "CLEVER" (p3).
