# RTD first read: notes

PDF: arXiv 2608.03135v1, Rectify Then Diffuse, 28 pages. Page numbers are PDF pages. `R2F` is Rare-to-Frequent, which is off the rival list and stays a `baseline`.

## Numbers that could not be read

None. Every printed table value is legible at 300 dpi and agrees with the PDF's text layer.

## Plot-only

Figure 5 (p7) plots BLIP-VQA and ImageReward against the relative correction ratio and the pilot timestep. It prints no point values, so it has no rows.

The curve in Figure 2 (p3) has no point values either. Its three printed labels are rows: Vanilla SDXL 0.269 and RTD 0.170 at the step 49 end of the curves, and a badge reading 29.5% lower average overlap. The two end labels alone give a 36.8% reduction at step 49. The 29.5% is an average over the curve, and the paper does not print the values it averaged.

## Which rows were copied and which were rerun

**The footnote.** Footnote 1 reads: "All CO3 and SDXL results in this table are taken directly from the original paper (Dutta et al. 2026)." Its marker is on the caption of Table 4 (p7). Its text is printed at the bottom of p6, under Table 2. Tables 1, 2, 3, 5 and 6 carry no marker.

**What the paper states as its own runs.** The RTD rows are the paper's runs. Section C.1 (p13) says each baseline uses "the guidance rule and method-specific hyperparameters from its official implementation". Section C.2 (p13) says "all metrics for a method are computed on the same generated set". No sentence names a baseline row in Tables 1, 2 or 5 as rerun or as copied.

**Table 1 against CO3 Table 1.** All 66 BLIP-VQA and ImageReward cells of the 11 non-RTD rows are digit-for-digit equal to CO3's Table 1. The footnote does not cover them. The S-IoU5 column is new in this paper.

**Table 2 against CO3 Table 2.** 19 of the 20 ImageReward cells for the five non-RTD methods are equal. All 17 human-evaluation cells that CO3 prints differ. SynGen's three RareBench human scores (0.3755, 0.4378, 0.5804) have no counterpart, because CO3 prints a dash there. The S-IoU5 columns are new.

**Table 3.** The `No rectification` row equals the SDXL row of Table 1 in all six cells.

**Table 4 against CO3 Tables 6 and 7.** Of the 72 SDXL and CO3 cells the footnote covers, 59 equal a CO3 cell for the same solver and step count, and 13 do not:

- The 12 cells in the `DPM++ 2M`, 50-step rows for SDXL and CO3 are CO3's Table 6 rows labelled `SD-DDIM` and `CO3-DDIM` at 50 steps. CO3 prints no DPM++ result at 50 steps. Those two rows are DDIM numbers under a DPM++ heading, and RTD's own 50-step DPM++ row (p7) is compared against them.
- SDXL DDIM 50-step BLIP-VQA A-O is 0.8654 here. CO3 Tables 6 and 7 print 0.8658 at that setting, and CO3 Table 1 prints 0.8654.

**Table 5.** The SDXL `Vanilla` BLIP-VQA row (0.6951, 0.8654, 0.4926) equals CO3 Table 3's SDXL row, and the SDXL `CO3` row equals CO3 Tables 1 and 3. Table 5 has no footnote. CO3 reports no SD 2.1 result, so the SD 2.1 rows have no counterpart there.

## Numbers that differ from CO3's Table 2

Every method and dataset pair with a CO3 value is listed. Values are RTD Table 2 against CO3 Table 2.

| Method | Dataset | Metric | RTD prints | CO3 prints |
|---|---|---|---|---|
| SynGen (SDXL) | T2I-CompBench Complex | ImageReward (IR) | 0.2899 | 0.2889 |
| SDXL | T2I-CompBench Complex | Human Evaluation (HE) | 0.4678 | 0.5983 |
| SynGen (SDXL) | T2I-CompBench Complex | Human Evaluation (HE) | 0.4127 | 0.5133 |
| ToMe | T2I-CompBench Complex | Human Evaluation (HE) | 0.4361 | 0.5866 |
| R2F | T2I-CompBench Complex | Human Evaluation (HE) | 0.3749 | 0.4816 |
| CO3 | T2I-CompBench Complex | Human Evaluation (HE) | 0.6278 | 0.8033 |
| SDXL | RareBench Concat | Human Evaluation (HE) | 0.4590 | 0.5416 |
| ToMe | RareBench Concat | Human Evaluation (HE) | 0.3974 | 0.4500 |
| R2F | RareBench Concat | Human Evaluation (HE) | 0.3862 | 0.4300 |
| CO3 | RareBench Concat | Human Evaluation (HE) | 0.5246 | 0.6167 |
| SDXL | RareBench Relations | Human Evaluation (HE) | 0.4133 | 0.5016 |
| ToMe | RareBench Relations | Human Evaluation (HE) | 0.4607 | 0.4983 |
| R2F | RareBench Relations | Human Evaluation (HE) | 0.4796 | 0.5200 |
| CO3 | RareBench Relations | Human Evaluation (HE) | 0.4889 | 0.6200 |
| SDXL | RareBench Complex | Human Evaluation (HE) | 0.6174 | 0.6283 |
| ToMe | RareBench Complex | Human Evaluation (HE) | 0.4653 | 0.48666 |
| R2F | RareBench Complex | Human Evaluation (HE) | 0.5536 | 0.53666 |
| CO3 | RareBench Complex | Human Evaluation (HE) | 0.5977 | 0.61166 |

CO3's score for itself on T2I-CompBench Complex is 0.8033 in its own paper and 0.6278 here. RTD's 0.6615 beats the 0.6278 and sits below the 0.8033. Human scores from the two papers come from different rating panels and should not be compared across papers.

## Protocol facts for Tables 1 and 2

**Sampler and steps.** Deterministic DDIM with 50 inference steps (p4, p13).

**Guidance.** Classifier-free guidance at scale 5.0 with an empty negative prompt (p13).

**Resolution.** 1024 x 1024 (p4, p13).

**Model.** Stable Diffusion XL Base 1.0 with the SDXL VAE fix (p13).

**Seeds.** "For each method and prompt, we generate images using three fixed random seeds and report the mean performance" (p4). The seed values are not stated.

**Images per prompt.** No count is printed. The human protocol rates "the output from every seed" (p13), which reads as one image per seed and three per prompt per method.

**Who that setting covers.** Section C.1 says the configuration "applies to RTD and the controlled SDXL comparisons". Each other baseline uses its official guidance rule and hyperparameters (p13), so 5.0 and 50 steps are not stated for them.

**RTD settings.** One pilot forward pass and one backward pass at t_pilot = 980, rho = 0.02, one update, using cross-attention at 32 x 32 (p4, p12, p13).

**Prompt sets.** AE-Bench has 276 prompts across A-A, A-O and O-O. T2I-CompBench uses 1,000 complex prompts. RareBench has 120 prompts across Concat, Relations and Complex (p4, p13). Per-category counts are not printed.

**Backbone of each baseline.** Page 4 says SDXL is the base model "for all main experiments", yet Table 1 has a separate SD 1.5 row. Only `SD 1.5`, `SDXL`, `SynGen (SDXL)`, `CO3` and `RTD` name or imply a backbone, so `model` is empty for the other baseline rows in Tables 1 and 2. CO3 (p17) lists Attend-and-Excite, Divide & Bind and InitNO as SD 1.5 methods, and their cells here equal CO3's.

## Who judged the human scores

11 participants (p4, p13). The paper does not say who they were, how they were recruited, or whether they were experts. It reports no inter-rater agreement and no uncertainty.

For each prompt, the outputs of RTD and the baselines are shown side by side with method names hidden and display order randomized independently. Every participant rates "the output from every seed for each prompt and method" on an integer scale of 0 to 5. Each rating is divided by 5 and averaged across participants and seeds (p13). The protocol follows CO3 and is adopted from R2F (p4).

Taken literally, 1,120 prompts x 3 seeds x 6 methods is 20,160 ratings per participant. The paper does not say whether a subset was used.

## Hardware and cost

All main experiments run on one NVIDIA H200 GPU with Python 3.10.20, PyTorch 2.5.1 and "CUDA 12.1" as printed (p13). RTD is training-free. Table 6 (p7) gives per-image time at 1024 x 1024 with 50-step DDIM: SDXL 4.89 s, CO3 12.02 s, RTD 5.20 s including 0.31 s of pilot work. CO3's own Table 8 used an A100 and printed 7.20 s for SDXL and 19.90 s for CO3, so the two papers' times are on different hardware.

## Contradictions inside the paper

**DPM++ rows that are DDIM numbers.** Table 4's SDXL and CO3 rows at 50 DPM++ steps carry CO3's 50-step DDIM values, as listed above. The p7 claim that RTD has the best 50-step A-A and O-O ImageReward "under both solvers" rests on those rows for DPM++.

**RTD's BLIP-VQA is identical across the two solvers at 10 and 20 steps.** DDIM and DPM++ 2M both print 0.7895, 0.8334, 0.7084 at 10 steps and 0.8015, 0.8546, 0.6907 at 20 steps. ImageReward differs between the solvers in those same rows, and the 50-step BLIP-VQA row differs (0.8081, 0.8422, 0.7503 against 0.8131, 0.8742, 0.7476). Six equal four-digit scores from two different samplers suggest a duplicated column. The p7 sentence quoting 0.7895 and 0.7084 relies on them.

**A text claim the pictures contradict.** Page 5 says that for "a red backpack with orange glasses" RTD "produces clearly separated entities with the requested attributes". In Figures 1 (p1) and 2 (p3) the RTD picture has red glasses and a red backpack with an orange pocket.

**The same run differs by a digit across tables.** CO3 ImageReward A-A and A-O are 1.2342 and 1.6744 in Table 1 and 1.2341 and 1.6743 in Table 4. SDXL BLIP-VQA A-A is 0.6950 in Tables 1, 3 and 4 and 0.6951 in Table 5. SDXL BLIP-VQA A-O and O-O at 50 steps are 0.8654 and 0.4926 under DDIM and 0.8658 and 0.4925 under DPM++. These match differences already present between CO3's own tables.

**An Overall that does not match its row.** Table 5's SD 2.1 CO3 Overall is 0.6650. The mean of its three printed scores is 0.66507, which rounds to 0.6651.

**"Consistent gains across backbones" (p2, p8).** On SD 2.1 in Table 5, vanilla beats RTD on A-O (0.8003 against 0.7954) and O-O (0.6237 against 0.6181). RTD leads vanilla on A-A and by 0.0049 on Overall (0.7110 against 0.7061). On SDXL, RTD is below vanilla on A-O BLIP-VQA (0.8422 against 0.8654).

## Text results that recompute

The percentages and differences on p5 and p7 recompute from the printed tables, to the digit printed: 16.5, 6.7, 45.8, 19.6, 8.6, 4.7, 5.8 and 5.4 percent, the differences 0.2854, 0.2172, 0.1131, 0.2577, 0.5103, 0.5355, 0.4220, 0.1379 and 0.4198, the 6.3% overhead and the 2.3 times speed factor.
