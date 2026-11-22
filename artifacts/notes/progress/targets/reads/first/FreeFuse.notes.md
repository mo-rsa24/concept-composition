# FreeFuse: notes on the first read

arXiv 2510.23515v3, 26 PDF pages, published in TMLR (09/2026). The CSV holds 284 rows: 249 from six result tables, 33
printed on Figures 3, 4, 6 and 17, and 2 from running text. Table 4 lists hyperparameters and holds no results.

## Numbers that could not be read

None. Every table value was typed from the page image and matched by script against the PDF text layer.

## How the rows are labelled

`Ours` and `Ours(Full)` are `FreeFuse` and `published`. The four ablated variants in Table 2 are `baseline` and keep their
printed names with `Ours` replaced by `FreeFuse`, because each one is a different method from the published one. None of
the compared methods (OmniGen, UMO, Multi-Redux, Multi-IP-Adapter, LoRAShop, Naive Flux) is one of the 19 rivals.

Some numbers are printed twice. Table 7 repeats the latencies of Table 3. The `Ours(Full)` row of Table 2 repeats the
`Ours` row of Table 1. The Figure 17 labels repeat the deltas of Table 5. The two text rows on p6 repeat the ratios on
Figure 4.

The ranges in the running text are not rows, because the table beside them prints the exact values: "1.8x to 2.1x" (p26),
"over 140 seconds" and "under 70 seconds" (p26), and "> 7.0 in Block 12-18" (p5).

## Plot-only

These curves and maps have no printed values and get no rows:

- Figure 3 right (p5): Precision@K for six attention methods.
- Figure 4 (p6): LoRA perturbation per double-stream block. Only the two ratios and their intervals are printed.
- Figure 6 (p7): the step-by-block precision map. Only the best cell, 0.704, is printed.
- Figure 10 (p18): Precision@K for four attention methods on SDXL, the only SDXL measurement in the paper.
- Figure 17 (p24): the curve heights. Their values are in Table 5.

## Protocol facts a comparison depends on

**Backbone.** FLUX.1-dev for the main benchmark (p9, p18). FreeFuse uses 28 denoising steps and extracts masks from block
18 during the first 5 steps (p9).

**Main benchmark (Tables 1 and 2).** 15 LoRAs, 10 character and 5 object. It combines 20 scenes with 4 concurrent LoRAs
each and character-object prompts, with 470 images per method (p9, p10). The paper states no seeds, guidance scale or
resolution for these tables. Table 2 does not restate its image count.

**LoRA-count test (Table 5).** k is 2, 3, 4 or 5 LoRAs, with the same LoRAs and prompts for both methods (p23). The image
count is not stated.

**Public protocol (Table 6).** The LoRAShop dual-subject protocol, which mainly evaluates two real-person LoRAs (p24). The
image count is not stated.

**These three settings are not comparable.** FreeFuse ArcFace is 0.4275 in Table 1, 0.6285 at 4 LoRAs in Table 5, and
0.6557 in Table 6 against 0.7002 at 2 LoRAs in Table 5.

**User study (Table 1, Q1 and Q2).** 17 participants aged 18 to 40 ranked 40 cases, 20 character-character and 20
character-object, with methods shown blind and shuffled (p25). Lower is better, and the six mean ranks sum to 21.0 for Q1
and 21.1 for Q2.

**Attention measurements.** Figure 3 left uses 100 samples with SAM3 masks. Figure 3 right, Figure 6 and Figure 10 use 300
samples (p5, p7, p18).

## Hardware and training time

Latency is measured on a single L20 (48GB) at 1024 x 1024, from prompt encoding to final pixel decoding (p11, p25). The
method is training-free, and the paper states no training time.

## Where the paper disagrees with itself

**Object fidelity.** p10 claims "a commanding lead" across DINOv2, DINOv3, Clip-I and Dreamsim and "superior object
fidelity", and the conclusion claims improved object consistency. Table 1 shows UMO ahead on Object DINOv2 (0.6535 against
0.6393) and Object DINOv3 (0.6884 against 0.6804).

**The ablations beat the full method on 10 cells of Table 2.** p11 describes each removal as harmful, naming identity,
boundary precision, alignment, and character and face metrics. The character and face columns support that. These object
and prompt columns go the other way:

- Without attention bias: Object DINOv2 0.6589, Object DINOv3 0.6939, Object Clip-I 0.7635, Clip-T 0.2882, HPSv2 0.2865,
  against 0.6393, 0.6804, 0.7499, 0.2766 and 0.2857 for the full method.
- Without the Router: Object DINOv2 0.6495 and Object DINOv3 0.7249.
- With plain cross-attention: Object DINOv2 0.6405 and Object DINOv3 0.6832.
- Without post-processing: Clip-T 0.2868.

**Aesthetics on the public protocol.** p24 says FreeFuse remains competitive on aesthetic metrics. In Table 6 it is last of
six methods on Aesthetic v2, at 5.7426 against 5.9921 to 6.2833 for the other five.

**"Speedup" means two different things.** Table 3 prints a ratio, LoRAShop time over FreeFuse time (1.80 at 2 LoRAs). Table
7 prints the time saved, 1 minus FreeFuse over LoRAShop (44.3% at 2 LoRAs), under the same label. Both follow from the same
latencies, so a target should name which one it uses.

**A repeated value.** OmniGen and UMO both print LVFace 0.1661 in Table 1 (p10). The image confirms both digits, and the
paper does not comment on it, so one may be a copying error in the paper.

**A one-digit gap in Table 5.** At 3 LoRAs the HPSv2 delta prints +0.0019, while 0.2804 minus 0.2786 is 0.0018. Rounding of
unrounded scores would produce this.

**Precision units.** Figure 6 defines Precision@10% as a percentage and prints it as the fraction 0.704 (p7).

**Significance.** p10 says 470 images were generated "to ensure statistical significance", and Tables 1, 2, 5 and 6 print no
interval or test.
