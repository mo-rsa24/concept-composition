# CO3: Steer Away From Mode Collisions

arXiv 2509.25940v2 (13 March 2026), Dutta, Chen, Rajagopalan, Wei and Roy Choudhury, University of
Illinois. Read from the full PDF on the shelf, 25 pages with appendix:
`~/goal-setting/learning/papers/deep-learning/diffusion-models/compositional-diffusion-models/region-and-binding-failures/steer-away-from-mode-collisions.pdf`.

Every claim below is marked **read** (read from the paper, page given) or **inferred** (this note's own
reasoning).

## What it does

CO3 targets the same failure Paper 2 targets, named with the same example: "a cat and a dog" giving an
image where "one concept is missing, faint, or colliding awkwardly with another" (read, p. 1). Its
explanation is that the joint prompt's distribution has modes that overlap one subject's own
distribution, so a picture of a big cat with an inconspicuous dog still scores as likely under the joint
prompt (read, p. 1, Figure 1 on p. 2).

The fix is a sampler. At each early step it forms a weighted sum of the denoised estimates (Tweedie means)
under the joint prompt and under each single subject's prompt, with weights that push the sample away from
any one subject's mode (read, pp. 3 to 5). The first three steps use weights summing to 0 and resample the
noise; later steps use weights summing to 1 and correct the latent between DDIM steps (read, p. 2, p. 5).
A weight schedule keeps the sum inside what the paper calls stable regions (read, p. 5).

## What it needs

No training, no gradients, no layout and no outside model (read, p. 1, p. 2, Table 1 property columns on
p. 7). It needs one extra noise prediction per subject per corrected step (inferred from the sum over
subjects on p. 4).

## Which model and settings

SDXL only, with a 50-step DDIM sampler, guidance scale 5.0, CO3 active in the first 20% of steps
(Tc = 0.8T) with the first three of those resampling, weight modulation β = 0.8, on one A6000 (read,
p. 6). Appendix A.5 repeats it with 10 to 20 step DDIM and the DPM++ solver (read, p. 17). The paper calls
CO3 "model-agnostic" (read, p. 2) but runs no model other than SDXL: no SD3, SD3.5, FLUX or any
transformer model appears in the experiments (read, full-text search of the PDF for SD3, FLUX and DiT finds
none).

## What it measured

| Benchmark and metric | CO3 | Best other | Count behind it | Where |
|---|---|---|---|---|
| Attend-and-Excite animal pairs, BLIP-VQA, 0 to 1 | 0.7441 | 0.7390 Tweediemix; SDXL 0.6950 | not stated | Table 1, p. 7 (read) |
| Attend-and-Excite animal pairs, ImageReward | 1.2342 | 1.0082 InitNO on SD1.5; SDXL 0.7820 | not stated | Table 1, p. 7 (read) |
| Attend-and-Excite object pairs, BLIP-VQA | 0.5146 | 0.6595 SynGen on SD1.5, 0.6440 ToMe on SDXL | not stated | Table 1, p. 7 (read) |
| T2I-CompBench complex, ImageReward | 0.4406 | 0.3561 R2F; SDXL 0.3083 | not stated | Table 2, p. 7 (read) |
| T2I-CompBench complex, human rating 0 to 1, 11 raters | 0.8033 | 0.5983 SDXL | not stated | Table 2, p. 7 (read) |

No GenEval number appears in the paper (read, full-text search). The PDF states no seed count and no
images per prompt for any table (read, full-text search for "seed" and "per prompt" finds none).

The ablation on p. 8 (Table 3) shows the corrector alone lowers SDXL's object-pair BLIP-VQA from 0.4926 to
0.4796 and only the weight modulation brings it to 0.5146 (read).

## Does it already claim Paper 2's contribution

**No.** Paper 2 claims that a classifier trained afterwards, reading "two separate things in frame" from
the noisy latent, lifts several base samplers (plain PoE, the joint prompt and CO3) with the model frozen
([Paper 2's design](../../manager/paper2-four-way-design.md), "The contribution"). CO3 has no learned
component and is itself one of the base samplers. Its "plug-and-play" claim means it sits beside
classifier-free guidance on any model (read, p. 1, p. 10). Its "generalization" appendix changes the ODE
solver under CO3, not the composition rule CO3 is added to (read, p. 17).

**What it does take from Paper 2 (inferred).** The problem statement is CO3's, with the same cat and dog
example on the same model. Paper 2 cannot present "frozen SDXL drops or fuses one of two subjects" as its
finding; it has to present the learned checker, and the cross-sampler lift, as the finding.

## What it leaves open for Paper 2 (inferred)

- CO3 is untested on any transformer model, so whether mode collision is still the failure on SD3 or FLUX
  is open in its own record.
- CO3 loses to the attention methods on object pairs with colours (0.5146 against 0.6440 for ToMe,
  Table 1, p. 7, read). Binding an attribute to each subject is outside what CO3 fixes.
- This project measured CO3 drawing a clean cat and dog on 89 of 128 seeds by the automatic scorer and
  101 by eye ([the report index](../../../../../report/00-INDEX.md), the kept push comparison), so 27 to
  39 of 128 still fail on the paper's own example pair.
