# Reproducing Projective Composition on CLEVR

## What this asks

Can the printed result be reproduced: 98 of 100 images showing all 5 objects in the right places, when five single-object models are
composed over an empty background? The count is by hand, on location-conditioned CLEVR at 128 by 128 pixels.

## The row

| Row | Benchmark | Published | Source |
|---|---|---|---|
| `Bradley-Projective-T1-008` | CLEVR location-conditioned, 5 objects composed | 98 of 100 images | Table 1, p17 |

## Status

The row is `blocked` by the owner's answer to board question 21, and nothing has run. 98 stays a published number that was not reproduced.

## Why it cannot run as published

**No code or weights.** The paper links neither, a search found neither, and its [Apple research page](https://machinelearning.apple.com/research/mechanisms-of-projective-composition)
links no repository.

**The data must be rendered.** The paper generates its own CLEVR sets with Blender at 320 by 240 and downsamples to 128 by 128: 50,000 empty
background images and 1,550,000 single-object images (appendix B.1.1).

**The training is large.** Each model trains on 128 × 2^20 images at a batch size of 2,048, which the paper says takes about 16 hours on 32 A100
cards, about 512 A100-hours (appendix B.1.3). This composition needs two models, a location-conditioned one and an empty-background one, so
about 1,000 A100-hours. The brief asks for 3 independent trainings of a retrained rival, so about 3,000 A100-hours, against a limit of 24 GPU-hours
per run and a cap of six `bigbatch` cards.

**The existing checkpoints do not match.** `/datasets/mmolefe/projective-composition/checkpoints/` holds `clevr-color-20k-64-xs-8Mi` and
`clevr-none-20k-64-xs-8Mi`. They are conditioned on colour, trained on 20,000 multi-object images at 64 pixels for 8 million images seen, so they
cannot test a location-conditioned count at 128 pixels.

## If the owner wants a run

A smaller location-conditioned experiment would be a different benchmark, so it could not count as a reproduction. Any such run gets its own
design, cost line and question first.
