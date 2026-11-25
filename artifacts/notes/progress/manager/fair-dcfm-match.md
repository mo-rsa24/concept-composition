# A fair comparison with DCFM on Colored MNIST

The owner chose to build this, reviewed first, then run on RunPod. It turns question 108's third
experiment into a table and a figure Paper 1 can carry. It is a design for review, not a result.

## Why

The project's coloured-digit results have so far been set beside DCFM's printed numbers (DCFM-A 11.41
novel-combination FID, its finished-expert baselines 25.19, Table 1, p. 7), which is context only. The
two setups differ in the number of data holders (2 against 10), the split, the model family (small
noise-prediction diffusion models against flow models), the FID code and sample count (DCFM states none),
and DCFM's code is not public (`targets/dcfm-compositional-flow-matching-from-separate-silos.md`). A fair
table runs every method in one pipeline on the same data, splits, models, FID code and sample counts.

## Rows

| Row | What it is | Source |
|---|---|---|
| 1 | plain product of the finished experts | the floor; Z9 arm 1 |
| 2 | DCFM's finished-expert baselines, rebuilt: Decentralized Diffusion Models (routing between experts) and Diffusion Federated Dataset | DCFM pp. 4 and 7 describe both |
| 3 | **a DCFM-style row**: the experts retrained with CoInD's public conditional-independence penalty, exchanged once as DCFM-A does | CoInD's public code; DCFM pp. 5 and 6 |
| 4 | the push that names the digit and the colour only, at row 5's length | Z9's and Z11's control a |
| 5 | **the gradient-trained classifier** | Z11's method, unchanged |
| context | DCFM's printed numbers, in their own column and never compared directly | DCFM Table 1 |

## Splits

DCFM's own Colored MNIST split (10 holders, half of the 100 digit-colour combinations on no holder; the
even split and the uneven Dirichlet 0.1 split), and the project's two-holder split from Z9 and Z11.

## Measures

Novel-combination FID and known-combination FID with one FID code and a stated sample count, plus the
digit-and-colour accuracy by Z9's judge, with intervals over seeds. One picture figure: the same unseen
combinations under every row.

## Open questions for the reviewer

1. Is row 3 a faithful stand-in for DCFM-A when built from CoInD's code on our diffusion models, and
   what must the paper say about the difference from DCFM's flow models?
2. Can rows 2 be rebuilt from DCFM's descriptions alone, or should they be dropped and DCFM's printed
   numbers for them kept as context?
3. The bar, written first: what lead over rows 3 and 4 counts, on which split, with which test.
4. Cost and card type: all rows on RunPod RTX 4090s, so every number shares one card type.
