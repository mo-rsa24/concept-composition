# Our methods on DCFM's Colored MNIST benchmark

The owner chose to compare with DCFM the way papers usually do: rebuild DCFM's exact dataset and splits,
run only the project's own methods on them, and set the results beside DCFM's printed numbers. DCFM
itself is not rerun. The reviewer read a larger version on 2026-11-25 ("The fair DCFM match,
2026-11-25" in `../reviewer/2026-11-24-review.md`); its findings on the data, the bar and the
ten-holder limit of the classifier carry over below.

## The benchmark, as DCFM prints it

Colored MNIST: ten digits by ten colours, 100 combinations, spread over 10 data holders so that half the
combinations appear at no holder. Two splits: an even split, and an uneven one drawn from a Dirichlet
distribution with alpha 0.1. FID on combinations some holder saw (known) and combinations no holder saw
(novel), lower is better; DCFM does not state the number of samples behind each FID (p. 7, Table 1;
`../targets/dcfm-compositional-flow-matching-from-separate-silos.md`).

| DCFM's printed rows | Even, known | Even, novel | Uneven, known | Uneven, novel |
|---|---:|---:|---:|---:|
| Federated flow | 9.41 | 20.83 | 15.02 | 20.02 |
| Decentralized Diffusion Models | 8.19 | 25.19 | 7.46 | 17.98 |
| Diffusion Federated Dataset | 8.81 | 29.71 | 7.02 | 38.27 |
| DCFM-A | 8.53 | 11.41 | 7.33 | 9.29 |
| DCFM-B | 9.32 | 12.24 | 8.49 | 9.15 |

## What runs

| Row | What it is |
|---|---|
| calibration | **Decentralized Diffusion Models rebuilt** as a sampling rule on the same finished experts, no retraining. If our FID lands near DCFM's printed 8.19 and 25.19 (even) and 7.46 and 17.98 (uneven), our measurement is close enough to set beside DCFM's printed numbers; if not, every comparison below is marked as measured differently |
| 1 | plain product of the finished experts |
| 2 | the push that names the digit and colour only |
| 3 | the gradient-trained classifier, extended from two experts to ten |

Each holder's expert is trained only on that holder's pictures, at DCFM's split, and never retrained.

## Claims and bar

Inside the project's own rows (2 and 3 against 1, and 3 against 2) the bar is the one Z11 used, written
before the run: at least 5 points more digit-and-colour accuracy on novel combinations, an exact
one-sided McNemar test at p below 0.05, and a lower novel FID. Against DCFM's printed rows nothing is
tested statistically: the paper reports our novel FID beside theirs, says the experts were never
retrained or exchanged, and states the calibration row's result.

## What must be built

- DCFM's splits, both of them, written to a file before any training.
- Ten experts per split, each on one holder's pictures.
- The classifier extended to ten experts. The reviewer notes the current one takes exactly two and
  trains on pictures pooled from every holder; the extension says what it trains on per holder.
- The calibration row's sampling rule, from DCFM's description (p. 4, p. 7).
