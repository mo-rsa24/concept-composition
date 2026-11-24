# A small repeatable environment for abstract checkers (step V1)


The verdict is in [the finding on whether the small environment is repeatable](../../../../../2-classifier-guidance/report/abstract-checkers-on-digits/is-the-small-environment-repeatable.md).

## What this asks

Later stages test whether a checker can learn a property that holds across concepts, starting with plurality: are the two asked
digits both on the canvas, apart? Those tests need data whose labels are exact and that anyone can rebuild byte for byte. They also
need a sampler where adding no push changes nothing. This stage builds that data and checks both conditions before any checker
trains.

## Words this uses

**Canvas.** A 32 by 64 pixel greyscale image holding pasted 28 by 28 MNIST digits.

**Asked pair.** The two digits a canvas is being judged against. The checker receives the asked pair as its condition.

**Plurality label.** 1 when the canvas holds both asked digits, one in each half, and 0 otherwise. It is exact because the canvas
is built from known digits at known places.

**Push.** A checker's gradient added to the noise prediction during sampling.

**Zero push.** A push function that returns all zeros. It runs through the same code path as a real push, including the cap.

**Data hash.** A SHA-256 over the build version, the four MNIST files' own hashes, and every array's name, type, shape and bytes.

## Run settings

**Group.** Runs that try new ideas (group 2). This stage builds the test bed and makes no claim in either paper. The idea under test
in later stages comes from Session 7's reading, in `abstract-checkers/03-design.md`.

**Varied axis.** For the rebuild check, nothing varies: the same command runs twice. For the sampler check, one axis varies: the
push is absent or zero.

**Model.** `/home-mscluster/mmolefe/poe_repair_min_scratch/mnist-canvas/models/share-0.0/unet.pt`, the class-conditional canvas
model with 40,000 training steps and label 10 as the null label. No diffusion model is trained.

**Sampler.** DDIM with 50 steps and eta 0, the product of the two digit labels at guidance 1, from `canvas_unet.sample`.

## The data

| Split | MNIST file | Digits | Asked pairs | Canvases per pair | Canvases |
|---|---|---|---|---|---|
| `train` | training file | 0 to 4 | the 10 pairs among them | 2,000 | 20,000 |
| `seen_test` | test file | 0 to 4 | the same 10 pairs | 200 | 2,000 |
| `unseen_test` | test file | 5 to 9 | the 10 pairs among them | 200 | 2,000 |

No digit from 5 to 9 enters `train`. Pairs that mix the two ranges appear in no split, so Paper 1's cross-range sets stay untouched.

Each asked pair gets the same mix of five kinds of canvas, so the asked pair alone says nothing about the label.

| Kind | Share | What is on the canvas | Plurality label |
|---|---|---|---|
| `apart` | 4 of 8 | both asked digits, one per half, either order | 1 |
| `overlaid` | 1 of 8 | both asked digits on the same spot | 0 |
| `same_half` | 1 of 8 | both asked digits in one half | 0 |
| `single` | 1 of 8 | one asked digit | 0 |
| `wrong` | 1 of 8 | one asked digit and a different digit from the same range, one per half | 0 |

The `wrong` kind is this toy's version of Learn-to-Steer's right and wrong prompt. Two digits sit apart on the canvas, and only the
condition decides the label, so a checker that ignores the condition cannot score above chance on it.

Every canvas stores its asked pair, the digits it holds, where each was pasted, which MNIST image each came from, its kind and its
label. The build stops with an error if a digit falls outside its split's range, a stored image's MNIST label disagrees with the
stored digit, or any asked pair's positive rate differs from one half.

## What counts as a result

| Result | What it takes |
|---|---|
| Support | two rebuilds print the same data hash, and the largest absolute pixel difference between the plain sampler and the zero push is under 1e-5 on all 20 pairs at 8 seeds, in one process on one card and dtype |
| Null | the hashes differ, or the zero push differs by 1e-5 or more while two plain runs agree to under 1e-5 |
| Inconclusive | two plain runs differ from each other by 1e-5 or more, because the card's own repeatability then hides anything the push does |

**What would surprise.** A zero-push difference above 1e-5 would mean the push path changes the sampler even when the push is zero,
and every later comparison on this sampler would need that fixed first.

## Cost and what it buys

**Cost:** about 0.2 GPU-hours of this loop's 30, budgeted before any timing: 3 sampler runs over 20 pairs at 8 seeds, plus model
loading and the job's checks. The data build runs on CPU. No API tokens.

**Buys:** step V1, a test bed whose labels and rebuilds can be trusted, so a later pass or miss is about the checker.

## The ladder

No checker trains in this stage, so the overfit rung has nothing to act on.

1. **Smoke test.** Build at 1% scale and run the zero-push check on one pair at 2 seeds.
2. **1% of the data.** Build twice at 1% scale and compare hashes.
3. **Controls.** The build's own audit, then the plain-against-plain repeat on the card.
4. **Full run.** Two full builds and the zero-push check on all 20 pairs at 8 seeds.

## How to run it

The job script is `2-classifier-guidance/scripts/abstract_checkers/v1.sbatch`, submitted from a code snapshot under
`/datasets/mmolefe/poe_repair_min/loop-8/code/<repo commit>/`. Outputs go to `/datasets/mmolefe/poe_repair_min/loop-8/v1/`.

The one command that rebuilds the data and prints its hash:

```bash
python 2-classifier-guidance/scripts/abstract_checkers/build_env.py --mnist <raw MNIST folder> --out <output folder> --twice
```

## What this cannot say

A passing environment says nothing about whether any checker can learn plurality. Pasted canvases also differ from what a sampler
draws, so a checker that reads them well may still read sampled canvases badly. V3 tests that on the sampler's own output.
