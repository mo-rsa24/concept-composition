# Does a linear probe on a frozen sentence model's vectors read pairs it never saw? (step W2b)

W2 reads the five properties with no training at all: a right sentence against a wrong one. This step trains the smallest possible thing on top of the same frozen image encoders, one linear layer per step, and asks whether that reads counting and prompt match on animal and object pairs that were never in its training data. It sits between W2, which trains nothing, and W4, which trains a whole encoder.

## Words this uses

**Image vector.** The normalised output of a model's image encoder for one noised latent: 768 numbers for NoisyCLIP and frozen CLIP, 640 for Latent CLIP.

**Linear probe.** A logistic regression from one image vector to one label. It has one weight per number in the vector, plus a bias.

**Seen and unseen pairs.** Seen pairs appear in the probe's training pictures. Unseen pairs appear in none of them.

**Balanced accuracy.** The mean of the share of label-1 and the share of label-0 pictures called correctly, from 0 to 1, where 0.5 is chance.

## The run

**Group.** Runs that try a new idea (group 2). A result here cannot change a claim in Paper 2.

**Varied axis.** The frozen model whose vectors the probe reads: NoisyCLIP, frozen CLIP or Latent CLIP. Pictures, labels, noise, splits and probe training are identical. NoisyCLIP against frozen CLIP is a clean comparison, and Latent CLIP against either is mixed, for the reasons in [the W2 design](../02-can-it-read/design.md#the-run).

**Cost:** about 0.7 GPU-hours on one `bigbatch` card, for one image vector per picture per step from three models with fp16 arithmetic, and seconds of CPU to fit the probes. **Buys:** a measured answer to whether these models' vectors hold counting and prompt match in a form a linear layer can read on new pairs. The owner approved this step on 2026-11-24, which takes the loop's planned total from 8.0 to about 8.7 GPU-hours.

## The pictures

| Split | Source | Pairs | Used for |
|---|---|---|---|
| Training | `2-classifier-guidance/scripts/rung2/training_set_v4.json`, less 10% of pictures held back by the SHA-256 of their `id` | 119 after the subject guard | fitting the probes |
| Seen test | the held-back 10% | the same 119 | the seen score |
| Unseen test | `2-classifier-guidance/scripts/rung2/eval_set_v4.json`, its 7 named held-out pairs only | bison and horse, donkey and duck, guitar and traffic cone, horse and pig, llama and alpaca, rhino and zebra, windmill and tractor | the unseen score, read once |

Every pair passes `scripts/shared/disallowed_subjects.py`, which removes 28 training pictures from 4 pairs. No picture has a seed from 7001 to 7008, and no benchmark prompt set is used. A latent that cannot be found on either disk is left out, and the count left out is reported.

**Labels.** The `kind` each picture already carries, with a `(stature rule)` kind counted as its base kind and `damaged`, `pair-prompt failure` and `rung 2 negative` left out:

| Property | Label 1 | Label 0 |
|---|---|---|
| Counting | `positive`, `wrong pair` | `hybrid`, `single` |
| Prompt match | `positive` | `wrong pair` |

Spatial relation, position and size, and binding have labels only on W2's 64 cat and dog pictures, so they have no unseen pairs to test on and are left out of this step.

## The probe

Each picture is noised to each of steps 0 to 15 with one noise draw, the same draw for every model, and read by each model's image encoder. For each model, property and step there is one logistic regression, fitted with L2 strength chosen by 5-fold cross-validation on the training split over the values 0.01, 0.1, 1 and 10. Classes are weighted so that each label counts equally. The threshold is 0.5 on the predicted probability.

## The bars

These match step V2's bar, so the two can be read side by side.

| Result, per model and property | What it takes |
|---|---|
| Support | On the unseen pairs, balanced accuracy averaged over steps 0 to 15 is at least 0.75 and within 0.05 of the seen score |
| Null | It misses either condition |
| Inconclusive | Either unseen class has fewer than 30 pictures, or the probe scores below 0.55 on the seen test at every step |

The bars are constants in `2-classifier-guidance/scripts/noise_text_critic/w2b_probe.py`.

**Control.** A probe on 768 random numbers per picture, drawn once and fixed, is fitted the same way and must score near 0.5. A probe above 0.6 on random numbers would mean the labels can be predicted from something other than the picture.

**What would surprise.** A probe on Latent CLIP passing at steps 0 to 5. Latent CLIP never trained on a noisy latent, so a pass would mean its vectors keep the property even when the latent is mostly noise.

## The order of work

1. Smoke test: 32 pictures at 2 steps.
2. Extract the vectors for all pictures, logged to W&B project `prime_lab/poe-repair-animals-compose`.
3. Fit the probes and the random-number control, then read the unseen pairs once.
4. Write `review.md` with seen and unseen balanced accuracy by step, for each model and property, and example pictures beside the numbers.

## What this cannot say

**It reads and does not steer.** A probe that reads unseen pairs can still have a gradient that does not move a picture, the same gap W3 tests for the zero-shot score.

**Noised finished pictures.** As in W2, a finished picture noised to step 5 is not the sampler's real path at step 5.

**Labels came from an earlier labeller.** The `kind` on these pictures was set by Claude-API calls and hand checks for an earlier checker, and their error rate on these exact pictures has not been measured here.
