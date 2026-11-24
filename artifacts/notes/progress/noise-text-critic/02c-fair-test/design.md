# Design: is W2's null fair, and is the setup readable at all? (step W2c)

W2 found that no sentence score reads counting, prompt match or left-right order at steps 0 to 15. Before W3 is stopped on that, this step checks two things:
- **The positive control.** Does a reader that is known to work read the same pictures under W2's exact rules? If it does, the information is there at those steps.
- **A fair test of the sentence scores.** Do they still fail when each gets its best fair chance?

The owner asked for both on 2026-11-24. This file is written after W2's scores were seen, so every addition below is a fixed practice from outside this data, and the bars are fixed before any W2c score exists.

## What stays identical to W2

- The same 220 labelled cat and dog pictures (`eval_set_catdog_v2.json`), noised by the DDIM scheduler's `add_noise`.
- The same four noise draws per step, generated with the seed `1000 × step + draw` over the full W2 latent stack, so each picture gets exactly the noise W2 gave it.
- The same split of seeds into a threshold half and a reading half, the same threshold rule, the same balanced accuracy, and the same 10,000-resample bootstrap over seeds. The code imports these from `w2_read.py` unchanged.

## What is added

**Group 1, the positive control.** The kept readout checker, iteration 4 (`readout_pair/readout-catdog-more3-long/head_step_002000.pt`), reads SDXL's own features at each step. Its checker loop recorded 0.63, 0.74, 0.78 and 0.79 at steps 0, 5, 10 and 15 on these pictures, under a different protocol. Here its logit is the score, under W2's protocol. Caveat: the 220 pictures were used to choose which iteration to keep, so this row is slightly optimistic.

**Group 2, sentence scores at their best fair setting.** Each change below is standard practice, and none is tuned on this data:

| Change | Why it is fair | Rows |
|---|---|---|
| Prompt ensembling: each sentence's text vector is the mean of its normalised vectors under 5 fixed wordings, "a photo of {}.", "a picture of {}.", "an image of {}.", "a photograph showing {}." and "{}.", where {} is the sentence without a leading "a photo of " | CLIP's own recommended practice for zero-shot scoring (CLIP paper, section 3.1.4) | `<model>_ens` for NoisyCLIP, frozen CLIP and Latent CLIP |
| Every step from 20 to 29 is read as well as 0 to 15 | NoisyCLIP trained on latents 20 to 29 of a 50-step run (`01-reading/noisyclip.md`, p. 6) | reported as the mean over steps 20 to 29 |
| The sampler's real states: 195 held-out cat and dog joint runs, 105 clean and 90 rejects by `trajectories/labels.json`, read at steps 0 to 20 as saved, with no added noise | NoisyCLIP trained on real trajectory states, and W3's push acts on them | Group 3 |

**A third task, the clean pair.** Positive against wrong pair, hybrid and single. It is the readout checker's own task, so the control is judged on what it was trained for. The sentence scores use the right sentence "a photo of a cat and a dog" against all five wrong sentences from counting and prompt match.

**Group 3, the real trajectory states.** The same models, sentences and ensembles, plus the readout checker, on the 195 held-out runs, on the clean-pair task only, with one draw because the states are fixed. These runs' labels come from `label_trajectories.py` (a model's call on the finished picture, not an eye call), so this group's labels are less sure than the 220 pictures'.

## The bars, fixed before any W2c score

| Result | What it takes |
|---|---|
| **Control holds** | The readout checker's mean over steps 0 to 15 on the clean-pair task of the 220 pictures is at least 0.65, and above frozen CLIP's plain sentence score by more than 1.96 standard errors |
| **A sentence score reads** | Any sentence row (plain or ensembled, any of the three models) reaches the W2 bar, a mean of at least 0.75 over steps 0 to 15, on any task in Group 2 or Group 3 |
| **W2 stands** | The control holds and no sentence row reads: the setup is readable and sentence scores do not read it at steps 0 to 15 |
| **W2 is inconclusive** | The control does not hold: this protocol cannot show reading at these steps even for a reader known to work |
| **The finished-picture check** | At step 49, where almost no noise is left, each sentence row should read counting (plurality, two separate animals against a hybrid or one animal) and the clean-pair task at 0.75 or more. W2 found NoisyCLIP at 0.71, frozen CLIP at 0.67 and Latent CLIP at 0.54 on counting at step 49. A row that fails this check cannot be expected to read plurality earlier, so the review reports it beside every early-step number |
| **In-window reading** | NoisyCLIP's mean over steps 20 to 29 reaching 0.75 on prompt match is reported as context; the push acts at 0 to 15, so it changes no W3 gate on its own |

There are 24 sentence rows here: 6 readers (3 models, plain and ensembled), on 3 tasks in Group 2 and 1 task in Group 3. The means have standard errors of about 0.02 to 0.04, and W2's best row was 0.56, so a pass by chance is unlikely (inferred).

## Group 4: counting with sentences that name no species

W2's counting sentences mix the species with the count. The wrong sentence "one animal that is part cat and part dog" shares both species words with the right one, so two cats or two dogs match neither, and NoisyCLIP calls them one animal on 23 of 46 finished pictures (`02-can-it-read/review.md`). The owner chose on 2026-11-24 to rescore counting with sentences that name no species, fixed here before any score:

| Role | Sentence |
|---|---|
| Right | a photo of two animals |
| Wrong | a photo of one animal |
| Wrong | a photo of one animal with mixed features |

- **What stays the same.** The pictures, labels, draws, halves and analysis are W2's counting task. The image side is W2's saved image vectors (`loop-9/w2/read/vectors-<model>.npy`), so only the text changes.
- **Rows.** Each of the three models, plain and with the five wordings above.
- **Bars.** The same: a mean of at least 0.75 over steps 0 to 15 reads, and 0.75 at step 49 passes the finished-picture check.
- **Code and cost.** `w2c_species_free.py`, as stage `species` of `w2.sbatch`. It encodes three sentences in five wordings per model and reuses saved vectors, so it takes a few GPU-minutes.

## Run

- **Group:** `loop9-w2c`. **Varied axis per comparison:** the reader, with pictures, draws, halves and labels held fixed.
- **Cost:** one `bigbatch` job, about 0.5 GPU-hours (estimate: about 30,000 SDXL U-Net passes for the readout checker, plus the CLIP models). Smoke test first, on 24 pictures, steps 0 and 15, and 8 trajectory runs.
- **Buys:** whether W2's null means "sentence scores cannot read these steps" or "nothing can read these steps under this protocol", and whether a fairer setting changes it. That decides whether W3 runs.
- **Code:** `2-classifier-guidance/scripts/noise_text_critic/w2c_fair.py`. **Output:** `/datasets/mmolefe/poe_repair_min/loop-9/w2c/`. **Review:** `02c-fair-test/review.md`.
