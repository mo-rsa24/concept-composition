# Reproducing Rectify Then Diffuse on T2I-CompBench Complex and RareBench Concat

## What this asks

Does the authors' released code, run at the settings they printed, give the ImageReward scores they printed for Rectify
Then Diffuse (RTD)? ImageReward is a learned score of how well a picture matches its prompt, averaged over pictures, and
higher is better.

RTD makes one small change to the starting noise of an SDXL render, then samples normally. The result feeds the two
Paper 2 targets that RTD sets.

## The run

**Group.** Reproduction (group 4). Each number freezes as a baseline (group 5) once it lands, and no RTD setting changes
after that.

**Varied axis.** None for RTD. Plain SDXL runs on the same prompts, seeds and starting noise as a separate run, because
its printed scores check our scorer and the comparison protocol requires a rerun of plain SDXL on our setup.

**Buys.** `RTD-T2-051` and `RTD-T2-054` in `table.csv`, the targets on T2I-CompBench Complex and RareBench Concat.

**Cost.** The subject guard allows 988 of the 1,000 Complex prompts and 33 of the 40 RareBench Concat prompts. With 3 seeds
each, that is 3,063 pictures per method. At 15 seconds per picture, the time measured on an RTX 3090 in the smoke rung, RTD
costs about 13 GPU-hours and plain SDXL about 13 GPU-hours.

## The prompts the subject guard keeps out

`scripts/shared/disallowed_subjects.py` is a hard rule for every render and score in this project, and each script here
imports it. On these two benchmarks it keeps out 12 Complex prompts (6 blocked permanently, 6 needing a decision in the pair
ledger) and 7 Concat prompts (1 blocked, 6 needing a decision). No ledger row covers this reproduction.

A published score averages all its prompts, so a score on fewer prompts is a weaker test of it. By the owner's answer to board
question 14, RTD, the other rivals and our method are all scored on the same allowed prompts, each row is judged on them, and
every number carries its count, such as "scored on 988 of 1,000 prompts". The prompts needing a decision stay out, and the
pair ledger does not change.

## The numbers to reproduce

| Row | Method | Benchmark | Published ImageReward | Source |
|---|---|---|---|---|
| `RTD-T2-051` | RTD | T2I-CompBench Complex | 0.4661 | RTD Table 2, p6 |
| `RTD-T2-054` | RTD | RareBench Concat | 0.4464 | RTD Table 2, p6 |
| scorer check | plain SDXL | T2I-CompBench Complex | 0.3083 | RTD Table 2, p6, equal to CO3's Table 2 |
| scorer check | plain SDXL | RareBench Concat | 0.1018 | RTD Table 2, p6, equal to CO3's Table 2 |

## The protocol

| Setting | Value | Where it comes from |
|---|---|---|
| Model | SDXL Base 1.0 with `madebyollin/sdxl-vae-fp16-fix` | paper p4 and `rtd/config.py` |
| Size | 1024 by 1024 | paper p4 and `rtd/config.py` |
| Sampler | deterministic DDIM, 50 steps | paper p4 and `rtd/config.py` |
| Guidance scale | 5.0, empty negative prompt | scoreboard protocol and `rtd/config.py` |
| Update size | ρ = 0.02 of the starting noise's length | paper p4 and `rtd/config.py` |
| Prompts | the authors' `download_data.py` and `prepare_data.py` | repository commit `241eb37` |
| Seeds | 0, 1, 2 | `generate_dataset.py` default; the paper says "three fixed random seeds" without naming them |
| Scorer | ImageReward v1.0 from the `image-reward` 1.5 package in `co3` | named in the paper; the package version is unstated |
| Libraries for rendering | diffusers 0.32.2, transformers 4.48.3, accelerate 1.3.0, as the authors pin them | `environment.yml` |

**Two places the paper and the code differ, both settled by the code as the brief requires.** The paper prints a pilot
timestep of 980 and the code requires 981. The paper says T2I-CompBench concepts come from benchmark annotations, and
`prepare_data.py` extracts them with Stanza for every benchmark.

**One difference we cannot remove.** The authors used an NVIDIA H200, and our cards are different models.

## The test, fixed before the first run

ImageReward is scored per picture, so the standard error comes from a bootstrap: resample prompts with replacement 10,000
times, keeping each prompt's 3 pictures together. The paper prints no standard error, so the published one is taken as
equal to ours.

On the allowed prompts, RTD counts as `reproduced` on a row when |ours − published| ≤ 1.96 × √(SE_ours² + SE_published²), and
otherwise as `not reproduced`, with both numbers and the prompt count recorded. The constants and this rule sit in
`scripts/reproductions/verdict.py` in the replications hub.

Plain SDXL takes the same test against its printed scores. If plain SDXL fails while the prompts and scorer are correct,
the RTD row is read as a difference in setup and goes through the diagnosis procedure before any verdict is written.

## Seeds

Seeds 0, 1 and 2 are the evaluation seeds. Nothing is tuned here, so this run has no tuning seeds, and no setting is
chosen on any seed.

## The ladder

Each rung saves its config, metrics, image sheet and commit under
`/datasets/mmolefe/poe_repair_min/loop-5/renders/RTD/<rung>/`.

1. **Smoke test.** Prepare the prompts, render 2 Complex prompts on seed 0 with both methods, and score them. It passes when
   every output exists and the time per picture is recorded.
2. **Overfit.** Does not apply, because nothing trains.
3. **1% of the data.** Render the first 10 allowed Complex prompts and the first 4 allowed Concat prompts on all 3 seeds with
   both methods. It passes when the
   image sheet opened at full size shows the prompts' subjects and the scores land in ImageReward's usual range.
4. **Controls.** With the update size at 0, the RTD path must match plain SDXL's final latent to under 1e-5 (largest
   absolute difference), in one loop on one card. With the update at 0.02, the same comparison must differ by more than
   1e-3, which shows the update does something. Nothing is believed until both pass.
5. **Full protocol.** All 3,063 allowed pictures per method, split across up to six `bigbatch` jobs.
