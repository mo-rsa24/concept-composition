# Reproducing CO3 on T2I-CompBench Complex and RareBench Concat

## What this asks

Does the authors' released CO3 code give the ImageReward scores printed in CO3's Table 2? CO3 corrects the early sampling
steps of SDXL by moving away from what each single concept would draw on its own.

## The run

**Group.** Reproduction (group 4). Each number freezes as a baseline (group 5) when it lands.

**Varied axis.** None. CO3 runs on the prompts and seeds of the RTD and plain SDXL renders, from the same starting noise.

**Buys.** `RTD-T2-041` (0.4406, Complex) and `RTD-T2-044` (0.1610, Concat) in `table.csv`. Both values equal CO3's own
Table 2, as do 19 of the 20 baseline ImageReward cells in RTD's Table 2, so RTD probably did not rerun CO3.

**Cost.** Three seeds per prompt on the 988 Complex and 33 Concat prompts the subject guard allows is 3,063 pictures. CO3 adds 40
corrector iterations of 4 to 5 U-Net rows each to SDXL's 100 rows, so a picture takes about 40 to 50 seconds at the 14 seconds
per 100 rows measured on an RTX 3090. The owner approved a budget of about 33 to 45 GPU-hours on board question 18. The smoke
rung measures the real time, and a run going materially past the budget stops and asks.

## The paper and the released code disagree

The paper states its settings in section 4.1 and appendix A.4. The script its README says to run, `run_sample_co3.sh` at commit
`35ea972`, uses different ones.

| Setting | Paper | Released script |
|---|---|---|
| Guidance | DDIM, λ = 5.0 | CFG++, 0.8 |
| Steps corrected | 10 (the first 20%) | 6 |
| Weight modulation β | 0.8 | 0.9 |
| Resampling steps | 3 | 3 |
| Corrector iterations per step | 5 | 10 on the first 2 steps, hard-coded in `Co3.py`, then 5 |
| Corrector guidance `lmda` | λ̃ = λ, so 5.0 | 0.8 |

The code does not settle this. Its corrector re-noises with the unconditional prediction whatever the sampler is, which is the
CFG++ form, and `attn_utils.py` documents `lmda` as between 0 and 1. Running the paper's λ = 5.0 through that corrector puts it
outside the range the code documents.

By the owner's answer to board question 16, this run uses the released script exactly as shipped and is judged against the
paper's Table 2. Each row's note states both sets of settings, because a miss could come from the settings rather than from CO3.

## Settings the paper and code leave open

**Seeds.** CO3's paper and code name no evaluation seeds, and the released script renders one seed, 1688. The row's protocol text,
from RTD's paper, states three seeds per prompt. By the owner's answer to board question 18, this run uses seeds 0, 1 and 2, the
same seeds as RTD and plain SDXL.

**Starting noise.** CO3 draws its own noise on the CPU. This run passes in the same starting latent the RTD renderer draws for
that seed, so the methods differ only in what they do from there.

**Concepts.** The paper extracts nouns with Stanza, and RTD's prompt manifests were built the same way, so CO3 reads the concepts
from those manifests.

## The test, fixed before the first run

ImageReward is scored per picture. The standard error comes from resampling prompts with replacement 10,000 times. CO3 prints
no standard error, so the published one is taken as equal to ours. On the allowed prompts, CO3 counts as `reproduced` on a row
when |ours − published| ≤ 1.96 × √(SE_ours² + SE_published²), and otherwise as `not reproduced`. Every number carries its prompt
count, such as "scored on 988 of 1,000 prompts". The rule is in `scripts/reproductions/verdict.py` in the replications hub.

## The ladder

Each rung saves its config, metrics, image sheet and commit under `/datasets/mmolefe/poe_repair_min/loop-5/renders/CO3/<rung>/`.

1. **Smoke test.** Two prompts on seed 0, with the time per picture recorded.
2. **Overfit.** Does not apply, because nothing trains.
3. **1% of the data.** The first 10 allowed Complex prompts and the first 4 allowed Concat prompts, with the sheet opened at full
   size beside plain SDXL.
4. **Controls.** With the corrector replaced by the identity, CO3's loop must match its own loop with correction off to under
   1e-5 (largest absolute latent difference), in one process on one card. The real corrector must change the final latent by more
   than 1e-3.
5. **Full protocol.** All 3,063 allowed pictures.
