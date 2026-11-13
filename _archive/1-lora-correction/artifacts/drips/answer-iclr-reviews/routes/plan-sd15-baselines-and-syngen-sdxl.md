# Plan: the SD 1.5 baselines for Table 4, and SynGen ported to SDXL for Table 1

Two GPU job plans for the ICLR 2027 review response of submission 56173. Plan 1 renders Attend-and-Excite, SynGen, Divide-and-Bind and PoE on Stable Diffusion 1.5 for Table 4 (slot R5, compute job 10 in [the compute plan](../../../../../../1-lora-correction/report/paper-review/prze-10-compute.md)) and its image grid, Figure 9 (slot R13). Plan 2 ports SynGen to SDXL so Table 1 (slot R1) can carry a SynGen row on the same latents as every other SDXL method. Nothing here is launched: each section ends with the commands that run it once you say yes.

Every time and GPU-hour figure below is an **estimate** unless it is marked measured. No SD 1.5 binding method has been timed on this cluster, and the smoke tests replace these estimates with measured rates before any production job is sized.

## Contents

- [What both plans start from](#what-both-plans-start-from)
- [Decisions to make before launch](#decisions-to-make-before-launch)
- [Plan 1: compute job 10 on SD 1.5](#plan-1-compute-job-10-on-sd-15)
- [Plan 2: SynGen ported to SDXL for Table 1](#plan-2-syngen-ported-to-sdxl-for-table-1)
- [Environment facts these plans depend on](#environment-facts-these-plans-depend-on)

## What both plans start from

**The prompts and seeds.** The 276 Attend-and-Excite prompts in `data/prompt-sets/attend-and-excite/a.e_prompts.json` (animals 66, animals with objects 144, objects 66), split into two concepts per line in `1-lora-correction/scripts/review_benchmark_pairs.txt`, at seeds 9 to 12: 1,104 images per method.

**Where the tiles go.** The existing layout is `<method>_<setting>/r1/<pair slug>/seedNN/<method>.png` with a JSON sidecar beside each tile (`1-lora-correction/scripts/review_baseline_cell.py`, function `tile`). `1-lora-correction/scripts/review_score_rows.py` reads any folder whose name ends in `/r1`, looks the pair slug up in the benchmark list, and copies `model`, `method`, `adapter`, `sampler`, `steps`, `guidance`, `resolution`, `seed`, `setting` and `nfe` from the sidecar. `co3_default/r1` holds 276 pair folders on `/datasets` (counted 2026-11-11), so a new method that writes the same layout is read by the row builder and the scorers with no change to either.

**The scorers.** BLIP-VQA (T2I-CompBench's script, `t2icomp` environment, RTX 3090), CLIP score and ImageReward (`co3`, CPU on `batch`), and the GroundingDINO animal count (`co3`), per [the protocol's scorer table](../../../../../../1-lora-correction/report/paper-review/protocol.md#the-scorers). Their outputs join into `scores/scores_long.csv` under `/datasets/mmolefe/poe_repair_min/outputs/paper_review/`, whose header is `image,prompt,run,table,model,method,adapter,sampler,steps,guidance,resolution,pair,seed,group,setting,nfe,clip,blip,ir,comp`. `1-lora-correction/scripts/review_build_tables.py` writes `model` as `SDXL` in its two table branches, so building Table 4 needs one R5 branch there; that change belongs to the scoring step after these renders, not to these plans.

**Whether "ours" on SD 1.5 exists.** It does not exist and is not scheduled. Route step 10 (unchecked in `plans/route-answer-the-iclr-reviews.md`) builds the SD 1.5 joint and PoE columns, the measured-correction sampler and a ceiling test; route step 12 trains an SD 1.5 adapter (7 to 10 GPU-hours, plus a pair filter, caches and evaluation) only if that ceiling test passes and you say yes. Until then the layout sheet's R5 row "ours (SD 1.5 adapter)" is dropped and the limit is stated in the caption. The ceiling bar is not yet written into the protocol. InitNO, CONFORM and the measured correction, which the layout sheet also lists for R5, are outside this file and stay owed to route step 10.

**Where these rows sit in the tables.** Attend-and-Excite, SynGen and Divide-and-Bind all denoise from the joint prompt "a cat and a dog". Tables 1 and 3 already place methods that read the joint prompt (joint, CO3) above a rule as references, with bold marking the best method that never reads it (commit `426bb86c`). The three methods here belong above that rule in Table 4, and the SDXL SynGen row belongs above it in Table 1.

## Decisions to make before launch

Each has a recommendation. Nothing launches until each is answered.

| # | Decision | Options | Recommended, and why |
|---|---|---|---|
| D1 | Sampler on SD 1.5 | (a) 50 DDIM steps for every column, the scheduler swapped from the thin runner with `DDIMScheduler.from_config(pipe.scheduler.config)`, no method code edited; (b) each method's released default, which is the PNDM scheduler stored in the SD 1.5 snapshot | (a). Protocol rule 2 fixes 50 DDIM steps per model, and PoE has to share one sampler with the methods it is compared against. The caption says so. The smoke test also renders one cell per method under PNDM so the size of the difference is on file. |
| D2 | SynGen's `include_entities` flag | (a) the released default, `False`; (b) `True`, which adds a separation loss between bare nouns | (a), per rule 3. Under it SynGen's loss is exactly zero on any prompt without an adjective, so it draws what plain SD draws from the joint prompt on 102 of the 276 prompts (all 66 animals prompts and 36 of the 144 animals-with-objects prompts; counted from the prompt file). The caption states this, and Table 4 reports SynGen per set so the animals set is read as the joint prompt. |
| D3 | Precision on SD 1.5 | fp32 for every column, or fp16 for PoE and joint | fp32 for all. The three binding methods load in fp32 by default, and an fp16 PoE beside them would differ on a second axis (about 3 grey levels over 50 steps, measured on SDXL). |
| D4 | SynGen on SDXL: step size | (a) the released 20.0 only; (b) a three-setting grid {10, 20, 40} on the tuning pairs only (lion and meerkat, camel and pelican, donkey and duck, chicken and rabbit; seeds 9 to 16; 96 images) | (a), matching how CO3 was treated (one row at its published default, no grid). (b) is the answer if the smoke test shows the loss not falling on SDXL. |

## Plan 1: compute job 10 on SD 1.5

Plan 1 renders four methods, 1,104 images each, on SD 1.5 at 512 by 512, guidance 7.5, 50 DDIM steps (per D1), fp32 (per D3), every image starting from one saved latent per seed. A joint-prompt column rides in the same job as PoE because it costs under an hour (estimate) and it is the identity check for every binding method below.

### The model weights

`runwayml/stable-diffusion-v1-5` is already in the cluster's Hugging Face cache at snapshot `451f4fe16113bff5a5d2269ed5ad43b0592e9a14` (`unet/diffusion_pytorch_model.safetensors` present, listed 2026-11-11). Every runner loads that snapshot path with `HF_HUB_OFFLINE=1`, because the `runwayml` repository has since left the Hub. The snapshot has no `safety_checker` folder, so every pipeline is loaded with `safety_checker=None, requires_safety_checker=False`; otherwise an offline load fails looking for it. Attend-and-Excite and SynGen default to SD 1.4 (`CompVis/stable-diffusion-v1-4`, also cached); Table 4 is SD 1.5, so both are pointed at the SD 1.5 snapshot, which is the only argument changed.

### The starting latent

`1-lora-correction/scripts/review_sd15_prep.py` (new, `co3`) draws one latent per seed, `torch.randn((1, 4, 64, 64), generator=torch.Generator("cpu").manual_seed(seed), dtype=torch.float32)`, and saves it to `/datasets/mmolefe/poe_repair_min/outputs/paper_review/latents_sd15/seedNN.pt` for seeds 9 to 12, printing each tensor's md5. A saved file is used because the four methods run under three torch versions (1.12.1, 2.0.1, 2.5.1), and a file removes any doubt that their random draws agree. This is the SD 1.5 counterpart of `1-lora-correction/scripts/review_seed_latents.py`; SDXL's latents (shape 1×4×128×128) cannot be reused at 512.

All three official pipelines take a `latents=` argument (Attend-and-Excite line 360, SynGen line 104, Divide-and-Bind line 489 of each pipeline file at the pinned commits), so every SD 1.5 column gets the saved tensor and none needs a dagger for noise. The pipelines multiply it by the scheduler's `init_noise_sigma`, which is 1.0 for DDIM, exactly as the joint and PoE loop does.

### The token indices

Attend-and-Excite and Divide-and-Bind need the token index of each subject noun. The released `run.py` asks for them at the keyboard. `1-lora-correction/scripts/review_sd15_prep.py` computes them once with SD 1.5's CLIP tokenizer: the index of the last word of each concept ("a horse with a glasses" gives `[2, 5]`, "a pink crown and a purple bow" gives `[3, 7]`), and for Divide-and-Bind the index of the colour word in front of each noun, or 0 where there is none (`color_index_list`; its code skips a subject whose index minus 1 is negative). The function asserts that every noun is a single word piece and writes `data/prompt-sets/attend-and-excite/token_indices_sd15.json`. SynGen finds its own indices with spaCy and needs none of these.

### The four methods

| Method | Code, pinned | Environment | Entry point and arguments (the paper's defaults) |
|---|---|---|---|
| PoE (and joint) | the repo's own rule, copied from `render_unet` in `1-lora-correction/scripts/same_rule_four_models.py`: ε = ε_uncond + 7.5 · Σ(ε_concept − ε_uncond) for PoE, plain CFG on the prompt as written for joint | `co3` (torch 2.5.1, diffusers 0.29.2), exists | `1-lora-correction/scripts/review_sd15_cell.py --method poe` and `--method joint`; prompts split per `review_benchmark_pairs.txt` |
| Attend-and-Excite | `github.com/yuval-alaluf/Attend-and-Excite` at `163efdfd341bf3590df3c0c2b582935fbc8e8343`; identical to the repo's `composition/aae/` (a recursive diff shows no difference) | new: `sd15-aae` | `AttendAndExcitePipeline` from `pipeline_attend_and_excite.py`, called as `run.py`'s `run_on_prompt` calls it, with `config.py`'s defaults: 50 steps, guidance 7.5, `max_iter_to_alter` 25, `attention_res` 16, thresholds {0: 0.05, 10: 0.5, 20: 0.8}, `scale_factor` 20, `scale_range` (1.0, 0.5), smoothing on with sigma 0.5 and kernel 3; `token_indices` from the saved file |
| SynGen | `github.com/RoyiRa/Linguistic-Binding-in-Diffusion-Models` at `50b8eab175dc9b31ce9c7c4b2eba28c817fc440e` | new: `syngen` | `SynGenDiffusionPipeline.from_pretrained(<SD 1.5 snapshot>, include_entities=False)`, called as `run.py`'s `generate` does: `syngen_step_size` 20.0, `attn_res` (16, 16), and the pipeline's own `num_intervention_steps` 25, 50 steps, guidance 7.5. The runner sets `pipe.subtrees_indices = None` before every prompt (see below). |
| Divide-and-Bind | `github.com/boschresearch/Divide-and-Bind` at `644dfda4ee4e3fbf8ac7128d47623b7274322042` | `sd15-aae` (shared) | `DivideAndBindPipeline` from `divide_and_bind/pipeline_divide_and_bind.py`, the pipeline the repository's `generate_images.ipynb` uses for the paper, called as `utils/test_utils.run_on_prompt` calls it with `loss_mode='tv_bind'` and `default_param('tv_bind')`: thresholds {0: 0.05, 10: 0.2, 20: 0.3}, `threshold_indicator` `tv_loss`, `max_refinement_steps_init_step` 50, `max_iter_to_alter` 25, `scale_factor` 20; its own `AttentionStore` with attention-map saving off (it only draws visualisations) |

**SynGen keeps the first prompt's parse unless told otherwise.** The pipeline stores the spaCy-derived indices in `self.subtrees_indices` on its first call and only recomputes them when that attribute is empty (`syngen_diffusion_pipeline.py` lines 46 and 412). The released `run.py` makes one pipeline per prompt, so it never meets this; a runner that loads the model once and loops over 276 prompts would apply the first prompt's word indices to all of them. Resetting the attribute before each call reproduces what `run.py` does.

**SynGen's `--include_entities` flag cannot be set to False from the command line.** It is declared `type=bool`, so any value passed reads as True. The runner calls the pipeline directly with `include_entities=False`, which is the flag's real default.

**The official code is cloned, never copied into the repo.** Attend-and-Excite is already at `composition/aae/`. SynGen and Divide-and-Bind are cloned to `/datasets/mmolefe/repos/<name>` at the commits above, beside `T2I-CompBench` and `geneval`. Divide-and-Bind is AGPL-3.0, so keeping it outside the repo also keeps its licence out of the paper's code release. The runner adds the clone to `sys.path` and imports its pipeline; no file in any clone is edited.

### The two new environments

Attend-and-Excite and Divide-and-Bind share one, because both repositories' `requirements.txt` pin the same stack. Neither runs in `co3`: Attend-and-Excite's `utils/ptp_utils.py` imports `diffusers.models.cross_attention` and its pipeline imports `randn_tensor` from `diffusers.utils`, and both are gone from diffusers 0.29.2. Divide-and-Bind's original pipeline imports the same two names.

| Environment | Path | Pins | Why each pin |
|---|---|---|---|
| `sd15-aae` | `/datasets/mmolefe/envs/sd15-aae` | python 3.9; torch 1.12.1+cu113, torchvision 0.13.1+cu113; diffusers 0.12.1; transformers 4.26.0; **huggingface_hub 0.13.4**; accelerate 0.16.0; numpy 1.23.5; pyrallis 0.3.1; opencv-python-headless 4.7.0.72; ipython; matplotlib; ftfy; einops 0.6.0; scipy | the two repositories' `requirements.txt` (torch 1.12, diffusers 0.12.1, transformers 4.26.0). `huggingface_hub` is pinned because diffusers 0.12.1 imports `cached_download`, which `huggingface_hub` 0.26 removed; an unpinned hub is exactly what stopped ToMe's official code from importing (protocol rule 2). cu113 wheels run on the RTX 3090 (sm_86). IPython is imported at module level by both repos' `ptp_utils.py`. |
| `syngen` | `/datasets/mmolefe/envs/syngen` | SynGen's `environment.yaml` as released: python 3.11; torch 2.0.1; diffusers 0.21.1; transformers 4.30.2; spacy 3.6.1; numpy 1.23.5; tqdm 4.65.0; accelerate 0.23.0; huggingface_hub 0.23.2; plus `en_core_web_trf` 3.6.1 from its release wheel | the repository's own pins. The transformer parser is what the README names; it runs on the CPU in under a second per prompt (estimate). |

The installer is `1-lora-correction/scripts/review_install_sd15_envs.sh` (new), written like `1-lora-correction/scripts/review_install_t2icomp.sh`: `conda create -p <path>`, pip installs with every version above, then an import line per environment that loads each pipeline class and prints `SD15_AAE_OK` or `SYNGEN_OK` with the versions. It runs as a `bigbatch` job because the compute nodes reach PyPI (13.5 MB/s measured on mscluster88) and the import line needs a card.

### The runner and its output

`1-lora-correction/scripts/review_sd15_cell.py` (new) takes `--method {joint,poe,aae,syngen,dnb}`, `--shard i/n` and `--smoke`. Its top level imports only the standard library and torch, and each method's imports sit inside that method's function, so the one file runs under all three environments. It loops over `review_benchmark_pairs.txt` × seeds 9 to 12, takes every n-th cell for its shard, and skips any tile already on disk, so a stopped job resumes.

Each tile goes to `/datasets/mmolefe/poe_repair_min/outputs/paper_review/sd15/<method>_default/r1/<pair slug>/seedNN/<method>.png`, beside a JSON sidecar with protocol rule 9's fields: `model` "SD 1.5", `method`, `adapter` "none", `sampler` "DDIM", `steps` 50, `guidance` 7.5, `resolution` 512, `pair`, `seed`, `group` "r1", `setting` "default", and `nfe` and `unet_calls` counted by the same `UNetCounter` wrapper `review_baseline_cell.py` uses (it replaces `unet.forward` on the instance, so Attend-and-Excite's and Divide-and-Bind's extra refinement passes are counted). It also records `prompt_a`, `prompt_b`, `joint_prompt`, `noise` (the latent file and its md5), `knobs`, `token_indices`, `color_indices` for Divide-and-Bind, SynGen's printed modifier indices, `seconds`, the method's commit and the environment's python path. The row builder then reads `sd15/<method>_default/r1` like any other baseline folder.

### The sbatch file

`scripts/review_sd15.sbatch` (new) takes `STAGE` (`smoke` or `run`), `METHOD` (`joint+poe` runs the two in turn in `co3`) and `SHARD` from `--export`, and picks the python by method: `co3` for joint and PoE, `/datasets/mmolefe/envs/sd15-aae/bin/python` for `aae` and `dnb`, `/datasets/mmolefe/envs/syngen/bin/python` for `syngen`. Header: `-p bigbatch`, `-t 1-00:00:00`, log to `/datasets/mmolefe/poe_repair_min/outputs/paper_review/sd15/logs/%x_%j.log`, and `--exclude=mscluster44,mscluster45,mscluster50,mscluster51,mscluster57,mscluster59,mscluster62,mscluster65,mscluster72,mscluster73,mscluster74,mscluster75,mscluster76,mscluster83` (the union of the faulted-card list in `environment/hpc/nodes.md` and the exclude line of `1-lora-correction/scripts/same_rule_four_models.sbatch`). Its guards, in order, each aborting in seconds:

1. `mkdir -p` the output folder, then write and delete a probe file in it, because mscluster72 and 73 could not see `/datasets` from inside a job while `ssh` saw it.
2. `df` on that same folder, abort at 90% full.
3. `nvidia-smi` lists a card, and the chosen python's `torch.cuda.is_available()` is True.
4. The python path exists.

Then it prints node, card, job id and the method's commit, and runs the runner.

### Smoke test: one prompt, one seed, per method

The smoke job runs on one RTX 3090 in this order: the prep script (latents and token indices), then one cell per method, "a cat and a dog" at seed 9, then the checks below. That cell is used because it has no adjective, so SynGen's loss is zero on it, and the joint-prompt column gives an exact reference.

| Check | What passes | What it proves |
|---|---|---|
| Each method writes one tile and one sidecar | 5 PNGs and 5 JSONs under `sd15/*_default/r1/a_cat__x__a_dog/seed09/` | the imports, the weights and the layout work in each environment |
| Attend-and-Excite with its own `run_standard_sd=True`, and Divide-and-Bind the same, against the joint tile | mean absolute pixel difference under 3 grey levels (the precision gap measured on SDXL; the stacks differ in torch and diffusers, so exact equality is not expected) | each pipeline took the saved latent and the DDIM swap; with its loss switched off it draws what the joint prompt draws |
| SynGen against the joint tile | the same bound, since its loss is zero on this prompt | SynGen took the saved latent, and D2's consequence is real |
| SynGen on a second prompt in the same process, "a pink crown and a purple bow" | two "Final modifier indices collected" lines with different indices | the per-prompt reset works |
| Each method's tile under the released PNDM scheduler | saved beside the DDIM tile as `<method>_pndm.png` | puts the size of D1's change on file |
| Seconds per image and peak memory, from the sidecar and `torch.cuda.max_memory_allocated` | recorded | replaces the estimates in the next table |

You judge the five tiles as a contact sheet before production starts.

### Time, measured or estimated

| Method | Seconds per image on an RTX 3090 | Images | GPU-hours | Basis |
|---|---|---|---|---|
| joint | 2 to 4 | 1,104 | 0.6 to 1.2 | estimate: one CFG pass of 50 steps at 512 |
| PoE | 3 to 5 | 1,104 | 0.9 to 1.5 | estimate: three conditionings per step |
| Attend-and-Excite | 20 to 45 | 1,104 | 6.1 to 13.8 | estimate: a forward and backward pass on each of the first 25 steps, plus up to 20 refinement iterations at steps 0, 10 and 20 |
| SynGen | 12 to 30 | 1,104 | 3.7 to 9.2 | estimate: a forward and backward pass on each of the first 25 steps; cheaper on the 102 prompts where the loss is zero |
| Divide-and-Bind | 25 to 70 | 1,104 | 7.7 to 21.5 | estimate: as Attend-and-Excite, with up to 50 refinement iterations at its first threshold step |
| Smoke and install | | 6 + 5 PNDM | about 0.5 | estimate |
| **Total** | | 5,531 | **19 to 48 GPU-hours (estimate)** | |

No measurement of any of these exists yet; the smoke test supplies one per method. Memory is estimated at 8 to 14 GB for the three binding methods in fp32 at 512, inside a 3090's 24 GB.

**Sharding within the six-job cap.** Six `bigbatch` jobs at once: joint and PoE in one job, Attend-and-Excite in two shards, Divide-and-Bind in two, SynGen in one. At the estimated rates the longest shard (Divide-and-Bind) takes 4 to 11 hours, so all four methods finish within about 11 hours of launch (estimate). If the smoke rates make one shard much longer than the rest, the shard counts are rebalanced before launch, keeping the total at six.

### Checks that each method finished

```bash
scripts/cluster.sh sh "for m in joint poe aae syngen dnb; do
  for s in animals animals_objects objects; do
    n=\$(grep \"|\$s\$\" /home-mscluster/mmolefe/Playground/PhD/poe_repair_min/1-lora-correction/scripts/review_benchmark_pairs.txt | wc -l)
    printf '%s %s: expect %d\n' \$m \$s \$((n*4))
  done
  printf '%s: %s tiles, %s sidecars (expect 1104 each)\n' \$m \
    \$(ls /datasets/mmolefe/poe_repair_min/outputs/paper_review/sd15/\${m}_default/r1/*/seed*/\$m.png | wc -l) \
    \$(ls /datasets/mmolefe/poe_repair_min/outputs/paper_review/sd15/\${m}_default/r1/*/seed*/\$m.json | wc -l)
done"
```

A method is finished when it has 1,104 tiles and 1,104 sidecars: 264 for animals, 576 for animals with objects and 264 for objects. The per-set split is read from the row builder's output, which assigns each pair to its set: `review_score_rows.py --folders sd15/<method>_default/r1 ...` prints its row count per folder. The count is read on the node that wrote the files, or after a few minutes, because the session node's view of `/datasets` lags. A sidecar whose `noise` field names a file other than `latents_sd15/seedNN.pt` fails the run.

After all five pass, scoring follows the existing route: `review_score_rows.py` over the five folders into a new `rows_sd15.csv`, BLIP-VQA with `review_score_blip.sbatch` (`BATCH=sd15`), CLIP and ImageReward with `review_score_clip_ir_cpu.sbatch`, and the animal count, then an R5 branch in `review_build_tables.py` and the Figure 9 grid with `1-lora-correction/scripts/paper_tile_figure.py`. These steps are listed for completeness and are not part of this plan's GPU budget (BLIP-VQA on 5,520 images is about 1 to 2 GPU-hours on a 3090, estimate).

### Launch commands for Plan 1, once approved

`scripts/cluster.sh run` refuses while any tracked file has uncommitted changes, and the working tree holds uncommitted trainer edits (`multi_pair_trainer.py`, `train_pooled.py`, `trainer.py`) and paper-review files. Those are committed or stashed by you first. The four new scripts (`review_install_sd15_envs.sh`, `review_sd15_prep.py`, `review_sd15_cell.py`, `review_sd15.sbatch`) are written and committed after your yes to this plan, and the commands below assume that commit.

```bash
# 0. Clones at the pinned commits, and the log folder (no GPU)
scripts/cluster.sh sh "cd /datasets/mmolefe/repos && git clone https://github.com/RoyiRa/Linguistic-Binding-in-Diffusion-Models && git -C Linguistic-Binding-in-Diffusion-Models checkout 50b8eab175dc9b31ce9c7c4b2eba28c817fc440e && git clone https://github.com/boschresearch/Divide-and-Bind && git -C Divide-and-Bind checkout 644dfda4ee4e3fbf8ac7128d47623b7274322042 && mkdir -p /datasets/mmolefe/poe_repair_min/outputs/paper_review/sd15/logs"

# 1. The two environments (one bigbatch job, about 20 minutes, estimate)
scripts/cluster.sh run "sbatch -p bigbatch -t 02:00:00 --exclude=mscluster44,mscluster45,mscluster50,mscluster51,mscluster57,mscluster59,mscluster62,mscluster65,mscluster72,mscluster73,mscluster74,mscluster75,mscluster76,mscluster83 -o /datasets/mmolefe/poe_repair_min/outputs/paper_review/sd15/logs/install_%j.log 1-lora-correction/scripts/review_install_sd15_envs.sh"

# 2. The smoke test: latents, token indices, one cell per method, the checks
scripts/cluster.sh run "sbatch --export=ALL,STAGE=smoke scripts/review_sd15.sbatch"

# 3. After the smoke passes and you say yes: six jobs, the bigbatch cap
scripts/cluster.sh run "sbatch --export=ALL,STAGE=run,METHOD=joint+poe,SHARD=0/1 scripts/review_sd15.sbatch"
scripts/cluster.sh run "sbatch --export=ALL,STAGE=run,METHOD=aae,SHARD=0/2 scripts/review_sd15.sbatch"
scripts/cluster.sh run "sbatch --export=ALL,STAGE=run,METHOD=aae,SHARD=1/2 scripts/review_sd15.sbatch"
scripts/cluster.sh run "sbatch --export=ALL,STAGE=run,METHOD=dnb,SHARD=0/2 scripts/review_sd15.sbatch"
scripts/cluster.sh run "sbatch --export=ALL,STAGE=run,METHOD=dnb,SHARD=1/2 scripts/review_sd15.sbatch"
scripts/cluster.sh run "sbatch --export=ALL,STAGE=run,METHOD=syngen,SHARD=0/1 scripts/review_sd15.sbatch"
```

**Plan 1 in one line:** 19 to 48 GPU-hours on RTX 3090s (estimate), two new environments, and the biggest risk is that the 2023-era stack in `sd15-aae` fails to install or import on the cluster, as ToMe's did, which would stop two of the three methods; the pinned `huggingface_hub` and the install job's import line exist to surface that in the first 20 minutes rather than in the production jobs.

## Plan 2: SynGen ported to SDXL for Table 1

No official SDXL SynGen exists. The CO3 paper reports a "SynGen (SDXL)" row from its own port, which protocol rule 10 lets appear only as a grey cited row. Plan 2 builds this project's own port so Table 1 can carry SynGen on the same 1,104 cells, the same saved latents (`paper_review/latents/seedNN.pt`, 1×4×128×128 fp16) and the same sampler as joint, PoE, ours and CO3. The caption says the row is a port by the authors, with its code released, rather than official code.

### What has to change from SD 1.4 to SDXL

**Two text encoders.** SDXL's cross-attention keys are the 77 per-token embeddings of CLIP ViT-L (768 wide) and OpenCLIP bigG (1,280 wide), concatenated to 2,048 per token. Both tokenizers use the same BPE vocabulary, so token *k* in the prompt is column *k* of every cross-attention map whichever encoder made it, and SynGen's loss, which works on per-token attention maps, needs no change. The port encodes the prompt with the repo's own `encode_prompt_sdxl`, the code that produced the joint column, and asserts that `tokenizer` and `tokenizer_2` give the same ids up to the end-of-text token on every prompt (they differ only in padding).

**The pooled embedding.** SDXL also feeds bigG's pooled embedding and the size-and-crop ids into the U-Net through `added_cond_kwargs`. SynGen's extra forward pass (`_syngen_step`, one per sample, conditional embedding only) has to pass the conditional half of both, or the U-Net call fails. They do not enter the loss.

**The attention hooks.** SynGen registers diffusers' `AttendExciteAttnProcessor` on every attention layer and stores maps in an `AttentionStore` at one resolution. In SD 1.4 at 512 that resolution is 16 by 16 (`--attn_res 256`), the U-Net's coarsest. SDXL at 1024 has no 16 by 16 layer: its cross-attention sits at 64 by 64 (`down_blocks.1`, `up_blocks.1`) and 32 by 32 (`down_blocks.2`, the mid block, `up_blocks.0`, 60 of its 70 transformer layers). The port uses 32 by 32, the coarsest SDXL has and the value the diffusers pipelines default to (`ceil(1024 / 32)`). That choice is the port's main judgement call and the caption names it. The port registers the store only on cross-attention layers (`attn2`), leaving self-attention on its fast kernel; only cross-attention maps enter the loss, so the arithmetic is unchanged and memory falls by several GB (estimate). diffusers 0.29.2, in `co3`, still ships `AttendExciteAttnProcessor` and `AttentionStore` in `pipeline_stable_diffusion_attend_and_excite`, so the port imports the same classes SynGen does.

**spaCy and the token indices.** `co3` has no spaCy, and `en_core_web_trf` 3.6.1 needs `spacy-transformers` 1.2, which needs transformers below 4.31, while `co3` has 4.44.1. Installing it into `co3` would break `co3`. So the parse runs once, in Plan 1's `syngen` environment: `1-lora-correction/scripts/review_syngen_indices.py` (new) loads SynGen's own `SynGenDiffusionPipeline._extract_attribution_indices` path (its `extract_attribution_indices`, `..._with_verb_root`, `..._with_verbs` and `unify_lists`, with `_align_indices` against SDXL's `tokenizer` loaded from the cached SDXL snapshot) and writes `data/prompt-sets/attend-and-excite/syngen_indices_sdxl.json`, one entry per prompt. The SDXL sampler reads that file. Because the vocabulary is shared, these indices equal SynGen's own SD 1.5 indices from Plan 1, which the script asserts against Plan 1's smoke log for the prompts the smoke ran.

### Which files

| File | New or changed | What it holds |
|---|---|---|
| `poe_repair/baselines/syngen_sdxl.py` | new | `run_cfg_syngen`: the body of `run_cfg` in `poe_repair/methods/_sampling.py` (plain single-prompt CFG on SDXL, the sampler behind the joint column) with SynGen's latent update inserted at the top of steps 0 to 24, and the attention-store registration on `attn2`. `run_cfg` itself is not edited. |
| the SynGen clone at `/datasets/mmolefe/repos/Linguistic-Binding-in-Diffusion-Models` | imported, not edited | `compute_loss.py` (the positive and negative losses, `get_attention_map_index_to_wordpiece`) and `_get_attention_maps_list`, imported as released |
| `1-lora-correction/scripts/review_syngen_indices.py` | new | the spaCy parse into `syngen_indices_sdxl.json`, run in `syngen` |
| `1-lora-correction/scripts/review_syngen_sdxl_cell.py` | new | the runner: R1's cells, the saved SDXL latents, tiles to `paper_review/syngen-sdxl_default/r1/<pair>/seedNN/syngen-sdxl.png` with sidecars built by the same `row()` fields as `review_baseline_cell.py` (`model` SDXL, `method` "syngen-sdxl", `sampler` "DDIM with SynGen latent update (port)", `resolution` 1024, counted `nfe`) |
| `1-lora-correction/scripts/review_syngen_sdxl.sh` | new | the shared-device launcher, with the same guards as Plan 1's sbatch file plus the device check of `scripts/launch_sdlora.sh` (abort if the pinned card holds more than 1 GB or reads `[N/A]`), and `PY=` chosen by node: `co3` on mscluster108, `co3_bw` on 110 to 112 |

### How faithful the port can be

| Part | Faithful? | Note |
|---|---|---|
| Loss: positive symmetric KL between modifier and noun maps, negative loss against other words | yes | the released functions, imported |
| Latent update `z ← z − 20 · ∇L` on steps 0 to 24 of 50, one conditional forward pass per update | yes | the released step size and schedule (D4). SynGen does one update per step and no iterative refinement, so nothing else is carried over. |
| Parse: `en_core_web_trf`, the three extractors, `include_entities` False | yes | same code, same parser, run beforehand; on the 102 prompts with no adjective the loss is zero, as on SD 1.5 |
| Attention resolution | adapted | 32 by 32 in place of 16 by 16, since SDXL has no 16 by 16 layer at 1024 |
| Step size relative to SDXL's latent scale | open | 20.0 was set for SD 1.4's 64 by 64 latents (VAE scale 0.18215); SDXL's latents are 128 by 128 with scale 0.13025, and its attention maps spread over four times as many positions, so the same step moves the latent by a different amount. The smoke test measures whether the loss falls. |
| Precision | adapted | SynGen runs in fp32; protocol rule 2 runs SDXL in fp16. The port keeps the U-Net in fp16, computes the attention maps and the loss in fp32, applies the update to an fp32 copy of the latent, and aborts the cell on a NaN or infinite gradient. |
| Sampler and starting latent | protocol | 50 DDIM steps, guidance 7.5, the saved latent per seed: the same as every other Table 1 column, so no dagger |

### Smoke test

On mscluster108 card 0, the RTX 8000 that rendered the `r1_joint` column, with `co3`:

1. **The identity check.** Run the port with step size 0 on "a cat and a dog" at seed 9. It passes if the tile equals `r1_joint/a_cat__x__a_dog/seed09/mono.png` exactly, or within 3 grey levels if the GPU model differs (the precision gap measured on SDXL). This proves the port is the joint sampler plus nothing.
2. **The loss falls.** Run step size 20 on "a pink crown and a purple bow" and on "a monkey and a red car" at seed 9, logging the loss at each of the 25 update steps beside Plan 1's SD 1.5 SynGen loss on the same prompts. The port is called working when its loss at step 24 is below its loss at step 0, as on SD 1.5. If it is flat, D4's grid is the next step and you decide.
3. **Memory and time.** Peak memory from `torch.cuda.max_memory_allocated` and seconds per image, written to the sidecar.
4. **The pictures.** The three tiles beside their joint and PoE tiles, sent as a contact sheet for you to judge.

### Time and memory, estimated

A joint SDXL image takes 5.0 to 5.2 s on the RTX PRO 6000 Blackwell (measured, step 4 logs). The port adds 25 forward and backward passes at batch 1 and the attention store, estimated at 5 to 10 s more, so 10 to 15 s per image on a Blackwell (estimate). The RTX 8000 runs this kind of work 3.2 to 4.6 times slower (measured on an eval pass), so 32 to 70 s per image there (estimate). Peak memory is estimated at 20 to 35 GB, because backpropagation through SDXL's U-Net at 128 by 128 holds its activations; this rules out a 24 GB RTX 3090 until the smoke test says otherwise.

| Where | Seconds per image | GPU-hours for 1,104 images |
|---|---|---|
| one Blackwell card (mscluster110, 111 or 112, `co3_bw`) | 10 to 15 (estimate) | 3.1 to 4.6 (estimate) |
| one RTX 8000 card (mscluster108, `co3`) | 32 to 70 (estimate) | 9.8 to 21.5 (estimate) |
| smoke test | | about 0.2 (estimate) |
| D4's grid, only if chosen: 96 images | | 0.3 to 0.4 on a Blackwell (estimate) |

Recommended: production on one free Blackwell card, split across two cards if two are free, which halves the wall time to about 2 hours (estimate).

### Risks a reviewer would raise

**"Your SynGen is a weak port."** The step size and the 16 by 16 resolution were chosen for SD 1.4. The port keeps the first and adapts the second, and either can make SynGen look worse on SDXL than it could be. The answer on file is the loss curve from smoke step 2, the released code in the supplement, and Table 4's SD 1.5 row from SynGen's own code, which shows the method at home.

**"SynGen does nothing on your animals set."** True by construction: with no adjective, its loss is zero on 102 of 276 prompts, so its row equals the joint row there. The caption says so and the table reports SynGen per set. A reviewer may ask for `include_entities=True`; that is a different method setting and stays out unless you choose D2 (b).

**"Your numbers differ from CO3's SynGen (SDXL) row."** CO3's port, settings and seeds are not stated (protocol rule 10), so the two rows are not comparable; CO3's number appears only as a grey cited row.

**"SynGen reads the joint prompt."** It does, so in Table 1 it sits above the rule with joint and CO3, and bold continues to mark the best method that never reads the joint prompt.

**fp16 gradients.** The loss is computed in fp32 and NaN cells abort rather than writing a tile; a count of aborted cells goes in the sidecar summary, and a cell missing for that reason is counted as missing under the bar's 10% rule.

### Launch commands for Plan 2, once approved

These need Plan 1's `syngen` environment and clone (steps 0 and 1 of Plan 1). If Plan 2 runs alone, those two steps run first and the environment count for Plan 2 becomes one. `biggpu` allows one Slurm job per user and job 83423 holds mscluster107, so the GPU work takes the shared-device path over SSH, harvested with `pgrep`. Each card is read with `nvidia-smi` immediately before its launch; the card named below is the one to check first, not a promise that it is free.

```bash
# 0. Push the new code and fast-forward the cluster checkout (after the uncommitted edits are committed or stashed)
scripts/cluster.sh sync

# 1. The spaCy parse into syngen_indices_sdxl.json (CPU, minutes)
scripts/cluster.sh sh "/datasets/mmolefe/envs/syngen/bin/python /home-mscluster/mmolefe/Playground/PhD/poe_repair_min/1-lora-correction/scripts/review_syngen_indices.py"

# 2. The smoke test on mscluster108 card 0 (same card model as the r1_joint column, for the identity check)
scripts/cluster.sh sh "ssh mscluster108 nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv"
scripts/cluster.sh sh "timeout 30 ssh mscluster108 'GPU=0 STAGE=smoke PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python nohup bash /home-mscluster/mmolefe/Playground/PhD/poe_repair_min/1-lora-correction/scripts/review_syngen_sdxl.sh > /datasets/mmolefe/poe_repair_min/outputs/paper_review/logs/syngen_sdxl_smoke.log 2>&1 < /dev/null & disown'"

# 3. After the smoke passes and you say yes: production on a free Blackwell card, two shards if two cards are free
scripts/cluster.sh sh "ssh mscluster112 nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv"
scripts/cluster.sh sh "timeout 30 ssh mscluster112 'GPU=0 STAGE=run SHARD=0/2 PY=/home-mscluster/mmolefe/miniforge3/envs/co3_bw/bin/python nohup bash /home-mscluster/mmolefe/Playground/PhD/poe_repair_min/1-lora-correction/scripts/review_syngen_sdxl.sh > /datasets/mmolefe/poe_repair_min/outputs/paper_review/logs/syngen_sdxl_0of2.log 2>&1 < /dev/null & disown'"
scripts/cluster.sh sh "timeout 30 ssh mscluster111 'GPU=0 STAGE=run SHARD=1/2 PY=/home-mscluster/mmolefe/miniforge3/envs/co3_bw/bin/python nohup bash /home-mscluster/mmolefe/Playground/PhD/poe_repair_min/1-lora-correction/scripts/review_syngen_sdxl.sh > /datasets/mmolefe/poe_repair_min/outputs/paper_review/logs/syngen_sdxl_1of2.log 2>&1 < /dev/null & disown'"

# 4. The finished check: 1,104 tiles and sidecars, counted on the node that wrote them
scripts/cluster.sh sh "ssh mscluster112 'ls /datasets/mmolefe/poe_repair_min/outputs/paper_review/syngen-sdxl_default/r1/*/seed*/syngen-sdxl.png | wc -l; ls /datasets/mmolefe/poe_repair_min/outputs/paper_review/syngen-sdxl_default/r1/*/seed*/syngen-sdxl.json | wc -l'"
```

**Plan 2 in one line:** 3.1 to 4.6 GPU-hours on a Blackwell card, or 9.8 to 21.5 on an RTX 8000 (estimates), no new environment beyond Plan 1's `syngen`, and the biggest risk is a reviewer reading the port as a weakened SynGen, because the step size was set for SD 1.4 and the attention resolution had to be changed.

## Environment facts these plans depend on

- `co3` is `/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python` (torch 2.5.1+cu118, diffusers 0.29.2, transformers 4.44.1, no spaCy; read 2026-11-11), for the RTX 3090, RTX 8000 and A6000 nodes; `co3_bw` for the Blackwell nodes 110 to 112 ([the nodes fact](../../../../../../environment/hpc/nodes.md#python-environments)).
- `bigbatch` cards are RTX 3090s with 24 GB; faulted nodes are excluded by name, and a two-minute in-job write to `/datasets` is the only probe that catches the mscluster72 and 73 fault ([the partitions table](../../../../../../environment/hpc/nodes.md#partitions)). The cap is 6 `bigbatch` jobs per user.
- `biggpu` allows one Slurm job per user, so a second GPU job there goes over SSH with `nohup` and an absolute path on every part of the launch line ([the execution protocol](../../../../../../environment/hpc/execution-protocol.md)).
- Everything a run writes goes to `/datasets/mmolefe/poe_repair_min/outputs/paper_review/`, and the `df` guard reads that same folder ([the storage fact](../../../../../../environment/storage.md#the-disk-guard-rule-and-the-failure-that-motivates-it)). `/datasets` was 45% full with 211 TB free on 2026-11-11.
- SD 1.5 (`runwayml/stable-diffusion-v1-5`, snapshot `451f4fe`), SD 1.4 and SDXL are in `/home-mscluster/mmolefe/.cache/huggingface/hub/`, and jobs run with `HF_HUB_OFFLINE=1` because online checks from a compute node stall.
- fp16 SDXL renders differ from fp32 by about 3 grey levels over 50 steps, so identity checks compare like with like.
