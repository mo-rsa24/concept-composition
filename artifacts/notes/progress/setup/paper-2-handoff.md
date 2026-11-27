# 🔬 Paper 2 setup: the code written, its checks, and the commands waiting for a yes

Loop 17 writes the laptop half of [Paper 2's setup plan](../../../../plans/paper-2-dials/plans/01-setup.md): the seven
rules and four crop judges, the identity dial trainer and the blind-sheet builder, each checked with no GPU, plus the job
scripts for the cluster half. Loop 18 reviews every task on the [supervisor page](supervisor.md). Nothing here has run
on the cluster, and nothing is committed.

## Table of contents

- [Where each task stands](#where-each-task-stands)
- [Task 1: the plan checked](#task-1-the-plan-checked)
- [Task 2: the seven rules and the four crop judges](#task-2-the-seven-rules-and-the-four-crop-judges)
- [Task 3: the identity dial trainer](#task-3-the-identity-dial-trainer)
- [Task 4: the blind-sheet builder](#task-4-the-blind-sheet-builder)
- [Tasks 6 to 8 and the vision-language environment: the scripts](#tasks-6-to-8-and-the-vision-language-environment-the-scripts)
- [CO3's loader, co3_common.py](#co3s-loader-co3_commonpy)
- [Choices made where the plans were silent](#choices-made-where-the-plans-were-silent)
- [Questions sent to the owner](#questions-sent-to-the-owner)
- [The species judge's scripts, written after the setup](#the-species-judges-scripts-written-after-the-setup)
- [What is still open](#what-is-still-open)
- [The commands, ready for the owner's yes](#the-commands-ready-for-the-owners-yes)

## Where each task stands

| Task | What it writes | Loop 18 |
|---|---|---|
| 1 | the plan checked, in report mode | accepted 2026-11-26 |
| 2 | the seven rules and the four crop judges | accepted 2026-11-26, after one fix |
| 3 | the identity dial trainer | accepted 2026-11-26, after two guard fixes and the rework to the reworded plan text |
| 4 | the blind-sheet builder | accepted 2026-11-26 |
| 6 to 8 | job scripts written and checked on the laptop, none run; they run at the owner's launch | accepted 2026-11-26 |
| the vision-language environment | build script and load check, written, not run (the owner's yes, question 120) | accepted 2026-11-26 |

## Task 1: the plan checked

**Files.** None. This session may edit only the plan's checkboxes, so both checks ran in report mode and their findings
go to the owner (listed on the supervisor page under "Waiting on you").

**Accepted** by Loop 18 on 2026-11-26, as run in report mode. 1.1 and 1.2 are ticked.

**Plan check findings (`/verify-plan`, doing-plan skeleton):**

1. Every required section is present in order; no anchor or file link is broken; the Environment Facts field is present.
2. Tasks in groups 2, 3, 4, 6, 7 and 8 are plain checkboxes, not numbered 2.1, 3.1 and so on.
3. The Error Matrix has no "From global catalog" and "From project catalog" subsections.
4. The Figure Catalog has no "Organization workflow" subsection.
5. The close out has `/sync-plan-tree` but no `/ingest-error-pattern`.
6. The invocations in 1.1, 1.2 and the close out are inline code, not fenced blocks.
7. Task 8 has no "Needs: task 6" pointer, though it reads the SDXL weights task 6 fetches. The task 8 job script guards
   this itself: it stops with a clear message when the SDXL snapshot is missing under `HF_HOME`.
8. No paired `review/01-setup.md`. A setup plan carries no verdict, and Loop 18 agreed it needs none.

**Cross-reference findings (`/xref-pillar`), none applied:**

| Mention | Proposed target |
|---|---|
| "disk guard" (line 100) | `environment/storage.md`, "The disk guard rule, and the failure that motivates it" |
| "the node table's exclude list" (line 365) | `environment/hpc/nodes.md` |
| `scripts/cluster.sh run "sbatch ..."` (line 183) | `runbook/running-things-on-the-cluster/launching-and-harvesting-a-run.md` |

"LoRA" is left unlinked: `context/world/lora-corrector.md` describes the earlier pooled correction, a different adapter.
Loop 18 agreed.

## Task 2: the seven rules and the four crop judges

**Files**, all in `2-classifier-guidance/scripts/paper2/` except the test:

| File | What it does |
|---|---|
| `paper2_kinds.py` | the seven rules as one function, `kind_of`, from each body's call and flag to the picture's kind and the rule that decided it; `paper2_bars.py` imports `KINDS` and `kind_of` |
| `paper2_common.py` | what the four judges share: the picture list, `bodies.json`, the crop with its 10% margin and the other bodies painted out, the loop, `calls.json` and `settings.json`, `redecide` (new settings applied to stored raw scores with no model), the `/datasets` path constant `PAPER2_DATASETS`, and `m14-list`, which writes a picture list from the joint prompt's blind round |
| `paper2_judge_clip.py` | CLIP ViT-L/14 on crops: four phrasings, the "neither" names, margin m and flag share p |
| `paper2_judge_imagenet.py` | `convnext_large.fb_in22k_ft_in1k` on crops through the ImageNet class table; uncovered pairs kept as not covered |
| `paper2_judge_dac.py` | Detect-and-Compare: DINOv2 large features of each crop against 8 or 16 real photographs per species, borrowed-part gap b |
| `paper2_judge_vlm.py` | Qwen3-VL-8B-Instruct asked three yes-or-no questions per crop, yes threshold t |
| `tests/test_paper2_kinds.py` | the five worked examples in order, the other rules, the crop's margin and paint-out, and each judge's dry run on 2 pictures |

**Check output** (laptop, no torch, 2026-11-26):

```
$ python3 -m py_compile <each of the 7 files>      # all compile
$ python3 -m pytest tests/test_paper2_kinds.py -q
......................                                                   [100%]
22 passed in 1.81s
$ python3 2-classifier-guidance/scripts/paper2/paper2_common.py m14-list --n 2 --seed 0 --out outputs/paper2/setup-checks/two-pictures.csv
.../pictures/95b7a187cfa1.png  asked boar and pheasant
.../pictures/a361aef39cf6.png  asked crow and raven
$ python3 2-classifier-guidance/scripts/paper2/paper2_judge_clip.py --pictures outputs/paper2/setup-checks/two-pictures.csv --out outputs/paper2/setup-checks/judge-clip --dry-run
1/2 95b7a187cfa1.png asked boar and pheasant: 2 bodies ['A', 'B'] -> two_clear (rule 4), two-clear 0.789
2/2 a361aef39cf6.png asked crow and raven: 2 bodies ['A', 'B'] -> two_clear (rule 4), two-clear 0.789
wrote outputs/paper2/setup-checks/judge-clip/calls.json (2 pictures, 2 covered) and 4 crops under outputs/paper2/setup-checks/judge-clip/crops
```

The ImageNet, Detect-and-Compare and vision-language dry runs print the same lines with two-clear scores 0.800, 0.580 and
0.874. Each folder `outputs/paper2/setup-checks/judge-<name>/` holds `settings.json`, `calls.json` and `crops/` (4 crops).
`outputs/paper2/setup-checks/dry-run-crops-sheet.png` shows both pictures with the two stand-in boxes (thick: the body;
thin: the 10% margin) beside their two crops, the other body's box painted in the picture's mean colour.

**Found and fixed while checking.** The first dry run called picture 2's bodies "other": each stand-in's body counter ran on
across pictures. Each judge now resets its stand-in at the start of every picture, and the test checks both pictures get
the same kind.

**Loop 18's fix.** A picture with fewer than two bodies now scores 0 two-clear for CLIP and ImageNet
(`paper2_common.share_two_clear`) and Detect-and-Compare (`paper2_judge_dac.py`), as the old judge does; the
vision-language judge already scored 0 unless there are exactly two bodies. `test_fewer_than_two_bodies_scores_zero`
checks one body and no body score 0 for all four judges, and two clear bodies score above 0:

```
$ python3 -m pytest tests/test_paper2_kinds.py -q
..........................                                               [100%]
26 passed in 1.80s
```

**Accepted** by Loop 18 on 2026-11-26 after the fix (rechecked there: 7 files compile, 26 passed). Ticked in the plan.

## Task 3: the identity dial trainer

**Files:**

| File | What it does |
|---|---|
| `2-classifier-guidance/scripts/paper2/paper2_identity_dial.py` | the rank-4 adapter on every attention projection with a strength knob; `closeness-cache`, the frozen side of the closeness term from a U-Net with no dial, in its own process; `train`, on CO3's own rollout with the reward as an argument, both routes; `random`, a random dial of matched norm; `draw`, the dial and random-dial rows with the same on-switch |
| `2-classifier-guidance/scripts/paper2/paper2_closeness_prompts.txt` | 16 prompts outside the dial's purpose: 8 single animals and 8 unlike pairs, none a pair named in the species judge plan or the pooled LoRA dial's pair file, each passing the subject guard |
| `tests/test_paper2_identity_dial.py` | 30 tests, listed under the check output below |

**The five requirements of the setup plan's task 3** (`plans/paper-2-dials/plans/01-setup.md`, as reworded with the
owner's yes on 2026-11-26):

| Requirement | Where it is met |
|---|---|
| 1. The rollout is CO3's, corrector included, never the plain product | `co3_sampler` loads CO3 through `paper2_rows.py` (`load`, `set_prompt`, `start_noise`, `encode`); `Sampler.rollout` calls CO3's own `denoise_step` at every step; the step with the graph is CO3's guided joint-prompt prediction, and the window may not start inside CO3's 6 corrected steps. `poe_eps` appears nowhere in the file |
| 2. Closeness outside the dial's purpose, frozen side from an untouched U-Net in another process | `closeness-cache` builds CO3 with no dial (it refuses a U-Net holding one), rolls each closeness prompt out from each fixed starting noise to a step drawn once from the window, and saves the state and the frozen prediction to one file; `train` draws `--closeness-batch` (4) of them a step and computes only the dial's prediction on the stored state. `closeness.json` at step 50 (`--closeness-report-step`) gives the share and the weight that would make the term a tenth of the loss; `--closeness-weight` is the argument |
| 3. The dial on only from the window's start, in training, in every picture drawn with it, and for the random dial | `--dial-from-step` (default and minimum: the window's first step); before it the U-Net runs without the dial; `draw` takes the same switch, read from the checkpoint by default; the random dial carries the trained dial's value |
| 4. A silent term stops the run with its name | a closeness weight of 0 or less exits before step 1; the reward term and the adapter's gradient norm must not be exactly 0 at step 1, or at the first step the reward can act when step 1 cannot (fewer than two bodies, or a skipped kept draw), within 10 steps; the closeness term is checked at the first step after the dial first moved. This is how item 4 is read, because item 2 says "a new adapter's drift is zero at step 1": up starts at zero, so a closeness check at step 1 would stop every run (Loop 18's ruling) |
| 5. Pairs from a file with a weight each; each pair's reward logged on its own to W&B | `--pairs-file` ("slug weight" per line, or JSON `[{"pair", "weight"}]`); draws follow the weights; each step logs `reward/pair/<slug>` and `reward/all` to W&B (`prime_lab/poe-repair-animals-compose`); the end of a run prints each pair's count, mean and spread, kept in `summary.json` |

**Check output** (laptop; the trainer needs torch, so these ran under a CPU-only torch installed for this session in
the job's temporary folder, outside the repository; on the laptop's own python the file skips):

```
$ python3 -m py_compile 2-classifier-guidance/scripts/paper2/paper2_identity_dial.py tests/test_paper2_identity_dial.py
$ <cpu torch>/bin/python -m pytest tests/test_paper2_identity_dial.py -q
..............................                                           [100%]
30 passed in 31.58s
$ <cpu torch>/bin/python 2-classifier-guidance/scripts/paper2/paper2_identity_dial.py train --dry-run --out outputs/paper2/setup-checks/dial-dry-run
dial: 4 attention projections at rank 4, 256 trainable weights; route gradient; window (2, 5); dial on from step 2; pairs and weights {'cheetah|cougar': 3.0, 'duck|goose': 1.0}; 2 seeds; closeness on 4 of 4 cached states a step, weight 1.0
step 1/2 duck and goose seed 11002 k 2: reward -0.0262 (grad 0.0593); closeness on 4 cached states: drift 0, share 0.000 (grad 0)
step 2/2 duck and goose seed 11002 k 3: reward -0.0178 (grad 0.0734); closeness on 4 cached states: drift 2.82e-07, share 0.000 (grad 0.00105)
checkpoint outputs/paper2/setup-checks/dial-dry-run/ckpt-2/dial.pt (norm 0.00130312)
reward per pair (n, mean, spread):
  duck|goose                      2  -0.0220  0.0042
  all                             2  -0.0220  0.0042
$ ... train --dry-run --closeness-weight 0 ...
closeness weight must be above 0: every dial run carries the closeness term (the earlier reward fine-tune ran with it silently off until a72f748)
exit 1
```

The 30 tests: strength 0 gives the frozen layer bit for bit; strength 1 changes it; rank 4 with up at zero; the random
dial's norm equals the asked norm to 4 digits at 0.0123, 1.0 and 37.5; the rollout runs through CO3's step with the dial
off before its first step, and the dial leaves the output unchanged before that step and changes it after; a row refuses
strength 0; the torch crop matches the judges' crop on a real picture within 1.5/255; the torch two-clear score matches
the judges' formula; both routes' dry runs write a checkpoint; a zero closeness weight exits non-zero; a dial switched on
before the window, and a window inside CO3's corrected steps, are refused; `random` matches the norm and `draw` uses the
same switch; the cache is made with no dial and its frozen predictions equal a never-dialled model's; `train` reads the
cache and builds no second model; a cache marked as made with a dial is refused; pair weights read from lines and JSON;
over 60 dry-run steps at 3 to 1 the weighted pair is drawn 35 to 55 times and each pair's reward reaches its own W&B
key; a stand-in reward of exactly 0 stops the run with "reward term is exactly 0 at step 1"; two skipped kept draws, and
a first step with one body, do not stop a healthy run; a reward that never reaches the dial does; the closeness prompts
are single animals and unlike pairs, none a known pair, all passing the guard.

**Loop 18's review.** It asked for two changes to the guards (the closeness check at the first step after the dial
moves, and a skipped kept draw counted toward the 10-step allowance) and then, after the plan text was reworded with the
owner's yes, three more (the closeness term from a frozen cache made in another process, weighted pairs logged per pair
to W&B, and item 4 read as above). All are in the file as described.

**Accepted** by Loop 18 on 2026-11-26 after the guard fixes, and again after the rework (rechecked there: 65 passed and
1 skipped across the four Paper 2 test files under the CPU torch; `poe_eps` absent; the prompts file 8 single animals
and 8 unlike pairs). Its decisions on the four rework choices: all accepted. Ticked in the plan.

## Task 4: the blind-sheet builder

**Files:**

| File | What it does |
|---|---|
| `2-classifier-guidance/scripts/paper2/paper2_sheets.py` | `build`: shuffles a picture list by a seed, re-saves each picture from its pixels under a ten-character neutral name, and writes `sheet/` (pages with "Asked: a cheetah and a cougar" above the 1024 by 1024 picture, the plain pictures, `sheet.csv`, `calls-template.csv`, `rules.txt`) and, apart, `key/key.json`; `check`: searches every sheet file and file name for the source folders, file names and the list's other columns |
| `tests/test_paper2_sheets.py` | 5 tests: build then check passes, a text chunk in a source never reaches the sheet, a planted leak fails the check, the same seed gives the same sheet and another seed another order, and a picture not 1024 by 1024 stops the build |

**Check output** (laptop, 5 joint-prompt pictures with their pair, seed and row carried as extra columns, so the
check has real values to look for):

```
$ python3 2-classifier-guidance/scripts/paper2/paper2_sheets.py build --pictures outputs/paper2/setup-checks/five-pictures.csv --seed 0 --out outputs/paper2/setup-checks/sheets-five
{"pages": 5, "sheet": "outputs/paper2/setup-checks/sheets-five/sheet", "key": "outputs/paper2/setup-checks/sheets-five/key/key.json", "seed": 0}
$ python3 2-classifier-guidance/scripts/paper2/paper2_sheets.py check --out outputs/paper2/setup-checks/sheets-five
searched 13 files under outputs/paper2/setup-checks/sheets-five/sheet for 23 strings (source folders, file names, and the list's other columns): 0 found
pages: 5 at [(1024, 1104)] (the picture at 1024 by 1024 under a 80-pixel band); pictures: 5 at [(1024, 1024)]
check passed
$ python3 -m pytest tests/test_paper2_sheets.py tests/test_paper2_kinds.py -q
31 passed in 2.90s
```

The 23 strings searched for: the five original file names with and without `.png`, the folders `m14`, `blind` and
`pictures`, the five seeds, the four pair slugs and the row name `joint-prompt`. A copy of the sheet with
`seed 6013 a_cheetah__x__a_cougar` planted in one picture's PNG text chunk failed the check with both strings named.
The page `outputs/paper2/setup-checks/sheets-five/sheet/pages/16694ae9e4.png` was looked at: the band reads
"Asked: a cheetah and a cougar" and the page's own name, above the picture at full size.

**Found and fixed while checking.** The first check searched every byte of every PNG and found `m14` twice by chance
inside compressed pixel data. Compressed pixels cannot carry readable text, so for a PNG the check now searches every
chunk except the compressed pixels (`IDAT`), which is where a prompt or seed written by a renderer would sit.

**Accepted** by Loop 18 on 2026-11-26 (rebuilt there from the same list: 0 found, 31 passed). Ticked in the plan.

## Tasks 6 to 8 and the vision-language environment: the scripts

Every run goes inside one interactive allocation on one Blackwell node (mscluster110, 111 or 112, one 96 GB card), as
plain `bash <script>` from the cluster checkout, with Paper 1's process sharing device 0. No `sbatch`, no SSH launch
onto another node. The owner allocates the node and pushes the branch. None of these scripts has run.

| File | What it does |
|---|---|
| `paper2_guards.sh` | sourced by every script: the disk guard on the filesystem a folder sits on (stop at 90%), the python by host (`co3_bw` on mscluster110 to 112, `co3` elsewhere), the card guard (device 0 answers, is not faulted, has a stated number of MiB free; another process on it is allowed), the CUDA probe, `HF_HOME` on `/datasets` offline, CO3's `PYTHONPATH`, and the log under `/datasets/mmolefe/poe_repair_min/paper2/logs/` |
| `paper2_launch.sh` | runs one GPU step (a judge, the trainer, a draw) behind every guard, pinned to device 0; `co3` or `vlm` names the python |
| `paper2_fetch_weights.sh`, `paper2_fetch_weights.py` | task 6: five repositories into `HF_HOME`, each with whether it was cached, its size and an offline config check; fails if the home folder's caches grew past 1 GB |
| `paper2_upload_joint.sh` | task 7, run on the laptop: the 192 pictures by `rsync` through mscluster84 in the push direction (question 119's proposed route), then a count and a size check on the cluster; does nothing without `--go` |
| `paper2_co3_patch.py` | lets CO3 build without xformers: diffusers' own attention in its place, and the kernel on record |
| `paper2_co3_load.sh`, `paper2_co3_load.py` | task 8: stops before loading if the SDXL snapshot, its fp16 U-Net, the fp16-fix VAE or the sentence parser's model is missing; then builds CO3 offline through `paper2_rows.py`, sets one pair, draws no picture |
| `paper2_vlm_env_build.sh` | the vision-language environment at `/datasets/mmolefe/envs/paper2-vlm`, conda and pip caches on `/datasets`, every version pinned; stops if the folder exists |
| `paper2_vlm_env_check.sh`, `paper2_vlm_load_check.py` | loads Qwen3-VL-8B-Instruct in that environment through the judge's own code and asks one crop one question |
| `tests/test_paper2_co3_import.py` | `co3_common.py` defines what `paper2_rows.py` and the trainer take from it (read with `ast`, on the laptop), matches the hub's tracked copy, and imports for real where stanza and CO3's code exist |

**Check output** (laptop):

```
bash -n ok: paper2_guards.sh, paper2_launch.sh, paper2_fetch_weights.sh, paper2_upload_joint.sh,
            paper2_co3_load.sh, paper2_vlm_env_build.sh, paper2_vlm_env_check.sh
py_compile ok: paper2_fetch_weights.py, paper2_co3_load.py, paper2_vlm_load_check.py
$ bash 2-classifier-guidance/scripts/paper2/paper2_upload_joint.sh
laptop: outputs/loop-6/m14/blind/pictures/ holds 193 files, 192 pictures
route: rsync -az outputs/loop-6/m14/blind/pictures/ mscluster84:/datasets/mmolefe/poe_repair_min/paper2/paper2-upload-joint/
dry: nothing copied; run with --go once the owner has named the route (question 119)
```

The card guard, fed readings from a stand-in `nvidia-smi`:

| Reading (used MiB, total MiB, utilisation, power state) | Needs | Result |
|---|---|---|
| 2, 97887, [N/A], P0 | 20000 | exit 4, "utilisation reads '[N/A]': faulted" |
| 2, 97887, 0, [GPU requires reset] | 20000 | exit 4, power state refused |
| 95000, 97887, 40, P0 | 20000 | exit 4, "2887 MiB free, under the 20000" |
| 30000, 97887, 60, P0, one other process | 20000 | exit 0: a shared card with room passes |

The other guards on the laptop: the disk guard passed on a temporary folder (41% full); the python choice stopped with
exit 5 (no `co3` here); the CUDA probe under a CPU-only torch stopped with exit 6; CO3's path stopped with exit 7 (no
snapshot here) and, given a stand-in snapshot, set `PYTHONPATH` to CO3's code, the subject guard and the repo.

```
$ python3 -m pytest tests/test_paper2_kinds.py tests/test_paper2_sheets.py tests/test_paper2_co3_import.py tests/test_paper2_identity_dial.py -q
35 passed, 2 skipped        # laptop python: the dial needs torch; the real CO3 import needs stanza and composers
$ <cpu torch>/bin/python -m pytest <the same four files> -q
57 passed, 1 skipped        # the real CO3 import, which runs on the held node
```

**Task 8 departs from the plan's "on a CPU node".** CO3 builds on the held node's device 0, behind a 12,000 MiB card
guard, as every CO3 run does.

**CO3 without xformers.** CO3's `Co3.__init__` calls `pipe.enable_xformers_memory_efficient_attention()`
(`composition/debottam_co3/composers/Co3.py`, line 67), and `co3_bw`, the only environment that drives a Blackwell card,
has no xformers: the first run of task 8 on mscluster112 stopped there with `ModuleNotFoundError`
(`/datasets/mmolefe/poe_repair_min/paper2/logs/paper2-co3-load.log`, from line 89). `co3` has xformers 0.0.28.post3 but
cannot drive the card. The owner's decision: fall back to PyTorch's own attention. `paper2_co3_patch.py` does it,
called before CO3 is built by `paper2_co3_load.py` and by the trainer's `co3_sampler` (used by `train`, `draw` and
`closeness-cache`): when xformers cannot be imported, the pipelines' xformers switch prints "xformers missing: CO3 runs
on PyTorch scaled-dot-product attention" and returns, so CO3 keeps diffusers' default attention processor; when xformers
imports, nothing changes. `co3_common.py`, `paper2_rows.py` and CO3's own code are untouched. Every run records the
kernel: `attention` in `load.txt`, in the trainer's `run.json` and `summary.json`, in the closeness cache and in
`draw.json`, as `xformers` or `pytorch-sdpa`.

```
$ python3 -m pytest tests/test_paper2_co3_patch.py tests/test_paper2_kinds.py tests/test_paper2_sheets.py tests/test_paper2_co3_import.py tests/test_paper2_identity_dial.py -q
40 passed, 2 skipped        # laptop python
$ <cpu torch>/bin/python -m pytest <the same five files> -q
70 passed, 1 skipped
```

`tests/test_paper2_co3_patch.py` holds 5 tests: with xformers absent a stand-in pipeline's switch prints the line and
does nothing; with it present nothing changes; a subclass with its own switch is patched when named; the laptop's
xformers check agrees with the import system; and both CO3 entry points call the patch before importing `paper2_rows`.
On the real classes (diffusers 0.35.2 and transformers 4.57.6 in the temporary CPU torch), `StableDiffusionXLPipeline`
takes the switch from `DiffusionPipeline` and does not define its own, and after the patch the call CO3 makes printed
the line and returned.

**Accepted** by Loop 18 on 2026-11-26 (rechecked there: `bash -n` on all seven scripts, `py_compile` on the three Python
files, 57 passed and 1 skipped under the CPU torch). Its decisions: the `.sh` in place of `.sbatch`, the build on the
card, the fifth model, the card minimums each command states, and the upload left unrun until question 119.

## CO3's loader, co3_common.py

`2-classifier-guidance/scripts/method_search/co3_common.py` was restored on 2026-11-26 by the owner's session from the
teaching scene's copy (`2-classifier-guidance/artifacts/scenes/co3-sampler/source/co3_common.py.copy`). CO3's row
(`paper2_rows.py`), the dial trainer (`paper2_identity_dial.py`, through `paper2_rows.py`) and task 8 all import it.
All three files, the restored one, the scene's copy and the hub repository's tracked copy
(`replications/published-methods-on-our-pairs/scripts/reproductions/co3_common.py`, last changed at commit `b91076f`),
have SHA-256 `ac2960190df1251d5e2d587860008e7c89764905aee48888d6c1491d242fd225`. `b91076f` is the commit in the
name of the code snapshot the Paper 2 rows read (`/datasets/mmolefe/poe_repair_min/loop-5/code/67b0bafa-b91076f`), so
the restored loader is the one earlier CO3 runs used. Task 8 is where it is proven to load CO3 here.

## Choices made where the plans were silent

Loop 18 may overrule any of these.

| Task | Choice | Why |
|---|---|---|
| 2 | A sixth file, `paper2_common.py`, holds the crop, the loop and the output files | the plan names five files; one shared front end keeps the four judges from drifting apart |
| 2 | A "plain" body (rule 1) and a "clear" body (rule 5) of a species is a body called that species, whatever its flag | a flag marks a borrowed part on a body that is still that species; only "mixed" takes it out |
| 2 | The picture list is a CSV with columns `picture`, `a`, `b` (bare species names); `bodies.json` is keyed by the picture path exactly as the list writes it | file names repeat across sources (`seed-6001.png` under every pair), so only the full path is unique |
| 2 | The 10% margin is 10% of the box's own width and height on each side, clipped to the picture | the plan says "a 10% margin" without a base |
| 2 | Every judge stores each body's raw scores in `calls.json`; `redecide` applies other settings to them without the model | steps 0.18 to 0.20 choose settings on choosing pairs, and this makes the grid search free |
| 2 | CLIP's four phrasings are combined by averaging the normalised text embeddings per name; other is decided on the softmax over all names, and m and p on the two asked shares, s_a = p_a / (p_a + p_b) | the plan names the phrasings and the grids, not how they combine |
| 2 | CLIP, ImageNet and Detect-and-Compare score 0 two-clear for a picture with fewer than two bodies, and follow the plan's formula for two or more (decided by Loop 18 for the owner) | rule 4 cannot call such a picture two clear, and a score up to 0.5 would rank it above an uncertain two-body picture in the within-pair AUC; the raw scores stay in `calls.json`, so `redecide` can reverse it |
| 2 | The ImageNet class table is `IMAGENET_CLASSES` in `paper2_bars.py` (step 0.2) or a JSON file given with `--class-table` | the plan puts the table in step 0.2, which is not written yet |
| 2 | Detect-and-Compare: a species' similarity is the mean cosine over its n references; flagged when the two similarities sit closer than b. It never calls mixed; `--neither` is off by default, so it never calls other either | the plan gives no rule for mixed or other, and gives it references of the asked species only. Loop 18 chose to add no rule after seeing data: as planned this judge cannot name hybrid or wrong species, and each counts as a miss on that kind |
| 2 | Vision-language: a body is called mixed when both species answers reach t | the plan does not say so for this judge; it is the plain reading of "both at once" (Loop 18) |
| 2 | Vision-language: yes probability is the yes tokens' mass over yes and no together on the first answer token; "the other" in the body-parts question is the asked species with the lower yes; the flag counts only on a body called A or B; each question ends "Answer yes or no." | the plan names Soft-TIFA's reading without these details |
| 2 | Vision-language two-clear score is 0 unless there are exactly two bodies, else the better of the two ways to assign them | rule 4 needs exactly two bodies, and the plan says "the yes probabilities that rule 4 needs" |
| 2 | Dry-run outputs go to `outputs/paper2/setup-checks/` (git-ignored) | the check must write `calls.json` and crops somewhere Loop 18 can open; nothing there is committed |
| 3 | The adapter wraps every linear layer named `to_q`, `to_k`, `to_v` or `to_out.0` under a module path containing `attn`, so both self-attention (`attn1`) and cross-attention (`attn2`); its scale is 1 (no separate alpha); down starts Kaiming-uniform, up at zero; its weights are float32 | "SDXL's attention layers" names no subset; up at zero makes the untrained dial add nothing |
| 3 | The norm of a dial is sqrt(sum over layers of ||up @ down||^2), the weight change it adds | the factors are not unique (up times c, down over c is the same dial), so only the product is a fair size |
| 3 | CO3 is loaded through `paper2_rows.py`'s own imports, so the trainer, the drawer and the CO3 row share one loader; the joint prompt is "a {A} and a {B}" with "an" before a vowel | the owner's requirement 1 asks to take CO3's step from that file |
| 3 | The closeness cache holds one step per prompt and starting seed, drawn once from the training window | item 2 asks for "a fixed set of those prompts, starting noises and steps" without saying how many steps |
| 3 | Both sides of the closeness term use one prediction function: the prompt's own text encoding, with CO3's empty-prompt row as the unconditional; only the dial differs | so the term measures the dial and nothing else |
| 3 | Each closeness item is backpropagated as it is computed | four graphs through SDXL at once would not fit beside Paper 1 on the shared card |
| 3 | The adapter's total gradient norm is checked with the reward term at step 1 | item 4 names "the gradient norm of the adapter" |
| 3 | The task term and the closeness term are backpropagated one after the other, and each one's gradient norm is logged | the guard needs each term's gradient on its own |
| 3 | The gradient route's rollout runs with the dial on from `--dial-from-step`, as the old trainer samples with its adapter on | the dial is trained on the states it will meet when it draws |
| 3 | The kept route draws with no dial, noises the kept draw back to step k with noise fixed by the seed and k, and trains on the joint prompt's own prediction (no guidance) | the plan names training on kept draws but not the target; the guided prediction is not aimed at the noise |
| 3 | `--lr` defaults to 1e-4 | the old trainer's 1e-5 fine-tuned an adapter already trained; a fresh rank-4 adapter starts from zero |
| 3 | The SDXL VAE decodes in float32 in the trainer and the drawer | the fp16 VAE overflows (the environment's fp16 upcast rule) |
| 3 | The gradient route's boxes come from the body counter on the detached decoded estimate at each step | "each object's box held fixed" (the ladder) |
| 3 | A `draw` subcommand draws the dial and random-dial rows | requirement 3 asks the same on-switch for the test rows; the CO3 row stays an untouched U-Net in its own process |
| 4 | The sheet's reviewer folder and its key are separate folders (`sheet/`, `key/`) | a reviewer is handed `sheet/` alone |
| 4 | A neutral name is ten consonants (no vowels, no digits) drawn from the shuffle seed | a name derived from the path could be reversed; with digits a name held a seed by chance (batch 1, seed 1: `cdb6008a57`), and with vowels it can spell a short row word such as own or box |
| 4 | The check searches only the identifying columns (pair, seed, source, round, row, made, also_called_in) and the source folders and file name; a pure-letter value of 4 letters or fewer counts only as a whole word; `rules.txt` must equal the seven rules exactly and is not searched | the call columns hold ordinary words ("one", "split") that the rules use; batch 1 at seed 2 failed on "one" in `rules.txt`, and the rows "both" and "plain" appear in the rules too |
| 4 | Each page is the picture under an 80-pixel white band printing only the asked species and the page's name; the plain picture sits beside it in `images/` | "the two species printed above"; the plain copy is for a reviewer or a scorer that wants the picture alone |
| 4 | Pictures are re-saved from their pixels; a picture not 1024 by 1024 stops the build | a renderer's PNG text chunk can hold its prompt and seed; the plan's pass condition is 1024 by 1024 |
| 4 | Each key entry keeps its picture under `picture_path`, a field no list column may take (a list with that column stops the build), and `check` requires every entry's path and file, with `--key` to check against another key | the builder once stored the path under `source` and the collector's own `source` column overwrote it, so a sheet built from the collector's list lost the link from a neutral name to its picture, and its check searched the source kind in place of the real file names and folders |
| 4 | `rules.txt` is the reviewers' rubric by version: version 1 (`RULES_V1`, frozen as batch-01 was built, checksum pinned) and version 2, the default, which flags a body only for a named part of the other species clearly covering more than a small patch, and says when unsure, do not flag; `key.json` records the version and `check` compares against it | batch-01's reviewers called its 34 borrowed-part pictures alike 26 times (0.76, under the 0.80 bar), and the owner chose to tighten the flag and call those pictures again; the plan's six-kinds table still carries version 1's wording, for the owner's session |
| 6 | The weight fetch is `paper2_fetch_weights.sh`, plain bash, not the plan's `paper2_fetch_weights.sbatch` | the owner's launch mode: one held node, no `sbatch` |
| 6 | A fifth repository, `madebyollin/sdxl-vae-fp16-fix` | CO3 loads it beside SDXL (`composition/debottam_co3/composers/Co3.py`, line 69); with `HF_HUB_OFFLINE=1` task 8 would fail without it |
| 6 | SDXL is fetched as its configs, scheduler, tokenizers and `*fp16.safetensors` only; the other four drop only other frameworks' formats | CO3 loads SDXL with `variant="fp16"`; the full SDXL repository is far larger |
| 6 | A stray download is caught by the growth of `~/.cache/huggingface`, `~/.cache/torch` and `~/.cache/pip` | the plan's "`/home-mscluster` grew by no more than 1 GB" cannot be read from the shared filesystem's total, which other users move |
| 7 | The route is question 119's proposed default, and the script does nothing without `--go` | the owner has not answered question 119 |
| 8 | CO3 builds on the held node's device 0, not the CPU | every CO3 run uses the card, and the load check should build it as the runs do |
| 8 | Without xformers, CO3 runs on diffusers' default attention (PyTorch scaled-dot-product attention), and every run records which | the owner's decision after task 8 stopped on mscluster112: `co3_bw` has no xformers and `co3` cannot drive a Blackwell card |
| 8 | The check also stops if stanza's English model is missing at `${STANZA_RESOURCES_DIR:-~/stanza_resources}/en` | CO3's parser loads it with `download_method=None` |
| env | The vision-language environment pins python 3.11, torch 2.11.0 and torchvision 0.26.0 (cu128), transformers 4.57.6, huggingface_hub 0.36.2, tokenizers 0.22.2, accelerate 1.12.0, safetensors 0.7.0, pillow 12.1.1, numpy 2.3.5 | torch as in `co3_bw`; transformers 4.57.6 is the last 4.x and holds Qwen3-VL (its wheel has `qwen3_vl/modeling`); it needs huggingface_hub below 1.0 and tokenizers 0.22; each version exists on PyPI or the cu128 index (`pip index versions`, 2026-11-26) |
| all | Card minimums: 12000 MiB for the CO3 load, 20000 MiB for Qwen3-VL; each command below states its own | the card is shared with Paper 1's process, so each run asks for what it needs rather than an empty card |

## Questions sent to the owner

| Board question | What it asks | Proposed default | Answer |
|---|---|---|---|
| 118 | whether a person can replace reviewer 1 of the three blind reviewers | Claude sessions | not yet answered |
| 119 | how the 192 joint-prompt pictures reach `/datasets` from the laptop | `rsync` through mscluster84 in the push direction | not yet answered |
| 120 | whether the vision-language judge may have its own mamba environment if `co3` cannot load Qwen3-VL-8B, and yes or no to tasks 6 to 8 | yes to both | yes (relayed by Loop 18, 2026-11-26) |

## The species judge's scripts, written after the setup

Written on 2026-11-26 at Loop 18's request, under the same rules, each reviewed and accepted before the next.

| File | Step | What it does | Loop 18 |
|---|---|---|---|
| `paper2_bodies.py`, `tests/test_paper2_bodies.py` | 0.13 | the body counter (`count_instances`, "animal", as committed) over a picture list: `bodies.json` in the shape the judges read, the crops, a 24-picture box sheet, the measured seconds a picture | accepted |
| `paper2_cat_list.py`, `paper2_cats_64.csv` | 0.15 | the 64 single cats of the cat census (base SDXL, "a photo of a cat", seeds 0 to 63), listed from its committed per-seed record, asked cat against dog; `--check` confirms them on the node | accepted; on mscluster112 "64 listed, 64 found, 64 at 1024 by 1024" |
| `paper2_bars.py`, `tests/test_paper2_bars.py` | 0.2 | all 17 bars at the plan's values, the seven rules, the grids, every species' order and the look-alike rule, the 23 pairs and their types, eight new look-alike pairs, the renaming table, the old judge's mapping, and the ImageNet class table (60 species; 13 of 23 plan pairs covered), checked against torchvision's class list | accepted, with pig and boar replaced by otter and weasel |
| `paper2_fetch_judges.sh` (and `paper2_fetch_weights.py --set judges`) | 0.14 | CLIP ViT-L/14 and `timm/convnext_large.fb_in22k_ft_in1k` into `HF_HOME`, safetensors only, with the setup fetch's guards | accepted |
| `paper2_co3_patch.py`, `tests/test_paper2_co3_patch.py` | task 8 | CO3 on PyTorch's attention where xformers is missing, both of CO3's xformers calls covered | accepted; on mscluster112 "PASS: CO3 built offline", attention `pytorch-sdpa` |
| `paper2_crop_check.py`, `tests/test_paper2_crop_check.py` | 0.15 | reads every judge's calls on the 64 cats: the bar on the pictures where every body is called cat with no flag (62 of 64), beside the share of crops called cat and the bodies-per-picture histogram | used for step 0.15 on mscluster112 |
| `paper2_judge_old.py`, `tests/test_paper2_judge_old.py` | 0.15, 0.17 | the old judge's namer and verdict on the shared lists and bodies, its four departures from `species_judge.py` written in its docstring | used for step 0.15 on mscluster112 |
| `paper2_dac_diagnose.py` | after 0.15 | prints the Detect-and-Compare judge's similarities for a few pictures at its own crop, at the tight unpainted box and against the averaged reference, with the references' similarity among themselves | written for the wiring check below |
| `paper2_collect.py`, `records/`, `tests/test_paper2_collect.py` | 0.4 | the candidate pictures from the plan's six sources, each tied to its pair and seed by a committed record (the uncommitted joint-prompt key and calls and the old judge's labels copied into `records/` with their sources): 96, 192, 867, 200, 128 and 29 as the plan counts; 1,335 distinct on animal pairs after the lighthouse and sailboat (64) and one cell its finding records as failing the subject check; 149 hybrid and 242 two-clear candidates from earlier calls; the 100 unselected CO3-alone pictures by seed 2; `--check` on the node opens every file | sent for review |
| `paper2_label_set.py`, `tests/test_paper2_label_set.py` | 0.8 | settles each batch's calls, merges batches (a later batch replaces an earlier label), refuses a two-clear or borrowed-part label from a version-1 batch once a version-2 batch exists, reports agreement against both bars, splits the pairs by the first seed-0 draw meeting the size table, seals the confirmation slice's kinds in `confirm.json` (blank in `labelled-set.csv`), and writes the owner's 60 by seed 4; on thin data it names each short kind and lists the next candidates, and once split a short unselected sample gets its top-up drawn by seed 2 into `unselected-next.csv` | sent for review |
| `paper2_render_lookalike.py`, `tests/test_paper2_render_lookalike.py`; `paper2_collect.py --sources` | 0.9, 0.10 | the plain sum (z10's sampler at weight 0, no adapter attached) and CO3 alone (as `paper2_rows.py` draws its co3 row) on the eight new look-alike pairs at seeds 8001 to 8008, each with its record and contact sheet; the collector takes them as two new sources, named with `--sources new-plain new-co3`, while its default stays the plan's six | sent for review |

**What the collector's source map corrects in the species judge plan's table** (for whoever edits the plan): the box
rows' "rest" halves are tied to their pairs by committed keys (`rest/key-rest.json` in rounds 12 and 13), so all 240
resolve; the box rows' folders are `z2`, `z3`, `z7`, `z8`, `z12` and `z13` (round 03's pass rows are in `z3`); no
picture was drawn by a push whose checker read CLIP or ImageNet features (round 08's push reads SDXL's U-Net readout
features, and CLIP only named which detected box was which animal in round 12's `box` and round 13's `own` rows); and
the "491 two-body pictures with a body not keeping its own parts" does not reproduce (522, 497 or 466 by settling rule).

**Batch-01 and its recall sheet were built before the key path fix.** Their `key/key.json` entries have lost each
picture's path (the collector's `source` column overwrote it), so Loop 18 relinked them without guessing, each entry
matching exactly one row of the batch's `list.csv` on every other column, into `key/key-relinked.json` beside the
untouched key. Their first checks searched the identifying columns but not the real file names and folders. Rechecked
on the node with `paper2_sheets.py check --out <batch> --key <batch>/key/key-relinked.json` after the fix (commit
7f175c0c): batch-01, 200 entries, 0 with no path, 0 missing, passed; batch-01-recall, 97 entries, the same, passed. The
pages the reviewers saw carried nothing identifying.

**The first blind calls** (Loop 18, on the node). Batch-01, 200 pictures under rubric version 1: reviewers 1 and 2 agree
on 181 of 200 (kappa 0.876) and on two clear against borrowed part at 0.905 (bar 0.85), but on the pictures settled as
borrowed part at 0.76 (26 of 34), under the 0.80 bar; two clear 0.95, hybrid 0.94, two of one kind 0.89, extra subject
0.95. The recall of its 97 two-clear and borrowed-part pictures under version 2: 95 of 97 alike; two clear against
borrowed part 91 of 92; borrowed part 1.00 (9 of 9), two clear 0.99, hybrid 0.80. Under version 2 those pictures hold 9
borrowed parts where version 1 had 34, so borrowed part may be the scarce kind in the test set, which needs 40.

**Step 0.15 on mscluster112** (the 64 cats, one body each, shared bodies): the old judge, CLIP, ImageNet and the
vision-language judge each name 64 of 64. The vision-language judge's yes-cat median is 1.000, so its first-token yes
reading works. Detect-and-Compare names 8 of 64: all 64 crops are called cat, and 56 carry the borrowed-part flag
(references: 16 Wikimedia Commons photographs a species, 8 used, at
`/datasets/mmolefe/poe_repair_min/paper2/references/`, licences in `references.csv`).

**Detect-and-Compare fails the crop check at every grid value, and it is not a wiring fault.** Its feature is DINOv2
large's pooled CLS token, L2-normalised on both sides (a photo against itself scores 1.0000). The crop and the references
share one preprocessing, and a species' similarity is the mean of the per-reference cosines. On real photographs (8 COCO
cat and 8 COCO dog references from `4-inclusive-captions/data/training-set`, 24 other real cats as queries, on the
laptop) that reading gives cat a median of 0.254 and a gap over dog of 0.234, so only 14 of 24 real cats pass b = 0.2.
The references agree among themselves at 0.218 (cat against cat). The 64 SDXL cats (median 0.188, gap 0.140) sit a
little below. Squashing the picture in place of the centre crop changes the gap by 0.014, and the margin and paint-out by
0.006 at most. MuDI's own Detect-and-Compare (arXiv 2404.04243, section 4.3) compares a crop's similarities with the
references' similarities among themselves (S^DC against S^GT) rather than thresholding a raw gap. Any such change is a
new rule written after seeing step 0.15, so the judge is out of steps 0.17 to 0.21. Whether a follow-on plan tests a
version faithful to MuDI on new test pairs is the owner's call.

## What is still open

1. **Questions 118 and 119** have no answer. Task 7 does not run until 119 names a route.
2. **No real model has met this code.** CO3 on SDXL, the four judges' models (CLIP ViT-L/14, the ImageNet classifier,
   DINOv2 large, Qwen3-VL-8B), the body counter and the vision-language environment have each been written against
   but not run. The real-model paths are `co3_sampler`, `CropShareReward`, `JudgeKeep` and `closeness-cache` in the
   trainer, each judge's model class, and the scripts for tasks 6 and 8. The first launch tests them.
3. **CO3 on PyTorch's attention in place of xformers** has not drawn a picture yet: the two kernels compute the same
   attention, but whether CO3's pictures under `pytorch-sdpa` match its earlier xformers renders is unmeasured, so a
   comparison across rows keeps to one kernel, which each run records.
4. **CO3 on a single-animal prompt** is untried: `closeness-cache` points CO3 at "a cat" as well as at pairs, and CO3's
   parser has only ever been given two-subject prompts here.
5. **The CO3 code snapshot** at `/datasets/mmolefe/poe_repair_min/loop-5/code/67b0bafa-b91076f` is the path
   `m1_paper2.sbatch` names; it was not checked live. The guards stop with a message if it is missing.
6. **The sentence parser's model** (stanza, English) must already sit at `${STANZA_RESOURCES_DIR:-~/stanza_resources}/en`
   on the cluster; whether it does is not recorded. Task 8 stops if it is missing.
7. **The trainer's memory on the shared card** is unmeasured. The earlier reward fine-tune needed a 48 GB card, so the
   commands below ask for 48,000 MiB free; the smoke measures it.
8. **Owed by the species judge plan's own steps.** `paper2_collect.py` (0.4), `paper2_label_set.py` (0.8) and
   `paper2_score.py` (0.21), which wait for the reviewers, and the choice of each judge's settings on choosing pairs
   (0.18 to 0.20), which `paper2_common.redecide` turns into a re-scoring of stored raw scores with no model. Step 0.2's
   review questions, `plans/paper-2-dials/review/02-species-judge.md`, stay with the owner's session. The
   Detect-and-Compare judge needs real photographs, 8 or 16 per species, from a source the owner trusts, with their
   licences recorded; until a references folder exists it cannot run. The first result's CO3-alone row is drawn by an
   untouched CO3 in its own process, not by this trainer's `draw`, which refuses strength 0.
9. **The vision-language environment's pins** all exist on PyPI or the cu128 index, but the build is untried.

## The commands, ready for the owner's yes

Everything below waits for the owner's launch: the owner commits this work (Loop 18 lists the commit), pushes the
branch, and holds one Blackwell node (mscluster110, 111 or 112) with an interactive allocation. Each command runs from
the cluster checkout on that node, `/home-mscluster/mmolefe/Playground/PhD/concept-composition`, unless it says laptop.
Every script opens with its guards and logs to `/datasets/mmolefe/poe_repair_min/paper2/logs/<run>.log`.

**Tasks 6 to 8, and the vision-language environment**

| Order | Command | Passes when |
|---|---|---|
| 1, task 6 | `bash 2-classifier-guidance/scripts/paper2/paper2_fetch_weights.sh` | the last line reads `PASS: every config opens offline`; `outputs/paper2/paper2-fetch-weights/sizes.txt` lists five repositories with their sizes, and the home folder's caches grew by under 1 GB |
| 2 | `/home-mscluster/mmolefe/miniforge3/envs/co3_bw/bin/python -c "from transformers import Qwen3VLForConditionalGeneration"` | it imports: the vision-language judge can run on `co3_bw` (launch it with `co3`), and steps 3 and 4 are skipped. An `ImportError` means steps 3 and 4 |
| 3 | `bash 2-classifier-guidance/scripts/paper2/paper2_vlm_env_build.sh` | it prints every pinned version, "transformers has Qwen3-VL" and the folder's size at `/datasets/mmolefe/envs/paper2-vlm` |
| 4 | `bash 2-classifier-guidance/scripts/paper2/paper2_vlm_env_check.sh` | `PASS: Qwen3-VL-8B answered on the card`; `outputs/paper2/paper2-vlm-env/load.txt` holds the load time, seconds per answer, peak memory and one yes probability |
| 5, task 8 | `bash 2-classifier-guidance/scripts/paper2/paper2_co3_load.sh` | `PASS: CO3 built offline`; `outputs/paper2/paper2-co3-load/load.txt` names `method_search/co3_common.py`, the versions and `attention pytorch-sdpa` on `co3_bw`. This proves the restored `co3_common.py` loads CO3 |
| 6 | `bash -c 'source 2-classifier-guidance/scripts/paper2/paper2_guards.sh && p2_python && p2_co3_path && "$PY" -m pytest tests/test_paper2_kinds.py tests/test_paper2_sheets.py tests/test_paper2_co3_import.py tests/test_paper2_identity_dial.py -q'` | every test passes, none skipped: the real import of `paper2_rows` and `co3_common` runs here |
| 7, task 7, on the laptop | `bash 2-classifier-guidance/scripts/paper2/paper2_upload_joint.sh --go`, only once question 119 names this route | `PASS: 192 pictures at 1024 by 1024 on /datasets`; then record the route in `environment/storage.md` |

**The species judge** (`plans/paper-2-dials/plans/02-species-judge.md`), the steps that run code written here. Each
judge's settings default to the middle of its grid, which is what step 0.15 asks.

| Step | Command |
|---|---|
| 0.5, the blind sheets | `"$PY" 2-classifier-guidance/scripts/paper2/paper2_sheets.py build --pictures <step 0.4's candidates as picture,a,b plus any other columns> --seed <the batch's seed> --out /datasets/mmolefe/poe_repair_min/paper2/paper2-label-sheets/<batch>`, then `... paper2_sheets.py check --out <the same folder>`; the reviewers get `<batch>/sheet/` only |
| 0.6 and 0.7, the reviewers | wait on question 118 (whether a person replaces reviewer 1); the reviewers get each batch's `sheet/` folder only, never `key/` |
| 0.8, the labelled set | `"$PY" 2-classifier-guidance/scripts/paper2/paper2_label_set.py --batches <manifest>.json --candidates /datasets/mmolefe/poe_repair_min/paper2/paper2-collect/candidates.csv --out /datasets/mmolefe/poe_repair_min/paper2/paper2-label-set`; the manifest lists each batch's key (the relinked key for batch-01 and its recall) and calls files in the order they were called. A short kind leaves `next-candidates.csv` for the next batch; a short unselected sample, once split, leaves `unselected-next.csv` |
| 0.9 and 0.10, a smoke first | on mscluster112, from the checkout: `bash 2-classifier-guidance/scripts/paper2/paper2_launch.sh paper2-render-plain-lookalike co3 16000 paper2_render_lookalike.py plain --pairs-limit 1 --seeds-limit 1` and the same with `paper2-render-co3-lookalike` and `co3`; each draws one picture (lion and tiger, seed 8001) and prints its seconds. A picture already drawn is skipped later, so the smoke's picture counts |
| 0.9, the plain sum on the new look-alike pairs | from the laptop, outside Slurm, behind the guards: `ssh mscluster112 'cd /home-mscluster/mmolefe/Playground/PhD/concept-composition && nohup bash 2-classifier-guidance/scripts/paper2/paper2_launch.sh paper2-render-plain-lookalike co3 16000 paper2_render_lookalike.py plain > /dev/null 2>&1 &'`. 64 pictures (the eight `NEW_LOOKALIKE_PAIRS` at seeds 8001 to 8008) under `/datasets/mmolefe/poe_repair_min/paper2/paper2-render-plain-lookalike/renders/`, with `record.json` and `contact-sheet.png`; the log is `/datasets/mmolefe/poe_repair_min/paper2/logs/paper2-render-plain-lookalike.log`. Passes when the last line reads "64 of 64 on disk" |
| 0.10, CO3 alone on the same pairs and seeds | `ssh mscluster112 'cd /home-mscluster/mmolefe/Playground/PhD/concept-composition && nohup bash 2-classifier-guidance/scripts/paper2/paper2_launch.sh paper2-render-co3-lookalike co3 16000 paper2_render_lookalike.py co3 > /dev/null 2>&1 &'`, into `/datasets/mmolefe/poe_repair_min/paper2/paper2-render-co3-lookalike/`, the same files. A pair CO3's parser does not split into 2 concepts is skipped and named in `record.json`. Both rows can run at once beside Paper 1 on the 96 GB card (each asks for 16,000 MiB free). Both are drawn on the Blackwell card, where z10's pictures came from RTX 3090s; the card is in each `record.json`, and these pictures only join the candidate pool, so no comparison crosses cards |
| batch 3's candidates | `"$PY" 2-classifier-guidance/scripts/paper2/paper2_collect.py --sources new-plain new-co3 --check --out /datasets/mmolefe/poe_repair_min/paper2/paper2-collect-lookalike`: its `candidates.csv` holds the 128 new pictures, all look-alike and uncalled, and the run passes when all 128 open at 1024. Build batch 3 from it with `"$PY" 2-classifier-guidance/scripts/paper2/paper2_sheets.py build --pictures /datasets/mmolefe/poe_repair_min/paper2/paper2-collect-lookalike/candidates.csv --seed <a new seed> --out /datasets/mmolefe/poe_repair_min/paper2/paper2-label-sheets/batch-03` (rubric version 3 by default), then `check`; add batch 3 to the label-set manifest after the batches before it |
| 0.11, your 60 | `owner-list.csv` stands in for the plan's `owner-sheet/` folder: build your sheet from it with `"$PY" 2-classifier-guidance/scripts/paper2/paper2_sheets.py build --pictures /datasets/mmolefe/poe_repair_min/paper2/paper2-label-set/owner-list.csv --seed <a new seed> --out /datasets/mmolefe/poe_repair_min/paper2/paper2-label-set/owner-sheet`, then `... paper2_sheets.py check --out <the same folder>`; you get its `sheet/` folder and call into its `calls-template.csv` |
| renamed wrong species | made only after the whole candidate pool has been called and wrong species is still short of 65 natural pictures: the session running step 0.8 (Loop 18, or whoever holds the species judge) writes a picture list of settled two-clear pictures from pairs holding a species in `paper2_bars.RENAMING`, with that species' name replaced by its renaming (cougar to leopard, goose to swan, fox to coyote, flamingo to spoonbill), builds it with `paper2_sheets.py` like any batch, and has it called blind; at most 20 of the 40 read, reported apart by renaming entry. No script makes them yet; that list is owed then |
| 0.15, CLIP | `bash 2-classifier-guidance/scripts/paper2/paper2_launch.sh paper2-judge-check co3 6000 paper2_judge_clip.py --pictures <the 64 cats, a=cat, b=dog> --bodies <paper2-find-bodies>/bodies.json --out /datasets/mmolefe/poe_repair_min/paper2/paper2-judge-check/clip` |
| 0.15, ImageNet | the same with `paper2_judge_imagenet.py`, `--out .../paper2-judge-check/imagenet`, and the class table from `paper2_bars.py` or `--class-table <json>` |
| 0.15, Detect-and-Compare | the same with `paper2_judge_dac.py`, `--references <a folder of <species>/ real photographs>`, `--out .../paper2-judge-check/dac` |
| 0.15, vision-language | `bash 2-classifier-guidance/scripts/paper2/paper2_launch.sh paper2-judge-check vlm 20000 paper2_judge_vlm.py --pictures <the 64 cats> --bodies <bodies.json> --out .../paper2-judge-check/vlm` (`co3` in place of `vlm` if step 2 above imported) |
| 0.18 to 0.20 | each judge as in 0.15 on the labelled set's choosing and test pairs, with the run names `paper2-judge-clip`, `paper2-judge-imagenet`, `paper2-judge-dac`, `paper2-judge-vlm`, and the settings chosen on choosing pairs passed as `--margin`, `--flag-share`, `--borrow-gap`, `--n-refs` or `--yes-threshold` |

**The identity dial's first result** (`plans/paper-2-dials/plans/03-identity-dial-first-result.md`). `<LO> <HI>` is the
noise window from the gradient judge's half-drawn read (species judge step 0.24); `<pairs>` is the training pairs file
with a weight per pair from step 2's `sets.json`; `<judge>` and its `settings.json` are the gradient judge the owner kept.

| Step | Command |
|---|---|
| 3, the closeness cache | `bash 2-classifier-guidance/scripts/paper2/paper2_launch.sh paper2-first-closeness-cache co3 16000 paper2_identity_dial.py closeness-cache --closeness-prompts 2-classifier-guidance/scripts/paper2/paper2_closeness_prompts.txt --seeds 21001:21005 --window <LO> <HI> --out /datasets/mmolefe/poe_repair_min/paper2/paper2-first-closeness-cache/closeness-cache.pt` |
| 3, the smoke | `bash 2-classifier-guidance/scripts/paper2/paper2_launch.sh paper2-first-identity-smoke co3 48000 paper2_identity_dial.py train --pairs-file <pairs> --seeds 11001:11065 --route gradient --reward <judge> --judge-settings <its settings.json> --window <LO> <HI> --closeness-cache /datasets/mmolefe/poe_repair_min/paper2/paper2-first-closeness-cache/closeness-cache.pt --closeness-weight 1 --steps 50 --ckpt-steps 50 --out /datasets/mmolefe/poe_repair_min/paper2/paper2-first-identity-smoke`; then read `closeness.json`'s `weight_for_a_tenth` |
| 4, training | the same with run name `paper2-first-identity-train`, `--closeness-weight <from the smoke> --steps 1000 --ckpt-steps 500 1000 --out /datasets/mmolefe/poe_repair_min/paper2/paper2-first-identity-train` |
| 6, the random dial | `/home-mscluster/mmolefe/miniforge3/envs/co3_bw/bin/python 2-classifier-guidance/scripts/paper2/paper2_identity_dial.py random --like /datasets/mmolefe/poe_repair_min/paper2/paper2-first-identity-train/ckpt-<picked>/dial.pt --out /datasets/mmolefe/poe_repair_min/paper2/paper2-first-random-dial/dial.pt --seed 0` (no card) |
| 6, the dial row | `bash 2-classifier-guidance/scripts/paper2/paper2_launch.sh paper2-first-identity-test co3 16000 paper2_identity_dial.py draw --ckpt <paper2-first-identity-train>/ckpt-<picked>/dial.pt --strength 1 --pairs-file <the 16 test pairs> --seeds 13001:13004 --out /datasets/mmolefe/poe_repair_min/paper2/paper2-first-identity-test/dial` |
| 6, the random-dial row | the same with `--ckpt /datasets/mmolefe/poe_repair_min/paper2/paper2-first-random-dial/dial.pt` and `--out .../paper2-first-identity-test/random` |
| 6, the sheets | `paper2_sheets.py build` over the 144 pictures (columns picture, a, b, row, pair, seed), then `check`, as in step 0.5 |
