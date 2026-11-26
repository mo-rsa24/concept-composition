# 🧭 Progress: where the project's own sources disagree

Every place where two of this project's sources say different things, found while checking the
pillar pages against their pictures, cards, code and cluster files on 2026-11-21. Nothing here has
been resolved. Each page named below states both sides. Step B6 reads this list before it writes
what we know.

## Table of contents

- [Numbers a paper or finding could quote](#numbers-a-paper-or-finding-could-quote)
- [Pictures that disagree with their cards](#pictures-that-disagree-with-their-cards)
- [Code and system facts](#code-and-system-facts)

## Numbers a paper or finding could quote

Navigation: 📋 [TOC](#table-of-contents) | [Next](#pictures-that-disagree-with-their-cards) ➡️

**These nine can change a sentence in a paper, so they come first.**

| What | One source says | The other says | What would settle it |
|---|---|---|---|
| Animal pairs in the current pool | 17, in the cluster's `fail_rate.md` | 20, in `plans/standing/retrofit-poe-repair-min.md` | List the pair folders in the current pool |
| Real errors in the 32-render scorer audit | 1 of 32, on [the problem page](../../../context/purpose/01-the-problem.md) | "One definite error in 30", on the audit card | Recount the audit's `calls.json` |
| True compose rate at correction strength 1 | 87% to 94%, on [the compose-rate page](../../../context/world/compose-rate.md) and its card | 75% to 94%, in an open review task | Close that review task |
| The audit's two failed tiles | The card says both were scored correctly | The seal and walrus tile (seed 9, count 1) shows two separate animals | Read that tile at full size |
| Elephant and penguin, seed 9 | The card says two elephants merged into one box | The picture shows one elephant | Read that tile at full size |
| Both-ness at seed 12, product plus 1.2 times the correction | 0.4714, in the [dictionary](../../../context/data/02-dictionary.md) and the cluster JSON | 0.4667, in the scene's `public/data.json` | Find which step and rounding each file uses |
| Length of `d_t` | 50 values, in the dictionary | 51 values (indexes 0 to 50), in the pilot `summary.json` | Correct the dictionary entry |
| Pairs drawn on the correction-size figure | Five named pairs, in the [research guidelines](../../../context/research-guidelines.md) | Two named pairs on the card, and none in the saved picture | Decide which version of the figure is current |
| Freshness of the 10 Inkscape recipes | An earlier index count named 1 as `ran` | No record says which, so [the runbook](../../../runbook/00-INDEX.md) marks all 10 `unverified` | Say which recipe was run |

## Pictures that disagree with their cards

Navigation: ⬅️ [Numbers](#numbers-a-paper-or-finding-could-quote) | 📋 [TOC](#table-of-contents) | [Next](#code-and-system-facts) ➡️

**In each row the picture was opened and read by eye, and its card says something else.**

| Picture | The card says | The picture shows |
|---|---|---|
| Every adapter by seed, version 2 | 10 adapter columns | 9 adapter columns plus "no adapter" |
| Corrector eight-seed sheet | 5 of 8 tiles are cartoons or line drawings | 4 of 8 (seeds 13 to 16) |
| Corrector across-counts strip | Runs from photograph to drawing | The first tile (k = 0) is not a photograph |
| SuperDiff cosine profile | The two norms are the same size throughout | Bucket medians of 10.7 to 47.2 for the adapter and 8.6 to 17.1 for the residual |
| SuperDiff baseline, 50 steps | Composes at 50 steps | The cat and dog tiles are one stylised cat head |
| SuperDiff baseline, 200 steps | The clamp changes nothing, and the meadow holds a blurred butterfly | Clamped and unclamped cat and dog differ, and no butterfly is visible |
| SuperDiff loss curves | Recipe 2 sets 100,000 steps | The x axis ends near 61,000 |
| SuperDiff cosine sidecar | The rank-8 sheet row names step 410,000 | The sidecar names step 420,000 |
| Twist head accuracy and loss | Loss reaches zero at high noise | The curve keeps swinging, and the sidecar records 0.034 at step 100,000 |
| SuperDiff per-bucket loss | Rank 8 is lowest everywhere | Rank 8 is at times above the others in the early panel |
| Same-concept-twice grid | Five cells and three columns | Four rows and four columns |
| Published-models grid | Three rows | One row, cat and dog |
| Where each condition lands | Recipe 4 names three screenshots | The finding embeds two |

## Code and system facts

Navigation: ⬅️ [Pictures](#pictures-that-disagree-with-their-cards) | 📋 [TOC](#table-of-contents)

**The first two rows can send a run to the wrong place or onto a dead card.**

| What | One source says | The other says | What would settle it |
|---|---|---|---|
| Where the training cache lives | `outputs/training_cache`, in `poe_repair/training_cache.py` and [the storage page](../../../environment/storage.md) | `artifacts/caches/training_cache`, in `poe_repair/paths.py`. Both folders exist on `/datasets` | Compare the two folders and name one |
| Launching on a faulted GPU | [The nodes page](../../../environment/hpc/nodes.md) says the shared-device launcher refuses `[N/A]` | `scripts/launch_sdlora.sh` passes any card under 100 MiB, whatever utilisation reads | Decide whether to add the refusal |
| Memory limit for sharing a GPU | 1 GB, in the safety rule | 4,096 MiB at 0% utilisation in `launch_sdlora.sh`, 1,024 MiB in `run_corrector_curve.sh` | Pick one limit |
| `co3` path check in job scripts | Every job script has it, on [the overview](../../../environment/overview.md) | `review_bench_co3.sbatch` does not | Correct the page or the script |
| Checks in the plan-tree script | Four, in the project's `CLAUDE.md` and the [recipe](../../../runbook/checking-the-plan-tree.md) | Nine, in `scripts/plan_pulse.py` | Correct `CLAUDE.md` and the recipe |
| The script's usage text | `--checks 1,2` | That form crashes, and `--checks=1,2` works | Correct the usage text |
| Control pairs for the fail rate | One, on [the animal-pair page](../../../context/world/animal-pair.md) | Three, in `fail_rate.py` | Correct the page |
| Smallest allowed figure text | 8 pt, in the presentation retrospective | 7 pt, as `MIN_PT` in `figure_text_audit.py` | Pick one floor |
| SuperDiff recipe commands | `$REPO/scripts/...`, as recorded in the [recipe](../../../runbook/running-things-on-the-cluster/training-and-watching-a-superdiff-adapter.md) | The scripts now sit under `1-lora-correction/scripts/`, and the recipe gives that path beside each recorded command | Settled for the two moved script sets |
| Scene served by the browser recipe | Root `artifacts/scenes/`, in the [recipe](../../../runbook/looking-at-what-a-run-produced/showing-a-scene-in-the-local-browser.md) | `1-lora-correction/artifacts/scenes/`, in both checkouts | Correct the recipe's path |

<!-- built-to: PAGE_LOOK_FORMAT v1 -->
