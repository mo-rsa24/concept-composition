# Retrospective: the SDXL product sampler's classifier push repaired (route step 5)

The walk replaced the product sampler's rescaled classifier push with Dhariwal and Nichol's unrescaled form, capped it at a tenth of the product's own prediction, and checked that at strength 0 the sampler still renders cat × dog seed 9 byte for byte like plain product. Both pieces passed. The plan was [route step 5](../../../plans/route-classifier-guided-poe-from-toy-to-animals.md#5--repair-the-sdxl-product-samplers-push-and-rerun-the-canary), and the record is the commits.

## T1: the cap piece

**The request.** "Replace the rescaled push in the product sampler with epsilon minus s times sigma_t times the classifier gradient, the form plan 17 uses. Cap the push at a fraction kappa of the product's own step, with kappa a named constant. Log the raw gradient size and the applied push at every step."

**The problem.** `run_poe_classifier_guided` already had a `strength` branch in the paper's form, but its default `weight` branch still stretched the push to a fixed share of the product prediction's norm, about 44,000 times too large. The composer, the canary and three scripts (the weight sweep and both head-training probes) only spoke `weight`. Nothing capped the push.

**What was done.** The `weight` branch was deleted, and a nonzero `weight` now raises with a message naming `strength`. A function `capped_push` scales the push down so its norm is at most `PUSH_CAP_FRACTION` (0.1) times the norm of the product prediction, keeping its direction. Each step logs the raw gradient norm, the push before and after the cap, and whether the cap fired. The composer gained `strength` and `push_cap`, and its output folders are named `poe_classifier_st<strength × 100>`. Commit `445cef9f`.

**Where the results live.** The code is in `poe_repair/methods/_classifier_guided.py` and `poe_repair/composers/poe_classifier_guided.py`. The check output is on the T1 line of route step 5.

**Before and after.** Before: any `weight` gave a push of weight times the prediction's norm, whatever the classifier said. After, on the mscluster login node: a push 0.871 the size of the prediction (plan 17's largest at strength 300, seed 9) comes out at 0.100 with direction cosine 1.000000; a push of 0.051 (strength 10's largest) passes untouched; a nonzero `weight` raises.

**Links to.** This is the first piece; the canary below runs on this sampler.

## T2: the canary piece

**The request.** "Rerun the canary: strength 0 against plain product on cat × dog seed 9."

**The problem.** The canary script called the composer with `weight=0.0`, so it tested a branch that no longer existed. Seed 9 has no cached starting latent on the cluster (only seed 42 does), so its noise comes from `manual_seed(9)`, as in plan 17. The cluster checkout could not fast-forward because step 4's walk map was edited there; the edit matched the laptop's commit line for line.

**What was done.** The canary moved to `strength=0.0` and records the cap in `canary.json`; a job script `presence_head_canary.sbatch` and a VS Code launch entry "canary strength 0" were added (commit `d1b8b61f`). The cluster's walk map was restored and the checkout fast-forwarded. The job ran on an idle node, mscluster108, as Slurm job 89341 (Quadro RTX 8000, 1 min 41 s).

**Where the results live.** `/datasets/mmolefe/poe_repair_min/outputs/presence_head/canary/canary.json` and the three renders it names.

**Before and after.** Before: the strength-0 identity was untested on the product sampler. After: plain product, strength 0 with no head, and strength 0 with an untrained head share md5 `c291b259647d30a450e87b234423fe99`, with 0 pixels differing. The plain render is one fused animal with a cat's ears and a dog's muzzle, which is what plain product draws on this seed.

**Links to.** It runs the sampler T1 built, at the strength where the classifier is never called, so it checks the wiring around the push rather than the push.

## The stop point

- Bar: `canary.json`'s strength-0 md5 equal to plain product's. Met.
- Time box: 1 sitting, about 90 minutes. About 100 minutes used, with the cluster fast-forward and the latent-cache check the main overrun.
- The bar arrived first. Way on: ship.

## Decisions

| Decision | Chosen | Options | Who | Why |
|---|---|---|---|---|
| the old rescaled branch | deleted; a nonzero `weight` raises | keep it beside `strength` · delete and move the three callers | you, by defaults | old callers fail loudly instead of drifting |
| κ | 0.1, `PUSH_CAP_FRACTION` | your own value | you, by defaults | the paper caps nothing; route step 9 sets the final value |
| what the cap measures | whole-latent norm, scaled down | a per-element clamp | you, by defaults | keeps the push's direction |
| the plan file | route step 5 | a new file in `plans/` | you, by defaults | a repair whose record is the commit |
| the route's uncommitted thumbnails | committed on their own, `980c5ff4` | wait for that session | you, by defaults | step 4's closing pass was already committed |
| the launch | sbatch on idle mscluster108 | nohup on a shared card | you, by `fix` | the launch recipe puts an idle node first |
| predictions | none asked | a prediction before each test | you | you asked not to predict |

## Dead ends and not done

- The weight sweep (`presence_head_weight_sweep.py`) and both head-training probes (`train_presence_head.py`, `train_strong_presence_head.py`) still pass `weight` and will raise until they move onto `strength`.
- `classifier_guidance_on_the_product.py` now runs with the cap on by default; its earlier renders were uncapped.
- Plan 17's `run_cfg_classifier_guided` has no cap, which is kept as the paper's reproduction; board row R0's wording says every sampler.
- No delivery log row: this project keeps no `report/delivery-log.md`.
- The cap fired on no step of this run, since strength 0 never calls the classifier; whether 0.1 binds on cat × dog is for step 9.

## The family table

| Kind | Id | What | State |
|---|---|---|---|
| this walk | `0bfb147d-0b70-4993-b3e9-330cad1293b2` | route step 5 | closed |
| Slurm job | 89341 | the canary on mscluster108 | completed, exit 0 |

---

Validation: 9 paths checked on disk (the route, the two code files, the canary script and job script, the launch file, the board, the walk map, this folder's card), 0 missing; the md5 checked against `canary.json` and against the render fetched to the laptop; the cap numbers checked against the T1 line.
