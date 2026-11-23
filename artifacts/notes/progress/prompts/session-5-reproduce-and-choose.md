# Session 5: reproduce the rivals

This runs as a background session, in the same permission mode as the other loops, and starts when step D1 is green.
`scripts/start_when_step_done.sh` starts it. To start it by hand from `/home/molef/PhD/concept-composition`:

```text
claude --bg --permission-mode auto --model claude-opus-5-5 --effort high --name "🔬 PhD Loop 5: Reproduce Rivals And Choose Checker" "/loop Follow artifacts/notes/progress/prompts/session-5-reproduce-and-choose.md to its end. Read artifacts/notes/progress/prompts/shared-rules.md on every tick."
```

## Background

The two-paper route cannot reach its go or no-go (step D4) until each rival's result has been reproduced on our protocol
and Session 6 has frozen a recipe per paper. Sessions 5 and 6 are the only loops allowed to spend GPU time, and they share the
job caps. Session 4 has already read the rivals and the owner has approved the targets on the scoreboard. Your owner code on the
board is `s5`, and your step is D2. By the owner's choice in board question 43, reruns continue only for the rival with the best
printed score on each benchmark. Session 6 owns the checker comparison (D3) and the fine-tune test (D5).

## Objectives

1. Step D2: for every rival with a row in `qualitative` in `artifacts/notes/progress/scoreboard.json`, write `artifacts/notes/progress/reproductions/<rival>/design.md` before its first run.
2. Step D2: run each rival up the ladder below on every headline dataset it has a `qualitative` row for, with its published metric, sample count and settings, and write the verdict to `artifacts/notes/progress/reproductions/<rival>/review.md`.
3. Step D2: write one row per `qualitative` row per `targets` row with the same `paper` and `dataset` to `artifacts/notes/progress/reproductions/table.csv`, and tell Session 4 in `artifacts/notes/progress/requests.md` where each number and render folder is.

## What to read

- `artifacts/notes/progress/prompts/shared-rules.md`: the rules, read on every tick.
- `artifacts/notes/progress/scoreboard.json` and `artifacts/notes/progress/targets/`: the targets, `numbers.csv`, `limits.md`, `failures.md` and `comparison-protocol.md`.
- `plans/route-two-papers.md`: the two papers and the scale ladder.
- `environment/overview.md`, `environment/hpc/nodes.md`, `environment/hpc/execution-protocol.md`, `environment/hpc/throughput.md`, `environment/storage.md` and `environment/known-failures.md`: how anything runs on this cluster.
- `runbook/running-things-on-the-cluster/launching-and-harvesting-a-run.md`: the launch and harvest recipes.
- `~/.claude/EXPERIMENT_CONVENTIONS.md` and `~/.claude/RESEARCH_PRACTICE.md`: run groups, design and verdict files, cost lines and the diagnosis procedure.
- `context/research-guidelines.md`: the mistakes this project has already made.
- `/home/molef/PhD/replications/published-methods-on-our-pairs/`: the hub for rival ports, with `scripts/launch_on_device.sh` and its own scoreboard. Each paper's code sits beside it in `/home/molef/PhD/replications/<paper>/`.
- `2-classifier-guidance/scripts/rung2/` and `2-classifier-guidance/artifacts/drips/rung2-pair-aware-discriminator/checker-loop.md`: the existing checker and how it was improved.
- `artifacts/notes/progress/void-list.md`: results that may be false. Nothing on it may justify a choice.

## What is already known

- `scripts/cluster.sh run` and `sync` push the branch to GitHub and refuse to run with uncommitted tracked files. The shared rules forbid pushing, so this session never uses them. `scripts/cluster.sh sh` and `pull` do not push and are allowed.
- Code reaches the cluster as a snapshot. Commit in this repository and in the replications hub. Both working trees hold other sessions' edits, so export each commit with `git archive <commit>`, `rsync` the two exports to `/datasets/mmolefe/poe_repair_min/loop-5/code/<repo commit>-<hub commit>/` through `mscluster84`, run from that folder, and record both commits. The cluster checkout is far behind this one, so nothing runs from it.
- `scripts/launch_on_device.sh` in the hub hard-codes the cluster checkout. Copy it into the snapshot, point its `HUB` and `CC` at the snapshot, add `export CC_ROOT=$CC` because `render_cell.py` reads that path, and make it abort on any empty GPU or disk reading. End it with `"$PY" ...; rc=$?; echo "... exit=$rc"; exit $rc`, because the original reports success when its runner crashes. Pass `--out-root` under `/datasets/mmolefe/poe_repair_min/loop-5/`, because the default folder already holds renders that would be skipped.
- The hub's Feynman-Kac port is uncommitted, and most upstream rival repositories are not on the cluster. Commit a port by named paths before its first run, export each upstream repository at its commit into the snapshot, and record that commit in `note`. If a rival's working tree holds uncommitted changes the run needs, file an `ask`.
- `scripts/cluster.sh pull` cannot fetch `/datasets` paths. Use `rsync -az mscluster84:<absolute path> <local folder>`, and leave model weights on the cluster.
- Partitions are `bigbatch` and `biggpu` only, each with a 3-day limit. `biggpu` allows one job per user and `bigbatch` allows six.
- `sinfo` cannot show GPUs, and some idle nodes have faulted GPUs, so every launch probes its device first.
- The kept push's 54 of 128 was selected on seeds 6001 to 6128, which is why it is on the void list. Those seeds count as tuning seeds from now on.
- The checker's 4,400-picture pool came from Attend-and-Excite renders and covers 275 pairs, so no checker trained on it is scored on those prompts.
- A guiding classifier needs negatives with neither subject, because one trained without them called a fridge a cat. Ask each candidate about a picture with neither subject before it guides anything.
- With the correction or push at zero, the sampler must reproduce the plain product: the largest absolute latent difference stays under 1e-5, measured in the same loop on the same card and dtype. Two separate loops can differ by about 3 grey levels with no bug. This check passes before any other result is believed.
- Bradley's CLEVR training reportedly took about 16 hours on 32 A100 cards, so released weights come first.

## When something is missing

- **A rival has no code or no released weights.** Try the authors' repository and project page. Then file `python3 scripts/progress_board.py ask "<question>" --holds D2 --default "mark it blocked" --by s5`, mark that row `blocked`, and carry on with the next rival.
- **A run would cost more than 24 GPU-hours.** File an `ask` with the cost, what it buys and a cheaper option as the default, and carry on with other work.
- **A dataset needs a licence or a login.** File an `ask`, because accepting a licence is the owner's step.
- **A published protocol leaves a setting unstated.** Use the authors' code default and record it. If the code does not settle it, file an `ask`, and count the row as `not comparable` until it is answered.
- **The cluster is unreachable.** Record it in `review.md`, and wake again in 1,800 seconds.
- **A job fails.** Read its log and classify it per the execution protocol. Retry at most twice, then mark the row `blocked` with the reason.

## Rules

- Run only the runs this brief names. Every run has a design first, with its group, its one varied axis, a `Cost:` line (cells times time per cell, in hours) and a `Buys:` line (the scoreboard row it feeds).
- Climb the ladder per rival and candidate, and save the config, metrics, image sheet and commit at each rung: smoke test, overfit 16 images if it trains, 1% of the data, controls, then the full published protocol. Show the owner nothing as a result until the controls pass.
- Rival runs are reproductions and then frozen baselines. Never tune a rival after our number is known, and never change its settings to move its number.
- For a metric scored per image or prompt, such as accuracy, bootstrap our standard error, resampling prompts when a prompt has several images. The rival counts as `reproduced` when |ours − published| ≤ 1.96 × √(SE_ours² + SE_published²). Use the standard error the paper prints or implies, and use SE_ours for SE_published when it prints none.
- For FID, Inception Score and other metrics computed over a whole set, bootstrapping is invalid because duplicated images inflate them. Render 5 independent sample sets at the published count. The rival counts as `reproduced` when its published value lies within mean ± 2.776 × sd × √1.2 of the five values.
- For a rival we retrain, sample sets from one model leave out the variation between trainings. Use at least 3 independent trainings, with the t value and √(1 + 1/n) for that number of trainings, or mark the row `not comparable`.
- Otherwise it is `not reproduced`, and both numbers are recorded. Write the test that applies into each design before the run.
- Write `table.csv` with one row per `qualitative` row per `targets` row with the same `paper` and `dataset`, and the columns `rival`, `number_id`, `dataset`, `metric`, `protocol`, `samples`, `published`, `ours`, `interval_low`, `interval_high`, `verdict`, `commit`, `run_id`, `render_folder` and `note`. `verdict` is `reproduced`, `not reproduced`, `not comparable` or `blocked`. `number_id` is the published `numbers.csv` row for that rival with the same `dataset`, `metric`, `protocol` and `task` as the row behind the target's `number_id`. When the rival is the target's own `rival`, use the target's `number_id`. For any other rival, if zero or several rows match, file an `ask` and mark the row `not comparable`, and never pick among them. Copy `dataset`, `metric`, `protocol` and `samples` byte-identical from that row, and put both snapshot commits in `commit`.
- Declare tuning seeds and held-out seeds in the design before the first run. Never choose anything on held-out seeds.
- Launch only through `sbatch`, inside the administrator's caps of one `biggpu` job and six `bigbatch` jobs. If `sbatch` is denied, wait. The shared-device `nohup` path in the execution protocol sits outside those caps, so use it only after the owner approves it through `ask`.
- Run only on `bigbatch` and `biggpu`. Use the absolute `co3` or `co3_bw` python the environment names, a disk guard on `/datasets`, and absolute paths on every SSH line.
- Never run `find /` or `pkill -f`, never remove a file on the cluster, and never start on a device with another user's active process.
- Never push. Commit only named paths with `flock artifacts/notes/progress/.git.lock sh -c 'git add -- <paths> && git commit -m "<message>" -- <paths>'` in this repository, and only named paths in the replications hub.
- Write rival code to the replications hub, outputs to `/datasets/mmolefe/poe_repair_min/loop-5/`, and notes to `artifacts/notes/progress/reproductions/`. Session 4 owns `scoreboard.json` and `targets/`, so send it numbers through `requests.md`.
- Log every run to W&B project `prime_lab/poe-repair-animals-compose`, and copy only its verdict, run id and bar into the review file.

## What to hand back

- `artifacts/notes/progress/reproductions/table.csv`, with one row per rival on each headline dataset.
- A `design.md` and `review.md` per rival, and a render folder per rival on the same prompts and seeds Session 4 compares.
- One line per result in `artifacts/notes/progress/requests.md`, addressed to `s4`.
- The proof for D2 is that every pair of a `qualitative` row and a `targets` row with the same `paper` and `dataset` has a row in `table.csv` with a verdict, both commits, and a run id or a filed question.

## What you will see

Along the way, each rival's verdict and image sheet land in `reproductions/`, and anything needing a decision appears on the
waiting picture. At the end the session reports what was asked, what was done with paths, how many rivals reproduced against how
many did not, what is still open, and at most one question.

The owner judges it from `table.csv` and the image sheets, and it is done when D2 is green.
