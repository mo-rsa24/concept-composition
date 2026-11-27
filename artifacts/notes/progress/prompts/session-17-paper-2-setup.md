# Session 17: write Paper 2's setup code

Launch from the repository root:

```text
claude --bg --permission-mode auto --model claude-opus-5-5 --effort high --name "🔬 PhD Loop 17: Paper 2 setup code" "/loop Follow artifacts/notes/progress/prompts/session-17-paper-2-setup.md to its end."
```

## Background

Paper 2 trains an identity dial, a small rank-4 adapter on SDXL's attention layers, that stops CO3
borrowing body parts between two look-alike animal species. The dial trains against a species judge
that step 0 builds and tests. The plan folder `plans/paper-2-dials/` runs in stages: setup code first,
then the species judge and the identity dial's first result (about 13 GPU-hours), then the full
ladder only after that result says yes. None of `2-classifier-guidance/scripts/paper2/` exists yet.
On 2026-11-26 the owner said yes to writing the setup code now, on the laptop, with nothing run on the
cluster. Loop 16 writes Paper 1's setup code at the same time. Loop 18 is the supervisor that
reviews both loops' code and answers their questions.

## Objectives

1. Work through tasks 1 to 4 of `plans/paper-2-dials/plans/01-setup.md` in order, one task per tick.
2. Make each task's code complete, and run the check the task names on the laptop. Every file passes
   `python3 -m py_compile`. The laptop has no torch, so a check that needs torch is written as a test
   and marked to run on the cluster later.
3. After each task, send Loop 18 one message naming the files written and pasting the check's output,
   then start the next task.
4. At the start of every tick, read Loop 18's replies first, and do any fix it asked for before new
   work.
5. Tick a task's checkbox in `01-setup.md` only after Loop 18 has accepted it.
6. For tasks 6 to 8 (fetch the four models, move the 192 pictures, check CO3 loads), write the job
   scripts and stop: they run only after the owner's yes in instruction 5.
7. Close by writing the hand-off file.

## What to read

- `plans/paper-2-dials/README.md`: the claim, the two contributions, the stages and the words.
- `plans/paper-2-dials/plans/01-setup.md`: the tasks, read whole.
- `plans/paper-2-dials/plans/02-species-judge.md`: the six kinds, the seven rules in order, the five
  worked examples and the four judges.
- `plans/paper-2-dials/plans/03-identity-dial-first-result.md`: what the dial's trainer must serve,
  including step 2's choice between training on the judge's gradient and training on kept draws.
- `plans/paper-2-dials/plans/04-full-ladder.md`, section "How a dial is trained": the closeness term.
- `environment/overview.md`, `environment/storage.md` and `environment/hpc/execution-protocol.md`:
  the environments, the two filesystems and the checks every job script opens with.
- The code the tasks reuse: `3-understanding-compositionality/scripts/species_judge.py` (the crop
  and paint-out at lines 72 to 82), `poe_repair/experiments/compose_scorer_validation/detection_scorer.py`
  (the body counter) and `poe_repair/experiments/reward_finetune/train.py` (the one-step trainer).

## What is already known

- The judges are CLIP on crops, an ImageNet classifier on crops, Detect-and-Compare (DINOv2 features
  against real photographs of each species) and a vision-language model on crops, beside the old
  judge (`02-species-judge.md`, "The judges").
- The species judge's half-drawn reads scored 0.602 to 0.724 against a bar of 0.75, so the dial may
  have to train on kept draws instead of the judge's gradient. The trainer takes the reward as an
  argument so either route uses the same loop (`03-identity-dial-first-result.md`, step 2).
- The earlier reward fine-tune ran with its fidelity term silently off until commit `a72f748`, and
  needed a 48 GB card. Check the closeness term is on in every run the trainer starts.
- "Off" for a dial is an untouched U-Net in another process, because a disabled adapter still acts
  (`04-full-ladder.md`, rule 14).
- The Blackwell nodes mscluster110 and 112 need the `co3_bw` environment. Under `co3` a CUDA
  operation runs on the CPU with no error (`environment/hpc/nodes.md`).

## When something is missing

- **The owner's three questions** (Claude sessions as reviewers, the route for the 192 pictures, a
  separate environment for Qwen3-VL) belong to instruction 5. Ask each once with
  `python3 scripts/progress_board.py ask "<question>" --by s17`, then carry on with tasks 1 to 4.
- **A setting the plans leave open,** such as an argument name or a file layout. Choose one, tell
  Loop 18 in the task's message, and record it in the hand-off. Loop 18 may overrule it.
- **Two plans disagree.** Follow the more specific plan (`02-species-judge.md` for the judges,
  `03-identity-dial-first-result.md` for the trainer), tell Loop 18, and record it in the hand-off.

## Rules

- Write only in `2-classifier-guidance/scripts/paper2/`, in tests named `tests/test_paper2_*.py`, in
  the task checkboxes of `01-setup.md`, and in the hand-off file.
- Never touch Paper 1's files or `2-classifier-guidance/scripts/paper1/`.
- Run nothing on the cluster or RunPod. Do not use `sbatch`, `ssh` or `scripts/cluster.sh`.
- Commit nothing, push nothing and delete nothing. Loop 18 lists the commit for the owner at the end.
- Every job script opens with the checks in step 4 of `environment/hpc/execution-protocol.md`: a
  disk guard on the filesystem it writes to, a python path check that picks `co3_bw` on mscluster110
  to 112 and `co3` elsewhere, and an `nvidia-smi` guard on GPU jobs.
- Large outputs go under `/datasets/mmolefe/poe_repair_min/paper2/`, named by one path constant.
- Messages use plain English and no dash as a clause separator.

## What to hand back

- The code files by path, each with one line on what it does.
- `artifacts/notes/progress/setup/paper-2-handoff.md`. It holds each task with its files, its check
  output and Loop 18's accept. It records every choice made where the plans were silent. It ends
  with the exact commands for tasks 6 to 8 and for the species judge, ready for the owner's yes.
- `plans/paper-2-dials/plans/01-setup.md` with tasks 1 to 4 ticked.

## What you will see

Along the way, after each task, a short note in chat names the files, shows the check's output, and
gives Loop 18's answer when it has come.

At the end, a closing report in chat:
1. what was asked;
2. what was done, with each file's path;
3. the outcome, as tasks accepted out of 4, and the job scripts written for tasks 6 to 8;
4. what is still open, with each question sent;
5. at most one question for the owner.

Loop 18 judges each task from its code and check. The loop is done when tasks 1 to 4 carry Loop 18's
accept, the scripts for tasks 6 to 8 exist, and the hand-off file is written; then it stops its own
loop.
