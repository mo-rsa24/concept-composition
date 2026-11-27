# Session 16: write Paper 1's setup code

Launch from the repository root:

```text
claude --bg --permission-mode auto --model claude-opus-5-5 --effort high --name "🧪 PhD Loop 16: Paper 1 setup code" "/loop Follow artifacts/notes/progress/prompts/session-16-paper-1-setup.md to its end."
```

## Background

Paper 1 composes two CLEVR image generators trained apart, one that draws red objects and one that
draws blue objects, into scenes where their objects touch. It does this by adding a small
discriminator's push to a region sampler. The plan folder `plans/paper-1-clevr-touching/` runs in
stages: setup code first, then a first result of about 3 GPU-hours, then the full plan only after
that result says yes. None of `2-classifier-guidance/scripts/paper1/` exists yet. On 2026-11-26 the
owner said yes to writing the setup code now, on the laptop, with nothing run on the cluster.
Loop 17 writes Paper 2's setup code at the same time. Loop 18 is the supervisor that reviews both
loops' code and answers their questions.

## Objectives

1. Work through the Tasks of `plans/paper-1-clevr-touching/plans/01-setup.md` in order, tasks 0 to
   7, one task per tick.
2. Make each task's code complete, and run the check the task names on the laptop. Every file
   passes `python3 -m py_compile`. The laptop has no torch, so a check that needs torch is written
   as a test and marked to run on the cluster later.
3. After each task, send Loop 18 one message naming the files written and pasting the check's
   output, then start the next task.
4. At the start of every tick, read Loop 18's replies first, and do any fix it asked for before new
   work.
5. Tick a task's checkbox in `01-setup.md` only after Loop 18 has accepted it.
6. Write task 10's model-check job script, and never submit it.
7. Close by writing the hand-off file.

## What to read

- `plans/paper-1-clevr-touching/README.md`: the claim, the stages and the words the plans use.
- `plans/paper-1-clevr-touching/plans/01-setup.md`: the tasks, read whole.
- `plans/paper-1-clevr-touching/plans/02-first-result.md`: every setting the code must serve.
- `plans/paper-1-clevr-touching/plans/03-full-plan.md`, sections "The models" and "The region
  sampler": the head's layers and the sampler's formula.
- `environment/overview.md`, `environment/clevr-and-edm2.md` and `environment/hpc/execution-protocol.md`:
  the environments, the renderer, the model files and the checks every job script opens with.
- The code the tasks port: `2-classifier-guidance/scripts/method_search/m10_pod/m10_compose.py`,
  `2-classifier-guidance/scripts/method_search/clevr_render.py` and
  `2-classifier-guidance/scripts/leakage/z11_gradient_checker.py`.

## What is already known

- The Blackwell nodes mscluster110 and 112 need the `co3_bw` environment. Under `co3` a CUDA
  operation runs on the CPU with no error (`environment/hpc/nodes.md`).
- The empty-scene model has no recorded cluster path. It was last at
  `models/empty/network-snapshot-0002000-0.100.pkl` on a RunPod machine (`environment/clevr-and-edm2.md`).
- The push is D' = D + w σ² ∇ logit, applied on sampler steps 10 to 22 of a 32-step Heun sampler
  (`03-full-plan.md`).
- The head joins the 8 by 8 decoder features, upsampled to 16 by 16, with the 16 by 16 features
  along the channels. Two 3 by 3 convolution blocks of 256 channels follow, each with a scale and
  shift from log σ and the count. Then an average over the grid and one linear layer give the
  logit (`03-full-plan.md`, "The models").
- Feynman-Kac reweighting is not run in this paper.
- The first result renders 512 joint scenes, trains one head at k = 64, and samples five rows of 768
  pictures each (`02-first-result.md`).

## When something is missing

- **The empty-scene model's path.** Write the code to take it as an argument. Ask the owner once with
  `python3 scripts/progress_board.py ask "Where is the empty-scene model on the cluster?" --by s16`,
  then carry on.
- **A setting the plans leave open,** such as an argument name or a file layout. Choose one, tell
  Loop 18 in the task's message, and record it in the hand-off. Loop 18 may overrule it.
- **Two plans disagree.** Follow `02-first-result.md`, tell Loop 18, and record it in the hand-off.

## Rules

- Write only in `2-classifier-guidance/scripts/paper1/`, in the joint-mode edit to
  `2-classifier-guidance/scripts/method_search/clevr_render.py`, in tests named
  `tests/test_paper1_*.py`, in the task checkboxes of `01-setup.md`, and in the hand-off file.
- Never touch Paper 2's files or `2-classifier-guidance/scripts/paper2/`.
- Run nothing on the cluster or RunPod. Do not use `sbatch`, `ssh` or `scripts/cluster.sh`.
- Commit nothing, push nothing and delete nothing. Loop 18 lists the commit for the owner at the end.
- Every job script opens with the checks in step 4 of `environment/hpc/execution-protocol.md`: a
  disk guard on the filesystem it writes to, a python path check that picks `co3_bw` on mscluster110
  to 112 and `co3` elsewhere, and an `nvidia-smi` guard.
- Large outputs go under `/datasets/mmolefe/poe_repair_min/paper1/`, named by one path constant.
  Nothing writes large files to `/home-mscluster`.
- Messages use plain English and no dash as a clause separator.

## What to hand back

- The code files by path, each with one line on what it does.
- `artifacts/notes/progress/setup/paper-1-handoff.md`. It holds each task with its files, its check
  output and Loop 18's accept. It records every choice made where the plans were silent. It ends
  with the exact commands that will run the first result once the owner says yes.
- `plans/paper-1-clevr-touching/plans/01-setup.md` with tasks 0 to 7 ticked.

## What you will see

Along the way, after each task, a short note in chat names the files, shows the check's output, and
gives Loop 18's answer when it has come.

At the end, a closing report in chat:
1. what was asked;
2. what was done, with each file's path;
3. the outcome, as tasks accepted out of 8;
4. what is still open, with each question sent;
5. at most one question for the owner.

Loop 18 judges each task from its code and check. The loop is done when tasks 0 to 7 carry Loop 18's
accept and the hand-off file is written; then it stops its own loop.
