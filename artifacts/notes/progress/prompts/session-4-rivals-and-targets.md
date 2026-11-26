# Session 4: read the rivals and set every target

This runs as a background session, in the same permission mode as the other three loops. Start it from
`/home/molef/PhD/concept-composition` with:

```text
claude --bg --permission-mode auto --model claude-opus-5-5 --effort high --name "🎯 PhD Loop 4: Read Rivals And Set Targets" "/loop Follow artifacts/notes/progress/prompts/session-4-rivals-and-targets.md to its end. Read artifacts/notes/progress/prompts/shared-rules.md on every tick."
```

## Background

The two-paper route needs targets that come from what published methods actually achieve. The owner removed the
earlier bars on 2026-11-22 because they were guesses and did not rest on the limits of the methods. This session reads
every rival paper ahead of the owner, including each paper's tables, figures and appendices. It then sets one target
per dataset and metric on the scoreboard, and those targets direct the other three loops. Your owner code on the board
is `s4`, and your steps are K1 to K6. This loop keeps running through K6 until both papers' full tables are done.

## Objectives

1. Step K1: list every rival for each paper in `artifacts/notes/progress/targets/rivals.md`, with its arXiv id, PDF path, datasets, the paper it rivals and its read state.
2. Step K2: read every reported number from the papers into `artifacts/notes/progress/targets/numbers.csv`, with the page and table each came from.
3. Step K3: look at every qualitative figure at full size and write how each method succeeds and fails into `artifacts/notes/progress/targets/failures.md`.
4. Step K4: write `artifacts/notes/progress/targets/limits.md`, which names per dataset and metric the best published result, who holds it, and the condition where the methods break.
5. Step K5: choose 4 to 6 headline targets for each paper, write them and one side-by-side row per rival per headline dataset into `artifacts/notes/progress/scoreboard.json`, write how the comparison is judged into `artifacts/notes/progress/targets/comparison-protocol.md`, and show the owner the scoreboard for a yes.
6. Step K6: after that yes, keep the scoreboard current until steps E1 and E2 are done. On each tick, read `artifacts/notes/progress/field-table.md` for new rivals, `artifacts/notes/progress/benchmark-spec.md`, and the board notes on D2, E1 and E2 for reproduced numbers, results and render folders. Update the row and tell the affected session in `artifacts/notes/progress/requests.md`.
7. Also in step K6, make the blind side-by-side calls as soon as both methods' renders exist on the same prompts and seeds, and record them in `artifacts/notes/progress/targets/side-by-side.csv`.

## What to read

- `artifacts/notes/progress/prompts/shared-rules.md`: the rules, read on every tick.
- `plans/route-two-papers.md`: what each of the two papers claims.
- `artifacts/notes/progress/rival-datasets-first-pass.md`: 11 papers from web summaries. It is a starting list, and none of its numbers count until read from the paper.
- `/home/molef/goal-setting/learning/papers/Papers.md`: the paper shelf, with 71 PDFs and the pages that carry each idea.
- `/home/molef/PhD/replications/published-methods-on-our-pairs/`: published methods already run on our pairs, which shows their failures on our data.
- `1-lora-correction/report/against-other-methods/`: our earlier benchmark runs, including CO3 and plain PoE. These are our runs and are recorded as reproduced numbers, never as published ones.
- `artifacts/notes/progress/void-list.md`: results that may be false. Nothing on it may set a target.
- `~/.claude/PROGRESS_MAP_FORMAT.md`, under "The scoreboard": the file's fields and how targets are computed.

## What is already known

- Paper 1 composes separately trained models at inference. Its rivals include Liu, Du, SuperDiff, Feynman-Kac Correctors, both Bradley papers, CoInD, MultiDiffusion, PiCo, FreeFuse, Diffusion Domain Expansion (2605.23275), Compositional Sculpting (2309.16115), Catastrophic Compositional Generation (2606.23920) and Test-Time Compositional Generalization (2605.07078).
- Paper 2 restores plurality in one conditional model. Its rivals include CO3 (2509.25940), ToMe (2411.07132), Attend-and-Excite (2301.13826), FlipConcept (2502.15203), SuperDiff and the re-evaluation of CFG alternatives (2608.16786).
- The shelf has no entry for CoInD, PiCo, FreeFuse, FlipConcept, Diffusion Domain Expansion or 2608.16786.
- The owner's target rule: if the best rival has FID 25, our target is 24. If it has a compose rate of 80%, our target is 81%. A score on a 0 to 1 scale moves by 0.01.
- The owner settled on 4 to 6 headline targets per paper and left the choice to this session. Prefer a dataset and metric that at least two rivals report, that a reviewer of that paper would expect, and that we can afford to run. Cover more than one dataset per paper, and keep every other number in `numbers.csv`.
- The owner also wants to know how often ours looks better than each rival, how it is better, and on which pairs, on CLEVR and every other dataset.
- The checker's 4,400-picture pool came from Attend-and-Excite renders and covers 275 pairs (`2-classifier-guidance/artifacts/drips/rung2-pair-aware-discriminator/checker-loop.md`, line 205), so a checker trained on it cannot be scored fairly on those prompts. Keep Attend-and-Excite off Paper 2's headlines until a question filed with `ask` settles which prompts are held out, and put that question's id in the row's `held_out_question`.
- Our T2I-CompBench runs use 2 seeds and 200 images, and CO3's sampler settings are on the void list, so those runs do not count as reproduced numbers unless their protocol matches the published one.
- A picture is called on this ladder: one thing, a fused hybrid, two bodies or heads, two clear things. Clear pairs are called from full-resolution tiles, because downscaled sheets overcount them.
- Bradley's CLEVR training reportedly took about 16 hours on 32 A100 cards, so the cost of matching a benchmark belongs beside its row in `limits.md`.

## When something is missing

- **A paper is not on the shelf.** Download its PDF from arXiv to `artifacts/notes/progress/targets/pdfs/`, and list it in `rivals.md` for the shelf to take in later.
- **A PDF will not load.** Try its arXiv HTML page. If that fails too, mark the paper `could not read` and carry on.
- **A number appears only in a plot.** Record it with `kind` `plot` and the figure number. It cannot set a target until a table or the authors' data confirms it.
- **A rival cannot be run for a side-by-side comparison.** File the reason with `ask`, and once it is answered put that question's id in the row's `skip_question`.
- **Two papers report one benchmark under different protocols.** Keep separate rows, and never compare them.
- **A choice would change which benchmark a paper must beat.** File it with `python3 scripts/progress_board.py ask`, give your recommendation as the default, and carry on with the other papers.
- **Codex is unavailable.** Use a separate Claude agent for the second read, and record that in `read_by`.

## Rules

- Run no experiments. This session reads papers and files that already exist.
- Read each number from the PDF page itself. A search summary or an abstract never counts.
- Have every number that sets a target read a second time by GPT-6 Astra through `codex` or by a separate agent, with neither seeing the other's answer. Open the page to settle any disagreement.
- Never invent, round or estimate a number. Copy it as printed, with its interval and sample count if the paper gives them.
- Give every row in `numbers.csv` an `id`, with these columns: `id`, `kind`, `paper`, `arxiv`, `dataset`, `task`, `metric`, `better`, `value`, `interval`, `samples`, `model`, `protocol`, `source_page`, `source_table`, `read_by`, `second_read`. `kind` is `published`, `reproduced`, `plot` or `baseline`, where a baseline is a method that is no rival, such as plain SDXL or plain PoE. Rows on one benchmark use byte-identical `dataset`, `metric` and `protocol` text, because the check compares them exactly. `paper` is the short name of the method measured, even when another paper printed the table, and `source_table` names the paper and table that printed it. `value` is a bare number.
- A headline needs a printed rival number on one protocol. When no rival prints one, write `no rival number` for that dataset and metric in `limits.md`, and leave it off the headlines.
- Give each scoreboard target the `number_id` of the published row it rests on, `unit` (exactly `%` for a percentage), and explicit `better` (`higher` or `lower`), `step` (1 for FID, Inception Score, percentages and whole-number counts, 0.01 for any other score printed to two or more decimals, and an `ask` if unclear) and `range` (the metric's lowest and highest possible values, with `null` for no limit).
- A target that falls outside its metric's range, such as 100% becoming 101%, goes to `ask` with your recommendation and cannot be a headline.
- Write one side-by-side row per rival per headline dataset, with whole-number `pairs`, `better`, `tied` and `worse`, all zero until both methods have run. `pairs` counts calls, one per prompt and seed. At K5, fill `failure` from `failures.md` with how that rival fails and on which pairs, in at most 140 characters. Once calls exist, fill `why` with how ours is better and on which pairs, in at most 68 characters. Each comparison call goes in `artifacts/notes/progress/targets/side-by-side.csv`, with the columns `paper`, `dataset`, `rival`, `pair`, `seed`, `call` (`better`, `tied` or `worse`) and `reason`. In that file and the scoreboard, `rival` matches the `paper` name in `numbers.csv`.
- Leave `target` out of `scoreboard.json`. The drawing script computes it from the best rival number and the step, so it cannot be set by hand.
- Set `reproduced` only when the run matches the published dataset, split, metric, sample count and settings, and give it its own `numbers.csv` row, named by `reproduced_id`. When the two numbers differ, the script takes the harder one.
- Count a qualitative comparison only when both methods ran on the same prompts and seeds, judged at full size, with the method names hidden until the call is written.
- Write only to `artifacts/notes/progress/targets/`, `artifacts/notes/progress/scoreboard.json` and `artifacts/notes/progress/requests.md`, and change the board only through its script.
- After each paper, update `papers_read` in `scoreboard.json` and run `python3 -I ~/.claude/skills/session-status/scripts/session_status.py --repo "$PWD" --draw`.
- When every paper is read or marked `could not read`, so `papers_read.read` equals `total`, and the check prints PASS, show the owner the scoreboard and file `python3 scripts/progress_board.py ask "Accept these targets?" --holds K5 --default "targets stay provisional" --by s4`. Keep K5 in progress until the owner answers, and record the answer with `progress_board.py answer`. On a yes, set K5 done. On a no, revise the targets and ask again. No other loop treats a target as final before that.
- On each tick, read `questions` in `artifacts/notes/progress/board.json` for the owner's answers.
- A change to a target the owner has approved stays provisional until the owner says yes to it through a new `ask`.
- If `/loop` reaches its time limit, tell the owner to restart it with the same line. A restart resumes from the board at K6.
- While that question is open, wake every 1,800 seconds and do not stop, even though `stop_when` in the board's brief would allow it. Stop only when K6 is done.
- When K2 lands, tell Session 3 in `artifacts/notes/progress/requests.md` to build step C3 from `artifacts/notes/progress/targets/numbers.csv`.

## What to hand back

- `artifacts/notes/progress/targets/rivals.md`, `numbers.csv`, `failures.md`, `limits.md`, `comparison-protocol.md` and `side-by-side.csv`.
- `artifacts/notes/progress/scoreboard.json`, drawn to `artifacts/notes/progress/scoreboard.png`.
- One line per affected session in `artifacts/notes/progress/requests.md`, naming the target it now works toward.
- The proof for K5 is `~/.claude/venvs/figures/bin/python -I ~/.claude/skills/session-status/scripts/progress_views.py --check artifacts/notes/progress`, which prints `PASS` when every paper has been read, each paper has 4 to 6 valid targets on the hardest rival number, each traces to a page read twice, and each rival reporting a headline dataset and metric has a side-by-side row that names how it fails and matches its calls.
- The proof for K6 is the same command with `--check-final`, which also needs our result and side-by-side calls on every headline row.

## What you will see

Along the way, the scoreboard's count of papers read rises after each paper, and any decision that changes a benchmark
appears on the waiting picture. At K5 the session shows the scoreboard and asks for a yes. At the end it reports what
was asked, what was done with paths, the targets per paper, what is still open, and at most one question.

The owner judges it from the scoreboard and `limits.md`, and it is done when K1 to K6 are green, after the owner has said yes to the targets.
