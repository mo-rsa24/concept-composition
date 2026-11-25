# Shared rules for the looping sessions

Every looping session reads this file on every tick. It holds what the sessions have in common.

## The route

The [two-paper route](../../../../plans/route-two-papers.md) says what the work is for. The board at
`artifacts/notes/progress/board.json` says where it stands. Change the board only through
`scripts/progress_board.py`.

## One tick

1. Run `python3 scripts/progress_board.py ready --owner <your owner code>` from the repository root.
2. If a step of yours is already `doing`, continue it. If none is, set the first ready step to `doing`.
3. Do one piece of work that can be checked.
4. Check the step's proof, printed by `ready` or `show`. Set it to `done` only when the proof is true.
5. Run `python3 scripts/progress_board.py blocks --todo`. If it lists a block, start the
   `calendar-cadence` agent in the background with the path to `scripts/progress_board.py`.
6. Say in chat what landed, then print `python3 scripts/progress_board.py line` and
   `python3 -I ~/.claude/skills/session-status/scripts/session_status.py --repo "$PWD" --line`. The second
   line shows every session's step, steps left, ETA and flags. A flag naming this session is this session's
   to fix. A flag naming another session goes to `artifacts/notes/progress/requests.md`.
7. Schedule the next wake. Use 1,200 seconds while work is running. Use 1,800 seconds while waiting on
   another session. Stop the loop when all your steps are done.

## Who owns what

| Session | Owner code | Writes to |
|---|---|---|
| 1. Records | `s1` | git commits, `artifacts/notes/session-closes/`, `DRIFT_LOG.md`, `RENAMES.md`, Claude session transcripts |
| 2. Artifacts | `s2` | each project's `report/`, the drain ledger, `artifacts/notes/progress/void-list.md`, `prune-list.md` |
| 3. Pillars and repo | `s3` | `context/`, `environment/`, `runbook/`, `plans/`, `artifacts/notes/progress/what-we-know.md`, the new paper repository |
| 4. Rivals and targets | `s4` | `artifacts/notes/progress/targets/`, `artifacts/notes/progress/scoreboard.json` |
| 5. Reproduce the rivals | `s5` | `artifacts/notes/progress/reproductions/`, the replications hub, `/datasets/mmolefe/poe_repair_min/loop-5/` |
| 6. Find the best method | `s6` | `artifacts/notes/progress/method-search/`, `reproductions/checker/`, `reproductions/inclusive-finetune-and-push/`, `2-classifier-guidance/scripts/checker_choice/`, `2-classifier-guidance/scripts/method_search/`, `/datasets/mmolefe/poe_repair_min/loop-6/` |
| 7. Read the abstract-checker papers | `s7` | `artifacts/notes/progress/abstract-checkers/01-reading/`, `02-score-view.md`, `03-design.md` |
| 8. Test abstract checkers | `s8` | `artifacts/notes/progress/abstract-checkers/04-environment/` and later stage folders, `2-classifier-guidance/scripts/abstract_checkers/`, `/datasets/mmolefe/poe_repair_min/loop-8/` |
| 9. Turn concepts into scores | `s9` | `artifacts/notes/progress/noise-text-critic/`, `2-classifier-guidance/scripts/noise_text_critic/`, `/datasets/mmolefe/poe_repair_min/loop-9/` |
| 10. The score path as a third space | `s10` | `artifacts/notes/progress/score-space/`, `2-classifier-guidance/scripts/score_space/`, `/datasets/mmolefe/poe_repair_min/loop-10/` |
| 11. Drain the loops into the pillars | `s11` | each project's `report/`, `context/methods/`, `artifacts/notes/progress/drain/`, the drain ledger, `held-pictures.md` |
| 12. Where CO3 leaks | `s12` | `artifacts/notes/progress/leakage/`, `2-classifier-guidance/scripts/leakage/`, `/datasets/mmolefe/poe_repair_min/loop-12/` |
| 13. Manager | `s13` | `artifacts/notes/progress/manager/`, `artifacts/notes/progress/reviewer/`, and nudges to other loops; no other loop's files |
| 14. Gaps and reviewer questions | `s14` | `artifacts/notes/progress/gaps/` |
| 15. Directions and dials | `s15` | `artifacts/notes/progress/directions/` |

Sessions 2 and 3 have finished their steps and are not running, so Session 11 holds `report/` and the drain ledger in Session 2's place, and writes `context/methods/` beside Session 3's other `context/` pages.

A session that needs a change in another session's area writes one line to
`artifacts/notes/progress/requests.md`, naming the owner. Each session reads that file on every tick.

## Targets

`artifacts/notes/progress/scoreboard.json` holds each paper's 4 to 6 targets, set by Session 4 from what rival papers report. Work toward those targets. Treat a target as final only after the owner has said yes to it, and propose no bar of your own. Step C3 builds from `artifacts/notes/progress/targets/numbers.csv`.

## Before any new design

Read [the consensus](../reviewer/consensus.md) and the [what-we-tried ledger](../reviewer/what-we-tried/README.md)
entries for the area first. A design names the ledger entries it builds on, follows the consensus's
rules for every new design (section 4), and says why it is not an idea the consensus lists as
settled. Add one entry to the matching ledger file when the attempt finishes, in the ledger's shape.

## Rules that never bend

- **Never delete anything.** Move files only after the user approves, and keep a restore manifest. Most
  artifact files are outside git, so a deletion has no undo.
- **Never prune by file type.** A file is a candidate only when no finding, card, plan, paper or script
  names it.
- **Commit only named paths, never `git add -A`.** Hold the lock:
  `flock artifacts/notes/progress/.git.lock git commit -m "<message>" -- <paths>`.
- **Commit messages carry no attribution line.**
- **Never push, and never publish anything.** Pushing is the user's decision.
- **Run no experiments in Sessions 1 to 4, Session 7, Session 14 or Session 15.** They read the cluster and do not train or render. Sessions 5, 6, 8 and 9 run only what their briefs name, and share the job caps.
- **On the cluster, use only `bigbatch` and `biggpu`.** Never run `find /`, never run `pkill -f`, and
  never remove a file there. If `mscluster84` is unreachable, record it and carry on with local work.
- **Launch GPU work only through `sbatch`, inside the administrator's caps** of one `biggpu` job and six
  `bigbatch` jobs per user. If `sbatch` is denied, wait. Starting work with `nohup` over SSH, outside your own
  Slurm job, needs the owner's yes through `progress_board.py ask`.
- **Keep these paths where they are:** `artifacts/_shared/cross_pair_pool_configs/`,
  `artifacts/_quarantine/`, `artifacts/caches/`, `artifacts/rung2-survive-noise/`,
  `artifacts/rung3-group-wise/`, `artifacts/rung4-scale/`. `poe_repair/paths.py` hard-codes them.
- **Keep `writing/overleaf-iclr/` untouched.** It is the Overleaf bridge, with its own git history.
- **Re-read a shared file before reporting from it.** Other sessions edit this repository.
- **Never rewrite a result.** A verdict, a number or a bar stays as recorded. A disagreement between two
  sources is reported to the user and never resolved alone.
- **Follow the writing rules in `~/.claude/CLAUDE.md`** in every file and message.

## The manager

Session 13 reads every loop on a timer and keeps `artifacts/notes/progress/manager/status.md`. It may send a loop a **method review request** after a run of bad results. A loop that receives one pauses new runs, writes a short review in its own folder before its next run (what the failures share, which unread papers bear on them, whether another loop's results already explain them, whether the step is still worth its cost, and what it would change in its design), and replies to the manager with the review's path. Any change of direction that follows is the owner's call.

The manager also decides open board questions for the owner when every option is reversible, inside the loop's budget, and changes no recorded bar or verdict (its prompt lists the rest, which stay the owner's). A loop treats such a decision, marked "Decided by Loop 13 for the owner", as the owner's answer, unless the owner answers again. Write every question so the owner, as the head of the project, can follow it without the loop's internal step codes: say what the work is and why it matters.

## Show a run while it runs

Any run longer than about 30 minutes reports as it goes, so the owner can stop one that is not
improving. At about each quarter of the run, send one picture grid of what the run is making now and
the one number it is about, beside the same number at the last report. Before the first report, write
a futility rule into the run's design: the condition under which it stops early, such as the number
not rising between two reports after the half-way point. A futility rule only stops a failing run; it
never declares success, which stays with the bar written before the run.

## When to stop and ask the user

Stop that step and ask one question, then carry on with any other ready step, when:

- anything would be deleted or moved outside the approved lists;
- two sources disagree on what a result was;
- a file looks like a secret, a client's data or a patient's data;
- a step's proof cannot be met.

File each of these with `python3 scripts/progress_board.py ask "<question>" --holds <step> --default "<default>" --by <owner code>`, so it appears on the waiting picture. A step this session adds with `add` names `--serves` and `--objective`.

## Telling the user

The user may be away from the computer. Load `PushNotification` with `ToolSearch` and send one short
notification in each of these moments:

- when this session stops to ask the user anything, naming the session and what it needs;
- when one of this session's steps turns green and the next step needs the user;
- when a failure stops this session's work;
- when all this session's steps are done.

Send one notification per stop, and do not repeat it on later ticks while the same question is waiting.
Begin it with the session's name, such as `Loop 1:`. Put no client name, patient detail or secret in it.
