# 🧭 Runbook: checking the plan tree's state

Navigation: 📋 [Index](00-INDEX.md)

Contents: [1. Run the state check](#1-run-the-state-check)

## 1. Run the state check

Navigation: 📋 [Index](00-INDEX.md) | [Next](#in-pictures-code-and-further-reading) ➡️

`ran 2026-08-24`

The four-check report-only pass named in `CLAUDE.md`'s "What the conventions look like in this
repo": stale in-flight claims, unharvested output, orphaned plan files, and jargon that won't
read cold. About 7 seconds over the whole tree. The session-start hook already runs checks 1
and 2 automatically; run this by hand for checks 3 and 4, or any time after a pass that moved
plan files. On the laptop this check cannot see Slurm, because `squeue` is missing there, so read its
`STALE` lines against `scripts/cluster.sh jobs`.

```bash
python3 scripts/plan_pulse.py
```

✅ Real output, 2026-08-24, 180 plan files:

```
plan pulse: 180 plan files under plans

STALE: task lines claiming a run is in flight with nothing in flight (1)
  1-lora-correction/plans/04-does-the-fix-reach-unseen-pairs/plans/instrument-02-three-live-curves-while-training.md:20

UNHARVESTED: output newer than the plan that owns it (0)

ORPHAN: markdown no task names (5)
  plans/standing/retrofit-poe-repair-min.md
  1-lora-correction/plans/03-does-the-correction-cause-composition/diagrams/the-product-peaks-at-the-chimera.prompt.md
  1-lora-correction/plans/02-can-we-trust-the-compose-rate/claim-under-test.md
  plans/diagrams/process-versions/01-2026-08-18.md
  plans/diagrams/process-versions/02-2026-08-19.md

DEBRIS: narration a plan should not carry (6)
  [six lines, in artifacts/plans/parked/artifact-reconciliation/inventory/, superseded-decision narration]

RUNSTATE: run state in a design plan, belongs in its review file (0)

UNJUDGED: a finished run with review questions still unanswered (3)
  [three review files, scopes 03 and 04, listed with their open question lines]

JARGON: design plans that will not read cold (4)
  [four plan files, missing a plain opener or carrying over-long sentences]
```

**Reading the report.** Each heading in the output means one thing to do:

| Heading | Usually | What to do |
|---|---|---|
| `STALE` | almost always real from a cluster session; on the laptop a live cluster run can read as stale | check `scripts/cluster.sh jobs`, then fix the plan line |
| `RUNSTATE` | almost always real | fix the plan line |
| `ORPHAN` | intentional rather than a defect (a standing reference document, a superseded-and-marked section) | check each by name before touching it |
| `DEBRIS` | intentional rather than a defect (a standing reference document, a superseded-and-marked section) | check each by name before touching it |
| `UNJUDGED` | a pointer to review files with open questions | leave it: `verify-plan` and the scope owner close these, not this check |
| `JARGON` | a flag | send the plan for a `/plain-speak` pass |

No ❌ branch: this is a read-only report and has no failure mode beyond the interpreter or the
script itself not running.

<!-- related:start (drain) -->
<a id="in-pictures-code-and-further-reading"></a>

## 🖼️ In pictures, code and further reading

Navigation: ⬅️ [1. Run the state check](#1-run-the-state-check) | 📋 [Index](00-INDEX.md)

This page has one command and one report. The block below shows the script that prints it and where the page and the script differ.

### The code behind it

The report is printed by `scripts/plan_pulse.py`. Its opening docstring is the list of what each heading checks. Excerpt from `scripts/plan_pulse.py` (lines 4 to 14):

```python
Reads. Never writes. Four checks, in the order they cost you time:

1. STALE      a task line says a run is in flight, and nothing is in flight.
2. UNHARVESTED  a run's output is newer than the plan file that owns it.
3. ORPHAN     a markdown file in a scope that no task names.
4. DEBRIS     narration a plan file is not supposed to carry.
5. RUNSTATE   a design plan carrying run state that belongs in its review file.
6. UNJUDGED   a review file whose run finished with questions still unanswered.
7. JARGON     a design plan with the mechanical tells of prose that will not read cold.
8. UNLISTED   a live plan in none of the root master plan's four lists, or in two.
9. SLOTS      a reserved figure slot whose review question got answered (fillable or broken).
```

- The numbers 1 to 9 are the numbers `--checks` takes, written with an equals sign: `python3 scripts/plan_pulse.py --checks=1,2` runs only the stale and unharvested checks. The docstring's own example, `--checks 1,2`, stops with a `ValueError` and exit code 1 (both forms run 2026-11-21).
- The script also takes the name of one folder under `plans/`, such as `python3 scripts/plan_pulse.py standing`. The docstring's example scope, `does-the-correction-cause-composition`, prints `no such scope`, since that folder sits under `1-lora-correction/plans/` with a `03-` prefix.
- `--brief` prints an orientation summary. The docstring calls it three lines, and the code prints three fixed lines and up to three more (`Launch first`, `Write meanwhile`, `Figure slots`).
- The docstring says the exit code is always 0. Every run tried here exited 0 except the `--checks 1,2` form above.

**Two sources differ.** This page and line 4 of the script say four checks, while the same docstring lists nine, and the recorded output of 2026-08-24 above shows seven headings with no `UNLISTED` or `SLOTS`. Neither is changed here.

### Go deeper

| What | Link |
|---|---|
| The script itself | [the plan pulse script](../scripts/plan_pulse.py) |
| The one order across every plan, which `UNLISTED` reads | [the master plan](../MASTER_PLAN.md) |

### Still open

- No picture of the report exists; it is plain terminal text, quoted above.
- No recorded run after 2026-08-24 is filed, so the output for `UNLISTED` and `SLOTS` has no example here.
<!-- related:end -->

<!-- built-to: PAGE_LOOK_FORMAT v1 -->
