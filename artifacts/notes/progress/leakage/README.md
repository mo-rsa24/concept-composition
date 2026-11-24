# Loop 12: where CO3's subjects borrow each other's parts

This folder holds Loop 12's work, run by [the session 12 brief](../prompts/session-12-where-co3-leaks.md): find where in SDXL with CO3 one subject borrows the other's parts, test one fix there, and choose Paper 2's method from what that shows.

<a href="loop-12-progress.png"><img src="loop-12-progress.png" width="620" alt="Loop 12's six steps: Z1 read the leakage papers, done; Z2 what predicts borrowing, in progress; Z3 one fix; Z4 Paper 2's method; Z5 audit Loops 8 to 10; Z6 verdict."></a>

## What is here

| File | What it is |
|---|---|
| `01-reading/mechanism-table.md` | Z1: 13 papers on where leakage enters, one row each, with the recorded search |
| `02-what-predicts-borrowing/design.md` | Z2: three early readings set against W2d's borrowed counts, with the bars |
| `04-paper-2-method.md` | Z4: the candidates, the owner's constraint, filled in after Z2 and Z3 |
| `ideas/a-checker-on-the-failure-signal.md` | a candidate method: a checker that reads only a sampler's own early failure signal |
| `where-leakage-could-enter.png` | the five places leakage could enter, with the papers that blame each |
| `loop-12-and-the-other-loops.png` | how Loop 12 meets Loop 10 |
| `mechanism-checker-idea.png` | the checker idea as a loop |

## What waits on what

- Z2's run waits on the owner's answer to board question 82.
- **A zoom ladder on CO3 runs after Z2's smoke rung,** chosen by the owner on 2026-11-24: `/zoom-ladder` on CO3, from the whole sampler to one corrected step to where $d_t$ and the attention maps come from to the code lines, with each rung's equation typeset, one line per symbol, and one worked case using the smoke run's real numbers (eagle × fox, seed 6001). It runs in the background once `/datasets/mmolefe/poe_repair_min/loop-12/z2/smoke/` holds its numbers.
