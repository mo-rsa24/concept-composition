# Brief for one loop page (Loop 10, step X2)

Loop 10 asks whether the **score path** (the stack of $d_t(c) = \epsilon_\theta(x_t, t, c) - \epsilon_\theta(x_t, t, \varnothing)$ over sampling steps 0 to 15 for one canvas and one seed) can share a contrastive space with captions and with noisy canvases. Step X2 describes each loop that is running or testing, in one common layout, so step X3 can see which ideas each loop already covers and what the score path could add.

The two papers, named by what they do (the files number them differently, so never write "Paper 1" or "Paper 2" alone):

- **the separate-experts paper**: two diffusion models trained on disjoint data, composed with a trained checker guiding the product. Section "Paper 1" of `artifacts/notes/progress/problem-statements.md`.
- **the CO3 paper**: one text-to-image model, with a checker's push added to the product of experts or to CO3. Section "Paper 2" of the same file.

## Read first

- `artifacts/notes/progress/problem-statements.md`, `artifacts/notes/progress/what-we-know.md`, `artifacts/notes/progress/void-list.md`. Nothing on the void list may be stated as a result without saying it is on the void list.
- `artifacts/notes/progress/score-space/01-reading/reading-table.md` and the notes it links: the 12 papers this loop read, plus the 4 cited from other loops.
- `python3 scripts/progress_board.py show` (run from the repo root) for the loop's board steps, their states and their proofs.
- Your loop's own files, named in your task.

## The page

Write `artifacts/notes/progress/score-space/02-the-loops/<slug>.md` with exactly these headings, in this order:

```
# <Loop name>: <what it tries, in a few plain words>

<One or two sentences: what the loop is for and which paper it serves. Its owner code and board steps, with each step's state today.>

## The idea
## The principle
## The theory, with its equation
## The objective, with its bar
## Where it stands
## The closest papers from the reading
## What it suggests for the separate-experts paper
## What it suggests for the CO3 paper
```

- **The idea**: what the loop does, in plain words, with one real example (a real pair, a real digit canvas, a real prompt).
- **The principle**: why it should work, the assumption it rests on.
- **The theory, with its equation**: the loop's own equation as its design or review file writes it, typeset in `$$...$$`, then one line per symbol saying what moves when it moves.
- **The objective, with its bar**: the measured quantity, its scale and direction, and the bar exactly as the loop's design recorded it before the run. Copy it, never restate it from memory.
- **Where it stands**: each board step with its state, the review file that holds the verdict, and the verdict and numbers exactly as recorded there, each with its unit and meaning. Say which steps have no result yet. If the review and the board disagree, report both and do not resolve it.
- **The closest papers from the reading**: two to four papers from the reading table, each with one sentence on why it is close, linked to its note in `../01-reading/<slug>.md` and to the page it relies on.
- **What it suggests for the separate-experts paper / the CO3 paper**: one paragraph each. What this loop's results so far suggest that paper try, keep or drop, and what a score-path representation could add to it. Mark every inference (inferred). If it suggests nothing, say so in one sentence.

## Rules

- Every claim cites the review file, design file or paper page it comes from, as a relative link from the page's folder. A path cited as plain text is written from the repo root.
- Never rewrite a result. A verdict, number or bar stays as recorded. Where two files disagree on a result, write both with their sources under the heading where it arises, and add one line at the end of your report.
- Other sessions edit these files while you read. Re-read a file before quoting a number from it.
- Writing rules in `~/.claude/CLAUDE.md`: no dash used as a clause separator, colons in titles, no "X, not Y" emphasis, no fronted participles, no "crucially/importantly/notably", every number with its unit and meaning, plain sentences. No private labels: write "the count checker", not "V2's head", unless the label is the board step id.
- Write only your one page. Change no other file, run no experiment, use no GPU, do not commit.

## Report back

At most five lines: the page path, the loop's current state in one line, the one thing it suggests most strongly for each paper, and any disagreement between sources you found.
