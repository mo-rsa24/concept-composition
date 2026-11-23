# How the limits rows are written

`limits.md` says, for each dataset a paper could be judged on, the best published result, who holds it, where the methods
stop working and what matching the benchmark costs. The scoreboard's targets are chosen from it, so each row has to survive
a reviewer opening the cited page. This page is the instruction for whoever writes those rows.

## What to read

| File | What it gives you |
|---|---|
| `numbers.csv` | Every printed number, with its id, page, table, protocol and both reads |
| `failures.md` and `reads/figures/<Paper>.failures.csv` | How each method fails in the pictures, by figure and page |
| `reads/first/<Paper>.notes.md` | Where a paper contradicts itself, its protocol facts, and its stated hardware and training time |
| `reads/figures/on-our-data.md` | What our own runs show. Never use these as published numbers |
| `rivals.md` | The 22 rivals and their short names |

All paths are under `artifacts/notes/progress/targets/`. PDFs are listed in `reads/rivals.tsv`.

## Which rows to write

Write one row per subset, metric and protocol for each main comparison of a dataset. Skip ablations and hyperparameter
sweeps unless one of them holds the best number the method reports. Two papers on one benchmark under different protocols
get separate rows, and they are never compared.

A dataset and metric that no rival prints gets a row with `number_id` set to `no rival number`.

## The columns

Write `reads/limits/<file>.csv` with exactly these columns:

`paper,family,dataset,task,metric,number_id,outside_ids,dispute,breaks_at,break_ids,seen_in,cost,cost_source`

| Column | What goes in it |
|---|---|
| `paper` | `Paper 1` or `Paper 2` |
| `family` | The dataset's plain name, the same on all its rows, such as `CLEVR` |
| `dataset`, `task`, `metric` | Copied byte for byte from the `number_id` row |
| `number_id` | The id of the best `published` row on that table, dataset, task, metric and protocol |
| `outside_ids` | Ids, separated by `;`, of any `baseline` row in the same table and on the same base model that beats it. Say nothing about them here; the page prints them |
| `dispute` | One sentence if the paper contradicts itself about this number, with both values and pages. Otherwise empty |
| `breaks_at` | One or two plain sentences naming the condition where methods on this benchmark stop working, such as how accuracy falls as objects are added. Any number here must be the value of a row in `break_ids` |
| `break_ids` | The ids, separated by `;`, of the rows that show the break |
| `seen_in` | Where the pictures show it, such as `Bradley-Projective Fig 3 p9` |
| `cost` | What matching the benchmark takes, as the paper states it: training hardware, hours, steps, images. Empty if the paper does not state it |
| `cost_source` | The page that states it, written `<Paper> p<page>`. Every number in `cost` must be printed on that page |

## Rules

Copy, never compute. Do not round a number, average two numbers, or state a percentage the paper did not print. If a break
can only be stated by computing something, state it in words and cite the rows.

A `plot` row cannot be a `number_id`. A disputed number stays in the table with its dispute written out, and it is not
settled by choosing one side.

Say what the paper shows and nothing about what we expect to achieve.

## The check

```bash
python3 -I artifacts/notes/progress/targets/reads/build_limits.py --check --only=<file>.csv
```

Run it from the repository root. It prints `PASS` or lists each row that fails.
