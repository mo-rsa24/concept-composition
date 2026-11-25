# What we tried: every attempt, what it found, and what it rules out

One entry per finished attempt across every loop and project, so that no loop repeats a failure and
every new design starts from where the last ones stopped. The consensus that reads across all of them
is in [the consensus page](../consensus.md). A loop reads both before writing any new design.

## The groups

| File | What it covers |
|---|---|
| [method-search.md](method-search.md) | Paper 1's method search: CLEVR experts, checkers, vote and placement on digits, identity readers |
| [leakage.md](leakage.md) | Paper 2 on CO3: where it leaks, own-prediction, corrections, the dial, the gradient-trained classifier, the box tests |
| [reproductions.md](reproductions.md) | reproducing rival methods and their numbers |
| [noise-text-critic-and-score-space.md](noise-text-critic-and-score-space.md) | turning concepts into scores, and the score path as a third space |
| [abstract-checkers.md](abstract-checkers.md) | abstract checkers on digits |
| [projects.md](projects.md) | the five project folders' own reports: the ICLR LoRA correction, classifier guidance, the compositionality toy, and the nulls recorded there |

## The shape of one entry

```
### <plain name of the attempt>

- **Asked:** one plain sentence: the question this attempt put to the data.
- **Paper:** 1, 2, or both.
- **Tried:** what was done, in one or two sentences, with the data, the model and the number of pictures.
- **Bar, written first:** the pass condition as recorded before the run, or "none recorded".
- **Result:** the numbers, each with its unit and what it means.
- **Verdict:** support, null, inconclusive, not run, or closed.
- **Why it failed:** the explanation, marked verified (measured in a run) or inferred (argued, not measured).
- **Rules out:** what a future design must not repeat, said as an instruction.
- **Leaves open:** what this attempt did not settle.
- **Evidence:** links to the review, the picture and the W&B run id, with the board step code in brackets.
```

Entries are in the order the attempts ran. A support result gets an entry too, with "Why it failed"
replaced by **Why it worked**.
