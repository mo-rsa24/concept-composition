# Leakage rivals: what can serve as a bar for Loop 12

Loop 12 asks whether a fix stops CO3's two subjects borrowing each other's parts, scored by eye on W2d's rubric over Loop 6's 12 tuning
pairs at seeds 6001 to 6008. This page names which published methods can set a bar for that question. It draws on Loop 12's
[mechanism table](../leakage/01-reading/mechanism-table.md) and on a check of each method's released code on 2026-11-24.

## No printed number can be the bar

No paper prints a borrowed-part rate for SDXL. The SDXL papers report only user preference, and the papers that score mixing directly
run on other models. By the owner's answer to board question 47, a bar therefore comes from a rerun on our own pictures, with any printed
number reported beside it.

| Paper | Model behind its numbers | What it measures | Closest printed number | Released code | Use for Loop 12 |
|---|---|---|---|---|---|
| [Bounded Attention](../leakage/01-reading/bounded-attention.md) (arXiv 2403.16990) | SD, version not printed. SDXL is qualitative only | Users pick the better image, on 5 pairs of similar animals | Picked over MultiDiffusion with probability 0.95, BoxDiff 0.72, layout guidance 0.85 (p. 9) | [Released](https://github.com/omer11a/bounded-attention), with `run_xl.py` for SDXL | **The rival to rerun.** It needs one box per subject |
| [Be Decisive](../leakage/01-reading/be-decisive.md) (arXiv 2505.21488) | SDXL | Users pick the better image, on 25 prompts of 3 or 4 similar animals | Picked over SDXL with probability 0.74 (p. 7, Table 1) | [The repository](https://github.com/omer11a/be-decisive) holds only a README | Printed context only |
| [Self-Cross Guidance](../leakage/01-reading/self-cross-guidance.md) (arXiv 2411.18936) | SD1.4, SD2.1 and SD3, not SDXL | GPT-4o's rate of pictures that are not a mixture, 0 to 100 | 86.45 on similar pairs against 77.47 for plain SD1.4 (p. 7) | [Released](https://github.com/mengtang-lab/selfcross-guidance), with no SDXL pipeline | Context. Its score is the closest to borrowing, and an SDXL port would be our work |
| [DeLeaker](../leakage/01-reading/deleaker.md) (arXiv 2510.15015) | FLUX.1-dev and SANA | A 5-way automatic label on similar animals | 46.07% of images rated a major improvement (pages in its note) | A project page only | Context. It does not run on a U-Net |
| [CONFORM](../leakage/01-reading/conform.md) and [Divide & Bind](../leakage/01-reading/divide-and-bind.md) | SD1.5 and SD | Whether both subjects appear (TIFA) | CONFORM 0.95 on Animal-Animal (p. 7) | CONFORM [released](https://github.com/gemlab-vt/CONFORM) | Context. Neither scores borrowed parts |

## What this means for Loop 12

**The bar stays CO3 alone on the same pictures.** A fix wins by Loop 6's M1 rule on faithful pairs of 96: a lead of at least 5 and more than
1.96 standard errors.

**Bounded Attention is the one rival we can rerun on SDXL.** It is both Loop 12's default fix and the rival Loop 12's Paper 2 method must
beat. It needs a box per subject, so its row must say where the boxes came from. Boxes taken from finished renders make it an upper bound,
because Loop 6's step-10 box reader failed 11 of 24 reads.

**The pair-by-pair result looks new.** Loop 12's [recorded search](../leakage/01-reading/search-for-leakage-papers.md) found no paper that
measures which pairs borrow parts and why. The closest is [DAAM](../leakage/01-reading/daam.md), which ties attention overlap to a missing
object and does not score borrowed parts.
