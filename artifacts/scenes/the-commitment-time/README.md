# The Commitment Time

The formula T_s = ½ ln λ_max, the time scale on which a sample commits to a class, taken apart one press at a time on a 2D toy.
Two equal Gaussian classes sit at +m and −m with variance 1 and |m|² = 19, so λ_max = 20 and T_s ≈ 1.498. The reader turns a line
to find the widest spread, watches noising shrink it, solves the crossing one log move at a time, and copies a backward trajectory
early and late to see when its class stops changing.

| File | What it is |
|---|---|
| `index.html` | the page source, 8 frames and 28 presses, published as the private artifact below |
| `ts_numbers.py` | computes every number on the page; needs only NumPy |

**Which reading it serves**

Reading 11, on Biroli, Achilli and Albrychiewicz. The page is the first picture to open in the
[reading plan's section on when a sample commits](../../notes/progress/targets/reading-plan.md#sitting-11-when-a-sample-commits).

**Published at**

[The Commitment Time](https://claude.ai/artifact/KMf73EpZmTtaFBp268sbMh). It is private to its owner until shared from the page's
Share menu. Each frame opens at `#spread`, `#shrink`, `#crossing`, `#solve`, `#clones`, `#slider`, `#stops` or `#letters`.

**How it was made**

The two classes and every dot are an illustrative toy. A NumPy script computed the page's numbers. Its agreement curve simulates
40,000 cloned pairs per copy time, with each parent started from the exact noised distribution at t = 4 and run backward on the
exact score in steps of 0.002. The curve is drawn beside a numerical integration of Biroli's Equation 6, and the two differ by at
most 0.0048 on the 0 to 1 agreement scale. Codex reviewed all 28 presses against quoted passages from the two papers. The script is `ts_numbers.py` in this folder. `python3 ts_numbers.py` needs only NumPy and writes `ts_numbers.json` and `ts_page_data.json` beside it.

**What it does not show**

The 2D toy shows the mechanism, and the sharp transition needs high dimension. Agreement here rises gradually, reaching 76.7% at
T_s and 95% only by t ≈ 0.87. The 16-copy picture is one seeded parent trajectory, so its counts are one draw. The same-mean frame
states only Achilli's ¼ log N scaling and gives no verdict on which data the formula covers, and its dots (variances 0.7 and 1.3)
are illustrative.
