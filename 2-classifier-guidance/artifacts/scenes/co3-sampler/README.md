# CO3 sampler: the zoom ladder

The CO3 sampler (arXiv 2509.25940) read at three levels: the whole SDXL run, one figure per algorithm, and the authors'
code line by line. Start at [the reading page](index.md). The ladder's record is `ladder.json`; its symbols are in
[the notation file](NOTATION.md); the steps still to build are in [the route file](route-zoom-ladder-co3.md).

| File | What it is |
|---|---|
| `index.md` | the reading page: overview first, each deeper figure linked from the text about its part |
| `co3-overview.png`, `.svg` | level 0: one run of 50 steps annotated stage by stage, with code boxes and a small forward diagram |
| `co3-resampler.png`, `.svg` | level 0.5: Algorithm 2 as a small network, equations beneath |
| `co3-corrector.png`, `.svg` | level 0.5: Algorithm 3 as a small network, printed and coded rules boxed beneath |
| `co3-weights.png`, `.svg` | level 0.5: equations 14 and 15 as a small network beside equation 15's curve |
| `co3-schedule.png`, `.svg` | level 0.5: the co3-hybrid schedule over 50 steps, with the branch's rules boxed beneath |
| `code-walk.md` | level 3: the authors' code in running order, each block under its equation, and the disagreement table |
| `exact-plane/co3-steps-cheetah-cougar.png`, `.svg` | the worked case: per CO3 step, the decoded guesses under the empty, joint and each animal's prompt, with the step's two distances and weights, and the finished pictures with CO3 on and off |
| `exact-plane/co3-steps-eagle-fox.png`, `.svg` | the same grid for eagle × fox, smaller |
| `exact-plane/co3-exact-plane.png`, `.svg` | the worked case: the flat sheet through the joint's and both animals' Tweedie means at each step's last pass, true lengths, both pairs, with each point's distance off the sheet |
| `exact-plane/co3-path-difference.png`, `.svg` | the worked case: the gap between the CO3 and plain latents at each of the 50 steps, over the plain latent's length, both pairs |
| `exact-plane/worked-pass.md` | the worked case: one real corrector pass (cheetah × cougar, t = 921, first pass) with every equation filled in |
| `exact-plane/exact_plane_numbers.py`, `.json` | every number the worked case prints |
| `exact-plane/draw_steps.py`, `draw_exact_plane.py`, `draw_path_difference.py`, `write_worked_pass.py` | one script per figure, and the worked pass's generator |
| `co3_numbers.py`, `co3_numbers.json` | every number any figure or page prints |
| `co3_figkit.py` | the shared colours and the check that refuses a figure where a line crosses a node, box or label |
| `draw_*.py`, `gen_codewalk.py` | one script per figure, and the code walk's generator |
| `ladder.json`, `NOTATION.md`, `route-zoom-ladder-co3.md` | the ladder's record, its symbols, and the route for the rungs not built |
| `reference/` | the owner's three style references |
| `source/` | the paper's algorithm screenshots, and the copies the numbers script reads |

## How each item was made

**The numbers.** `python3 co3_numbers.py` reads `source/sdxl-scheduler-config.json` (copied from the Hugging Face cache
on mscluster84, SDXL snapshot `462165984030d82259a11f4367a4eed129e94a7b`) and recomputes the 50-step DDIM schedule in
numpy the way diffusers' `DDIMScheduler` builds it, because diffusers is not installed on this laptop. It parses the
release's settings from `source/co3_common.py.copy` (copied from the Loop 6 code snapshot on the cluster), reads the
hybrid branch's constants off `Co3.py` and `attn_utils.py` with each line asserted, and computes equation 15's curve
for two concepts. It then checks its schedule against six lines the real Loop 6 run printed, kept in
`source/run-log-excerpt-l6-m1-p2-93468.txt`, and stops if they differ. They match.

**The figures.** Each `draw_*.py` draws with matplotlib through `co3_figkit.py`, which registers every box, node,
label and line, and refuses to save a figure where a line crosses a node, box or label or where two labels overlap.
Every figure passed. Code shown inside a figure is read from the checkout and asserted line by line. The styles follow
the owner's references: the overview takes `reference/owner-annotated-progress-with-code-boxes.png`, the four part
figures take `reference/owner-network-weights-and-equations.png` (symbols on the parts, numbers on the edges,
equations beneath), and the corrector and schedule figures add the boxed rules of
`reference/owner-network-with-rule-boxes.png`. The equations are typeset by matplotlib's TeX engine, since this
laptop has no system LaTeX.

**The eagle and fox picture.** `source/co3-eagle-fox-seed-6001.png` and its sidecar were copied read-only with `scp`
from `/datasets/mmolefe/poe_repair_min/loop-6/m1/paper-2/full/renders/co3/an_eagle__x__a_fox/`. The overview shows
it as the run's output.

**The code walk.** `python3 gen_codewalk.py` writes `code-walk.md` from the checkout at
`/home/molef/PhD/replications/co3`, commit `35ea972313276014716dca3d821023728e8bc917`. Each block is copied byte
for byte with its line numbers, its anchor lines are asserted, and every number is read from `co3_numbers.json`. It
follows the code-walk shape of `hld-diagram --paper` as a static page. The paper walk's seeded capture, exact toy rig,
term figures, interactive page and stepper were not built, because this round ran nothing.

**The worked case's records.** Slurm job 94450 (code snapshot `44b51af4`) ran
`2-classifier-guidance/scripts/leakage/co3_inside.py` on cheetah × cougar and eagle × fox at seed 6001, M1's CO3
settings unchanged, per [the run's design note](../../../../artifacts/notes/progress/leakage/co3-inside/design.md). It
recorded each of the 40 passes' incoming latent and four noise predictions, the weights the code returned, the latent
at all 50 steps with CO3 on and off, and VAE-decoded clean-picture guesses at each step's first and last pass. Its
control recomputes every pass from the records and matched the next latent to a relative difference of at most 0.00077,
against a limit of 0.01. The outputs are on mscluster under `/datasets/mmolefe/poe_repair_min/loop-12/co3-inside/`; the
scripts here read a laptop copy, by default at `/home/molef/.claude/jobs/0071c318/tmp/co3-inside/`, set with
`--data` on `exact_plane_numbers.py`. No picture was copied into this folder; the figures embed the ones they show.

**The worked case's numbers.** `python3 exact-plane/exact_plane_numbers.py` reads each pair's `meta.json`,
`passes.npz` and `plain-joint-path.npy` and computes in float64 from the float16 records. For each step's last pass it
forms the three guided Tweedie means, an orthonormal basis of the sheet through them (Gram-Schmidt, origin at the joint's
mean, first axis toward the first animal's), each point's coordinates on it and its distance from it. For corrector steps
it adds the weighted sum $2\hat x^{C} + w_1\hat x^{c_1} + w_2\hat x^{c_2}$ and the same after the code's rescale; for
resampler steps the added noise term, which is a difference of points of the sheet. It also computes the path gap at all
50 steps and the worked pass. It stops if the worked pass misses the next recorded latent by 1% or more, if the weights
do not follow from the distances, or if a weighted sum leaves the sheet by more than 1e-4 of its length. The largest
remainder is 1.7e-5 of its length, from the logged float32 weights summing to $-1$ only to 7 digits; the three means sit
at most 2e-15 off their own sheet.

**The worked case's figures and page.** `draw_steps.py <pair> [--small]`, `draw_exact_plane.py` and
`draw_path_difference.py` draw through `co3_figkit.py` with the ladder's colour key and pass its crossing check; the
exact-plane script places each point's label at the first of twelve offsets that touches nothing. `write_worked_pass.py`
writes `worked-pass.md` from the JSON's `trace` key. The overview's bottom-right box shows one panel of
`co3-exact-plane.png` (eagle × fox, step 4), cropped in `draw_overview.py`.

**What waits.** The miniature and the illustrated rung are `skipped` in `ladder.json`.
