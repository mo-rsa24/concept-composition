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

**What waits.** The overview carries a marked slot for a worked case with a real latent, to be filled from the Loop 12
Z2 smoke run at `/datasets/mmolefe/poe_repair_min/loop-12/z2/smoke/`. The miniature and the illustrated rung are
`skipped` in `ladder.json`.
