# Walk: adding an MNIST stage to the inclusive-captions route

Reopens [the inclusive-captions route](../../../plans/route-inclusive-captions-pilot.md) with
`/drip-skill-route --from`, to add one step: the inclusive-captions question asked again on MNIST
digit canvases with a diffusion model trained from scratch, where the share of two-digit canvases
trained under each single label is a dial the project sets exactly.

Nothing trains or renders in this walk. It writes the step, its prompt and the route's rows.

## Rounds

| Round | Decision | State | Who chose | Why | Options not taken |
|---|---|---|---|---|---|
| 1 | where the MNIST stage sits on the route | settled: new step 4 beside step 3, waiting for nothing in it; steps 4 to 10 renumbered 5 to 11; its verdict meets step 3's in step 10's pressure-test | user, `settle` | MNIST answers the half SDXL cannot (labels control everything "7" learnt); SDXL answers whether the effect survives inherited support, a different question, so neither gates the other | MNIST gating step 3's scoring; MNIST replacing step 3's scoring; a separate route |
| 2 | the canvases, the two-digit share, and which pairs the two-digit canvases hold | settled: 32 × 64, 28-pixel boxes, 10% two-digit; half of those 7 and 8, half over the other 39 seen pairs; 0 & 3, 1 & 9, 3 & 4, 3 & 6, 4 & 7 held out; held-out pairs read against their own dial-0 rate | user, `settle` | keeps 7 × 8 at 8.8% of "7" draws at dial 0.3 (about half the pilot's 0.169) and adds a test of pairs never seen together, at no extra training | 7 and 8 only (16.7%, no unseen-pair test worth running); all 40 evenly (0.5%, 7 × 8 barely dosed); a wider canvas for a new-arrangement test (to Later) |

## Round 2, asked along the way

The user asked whether out-of-distribution composition on unseen pairs had been considered, and to
see the training data. Drawn by `4-inclusive-captions/scripts/mnist_training_sheets.py` from real
MNIST digits at `/home/molef/PhD/replications/original-superdiff/data/MNIST/raw`, no model involved:

- [the canvas and the pairs](figures/the-canvas-and-the-pairs.png): one-digit canvases, two-digit
  canvases of seen pairs, 7 and 8, and the five held-out pairs (0 & 3, 1 & 9, 3 & 4, 3 & 6, 4 & 7,
  the classifier-guidance route's `random.Random(0)` rule).
- [what the "7" label sees](figures/what-the-7-label-sees.png): 30 random draws labelled "7" at
  dial 0, 0.3 and 1, for two-digit canvases of 7 and 8 only against all 40 seen pairs.

Share of "7" draws whose canvas holds an 8, counted over 400,000 simulated draws (a canvas has two
digits 10% of the time, blank rate 10% everywhere):

| Two-digit canvases hold | dial 0.1 | dial 0.3 | dial 0.6 | dial 1 |
|---|---|---|---|---|
| 7 and 8 only | 0.083 | 0.167 | 0.225 | 0.263 |
| 7 and 8 half the time, the other 39 seen pairs the rest | 0.043 | 0.088 | 0.124 | 0.150 |
| all 40 seen pairs evenly | 0.002 | 0.005 | 0.007 | 0.009 |

The pilot's run B, for comparison: 0.169 of "a photo of a cat" draws held a dog.

## Round 5, asked along the way: the measurements as pictures

The user asked for the round in simpler words, a sheet of the measurement table, and figures in the
zoom ladder's recorded style (cheat-sheet devices; blue inputs, green hidden units, red outputs,
orange for the push, from the classifier-guidance ladder's `ladder.json`). Only the style was
borrowed; no ladder was started.

- [what each model draws and how tiles are called](figures/what-each-model-draws-and-how-tiles-are-called.png),
  from `4-inclusive-captions/scripts/mnist_measurement_sheet.py`: one row per measurement, one
  pasted example per call, and the real starting noise of seeds 17 and 18.
- [the label-conditional U-Net](figures/the-label-conditional-unet.png) and
  [one sampling step of the product](figures/one-sampling-step-of-the-product.png), from
  `4-inclusive-captions/scripts/mnist_network_figures.py`. Channel widths are drawn as c, 2c, 4c
  because round 4 left them to the smoke run.

## Round 3, asked along the way: the classifier-guidance route's MNIST walk

The user asked for a read of session `d40b9299-0d32-4c1f-a983-428221d67138` (the classifier-guidance
route's MNIST walk, read through its round 4) and its
[canvas sheet](../../../../2-classifier-guidance/artifacts/drips/route-classifier-guided-poe-from-toy-to-animals/figures/mnist-canvas-shares.png).
Read only; nothing there was edited.

**What it settled.** The same 32 × 64 canvas and the same five held-out pairs. Three models at a
two-digit share of 0, 0.01 and 0.1, two-digit canvases spread evenly over the 40 seen pairs. Each
two-digit canvas is trained once under each of its digits, with no joint label; the label is
dropped to null on 10% of canvases. Sampling is DDIM at 50 steps. Plain product first, before any
checker, called on the ladder one digit, fused, two digits with a wrong identity, two clear right
digits. The judge is a digit classifier on separate blobs, used only once it agrees with 58 of 64
eye calls.

**What it changes here.**

- Its labelling is this route's dial at full strength (single labels only, no joint), on a
  different pair spread. The two stages are two slices of one grid: how many canvases hold two
  digits (its axis) against which labels those canvases get (this one). No model is shared yet,
  because no setting coincides on both.
- Its support check (how often "7" alone draws a held-out partner beside it, which it needs above
  zero) is this route's leak read with the sign reversed. Under the settled round 2 design, "7"
  draws hold a second digit on 5.1%, 10.3%, 14.5% and 17.6% of training draws at dial 0.1, 0.3,
  0.6 and 1, so a model that learns its data exactly draws "7" alone at 94.9%, 89.7%, 85.5% and
  82.4%. A flat 90% purity bar fails every dial above 0.1 by construction. Rounds 6 and 8 must
  measure purity against what the training data shows.
- Its tile ladder adds "two digits, wrong identity". Round 5 adopts it so both stages call tiles
  the same way.
| 3 | the dial | settled: the toy's rule, blank 10% everywhere; a two-digit canvas's other 90% split joint weight 1, each single label weight equal to the dial, at 0, 0.1, 0.3, 0.6, 1 | user, `settle` | the joint label is trained at every setting, and the toy's counts at the same settings become the prediction | the brief's split (joint never trained at dial 1); the toy's rule plus a singles-only setting matching the classifier route (3 more trainings) |
| 4 | the model and its training seeds | settled: one label-conditional pixel U-Net per training, no VAE, 56-entry label table, blank label as the null branch; 3 training seeds × 5 settings; 30,000 steps at batch 128 for all, cut equally on overrun | user, `settle` | no VAE means a fusion comes only from the denoiser or the sampler; three seeds because the toy's counts swung 17 to 252 of 300 between trainings | the toy's VAE plus latent denoiser; one training per setting |
| 4, asked along the way | one conditional model or two separate unconditional ones | one conditional model; two separate models written to the route's Later list | user, `settle` | the label question lives inside one model, the blank is the null branch of the same weights, and the toy's two separate networks drew 0 of 300 | two separate models as a fourth column of step 4 |
| 5 | the measurements and the seeds they share | settled, simplified: the toy's sampler (200 steps, guidance 3, 20 corrector steps per level at 0.05 of the noise variance); 8 rows on seeds 0 to 255 shared by every model: "7", "8", "4" alone, "7 and 8", 7 × 8 plain and corrected, 4 × 7 plain and corrected; 2,048 pictures per model; the other four held-out pairs only if 4 × 7 moves | user, `simplify then settle` | the toy's counts stay a prediction; one held-out pair answers the unseen-pair question at a third of the 23-row cost | DDIM at 50 steps; 64 seeds; all ten singles and all five held-out pairs (23 rows, 5,888 pictures per model) |
| 6 | the leak read | settled: each single label judged against its own training line ("7" 100% to 82.4% lone across the dial); a two-panel figure, product rates on top, single-label purity below beside the dashed training lines; a leak is a gap below the line wider than two standard errors | user, `settle` | a model that learns its data exactly fails a flat 90% at every dial above 0.1 | the brief's flat 90% bar |
| 7 | the judge | settled: one judge shared with the classifier-guidance route, imported at a pinned commit, never edited from here; two checks before counting (95% of each call on 500 pasted canvases per call; 58 of 64 eye-call agreement per dial sheet); stop and ask if uncommitted or failing | user, `settle` | both routes call tiles the same way; the prototype misreads a fused 7-and-8 as one digit, so it is checked here first | this stage's own judge; its own judge written for the other route to import |
| 8 | the bar | settled: settings count only where "7" and "8" stay within 5 points of their training lines in all three trainings; supported at 128 of 256 corrected 7 × 8 in all three with dial 0 at 12 or fewer; null under 39 on average everywhere; stop if plain dial 0 exceeds 12; corrector and unseen-pair readings reported beside | user, `settle` | the brief's numbers kept, with purity judged against training lines and the noise of 256 seeds | the brief's flat 90% purity; the average of three trainings for support |
| 9 | time on both clocks, the schedule, FashionMNIST | settled: laptop RTX 4070; three 1.5 h sittings; you ~4.5 h, AI ~5 h (`estimate` until the smoke run); code in `poe_repair/experiments/inclusive_captions/mnist/`; FashionMNIST in Later, run only if MNIST is supported | user, `settle` | `/datasets` is down, MNIST is small, and the toy ran on this card | the cluster on a home-folder node |
| 10 | the step's prompt, the route rows, the checks | settled: step 4's prompt in eight sections, with Sees and Filing lines; five rows added to "What we know so far"; Achieve row 4 and lanes row 4 added | user, `settle` in a loop | the prompt format's eight checks pass; the route template's step shape is complete | a separate route file for MNIST |

## The checks at the close

- **verify-plan on the route:** the file carries `template: exempt` (a skill-route route built to
  route-file-template v2), so the plan-template structure checks are skipped by design. The checks
  that apply to a route were run by hand. Every link and anchor in step 4 resolves (0 missing). No
  runnable `/skill` line sits outside a fence anywhere in the route. No `pending` line is left.
- **Broken links found outside step 4, not touched by this walk:** four
  `/home-mscluster/mmolefe/goal-setting/...` links in step 3's learning-path tables (cluster paths
  that do not resolve on the laptop) and `../artifacts/results/can-we-trust-the-compose-score/README.md`
  in the compose-scorer fact row.
- **The prompt format's eight checks on step 4's prompt:** all pass; the only lines over 100
  columns are two paths, which the format says never to break.
- **The route template's step shape:** step 4 carries the time line, Objective, Outcome, Serves,
  Sees, the prompt, `added:` and `dropped:`, a Gate and Leaves.

## Questions still open

| Question | Who answers | Fallback while it is open |
|---|---|---|
| where the stage runs | answered in round 9: the laptop's RTX 4070 | none needed |
| where its code lives | answered in round 9: `poe_repair/experiments/inclusive_captions/mnist/`, data in `data/mnist/` ignored by git | none needed |
| whether the canvas maker and the judge are shared with the classifier-guidance route's MNIST row | answered in round 7: the judge is shared, imported at a pinned commit; the canvas matches that route's | none needed |
| how many draws per measurement | answered in round 5: 256 seeds per row | none needed |
