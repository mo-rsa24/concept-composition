# Route: a figure ladder for the toy relabelling experiment

**What this gets you.** Pictures and pages that let you answer questions about the toy relabelling
experiment precisely: how the training set is built, what the network is, how it is trained, how it
relates to Du et al. 2023, how a sample is drawn, and what "0 of 300" and "300 of 300" count.

**Where it lands.** `4-inclusive-captions/artifacts/scenes/how-the-toy-relabelling-experiment-works/`.
The ladder record is `ladder.json` there, and every number comes from `toy_relabelling_numbers.json`,
written by `make_toy_relabelling_numbers.py` in the same folder.

**Pace.** One rung at a time; each is shown before the next is built.

## Steps

1. [ ] **The architecture with examples.** The existing caption-model diagram grown so every part
   carries a real example picture: training pictures under each caption, a picture's latent, a
   noised latent, the denoiser's guess, the product's samples (purple and separate) and the
   scorer's call.
   - Reads: `3-understanding-compositionality/artifacts/scenes/how-the-caption-model-works/`, the
     numbers JSON, `toy-caption-threshold-samples.npz` and `toy-caption-dial-corrector-samples.npz`.
   - Leaves: the diagram (SVG, PNG, spec) and its published page.
   - Prompt: `/hld-diagram grow the caption-model diagram in 3-understanding-compositionality/artifacts/scenes/how-the-caption-model-works/ into 4-inclusive-captions/artifacts/scenes/how-the-toy-relabelling-experiment-works/ with a real example picture on every part`

2. [ ] **The network views.** Six figures in the ladder's style: data to latents with relabelling,
   what the denoiser reads, training with the empty caption, one sampling step with the product,
   the corrector, and the scorer that makes "of 300".
   - Reads: `ladder.json`, the numbers JSON, the code files it names.
   - Leaves: six SVG and PNG pairs and a page that embeds them in zoom order.

3. [ ] **Du et al. side by side.** Their composition rule and Langevin corrector beside ours, line
   for line, and what was reproduced from their code.
   - Reads: `environment/reduce-reuse-recycle-code.md`, the paper on the shelf, journey 02's notes.
   - Leaves: one figure and a short page.

4. [ ] **The miniature.** One sampling step with real numbers, and one batch of 300 counted into the
   scorer's five categories.
   - Leaves: the miniature scene and its answer key.

5. [ ] **The questions page.** The questions you expect, each answered with its figure and number,
   including the five-way breakdown behind every "0 of 300".
   - Leaves: one page, linked from `4-inclusive-captions/report/`.

## Board

| Step | What must be true | Status |
|---|---|---|
| 1 | every part of the architecture carries a real example | todo |
| 2 | six network views, every number from the JSON | todo |
| 3 | our rule and corrector shown equal to Du et al.'s, or the difference named | todo |
| 4 | one step and one count of 300 worked with real numbers | todo |
| 5 | every expected question answered with a figure and a number | todo |
