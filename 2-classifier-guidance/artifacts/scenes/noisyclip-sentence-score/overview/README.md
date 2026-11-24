# Overviews

One annotated-explainer overview for each model and training recipe loop 9 uses. Each is drawn by `annotated_explainer.py` in the zoom-ladder skill and saved as PNG and SVG. No number on them comes from a run of loop 9: noise and bar heights are illustrative, and sizes come from public configs or design files.

| File | What it shows |
|---|---|
| `ov0-the-sentence-score-push` | NoisyCLIP's sentence-score push. The gradient is taken at $x_t$ and the push ends at the noise prediction. Loop 7's sample at `artifacts/notes/progress/abstract-checkers/01-reading/style-annotated-explainer-sentence-score.png` is the same render. |
| `ov1-latent-clip-push` | Latent CLIP's push, with the same push lane |
| `ov2-the-linear-probe-on-frozen-vectors` | W2b's linear probe; it reads and does not push, and its seen and unseen bars are hollow |
| `ov3-our-own-encoder-on-digits` | W4's own encoder on a real digit canvas; it reads and does not push |

**The input pictures.** `ov0` to `ov2` show `data/pilot/seed_42/a_cat__x__a_dog/monolithic.png`, checked at full size against the subject guard. `ov3` shows `w4-input-canvas.png`.

**How the digit canvas was built.** `make_w4_canvas.py` calls `make_canvas` from step V1's `build_env.py` on the MNIST training files at `/home/molef/PhD/replications/original-superdiff/data/MNIST/raw/`. It draws one `apart` canvas for the seen pair 3 and 4 at 32 × 64, with the random stream V1 gives that pair in its training split. The result holds a 4 on the left and a 3 on the right, MNIST training images 8281 and 41843, recorded in `w4-input-canvas.json`. Nothing in V1's dataset on the cluster was read. The overview adds illustrative noise on top.

**To redraw them**, from the repository root:

```bash
export STYLE_SAMPLE_IMAGE=data/pilot/seed_42/a_cat__x__a_dog/monolithic.png
export STYLE_SAMPLE_CANVAS=2-classifier-guidance/artifacts/scenes/noisyclip-sentence-score/overview/w4-input-canvas.png
python3 ~/.claude/skills/zoom-ladder/scripts/style_samples.py <out> --family sentence-score
```

The families for the other three are `latent-clip-push`, `linear-probe` and `digit-encoder`.
