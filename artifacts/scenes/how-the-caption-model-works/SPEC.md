# Model 3, step by step: spec

Location:   artifacts/scenes/how-the-caption-model-works/
Ran under:  plans/10-what-poe-loses-when-it-composes/today-how-much-co-occurrence-the-product-needs.md
Built from: this repo; artifacts/notes/what-poe-loses-when-it-composes/toy_caption_dial_ldm.py (VAE, Denoiser), toy_caption_threshold.py, toy_caption_corrector.py, toy_sampler_hybrids.py
Why:        so supervisors' questions about model 3 itself (what each part takes in and gives out, its size, how a picture is made, how PoE is computed) can be answered by pointing at a screen
url:        https://claude.ai/artifact/GUr6NB6iSGAmTzgSbMCaAn

Run, from this folder:

    CUDA_VISIBLE_DEVICES= /home/molef/micromamba/envs/sdxl-local/bin/python -I build_scene_data.py   # about 4.5 minutes on CPU
    python3 -I build_page.py
    node ~/.claude/skills/codex-author/scripts/check_scene.mjs --page how-the-caption-model-works.html --out <dir> --budget 11

## Tree

Model 3 is small enough that the module tree is two levels: VAE (encoder of 3 convs, decoder of 3 convs with two upsamples)
and Denoiser (a 4 x 64 caption table, then 4 linear layers with SiLU). Shapes come from one hooked forward pass,
rung 1 of the extraction ladder (shape-traced); weights do not affect shapes.

| Level | Node | Treatment |
|---|---|---|
| L0 | the model | lane diagram: picture, VAE, latent, denoiser, decoder |
| L1 | VAE | volume funnel, face height from grid height, depth from channels |
| L1 | Denoiser | inputs-layers-output strip with real numbers as cell grids |
| L2 | caption table (nn.Embedding(4, 64)) | row lookup beside SDXL's token and text-encoder path |
| L2 | linear layers ×4 | one strip, parameter count per layer; not expanded further (identical kind) |

Untemplated node types: none needed a template beyond the generic block card.
Equation-bearing nodes: guidance (sampling), PoE combination (poe), Langevin step (corrector).
Floor: every screen's "the code" disclosure shows the real source lines, read verbatim at build time.

## Anchors

| Anchor | Screen | Shows |
|---|---|---|
| `#map` | Model 3, start to finish | the whole pipeline; each box jumps to its screen |
| `#data` | 6,000 captioned training pictures | the three kinds of training picture with caption ids |
| `#vae` | The VAE: 1,536 numbers to 128 | encoder and decoder shapes, the real 128-number latent, reconstruction |
| `#inputs` | Three inputs, one guess | the denoiser's three inputs, three layers of 1,024, its clean-latent guess |
| `#caption-is-a-label` | A caption is a row number | caption id as a row of the trained 4 x 64 table, beside SDXL's tokens and two text encoders |
| `#training-step` | One training step | noising to a random level, caption dropped to empty 1 in 10, squared error |
| `#sampling` | 200 steps from noise | one caption: the model's guess of the finished picture across 200 steps |
| `#poe` | PoE: three calls per step | the PoE combination and its trajectory from the same starting noise |
| `#relabelling` | What relabelling adds | training-set counts per caption as k changes |
| `#result` | PoE samples as relabelling grows | saved PoE samples and counts, every k and run |
| `#corrector` | A Langevin corrector finishes it | saved samples with no corrector, 5 and 20 corrector steps, 8 models |

Equation-bearing anchors, for a maths page to link back to: `#sampling` (classifier-free guidance at 3),
`#poe` (the PoE noise combination), `#corrector` (the Langevin update).

## What each screen's pictures are

- Saved samples, untouched from the experiments' npz files: `#result`, `#corrector`.
- Retrained here with the repo's code (2,000 relabelled, seed 20261022, CPU): the latent and reconstruction on `#vae`, all numbers on
  `#inputs`, the caption table on `#caption-is-a-label`, the noise-level row on `#training-step`, the trajectories on `#sampling`
  and `#poe`, and the finished picture on `#map`. This is not the saved run; CPU and GPU arithmetic differ.
- Exact training pictures from the drawing code plus pixel noise: `#data`, `#map`, `#relabelling`.
- Illustrative: the ten caption chips on `#training-step`. SDXL's encoder names on `#caption-is-a-label` are canonical
  (verify before relying); its token split and sizes are read from the stable-diffusion-xl-base-1.0 tokenizers and configs.

## Choices made without asking

- Shout face DynaPuff (a system and its sizes); the scope pins no palette, so red, blue and purple come from the pictures' own
  colours and teal is the mechanism.
- No first-run tour and no predict-before-reveal gates: the page is for presenting answers to questions, and a gate would
  stand between the presenter and the screen.
- No plumb prompts on any node.
- Default selections: 200 relabelled run 2 on `#result`, 2,000 run 2 on `#corrector` (the model the corrector moves most, 67 to 300).
