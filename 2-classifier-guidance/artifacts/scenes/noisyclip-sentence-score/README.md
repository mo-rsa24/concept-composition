# NoisyCLIP as a sentence-score push: a zoom ladder

This ladder explains one candidate for Paper 2's abstract checkers: a noise-aware CLIP reads a half-drawn SDXL latent, scores it against a right and a wrong sentence, and the gradient of the difference pushes the sampler. It climbs from the whole sampler down to a network of 8 latent numbers.

**Open [`noisyclip-ladder.html`](noisyclip-ladder.html) first.** It is one self-contained page holding every rung. The same content is in the markdown files below.

<a href="0.5-network-view/nc2-the-whole-read-and-push-path.png"><img src="0.5-network-view/nc2-the-whole-read-and-push-path.png" width="720" alt="A latent through a fixed map and a tuned image encoder, scored against a right and a wrong sentence, with the gradient taken at the latent and subtracted from the base noise prediction."></a>

## What is established and what is proposed

**NoisyCLIP's results are the paper's own.** It ranks seeds during sampling and reports its numbers on its own benchmarks. It never feeds a gradient to a sampler.

**The push is our proposal.** No result on this project's pictures exists yet. Steps W2 and W3 measure whether it reads or steers any of the five properties.

**The miniature is a toy.** Its weights are set by hand, so it shows what each part computes and predicts nothing about SDXL.

## The rungs

| Level | What it shows | Where |
|---|---|---|
| 0 | the real system | [the classifier guidance system picture](https://claude.ai/artifact/FBbwapTjp1dbEmzpLspRBT) and [CLIP's module tree](https://claude.ai/artifact/4mePVaMV9NuxENRkJphieQ), both linked, with [the three jobs of one model](0.5-network-view/nc1-one-model-three-uses.png) drawn here |
| 0.5 | the network view: what NoisyCLIP adds to CLIP | [the ten figures](0.5-network-view/README.md) |
| 1 | the idea, illustrated | [the six illustrations](1-illustrations/diagram-prompts.md) |
| 2 | the miniature: 8 latent numbers, every term on its part | [the worked miniature](2-miniature/README.md) |
| 3 | the real code, block by block | [the code walk](3-code/walk-noisyclip.md) at commit `016b913` |
| beside | each equation down to fundamentals | [the rebuild companion](rebuild-from-fundamentals.md) |
| close | the method read through the small network | [the closing page](read-through-a-small-network.md) |

## What is in this folder

| File | What it is |
|---|---|
| `noisyclip-ladder.html` | the whole ladder as one local page, with its pictures and maths built in |
| `ladder.json` | the ladder's record: each rung's state and files, the colour key and the code commit |
| `NOTATION.md` | the symbols every rung uses, beside the ones the paper prints |
| `style-reference.md`, `style-sample-clip-contrastive.png` | the approved drawing style |
| `numbers/make_numbers.py`, `numbers/numbers.json` | every full-scale number, with its source |
| `numbers/check_pages.py` | checks that every decimal on a page appears in a key, a numbers file or the paper |
| `figure_kit.py` | the drawing helpers, which refuse to write a figure where a line crosses a box or label |
| `build_page.cjs` | builds `noisyclip-ladder.html` from the markdown pages |
| `route-zoom-ladder-noisyclip.md` | the steps that rebuild this ladder |
| `reference/` | three third-party layout references, kept out of every page and never committed |

<details>
<summary>How it was made</summary>

The sizes come from the public configs of `openai/clip-vit-large-patch14` and SDXL's scheduler, and the parameter counts come from building that config on PyTorch's meta device, with no weights. The released code was read at commit `016b913` and never run. The checkpoint itself was not downloaded here, and the facts about its weights come from [the checkpoint comparison](../../../../artifacts/notes/progress/noise-text-critic/01-reading/tools/checkpoint-compare.txt). The illustrations were drawn by Codex's image tool from prompts built on the miniature's numbers, so every tile in them is a teaching drawing and none is an SDXL sample.

</details>
