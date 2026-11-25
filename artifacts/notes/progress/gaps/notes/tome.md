# ToMe: Token Merging for Training-Free Semantic Binding

arXiv 2411.07132v1 (11 November 2024), Hu, Li, van de Weijer, Gao, Khan, Yang, Cheng, Wang and Wang,
Nankai University with Barcelona, UCAS, MBZUAI and Linköping. The PDF prints no venue. Read from the full PDF on the shelf, 27 pages with appendix:
`~/goal-setting/learning/papers/deep-learning/diffusion-models/compositional-diffusion-models/region-and-binding-failures/token-merging-for-training-free-semantic-binding.pdf`.

Every claim below is marked **read** (read from the paper, page given) or **inferred** (this note's own
reasoning).

## What it does

ToMe fixes "semantic binding": an attribute or a sub-object attached to the wrong object, such as a hat on
the dog meant for the cat (read, p. 1, p. 2). It edits the text embedding, not the sampler or the weights.

1. **Token merging.** The embeddings of an object word and its attribute words are added into one
   composite token, so the object and its attributes share one cross-attention map (read, p. 6, Figure 4).
   The addition rests on a measured additivity of CLIP text embeddings (read, p. 5, Figure 3).
2. **End token substitution.** The end tokens, which carry a mix of every object's meaning, are replaced
   by end tokens computed from a prompt without the attributes (read, p. 2, p. 6).
3. **Two losses in the first 20% of steps.** An entropy loss sharpens each composite token's attention map
   and a semantic binding loss keeps the composite token close to the full phrase's meaning; the composite
   token is updated by gradient steps, the model's weights untouched (read, p. 2, p. 6, p. 7).

## What it needs

No training and no LLM or layout (read, p. 4). It needs a parser (SpaCy) to find each object and its
attributes in the prompt (read, p. 7), and gradients through the U-Net's cross-attention for the losses
(read, p. 6; CO3 lists it as gradient-based and architecture-dependent, CO3 p. 7).

## Which model and settings

SDXL only (read, p. 4, p. 7). The limitations section says the method is "optimized for inference based
on SDXL" and relies on the CLIP text encoder (read, p. 16). No SD3, SD3.5 or FLUX run appears (read,
full-text search).

## What it measured

| Benchmark and metric | ToMe on SDXL | Best other | Count behind it | Where |
|---|---|---|---|---|
| T2I-CompBench colour, BLIP-VQA, 0 to 1 | 0.7656 | 0.7774 CoMat (trained) | 300 validation prompts; images per prompt not stated | Table 1, p. 8; protocol p. 7 (read) |
| T2I-CompBench texture, BLIP-VQA | 0.6894 | 0.6686 ELLA (trained) | 300 prompts | Table 1, p. 8 (read) |
| T2I-CompBench shape, BLIP-VQA | 0.6051 | 0.5634 ELLA (trained) | 300 prompts | Table 1, p. 8 (read) |
| Own object-binding set, GPT-4o score | 0.9549 | 0.6481 ELLA on SD1.5; SDXL 0.4907 | 50 prompts of the form "a [object A] with a [item A] and a [object B] with a [item B]" | Table 1, p. 8; set p. 7 (read) |

SDXL's own colour score is printed as 0.6369 here (Table 1, p. 8, read), against 0.5879 in
T2I-CompBench++ Table XIII (p. 12, read) and 0.6007 in this project's own run on 200 images
([the standard benchmarks page](../../../../../1-lora-correction/report/against-other-methods/how-does-the-adapter-score-on-the-standard-benchmarks.md)).
Three sources, three numbers for one model on one metric; reported, not resolved.

The ablation (Table 2, p. 8, read) puts most of the colour gain in the entropy loss with token merging
(0.6369 to 0.7525); token merging alone gives 0.6577.

## Does it already claim Paper 2's contribution

**No.** ToMe binds attributes and sub-objects to objects; it does not try to make two subjects appear
separately, it has no learned reader of the picture, and it is tested only as a fix to plain SDXL sampling
(read, pp. 1 to 2, p. 7). It is not combined with any other sampler.

**Where a reviewer will place it (inferred).** ToMe and Paper 2's push both act in the first 20% of steps
with the model frozen. A reviewer will ask why a learned checker is needed when a text-side edit already
reaches 0.7656 on colour without training anything. Paper 2's answer has to be that ToMe's target
(attribute on the right object) differs from Paper 2's (two separate subjects rather than one hybrid), and
the four-way test has to show it on pairs where ToMe does not help.

## What it leaves open for Paper 2 (inferred)

- ToMe's rows on CO3's animal-pair benchmark are weak: BLIP-VQA 0.6257 against SDXL's 0.6950, ImageReward
  0.3895 against 0.7820 (CO3 Table 1, p. 7, read, ToMe run by CO3's authors). Merging tokens does not fix a
  missing or fused animal.
- ToMe depends on the CLIP text encoder's additivity (read, p. 5, p. 16); whether it carries to SD3's or
  FLUX's T5 encoder is untested in its own record.
