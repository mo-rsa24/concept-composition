# Follow the Flow: On Information Flow Across Textual Tokens in Text-to-Image Models

arXiv 2504.01137 (v3, 16 Apr 2026), read from the full PDF (19 pages, appendix included). No venue is printed on the PDF.

## What it does

The paper asks what each token at the text encoder's output carries, by generating an image from a chosen subset of token outputs while every other position is replaced by the encoder's output for an all-padding prompt, then asking a vision-language model what the image shows (p. 2, p. 3). It uses this to test whether one lexical item's tokens carry another item's meaning ("cross-item flow"), which it finds in about 11% of item pairs (p. 6). It then fixes a misread item by swapping its contextual token outputs for the encoding of that item with its own modifiers alone (p. 7).

## Where it says leakage enters

Text encoder. The paper's "semantic leakage" is a word read in the wrong sense because of another word in the prompt ("zebra" drawn as a zebra crossing beside "bus station", "bats" drawn as baseball bats beside "baseball stadium") (p. 6). That is a narrower thing than one subject borrowing another subject's parts.

The evidence is a measurement. For each item, an image is generated from the item's contextual token outputs alone and from the item encoded by itself; flow from item y into item x is counted when a VLM finds y in the contextual image and not in the context-free one (p. 16, Fig. 12 on p. 17). Over 15,950 item pairs from 1,053 DrawBench and PartiPrompts prompts on FLUX-schnell, 1,699 pairs (10.65%) show flow and 14,251 (89.35%) do not (Table 3, p. 6). On 110 curated prompts where the model draws the wrong sense, the item's contextual token outputs alone already draw the wrong sense in 93% of cases, while the item encoded with only its modifiers draws the right one, which the authors read as the error originating in the encoder (p. 7).

The VLM judge agrees with three human annotators at accuracy 0.810 and F1 0.740 on the cross-item task (p. 3), so the 10.65% carries judge error of that size.

## What it changes there

The probe (p. 2). For a prompt with $N$ tokens:

$$
\tilde t_i = \begin{cases} h_i & i \in S \\ p_i & \text{otherwise} \end{cases}, \qquad i = 1, \dots, N
$$

- $h_i$ is the final-layer output of the text encoder $E$ at position $i$ for the full prompt.
- $p_i$ is the encoder output at position $i$ for a sequence made only of pad tokens.
- $S$ is the set of positions kept; the sequence $\tilde t$ conditions the diffusion model in place of the normal encoding.

The fix (p. 7). The tokens of the misread item are replaced by the encoding of a "cleaned" prompt holding only the item and its local modifiers ("standing zebra", not "zebra" alone); all other positions keep their full-prompt encoding:

$$
\tilde t_i = \begin{cases} h_i^{\text{clean}} & i \in S_{\text{item}} \\ h_i & \text{otherwise} \end{cases}
$$

- $S_{\text{item}}$ is the misread item's token positions; $h_i^{\text{clean}}$ is the encoder output of the cleaned prompt at the matching token. How the cleaned tokens are aligned to positions is not printed (inferred).

Layers: only the text encoder's final output is changed (p. 4: "as probed from the final encoder layer"). Timesteps: no step range is printed; the patched sequence replaces the conditioning, so it acts at every step, 100% of the steps (inferred from p. 2). Cost: one extra text-encoder pass per patched item and no extra denoiser passes (inferred; no cost is printed). Finding which item is misread, as done in the paper, costs full image generations from each item plus a VLM call (p. 16); in the curated test the misread item was known in advance from human-annotated intended senses (p. 7).

## Does it need training, a layout or extra inputs

The patch trains nothing (p. 7). A separate token-redundancy classifier (a single linear layer on encoder outputs, binary cross-entropy, 25 epochs, 80/20 split of the token annotations) is trained for the in-item pruning result only (p. 15, Table 5 on p. 15). No boxes or masks. It needs the item and its local modifiers marked; the paper extracts items with GPT-4o and o3-mini-high and parts of speech with spaCy (p. 3, p. 13). The baseline it beats, RAG-Diffusion, uses boxes per item (p. 7).

## Which model it runs on

FLUX-schnell (main results) and FLUX-dev with T5-XXL, SDXL-Turbo with CLIP, and SANA 1.6B with Gemma (p. 3, p. 13). The leakage fix is evaluated on FLUX-dev and FLUX-schnell only (Table 4, p. 8). Resolution, sampler and step count are not printed. Five seeds per prompt (p. 3). About 480 GPU hours on eight A100 GPUs in total (p. 13, p. 14).

On SDXL-Turbo, whose CLIP encoder is causal (each token sees only earlier tokens), 55% of lexical items have no token that alone draws the item (11% on FLUX), and images drawn from the end-of-sequence token alone match the prompt 62% of the time against 73% for the full prompt (p. 15). The authors say this makes the masking probe less effective on CLIP, because masking removes the end-of-sequence token that carries most of the prompt (p. 15). No cross-item flow rate and no leakage-fix result is reported for SDXL-Turbo. The paper says its core results hold across all four models, with encoder-specific differences (p. 3, p. 8).

## What it measured on similar subjects

No similar-subject split. No two-animal or two-person prompts are reported. The examples of flow are an old man, tree reflection in the lake, a triangle with a smiling face (Fig. 5, p. 6), and the leakage set is word-sense cases built from Rassin et al. 2022 (the Seeing Double stimuli) plus GPT-4o expansions (p. 7, p. 14).

## Which pairs leak and why

- Flow is rare: 10.65% of 15,950 item pairs on FLUX-schnell (Table 3, p. 6), 11% on FLUX-dev (p. 14), 17.45% on SANA against 10.45% on FLUX (p. 15). After removing pairs a GPT-4o judge calls "perceptually bound" (old and man), 6.5% show unintended flow on FLUX-schnell (p. 13) and 3.1% on FLUX-dev (p. 14).
- Flow does not follow syntax: it appears between items with no direct syntactic relation (p. 6).
- Harmful flow concentrates on polysemous words, where the other item is associated with the wrong sense (bats with baseball stadium) (p. 6).
- Direction follows the encoder. In bidirectional T5 the source item comes first in 835 pairs and second in 823 (p. 6). In causal encoders (Gemma in SANA, CLIP) only earlier tokens shape later ones (p. 15, Fig. 10 on p. 16). On the closed Nano-Banana model, leakage is 26% of prompts overall and rises to 42% when the misread item comes after its cue (p. 18, p. 19).
- Text-embedding similarity between items, attention-map overlap and appearance similarity are not measured. Padding enters only as the masking value $p_i$ (p. 2).

## When in sampling leakage is set

Not measured. The paper studies only the text encoder output, which is fixed across sampling steps.

## Numbers

| Metric (scale, direction) | This method | Best baseline | Dataset | Model | Page, table |
|---|---|---|---|---|---|
| Prompts still showing leakage, % of 110 (lower is better) | 14% | RAG-Diffusion 43%; unpatched 94% | curated polysemy set | FLUX-schnell | p. 8, Table 4 |
| Prompts still showing leakage, % of 110 (lower is better) | 20% | RAG-Diffusion 39%; unpatched 79% | curated polysemy set | FLUX-dev | p. 8, Table 4 |
| Item pairs with cross-item flow, % (a rate, no direction) | 10.65% (1,699 of 15,950) | n/a | DrawBench + PartiPrompts, 1,053 prompts | FLUX-schnell | p. 6, Table 3 |
| Item pairs with cross-item flow, % | 17.45% | 10.45% FLUX | same | SANA | p. 15 |
| Misread items already misread from their own contextual tokens, % | 93% | n/a | curated polysemy set | FLUX | p. 7 |
| User-chosen sense drawn after patching, % (higher is better) | 95% | n/a | GPT-4o ambiguous prompts | FLUX | p. 8, p. 18 |
| Prompt-level alignment after dropping non-representative tokens, % (higher is better) | 87.02% | 83.48% full prompt | 339 prompts | FLUX-schnell | p. 5, Table 1 |
| Items with no token that alone draws the item, % | 55% | 11% FLUX | in-item set | SDXL-Turbo | p. 15 |
| Leakage rate on a closed model, % of 110 prompts | 26% (42% when item follows cue) | n/a | curated polysemy set | Nano-Banana | p. 18, p. 19 |

## What it suggests for this loop

- For the predictor step (inferred): the paper's flow test can be run on SDXL's CLIP encoders without images. Encode "a duck and a goose", take the second subject's token outputs, and compare them with the encoding of "a goose" alone. That contextual drift is a per-pair text predictor sharper than the cosine of the two bare names, because it measures what one subject's tokens absorb from the other.
- For the predictor step (inferred): CLIP is causal, so only the later-named subject can absorb the earlier one at the text level. If borrowing in CO3's pictures runs mostly from the first-named subject onto the second, and flips when the order is swapped, the text encoder is implicated. If it does not depend on order, the text encoder is an unlikely source.
- For the predictor step (inferred): the paper finds CLIP's per-token outputs weak and its end-of-sequence token carrying most of the prompt (p. 8, p. 15). SDXL also feeds a pooled text embedding to the UNet. A text-side predictor should include the joint prompt's end-of-sequence and pooled vectors, and a text-side cosine that sees only subject tokens may miss leakage carried there.
- For the one-fix step (inferred): CO3 already computes single-subject predictions whose prompts encode each subject without the other, which are the context-free encodings this paper patches in. If leakage enters at the text encoder, it enters through the joint prompt's prediction. A zero-cost check is to rebuild the joint prompt's conditioning with each subject's tokens taken from its single-subject encoding and see whether borrowing on cheetah and cougar, duck and goose, and puma and condor drops. This needs no boxes and no extra denoiser passes, so it can run before or beside Bounded Attention.
- For the one-fix step (inferred): the paper's leakage is a word misread in the wrong sense. CO3's borrowing (a goose with a duck's beak) may not be a sense error at all, so a null from token patching would point the search at cross-attention or the sum of scores.
- For Paper 2's method (inferred): any text-side intervention Paper 2 proposes has this token-patching swap as its nearest prior and should be compared against it on SDXL.

## Closer work it cites

- Circuit mechanisms for spatial relation generation in diffusion transformers (Wang, Fan, Pan 2026), arXiv 2601.06338, p. 8: reports that T5 fuses information from several lexical items into single tokens, the same flow measured here.
- Improving compositional attribute binding in text-to-image generative models via enhanced text embeddings (Zarei et al. 2025), arXiv 2406.07844, p. 9: cited as finding independently that attribute-binding failures originate in CLIP's text encoder, the encoder SDXL uses.
- Padding tone: a mechanistic analysis of padding tokens in T2I models (Toker et al. 2025), arXiv 2501.06751, p. 2 and p. 12: the masking framework this paper builds on; bears on whether padding positions carry subject information.
- Does liking yellow imply driving a school bus? Semantic leakage in language models (Gonen et al. 2024), arXiv 2408.06518, p. 6 and p. 9: the source of the behavioural definition of semantic leakage, measured in text-only language models.
