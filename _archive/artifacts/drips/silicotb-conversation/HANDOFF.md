# Silicotuberculosis conversation: what a session picking this up needs

This folder holds a ChatGPT conversation about chest imaging of silicosis, TB and other disease
pairs, saved so its claims can be walked and tested against the chest X-ray data on the cluster.
Read this file first, then the route.

## What is here

| File | What it is |
|---|---|
| `source/conversation.md` | the whole conversation, 8 turns, 19,004 words, fetched 2026-11-03 from https://chatgpt.com/share/6ae9602c-0378-83ea-a868-6df6c06a6c91 (the share page only renders in a browser; the text was decoded from the page's embedded data, longest saved copy per turn) |
| `source/screenshots/` | the three images that opened the session: a CPFE CT example, the three overlap regimes for silicosis with TB, the four CT patterns |
| `../../../plans/route-every-silicotb-chat-claim.md` | the route: settled facts with sources, the chain, every prompt |

## The turns

1. TB then silicosis: does one change how the other looks on X-ray and CT; what silicotuberculosis is
2. Where AI fails: mimicry, masking, interaction; what the label should be; South African data scarcity
3. Public-data pairs with the same three regimes: cardiomegaly plus effusion, effusion plus atelectasis; NIH, PadChest, VinDr
4. Truly interacting pairs: HIV plus TB, pneumonia plus pulmonary fibrosis, fibrosis plus emphysema (CPFE)
5. Stable diffusion and product of experts on X-rays: the interaction term R, inclusive labels and condition dropout, a LoRA interaction operator
6. Multi-label conditioning with LoRA on cats and dogs (already routed: `4-inclusive-captions/plans/route-inclusive-captions-pilot.md`)
7. Is there an m (mechanism) or h (context) for cats and dogs, or is co-occurrence enforced by captions
8. Does inclusive training generalise across seeds and pairs

## Decisions already made

**Home.** This repo (`concept-composition`), because turns 5 to 8 are this repo's own question. Paper3
(`~/Playground/PhD/Paper3` on the cluster) is read, never written: its Mission B already tests
single-disease classifiers on six comorbid pairs.

**Outcome.** Coverage of the whole conversation plus as much evidence as can be found on every
claim, with whatever figure fits each one (pair-figure, evidence-ladder, visualize-data-samples,
reference images). Not limited to one claim.

**Cadence.** No fixed windows; the learner steers with drip controls.

## Facts that change what can be tested

The full table with sources is in the route's "What we know so far". The ones that decide most:

- NIH has only 11 films labelled both pneumonia and fibrosis, so that claim cannot be scored on NIH.
- VinDr has no pneumonia or TB label; it has 187 films with fibrosis and consolidation, so the
  pneumonia-in-fibrotic-lungs test runs on VinDr with consolidation as the stand-in.
- No silicosis, TB-with-silicosis or HIV imaging labels exist in any set on disk.
- Unknown: whether a compute node can download model weights (TorchXRayVision, MedSAM). Ask.

## Running things from the laptop

The data exists only on the cluster, under `/datasets/mmolefe/` (`data/nih/`, `vinbigdata/`,
`chestxdet/`, `chexmask/`). A count, a `run` or a `prove` goes through
`scripts/cluster.sh sh "<command>"`, which runs on mscluster84 in the cluster checkout. Large
outputs stay on `/datasets`; bring back only figures and cards with `scripts/cluster.sh pull`.
