# Background for: answering the ICLR 2027 reviews of submission 56173

Navigation: [the route](route-answer-the-iclr-reviews.md)

## Found on disk

- The manuscript source and notes: `paper/iclr/` (tex, bibliography, `figures.md` register, `SPINE.md`, `DRAFT_MAP.md`), plus `paper/overleaf-iclr/` and `1-lora-correction/scripts/make_overleaf_bundle.sh --slim` for the upload zip.
- Scorers: `composition/focus/eval_metrics.py` holds CLIP, EVA-CLIP and ImageReward scorers. No code for GenEval or T2I-CompBench.
- Samplers: `poe_repair/methods/_poe_langevin.py` (Langevin corrector), `1-lora-correction/scripts/corrector_residual_curve.py`, `1-lora-correction/scripts/fk_steering/`, `1-lora-correction/scripts/twisted_smc/`, `composition/reduce_reuse_recycle/` (the authors' code), `composition/debottam_co3/` (CO3).
- Findings already filed: `1-lora-correction/report/is-the-gap-the-samplers-or-the-models/` (corrector, SuperDiff, steering, noise search runs).
- Pool files: `artifacts/_shared/cross_pair_pool_configs/`.
- Audit of the authors' code against their paper: `/home/molef/goal-setting/learning/deep-learning/diffusion-models/02-sampler-correctors-for-composition/drips/read-reduce-reuse-recycle/routes/code-vs-paper.md`.
- The reviews and the submitted paper: `paper/iclr-56173/Reviews.pdf` and `56173_Restoring_plurality_in_c.pdf` at the repo root.

## Plan weight

Short. This route file is the plan. Each experiment opens as its own request walk with its bar filled from this route. A full plan tree would cost more writing than the 2.5 working days allow, and the gated steps would force rewrites.

## Who you are for this route

Gloss tier for every benchmark and baseline: understand each well enough to use it and to know where it sits in the paper, not in depth. Deeper only where a number in the paper depends on it. Quick to run, thorough to check.

## Later

Reviewers UHz6 and k4nd get their own rows when their round starts. A three-concept adapter trained from scratch is a gated extra.

## Why this chain

Cheap tests go before training (the measured correction on SD 1.5, the adapter applied over early steps only, the adapter inside a three-expert product), so a training opens only when a ceiling is known. No curated flow fits, so the chain is proposed from your list.

**Where each time came from.** Step 3's open is measured (about 9 minutes, 4 runs); the 20 minutes at its gate and every other your-time figure is an estimate (pieces by about 30 minutes, plus half again for tangents). Sampling times are measured per image in `environment/hpc/throughput.md` (11 to 16 s on the Blackwell, 41 on an A6000, 51 on an RTX 8000). The 7 to 10 hours per training is your own check.

**Totals.** Your time about 14.0 h plus about 1 h posting, AI about 14 h (mostly overlapping the GPU), Blackwell 12 to 14 h, mscluster107 2 to 4 h, gated trainings 14 to 20 h if both open.

**Cuts, cheapest first, if your windows total less than 15 h.** Step 7 on two sources instead of four (saves about 25 min); step 15 without the quiz (saves about 15 min); step 3's diagram gate after the rebuttal (saves about 30 min). Below about 12 h the reply in step 14 would be written from the table without your sentence-by-sentence steering.

## References

- The CO3 paper, [arXiv 2509.25940](https://arxiv.org/abs/2509.25940): the protocol and Table 1 whose shape our table copies.
- T2I-CompBench++, [arXiv 2307.06350](https://arxiv.org/abs/2307.06350): where BLIP-VQA comes from.
- Reduce, Reuse, Recycle, [arXiv 2302.11552](https://arxiv.org/abs/2302.11552) and [the code-against-paper audit](/home/molef/goal-setting/learning/deep-learning/diffusion-models/02-sampler-correctors-for-composition/drips/read-reduce-reuse-recycle/routes/code-vs-paper.md): the corrector's reference.
- Attend-and-Excite, [arXiv 2301.13826](https://arxiv.org/abs/2301.13826), vendored in `composition/aae/`; the [CONFORM repository](https://github.com/gemlab-vt/conform), [arXiv 2312.06059](https://arxiv.org/abs/2312.06059).
- [The paper review board](../1-lora-correction/report/paper-review/board.md) and [the walk map](../1-lora-correction/artifacts/drips/answering-the-reviewers/answer-iclr-reviews/ROUTE_WALK.md).
