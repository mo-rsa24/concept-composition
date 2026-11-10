# PRZE-3a: the pair lists, exact

Answers PrzE's second weakness (the 48 pairs are not listed) and UHz6's question about cat and dog and about pairs kept only because CO3 composed.

## The adapter behind the paper

The adapter of record is **v58-03-self at step 50,000**, `/datasets/mmolefe/poe_repair_min/outputs/showcase/cohort3/v58-03-self/checkpoints/lora_step_050000.pt`. Every new score and pair list is read against it. Figure 5's "ours" column was drawn by three runs, read from each tile's record under `/datasets/mmolefe/poe_repair_min/outputs/ablation_grid/<pair>/seed<NN>/<key>.json` and each run's `config.json` (2026-11-10, `verified`):

| Figure 5 row | Run and step | Pool it trained on | Training pairs | Held out at training time | Epochs run |
|---|---|---|---|---|---|
| cat and dog, seed 9 | v57-11b-x0-self-decay, step 50,000 | `pair_pool_v57.yaml` | 29 | cat×dog, elephant×penguin | 1,000 |
| elephant and penguin, seed 10 | v58-05-self-contrast, step 15,000 | `pair_pool_v58.yaml` | 48 | bison×horse, cat×dog, piano×candle, skateboard×helmet, elephant×penguin | 300 of 1,000 |
| tent and snowy mountain, lighthouse and stormy sea, seed 9 | v58-03-self, step 50,000 | `pair_pool_v58.yaml` | 48 | the same five | 1,000 |

All three: rank 32, alpha 32, LoRA on `attn1` and `attn2` `to_q`, `to_k`, `to_v`, AdamW at 1e-4, x0-space loss, orthogonality weight 1.0, 50 optimiser steps per epoch, trained on step indices 0 to 9 of 50. The two `cross_pair_pooled__r32__lr1e-04__ep600__…` folders under `artifacts/_shared/cross_pair_pool_configs/all_groups/` are September runs on `pair_pool_v54.yaml` (24 training pairs, cross-attention only, 600 epochs) and are not behind any figure.

**The held-out list changed after training.** The v58 runs' own `pair_pool.json` (written 2026-09-23) holds out five pairs including bison×horse; `pair_pool_v58.yaml` dropped bison×horse from `heldout:` in commit 08d27d9b on 2026-10-21. The 48 training pairs are identical in both.

**Tent×snowy mountain and lighthouse×stormy sea are in neither list** of either pool, so they are unseen pairs outside any declared held-out set. Lighthouse×sailboat is a training pair.

## Outcomes

- [x] ✅ check passed: the training cache on the cluster is read and its pair list equals the 48 in `pair_pool_v58.yaml` and the paper, or the difference is stated with both paths. ✓ verified 2026-11-10: v58-03-self's `pair_pool.json` and `pair_pool_v58.yaml` list the same 48 training pairs (set difference empty both ways); all 48 have a folder under `/datasets/mmolefe/poe_repair_min/outputs/training_cache/train/`; `dataset_meta.json` reads 48 pairs and 182 cells. Held-out differs: the run held out bison×horse as well, the file no longer does
- [x] ✅ check passed: the checkpoint behind the paper's figures is identified (the folder name says 600 epochs, the paper says 1,000, and Figure 5 shows tent and lighthouse rows that sit in no held-out list). ✓ verified 2026-11-10: three runs, table above; the ep600 folders are unrelated v54 runs
- [ ] 🖼️ screenshot: Figure 5 re-rendered from the one frozen checkpoint, every row from that checkpoint and all four columns from the same starting noise including CO3, seed chosen by the fixed rule; the caption says so
- [ ] 📝 markdown created: an appendix table of every training pair and every held-out pair, with whether the joint prompt, CO3 or both passed the filter
- [ ] ✏️ markdown updated: the main text states the filter rule and how many pairs survive each stage
