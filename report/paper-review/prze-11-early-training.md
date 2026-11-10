# PRZE-11: does restricting the adapter to the early steps change the failure types

UHz6 asks: given the finding in Section 3 (the correction only works in the first ten steps), does restricting the adapter to the early steps change the failure types in Figure 7? Figure 7 shows four: a changed render under the joint prompt, intrusions of a third animal, counting errors, and style-dependent loss of plurality or instability.

Two tests, cheap one first. The inference test applies the existing adapter only over early steps. The training test trains a new adapter on states from all 50 steps and sets it against the paper's adapter, which trained on the first ten only. The training test opens only if the inference test shows the failure counts move, because that result says whether the window matters at all.

## Which steps the paper's adapter trained on

The adapter of record, v58-03-self at step 50,000, trained only on step indices 0 to 9 of the 50-step DDIM schedule: its `config.json` sets `train_step_range` to `[0, 10]`, the trainer keeps `lo <= step_index < hi` (`poe_repair/experiments/cross_pair_lora_pooling/train_pooled.py` line 532), and the cache's `meta.json` gives those indices the timesteps 981, 961, 941, 921, 901, 881, 861, 841, 821, 801, the noisiest ten. Each `dataset_meta.json` cell reads `n_steps: 10` (`verified` 2026-11-10). The two other Figure 5 runs used the same range.

So the paper's adapter is already an early-steps-trained adapter, and the training test's contrast is an adapter trained on all 50 steps.

## Outcomes

- [ ] 📝 markdown created: the failure types defined in writing before counting (intrusion, count error, style loss, joint-prompt corruption, instability), each with how a render is assigned to it
- [ ] 💻 code ran: the existing rank-32 adapter applied over steps 0 to 10, 0 to 25 and all 50, on the Figure 7 cases and the PRZE-3b held-out set, failures counted per type
- [ ] 🖼️ screenshot: qualitative grids from the inference test, one per group of pairs, rows are pairs and seeds, columns are the adapter off, on over steps 0 to 10, over 0 to 25 and over all 50, with the Figure 7 failure cases in their own grid
- [ ] 📝 markdown created: the bar for opening the training test, written before the inference run
- [ ] 💻 code ran: an adapter trained on states from all 50 steps (`train_step_range` unset), matching v58-03-self in everything else (`pair_pool_v58.yaml` with its 48 training pairs, the same 182 cells, rank 32, alpha 32, `attn1` and `attn2` q/k/v, learning rate 1e-4, x0 loss, orthogonality weight 1.0, 50,000 optimiser steps)
- [ ] 🖼️ screenshot: one figure of failure counts per type, the all-steps adapter against the paper's steps-0-to-9 adapter and against early-only application, with the full image grid in the appendix
- [ ] ✏️ markdown updated: a paragraph in the paper stating what changed and what did not
