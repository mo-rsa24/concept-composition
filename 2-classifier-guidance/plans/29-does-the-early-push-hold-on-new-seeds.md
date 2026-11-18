# 🧪 Does the early push hold on 32 new seeds

## Position in the plan tree

[Root](../../MASTER_PLAN.md) › [the classifier-guided PoE route](route-classifier-guided-poe-from-toy-to-animals.md) ›
this plan. Its pass marks are in [its review file](review/29-does-the-early-push-hold-on-new-seeds.md); results go into
[the checker loop file](../artifacts/drips/rung2-pair-aware-discriminator/checker-loop.md).

## Quick context

**Why this plan exists**

The best recipe so far (the readout checker's push at cap 0.3 on steps 0 to 15, then plain PoE) was chosen by looking at
held-out seeds 9 to 16. Eight seeds that guided the choice cannot show whether it works in general. This runs the recipe
unchanged on 32 seeds never used for anything, beside plain PoE, best of 2 and the size-matched random push.

## Environment Facts This Plan Depends On

- mscluster111 and mscluster112 are Blackwell cards driven only by `co3_bw` ([the node table](../../environment/hpc/nodes.md)).
- Outputs go under `POE_REPAIR_MOUNT=/home-mscluster/mmolefe/poe_repair_min_scratch` while `/datasets` is down
  ([storage](../../environment/storage.md)).
- A seed with no cached starting noise draws it from the seed, as plan 25's runs did.

## Tasks

- [x] **1.1 Draw.** `rung2/general.sh` on mscluster112: seeds 3301 to 3332, strength 30, cap 0.3, push on steps 0 to 15.
- [x] **1.2 Call.** Every full-size tile, plain, pushed and random, called separate or not, clean or near-clean or not;
  counts into the review file's Q1.
