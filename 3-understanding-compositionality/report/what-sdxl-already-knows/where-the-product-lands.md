# 📒 Report: where the product of "a cat" and "a dog" lands

**This is an illustration: the surface is drawn by hand and only its four renders are real.** It pictures the product of two prompts landing on the low ground between the "a cat" and "a dog" peaks, away from the separate peak that the joint prompt "a cat and a dog" reaches. Nothing here is measured, so it carries no verdict.

## Table of contents

- [The picture](#the-picture)
- [Where the sources disagree](#where-the-sources-disagree)
- [What this cannot tell you](#what-this-cannot-tell-you)
- [Concepts this uses](#concepts-this-uses)
- [Where this came from](#where-this-came-from)
- [Still open](#still-open)

## The picture

Navigation: 📋 [TOC](#table-of-contents) | [Next](#where-the-sources-disagree) ➡️

<a href="../../artifacts/results/what-sdxl-already-knows/poe-against-the-joint/where-the-product-lands/where-the-product-lands-on-the-manifold-seed12-surface.png"><img src="../../artifacts/results/what-sdxl-already-knows/poe-against-the-joint/where-the-product-lands/where-the-product-lands-on-the-manifold-seed12-surface.png" width="200" alt="A blue and red surface with peaks for a cat and a dog, a fused render above the saddle between them marked PoE lands here, and a cat and dog render on a lower front peak"></a>

**The fused render sits over the saddle, and the render with both animals sits on its own peak.** The two rear peaks carry the renders of "a cat" and "a dog". The X marks the product's peak, with its fused render above it, and the front peak carries the render of "a cat and a dog" `verified` (read on the picture; matches the card).

| Part of the picture | Real or drawn | Mark |
|---|---|---|
| the four renders | real: cat and dog pair, seed 12, one starting noise, 50 steps, guidance 7.5, no adapter | `stated` ([the card](../../artifacts/results/what-sdxl-already-knows/poe-against-the-joint/where-the-product-lands/README.md)) |
| the plane under the surface | drawn: it is not an embedding of anything | `stated` |
| the peaks | drawn: Gaussians placed by hand | `stated` |
| the X at the midpoint of the two rear peaks | follows from the maths of two equal-width Gaussians | `stated` ([the drawing script](../../scripts/where_poe_lands_on_the_manifold.py), lines 21 to 23) |

## Where the sources disagree

Navigation: ⬅️ [Previous](#the-picture) | 📋 [TOC](#table-of-contents) | [Next](#what-this-cannot-tell-you) ➡️

**The card and the drawing script differ on the height of the peaks, and the script settles it for this picture.**

| What | The card | The script |
|---|---|---|
| peak heights | "hand-placed Gaussians at equal height" | equal heights belong to the original layouts. This surface layout draws the joint peak lower, at 0.60 against 1.00, and marks that as a drawing choice (lines 18 to 19, 58 and 121 to 122) |

## What this cannot tell you

Navigation: ⬅️ [Previous](#where-the-sources-disagree) | 📋 [TOC](#table-of-contents) | [Next](#concepts-this-uses) ➡️

**The distances and heights mean nothing.** The picture cannot show how far the product's render is from either prompt, or how likely any of the three scenes is.

**It shows one seed.** Measured evidence for the fusion is on [what the fusion is not caused by](what-the-fusion-is-not-caused-by.md).

## Concepts this uses

Navigation: ⬅️ [Previous](#what-this-cannot-tell-you) | 📋 [TOC](#table-of-contents) | [Next](#where-this-came-from) ➡️

The toy numbers below are invented. The [product of experts](../../../context/world/poe-composition.md#what-poe-composition-is) rule itself is taught on [the cats and dogs page](does-poe-of-cats-and-dogs-match-the-joint.md#how-hard-a-product-pushes).

**The product of two equal bumps peaks at their midpoint**

Take two bell curves of equal width, one centred at -2 and one at +2, each with height 1 at its centre. Multiply their heights at three places:

| Position | First curve | Second curve | Product |
|---|---|---|---|
| -2, the first peak | 1 | 0.0003 | 0.0003 |
| 0, the midpoint | 0.135 | 0.135 | 0.018 |
| +2, the second peak | 0.0003 | 1 | 0.0003 |

The product is highest at 0, a place where neither curve is high on its own. Each height is e^(-d² / 2), where d is the distance from that curve's centre. In the drawing the two rear peaks sit at (-2.55, 1.05) and (2.55, 1.05), so the X goes at their midpoint, (0, 1.05) (script, line 120).

## Where this came from

Navigation: ⬅️ [Previous](#concepts-this-uses) | 📋 [TOC](#table-of-contents) | [Next](#still-open) ➡️

| What | Source | Mark |
|---|---|---|
| what is real and what is drawn | [the folder's card](../../artifacts/results/what-sdxl-already-knows/poe-against-the-joint/where-the-product-lands/README.md), read 2026-11-22 | `stated` |
| how the surface was drawn | [the drawing script](../../scripts/where_poe_lands_on_the_manifold.py), read 2026-11-22 | `verified` |
| 1 vector copy held out of the drain | the SVG of the surface figure placed above, in [its folder](../../artifacts/results/what-sdxl-already-knows/poe-against-the-joint/where-the-product-lands) | `stated` |

## Still open

Navigation: ⬅️ [Previous](#where-this-came-from) | 📋 [TOC](#table-of-contents)

- [ ] Correct the card's "equal height" for the surface layout.

<!-- built-to: PAGE_LOOK_FORMAT v1 -->
