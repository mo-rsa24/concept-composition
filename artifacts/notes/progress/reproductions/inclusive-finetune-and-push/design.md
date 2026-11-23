# Does a short inclusive fine-tune make the checker's push stronger?

## What this asks

Paper 2's appendix carries one paragraph on inclusive captions. The argument behind it is that a short fine-tune on inclusive captions gives the product of experts
some support for two animals, which the checker's push can then amplify. No run has combined the two. This test renders the two saved cat and dog LoRAs from the
inclusive-captions pilot, with and without the chosen checker's push, and decides what that paragraph may say. It trains nothing and starts after steps D3 and M2 have chosen
the checker.

## Words this uses

**Product of experts (PoE).** The sampler that adds SDXL's noise predictions for "a cat" and "a dog".

**Push.** The chosen checker's gradient added to the PoE prediction during the early sampling steps.

**Run A.** A rank-32 LoRA trained for 2,000 steps on 550 photos (200 cat, 200 dog, 150 both). A photo holding both animals is captioned "a cat and a dog" 90% of the time
and left blank 10% of the time.

**Run B, the inclusive fine-tune.** The same photos and training. A photo holding both animals is captioned "cat" 40%, "dog" 40%, "a cat and a dog" 10% and blank 10% of the
time, so the single-animal captions include pictures with both animals.

**Faithful pair.** One picture with two separate bodies, a clear cat and a clear dog, no borrowed features and an intact photograph.

## The run

**Group.** Runs that try ideas (group 2). The result decides the wording of one appendix paragraph and makes no claim about any benchmark.

**Varied axis.** How the photos holding both animals are captioned (run A against run B), read with the push off and on. The photos, training length, sampler, seeds and
checker are the same.

**Cost.** 4 conditions at 32 seeds is 128 pictures. At the 40 seconds per pushed picture budgeted for step D3, that is about 1.4 GPU-hours. It trains nothing and uses no API
tokens.

**Buys.** A measured answer for the appendix paragraph, by the owner's answer to board question 46.

| Condition | LoRA | Push |
|---|---|---|
| A, no push | `v4-A/step_02000` | off |
| A, push | `v4-A/step_02000` | on |
| B, no push | `v4-B/step_02000` | off |
| B, push | `v4-B/step_02000` | on |

The LoRAs are in `/datasets/mmolefe/poe_repair_min/outputs/inclusive_captions/`.

## The settings, fixed before the first render

| Setting | Value |
|---|---|
| Subjects | cat and dog, allowed by the subject guard |
| Seeds | 6001 to 6032, which are tuning seeds |
| Checker | the one steps D3 and M2 choose, unchanged and untuned for this test |
| Push | D3's settings for that checker: capped at 0.4 of the PoE prediction, on steps 0 to 15 |
| Sampler | the same sampler and step count as the D3 comparison |
| Counting | by eye at full size, in a shuffled order under neutral file names |

Each LoRA renders in its own process. With the push at zero, the sampler must reproduce that LoRA's plain PoE render to under 1e-5, in one loop on one card and dtype.

## What each result means, written before the first render

The comparison is faithful pairs of 32 for run B with the push against run A with the push.

| Outcome | What it takes | What the paragraph may say |
|---|---|---|
| Support | B with the push has at least 5 more than A with the push, and more than B without the push | an inclusive fine-tune made the push stronger on this one pair |
| Null | B with the push has no more than A with the push | no gain was measured from inclusive captions |
| Inconclusive | B with the push has 1 to 4 more than A with the push, or the support test's second half fails | the idea remains untested at this scale |

A lead of 5 matches the margin in the [checker comparison](../checker/design.md), where re-rendering one recipe moved its count by about 2. A surprising result would be run A
with the push beating run B with the push by 5 or more, because that would mean the inclusive captions weaken what the push can do.

## The ladder

1. **Control.** The zero-push check on each LoRA.
2. **Smoke test.** One seed through all four conditions, with the sheet opened at full size.
3. **Full test.** The 32 seeds and the shuffled count, written to `review.md` in this folder.

## What this cannot say

This is one animal pair, one training of each LoRA and 32 seeds. There is no base SDXL row, so it compares two ways of captioning and cannot say whether fine-tuning helps
compared with no fine-tuning. The checker was trained on plain SDXL's pictures, so a LoRA that shifts SDXL's features could weaken the push for a reason unrelated to the
captions. A pushed render is not bit-reproducible on the GPU, so only the counts are compared.
