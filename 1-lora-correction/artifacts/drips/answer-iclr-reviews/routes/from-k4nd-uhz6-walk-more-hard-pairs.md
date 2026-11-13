# For the route: grow the hard group of Table 2 to 66 pairs

From the walk answering Reviewers k4nd and UHz6 (2026-11-11), at the user's request.

## Why

Table 2 (slot R3) has 3 hard pairs, and their 95% intervals are about ±0.4 on a 0-to-1 scale, so the
methods cannot be told apart on exactly the pairs where the adapter matters most: the joint prompt
fails there, so no training target of that kind ever existed. k4nd asks for "a sufficiently large,
documented test set" and UHz6 for rates "over all held-out pairs and multiple seeds". The user's target
is 66 hard pairs, the size of the Attend-and-Excite animal set.

## What a hard pair is (unchanged)

A competing pair whose joint prompt shows both named things on fewer than 4 of 8 seeds
(1-lora-correction/report/paper-review/held-out-set.md). No concept pair may be a training pair or a tuning pair
(protocol rule 4), and the new pairs are fixed in held-out-set.md before any adapter result on them
exists.

## How

1. Census: render the joint prompt at seeds 9 to 16 on candidate pairs and keep those under 4 of 8.
   The first census found 3 hard among 32 competing candidates, about 1 in 11, so 66 needs roughly 700
   candidates at that rate. Drawing candidates that are rare or confusable together (two similar
   species, an object and an animal of similar shape) should raise the rate.
2. Render every Table 2 method on the new hard pairs at seeds 9 to 16 (528 images per method), score
   them as Table 2 does, and report the hard group with its interval.

## Cost (estimates, not measured)

From job 84832's timing (12 tiles in 9 minutes on an RTX 3090, about 0.27 s per U-Net pass per step):
a joint-prompt render takes about 27 s, so a 700-candidate census at 8 seeds is about 40 GPU-hours on
3090s. Then per method on 528 images: PoE about 6 hours, ours about 12 (the sampler runs the frozen and
adapted passes), CO3 and PoE + Langevin longer and not estimated here. Scoring is extra.

## Decisions for the user

- whether 66 is the target, or a smaller first wave (for example 20) to see the hit rate;
- whether the new hard pairs replace or join the current 3;
- the candidate source for the census.

The list goes to the code release, not the paper (Appendix C.2 now points there).
