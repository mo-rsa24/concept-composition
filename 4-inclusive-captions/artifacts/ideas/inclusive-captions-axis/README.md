# Do inclusive captions help the cat and dog experts compose?

**What this idea says**

Photos holding both a cat and a dog are also trained under "a photo of a cat" and "a photo of a dog", so each single-animal condition keeps images that hold the other animal. The product of the two conditions should then hold more pictures containing both animals `stated`.

**Where it stands**

[The claim-by-claim map](IDEA_MAP.md) marks six claims. Two need a check, including claim 1, which the idea depends on, and four are wrong as stated and settled `verified`. The SDXL pilot that tests them has a sheet and no compose rate `stated` ([the report page on the pilot](../../../report/do-inclusive-captions-help-the-product-compose/does-captioning-both-animal-photos-under-each-name-help-poe-compose.md)).

**What else is here**

[The checked design](COMPILED.md) is the standing version of the idea, and [the source design](source-design.md) is the proposal it was checked from, copied from a ChatGPT conversation.

**Where these files disagree**

- Matching run C's draw counts leaves the cat condition seeing 50 photos against run B's 200 in the checked design, and 52 against 131 in the map `stated`.
- The GPU cost is about 10 hours in the checked design and about 9 in the map `stated`.
- The checked design trains on 731 photos (131 with both animals), and the pilot trained on 550 (150 with both) `stated`.

**Proposed verdict: KEEP** `proposed`. [The pilot route](../../../plans/route-inclusive-captions-pilot.md) and [the pilot plan](../../../plans/11-do-inclusive-captions-help-poe-compose/plans/01-the-inclusive-captions-pilot.md) still read these files, and claims 1 and 2 are unsettled.
