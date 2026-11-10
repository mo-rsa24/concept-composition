# The frozen evaluation set for the review response

Fixed before any adapter result exists on these pairs or prompts, under [the comparison protocol](protocol.md) (rules 1, 4 and 5). The commit that adds this file is the proof of order: `git log -1 -- report/paper-review/held-out-set.md`.

## The adapter

**v58-03-self_s030000**, checkpoint `/datasets/mmolefe/poe_repair_min/outputs/showcase/cohort3/v58-03-self/checkpoints/lora_step_030000.pt`, shared seed **10**, chosen by the showcase rule ([the calls](showcase-calls.csv)); the live choice is `artifacts/_shared/cross_pair_pool_configs/adapter_of_record.yaml`.

## What is scored, and at which seeds

| Set | What | Seeds | Slot |
|---|---|---|---|
| benchmark | the 276 Attend-and-Excite prompts, `data/prompt-sets/attend-and-excite/a.e_prompts.json`: animals 66, animals with objects 144, objects 66 | 9 to 12 | R1, R5 |
| held-out ablation | 53 pairs in three groups, below | 9 to 16 | R3 |
| showcase | cat × dog, elephant × penguin: picked the adapter, never in a rate | 9 to 16 | R2 |
| tuning | lion × meerkat, camel × pelican, donkey × duck, chicken × rabbit (training pairs) | 9 to 16 | rule 3 only |
| training | the 48 pairs of `pair_pool_v58.yaml`, listed at the end | 1 to 8, 17 to 58 | never scored |

## The held-out groups

A competing pair is **hard** when its joint prompt shows both named things on fewer than 4 of 8 seeds, **easy but unseen** otherwise; object-and-place pairs are **not competing** by kind. Animal pairs were counted with the validated instance count and their species checked by eye; object pairs were called by eye. Columns: seeds of 8 where the joint prompt shows both; seeds of 8 where the plain product shows two animals (instance count, animal pairs only).

| Group | Pair | Joint prompt | Plain product | How judged |
|---|---|---|---|---|
| hard | cow × horse | 2 | 0 | instance count and species by eye |
| hard | seal × walrus | 0 | 0 | instance count and species by eye |
| hard | suitcase × desk fan | 3 |  | by eye |
| easy but unseen | baboon × gazelle | 6 | 0 | instance count and species by eye |
| easy but unseen | bathtub × streetlamp | 8 |  | by eye |
| easy but unseen | bear × salmon | 8 | 0 | instance count and species by eye |
| easy but unseen | bear × squirrel | 8 | 1 | instance count and species by eye |
| easy but unseen | birdcage × watering can | 8 |  | by eye |
| easy but unseen | bison × horse (held out at training) | 6 | 0 | instance count and species by eye |
| easy but unseen | briefcase × ceramic bowl | 8 |  | by eye |
| easy but unseen | cat × horse | 7 | 0 | instance count and species by eye |
| easy but unseen | cat × lion | 7 | 0 | instance count and species by eye |
| easy but unseen | chessboard × lantern | 8 |  | by eye |
| easy but unseen | deer × hedgehog | 7 | 0 | instance count and species by eye |
| easy but unseen | dog × horse | 8 | 1 | instance count and species by eye |
| easy but unseen | drum set × snowman | 8 |  | by eye |
| easy but unseen | feather pillow × cast iron pan | 4 |  | by eye |
| easy but unseen | fox × wolf | 4 | 0 | instance count and species by eye |
| easy but unseen | giraffe × lion | 7 | 0 | instance count and species by eye |
| easy but unseen | goose × swan | 6 | 0 | instance count and species by eye |
| easy but unseen | lab microscope × hay bale | 8 |  | by eye |
| easy but unseen | lion × dog | 7 | 0 | instance count and species by eye |
| easy but unseen | lion × horse | 4 | 0 | instance count and species by eye |
| easy but unseen | microwave × potted plant | 8 |  | by eye |
| easy but unseen | mug × wine glass | 5 |  | by eye |
| easy but unseen | peacock × sheep | 7 | 2 | instance count and species by eye |
| easy but unseen | piano × candle (held out at training) | 8 |  | by eye |
| easy but unseen | raccoon × salmon | 8 | 2 | instance count and species by eye |
| easy but unseen | skateboard × helmet (held out at training) | 8 |  | by eye |
| easy but unseen | tiger × dog | 4 | 0 | instance count and species by eye |
| easy but unseen | wolf × horse | 6 | 0 | instance count and species by eye |
| easy but unseen | wolf × husky | 8 | 0 | instance count and species by eye |
| not competing | bookcase × glacier |  |  | by kind |
| not competing | butterfly × flower meadow |  |  | by kind |
| not competing | camel × desert landscape |  |  | by kind |
| not competing | candle × waterfall |  |  | by kind |
| not competing | deer × forest clearing |  |  | by kind |
| not competing | dolphin × ocewave |  |  | by kind |
| not competing | duck × pond |  |  | by kind |
| not competing | fire hydrant × snowfield |  |  | by kind |
| not competing | flamingo × lagoon |  |  | by kind |
| not competing | horse × grassy field |  |  | by kind |
| not competing | lamppost × desert dune |  |  | by kind |
| not competing | lighthouse × desert dune |  |  | by kind |
| not competing | lighthouse × ocewith stormy waves |  |  | by kind |
| not competing | mailbox × snowfield |  |  | by kind |
| not competing | park bench × sand dune |  |  | by kind |
| not competing | phone booth × tropical beach |  |  | by kind |
| not competing | picnic table × snowstorm |  |  | by kind |
| not competing | polar bear × iceberg |  |  | by kind |
| not competing | rowboat × cactus garden |  |  | by kind |
| not competing | sailboat × harbor |  |  | by kind |
| not competing | tent × snowy mountain |  |  | by kind |

Counts: hard 3, easy but unseen 29, not competing 21. The hard group is under the 8 the plan asked for; the joint prompt rarely fails on these candidates, and the count is stated as found.

## Left out, with the reason

- eagle × hawk, leopard × jaguar: two birds or two spotted cats on every seed, species not separable by eye.
- frog × toad: not rendered; frog and toad are on the subject guard's ask-first list and the pair ledger allows frog only for the benchmark prompts.
- the 10 style pairs (barn × pencil drawing style, bicycle × sketch style, cactus × mosaic style, and others): fit none of the three groups.

## The 48 training pairs

badger × magpie, bear × boar, bear × heron, beaver × raccoon, beaver × woodpecker, bicycle × birdhouse, birdcage × hay bale, bison × weasel, buffalo × hippo, bull × rooster, cactus × cooler box, camel × pelican, chicken × rabbit, coat rack × potted plant, deer × heron, deer × wolf, donkey × duck, donkey × rooster, giraffe × hyena, giraffe × zebra, giraffe × ostrich, gorilla × chimpanzee, guitar × traffic cone, hamster × carrot, hay bale × ukulele, hedgehog × robin, horse × pig, horse × sheep, langur × peacock, lighthouse × sailboat, lion × meerkat, lion × oryx, mailbox × pumpkin, microscope × test tube, moose × chipmunk, peacock × pig, pony × red barn, rocking chair × cactus, snail × lantern, soccer ball × park bench, suitcase × globe, teddy bear × birthday cake, telescope × beach ball, typewriter × cactus, windmill × tractor, accordion × scarecrow, emu × koala, impal× a jackal.
