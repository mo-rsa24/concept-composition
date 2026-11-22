# How each rival fails

This page records what each method draws well and where it breaks, as seen in the pictures the papers print. It comes
from 266 figures in 19 papers, 196 of them generated samples, and 297 rows of per-method findings. The
scoreboard's `failure` line for each rival is taken from here.

**Authors choose their own figures.** A method's successes are over-represented, and a failure its authors did not print
stays unseen. `no failure shown in this paper's figures` means exactly that, and says nothing about how the method
behaves elsewhere. A method drawn by a rival's authors may also be run less carefully than by its own.

**Two-object pictures use one ladder:** one thing, a fused hybrid, two bodies or heads, two clear things.

What our own runs show on our pairs is kept apart, in [the record of our runs](reads/figures/on-our-data.md), because
those are our numbers and never published ones. Each paper's tile-by-tile calls are in `reads/figures/<Paper>.observations.csv`.
[The reading instructions](reads/how-the-figures-were-read.md) say how a tile was judged.

## Rivals

### Liu

| Dataset | Method as shown | Succeeds | Fails | On which | Where | Printed by |
|---|---|---|---|---|---|---|
| 2D two disjoint datasets | Liu | no success shown in this paper's figures | Averaged vector fields put the samples in the empty gap between the two training sets. | panel b | Fig 2 p5 | SuperDiff |
| 3D objects | Liu | A chair without legs comes out on a single pedestal. | Four of five two-object prompts give one fused object, such as an avocado-shaped chair; the authors present these as hybrids. | avocado AND chair; boat AND couch; couch AND monitor; toilet AND chair | Fig 7 p19 | Liu |
| Attend-and-Excite prompts | Liu | No success is shown on these prompts. | It fuses the turtle with the mouse and the crown with the clock, and it loses the bow. | a turtle and a mouse; a elephant with a bow; a gray crown and a white clock | Fig 3 p8 | CO3 |
| Attend-and-Excite prompts | Liu | One mouse sits in a red car, and frogs and bears are clear when the other subject is dropped. | It fuses the two subjects into one creature or object on most prompts shown. | A cat and a dog; A lion and a rabbit; A horse and a monkey; A purple chair and a red bow; A purple balloon and a white clock; A frog and a pink bench | Fig 5 p5; Fig 20 p20; Fig 21 p21; Fig 22 p22; Fig 23 p23 | AttendExcite |
| Attend-and-Excite prompts on PixArt-Sigma | Liu | It draws a brown balloon with a pink car. | It fuses subjects into one object on most prompts shown. | a horse and a monkey; a lion and a yellow clock; a dog and a black apple; a dog and a bear; a purple car and a pink apple; a blue crown and a red balloon | Fig 6 p9; Fig 8 p22 | CO3 |
| CLEVR | Liu | Every tile shown places all requested objects, from 4 up to 8, as separate shapes. | no failure shown in this paper's figures | 4, 5, 6, 7 and 8 object positions | Fig 1 p2; Fig 4 p11; Fig 14 p29 | Liu |
| CelebA | Liu | Draws three clean unsmiling men at smile weight 0. | At any smile weight of 1 or more the faces break into patches with long hair (printed as Composed GLIDE). | male AND smiling, smile weights 1, 2 and 6 | Fig 7 p10; Fig 11 p26 | CoInD |
| CelebA with SDv1.5 fine-tuning | Liu | Draws two smiling men for the joint prompt smiling male. | Composed prompts draw women or an unsmiling face (printed as Composed GLIDE). | smiling AND male; smiling NOT female | Fig 13 p27 | CoInD |
| Colored MNIST | Liu | Draws a clear pink 4 on all three supports, and correct digits on uniform support for three of four prompts. | On non-uniform and partial data it draws avoided digits and colours or broken strokes (printed as Composed GLIDE). | 4 AND Cyan; 4 AND NOT (Pink OR Green); NOT (3 OR 4) AND Pink | Fig 1 p1; Fig 5 p8 | CoInD |
| FFHQ | Liu | Faces match the asked smile, glasses and sex in every combination shown. | no failure shown in this paper's figures; the sex of two child faces cannot be judged | five rows in Fig 15; Fig 1; Fig 8 | Fig 1 p2; Fig 8 p20; Fig 15 p30 | Liu |
| Free-form prompts shown by the authors | Liu | It draws a clear single subject. | It drops the dog and the cake. | A brown dog and a grey mouse; A red cake and a bouquet of yellow tulips | Fig 20 p20 | AttendExcite |
| Relational CLEVR | Liu | Both pairs of relations hold in the two Fig 5 prompts and the first Fig 1 prompt. | It adds objects nobody asked for: three extras in the second Fig 5 prompt and a yellow cube in the second Fig 1 prompt. | brown sphere prompt in Fig 5; yellow cylinder prompt in Fig 1 | Fig 1 p2; Fig 5 p12 | Liu |
| Shapes3D | Liu | Gets the object colour right in all three Figure 6 rows. | Draws the wrong shape or size and adds colour noise or blotches (printed as Composed GLIDE). | uniform AND; partial AND; partial NOT | Fig 6 p9 | CoInD |
| Stable Diffusion AND pairs | Liu | As run by the SuperDiff authors, most tiles carry both concepts on 6 of 20 first-6-seed pairs, usually fused. | 14 of 20 first-6-seed rows are mixed or failed, with tiles keeping one concept or breaking up, such as a destroyed brown mass for chair and avocado. | all 20 AND pairs; weakest on a pineapple and a beehive, zebra and barcode, teddy bear and panda | Fig 1 p2; Figs A5 to A24 p34 to p53 | SuperDiff |
| Text prompts | Liu | An object placed in a scene usually works, such as a camel in a forest, a horse in a flower field and a polar bear under stars. | Subjects get dropped or fused: bird and flower become one lilac bird, the dog in the couch prompt vanishes, and only 4 of 16 train-on-bridge samples show a clear train while 6 show none. | bird AND flower; couch AND dog; bus AND person; train on a bridge AND river; dog AND sky; bear AND red tree | Fig 1 p2; Fig 3 p10; Fig 6 p14; Fig 9 p21; Fig 10 p25; Fig 11 p26; Fig 12 p27; Fig 13 p28 | Liu |
| molecules: GSK3beta and JNK3 | Liu | Its two best molecules print nonzero scores on both targets. | Printed products are 0.211 and 0.194, and each molecule is weak on one target. | two best molecules | Fig A4 p32 | SuperDiff |
| molecules: GSK3beta and QED | Liu | Its five best molecules print nonzero scores on both properties, with products up to 0.250. | Every printed product, 0.215 to 0.250, is below every SuperDiff molecule shown. | five best molecules | Fig A3 p32 | SuperDiff |

### Du

| Dataset | Method as shown | Succeeds | Fails | On which | Where | Printed by |
|---|---|---|---|---|---|---|
| 2D densities | Du | Products, mixtures and negations land on the right clusters when the factors overlap. | When the two factors share no region it still places samples in blobs or a bar where the product has no mass. | lower times upper half ring; two vertical bars with no overlap | Fig 1 p2; Fig 2 p5; Fig A1 p17; Fig A2 p18; Fig A3 p18 | Du |
| CLEVR | Du | Three of five requested positions hold a clear separate shape in the one sample shown. | The two upper requested positions stay empty in that sample. | 5 positions (the RRR baseline, one sample) | Fig 2 p5 | DomainExpansion |
| CLEVR cube positions | Du | All requested boxes hold a separate clear shape at 1, 3 and 5 positions in the one row shown. | no failure shown in this paper's figures | 1, 3 and 5 positions | Fig 3 p6 | Du |
| ImageNet 128 px | Du | Pizza, daisy, golden retriever and valley are each recognisable in both tiles. | no failure shown in this paper's figures | Pizza; Daisy; Golden Retriever; Valley | Fig 1 p2; Fig 4 p7 | Du |
| Text prompts | Du | Horse settings, negated forest colour, two-fruit counts and regional tapestries mostly show what was asked. | Composed subjects can warp or merge with the scene: the Taj Mahal turns lumpy, one car dissolves into shrubs and a horse gains a tangled extra muzzle. | Taj Mahal AND Snow falling down; red car AND pink forest; horse AND grass plains; sandy beach NOT coastal water | Fig 1 p2; Fig 6 p8; Fig 7 p8; Fig 8 p8; Fig A4 p20; Fig A5 p21; Fig A6 p22; Fig A7 p23 | Du |

### SuperDiff

| Dataset | Method as shown | Succeeds | Fails | On which | Where | Printed by |
|---|---|---|---|---|---|---|
| 2D two disjoint datasets | SuperDiff | The mixture sampler covers all four training clusters, and the equal-density sampler lands on two lines between them. | no failure shown in this paper's figures | panels c and d | Fig 2 p5 | SuperDiff |
| Stable Diffusion AND pairs | SuperDiff | On 14 of 20 pairs in the uniform first-6-seed grids, most tiles carry both concepts, usually fused into one object. | One fused object is the usual result on 17 of 20 first-6-seed rows, and a dog and a cat give one animal in all 12 tiles. | all 20 AND pairs; weakest on a dog and a cat, a helicopter and a dragonfly, a waffle cone and a volcano, zebra and barcode | Fig 1 p2; Figs A5 to A24 p34 to p53 | SuperDiff |
| Stable Diffusion OR pairs | SuperDiff | Each tile shows one clear concept, and pineapple or beehive splits three and three. | On candy cane or flamingo the split is five flamingo tiles to one candy cane. | a candy cane or a flamingo | Fig A25 p54 | SuperDiff |
| molecules: GSK3beta and JNK3 | SuperDiff | Its two best molecules print products of 0.359 and 0.216. | no failure shown in this paper's figures | two best molecules | Fig A4 p32 | SuperDiff |
| molecules: GSK3beta and QED | SuperDiff | Its five best molecules print GSK3beta and QED products of 0.254 to 0.376. | no failure shown in this paper's figures | five best molecules | Fig A3 p32 | SuperDiff |

### FKC

| Dataset | Method as shown | Succeeds | Fails | On which | Where | Printed by |
|---|---|---|---|---|---|---|
| 2D Gaussians | FKC | With analytic scores, the composed samples form the round ground-truth cloud. | With learned scores, the composed samples form a tilted band away from the origin. | composition of the two elongated Gaussians at K=16 | Fig 1 p9 | Catastrophic |
| 2D mixture of four Gaussians | FKC | Samples concentrate on the four annealed cores and on the two product clusters. | no failure shown in this paper's figures | annealed and product targets | Fig 1 p1 | FKC |
| ImageNet-512 | FKC | The dog, bird, coffee pot and lemon are sharp and recognisable. | The lemon picture has small glassy fragments and coloured smears on the plate. | lemon column | Fig 3 p7 | FKC |
| SDXL two coloured objects | FKC | All three prompts show both objects in their named colours. | The TV gains two extra toothbrushes and a green screen, and the red bag gains green handles. | blue TV and green toothbrush; red handbag and green mouse | Fig A5 p39 | FKC |
| docking: ATP1A1 and CPT2 | FKC | Its two molecules print strong P2 scores near -9.72. | Its best P1 score, -8.412, is weaker than the -8.870 drawn without FKC. | two best molecules | Fig A8 p43 | FKC |
| docking: P41594 and P23921 | FKC | It prints the strongest score on both pockets at 19 and 27 atoms. | At 15, 23 and 35 atoms it is beaten on at least one pocket, by up to 0.553 docking units, and Fig 5's filter of QED at least 0.4 pairs its 35-atom molecule with 23-atom and 27-atom baselines. | molecule sizes 15 to 35 atoms | Fig 5 p8; Fig A6 p41 | FKC |
| mixture of 40 Gaussians | FKC | Target Score + FKC puts samples on every mode visible in the ground truth. | Tempered Noise + FKC still leaves the modes near (36, 35) and (-28, -37) empty. | Target Score + FKC and Tempered Noise + FKC | Fig 2 p7 | FKC |
| rooms: couch and painting | FKC | In distribution, 4 of 4 and 9 of 10 rooms show both objects. | Out of distribution, 3 of 4 and 6 of 10 rooms show both, with bent couches and missing paintings. | panels a and c of Figs 2 and 4 at K=16 | Fig 2 p12; Fig 4 p33 | Catastrophic |
| rooms: couch and painting (separate models) | FKC | Couches are recognisable in seven of eight rooms. | No room of eight shows a painting, and one out-of-distribution tile is destroyed into noise. | panels a and c of Fig 3 at K=16 | Fig 3 p28 | Catastrophic |
| rooms: couch and table | FKC | In distribution, all 14 rooms show a couch and a table. | Out of distribution, none of 14 rooms shows a separate couch and table. | panels b and d of Figs 2 and 4 at K=16 | Fig 2 p12; Fig 4 p33 | Catastrophic |
| rooms: couch and table (separate models) | FKC | Couches appear in most rooms. | No room of eight shows a table, couches warp, and one out-of-distribution tile is destroyed into noise. | panels b and d of Fig 3 at K=16 | Fig 3 p28 | Catastrophic |

### Bradley-Projective

| Dataset | Method as shown | Succeeds | Fails | On which | Where | Printed by |
|---|---|---|---|---|---|---|
| CLEVR colour-conditioned | Bradley-Projective (empty background) | Composes yellow with blue in 15 of 16 samples. | Most colour pairs give one object or a blended colour, and even the marked successes shift colour. | yellow or red with blue, brown, cyan, gray, green and magenta | Fig 5 p5; Fig 11 p19 | Bradley-Projective |
| CLEVR colour-conditioned | Bradley-Projective (learned background) | Draws one object of the asked colour. | Asked for many colours it still draws about one to three objects, often single cubes in patches of two colours. | 2 to 8 composed colours | Fig 16 p38 | Bradley-Projective |
| CLEVR location-conditioned | Bradley-Projective (cluttered background) | Draws clean separate objects up to five locations. | At nine locations objects overlap and break up with bright artefacts. | 6 to 9 composed locations | Fig 3 p3 | Bradley-Projective |
| CLEVR location-conditioned | Bradley-Projective (empty background) | Draws the asked number of clean, separate objects up to four locations. | From seven locations on, objects melt into blobs of mixed colour. | 1 to 9 composed locations | Fig 2 p1; Fig 3 p3; Fig 10 p18 | Bradley-Projective |
| CLEVR location-conditioned | Bradley-Projective (learned background) | Places separate objects on the red dots up to four locations, and mostly up to six. | At nine and ten locations large objects pile through each other and hide dots. | 7 to 10 composed locations | Fig 16 p38 | Bradley-Projective |
| Synthetic Gaussian mixtures | Bradley-Projective (empty background) | Lights indices 0, 2, 4 and 6 in clean stripes. | No failure shown in this paper's figures. | compose indices 0, 2, 4, 6 | Fig 12 p23 | Bradley-Projective |
| Synthetic coloured squares | Bradley-Projective (empty background) | Draws both squares for red + blue on orthogonal conditions in all six samples. | Red + yellow gives one yellow square in all six samples, and some other tiles merge into one blended square. | red + yellow; red + green; green + blue | Fig 14 p29 | Bradley-Projective |

### Bradley-Local

| Dataset | Method as shown | Succeeds | Fails | On which | Where | Printed by |
|---|---|---|---|---|---|---|
| CLEVR location-conditioned | Bradley-Local (Experiment 2L) | Fills the conditioned dots at every count shown, including nine. | Adds objects at unasked places at low counts and piles objects through each other at high counts. | 0 to 10 conditioned locations | Fig 1 p2; Fig 8 p23; Fig 9 p24 | Bradley-Local |
| CLEVR location-conditioned | Bradley-Local (Experiment 3L) | Keeps adding objects as dots increase, up to ten dots. | Many objects sit beside their dots rather than on them, and scenes crowd at high counts. | 0 to 10 conditioned locations | Fig 13 p30 | Bradley-Local |
| CLEVR location-conditioned | Bradley-Local (LCS architecture trained on 1 object) | Places separate objects on up to four dots after training on single objects. | From five dots on, objects melt into merged blobs of mixed colour. | 5 to 10 conditioned locations | Fig 8 p23 | Bradley-Local |

### CoInD

| Dataset | Method as shown | Succeeds | Fails | On which | Where | Printed by |
|---|---|---|---|---|---|---|
| 2D Gaussian mixture | CoInD | Recovers all four modes, including the unseen one. | No failure shown in this paper's figures. | unseen mode at (+1, +1) | Fig 14 p29 | CoInD |
| CelebA | CoInD | Keeps three male faces intact while the smile widens from weight 0 to 6. | Its own failures include a shout in place of a smile and a barely visible smile on a warped face. | male AND smiling | Fig 7 p10; Fig 11 p26; Fig 16 p32 | CoInD |
| CelebA with SDv1.5 fine-tuning | CoInD | Draws two smiling men for all three prompt forms. | No failure shown in this paper's figures. | smiling male; smiling AND male; smiling NOT female | Fig 13 p27 | CoInD |
| Colored MNIST | CoInD | Draws the right digit and colour in every Figure 1 grid cell, including on partial support. | Its own failure panel shows wrong colours, forbidden digits and colours, and broken digits. | 7 AND Dark Green; 9 NOT Cream; Yellow NOT (2 OR 3); 4 AND Cream; 8 NOT (Cream OR Blue) | Fig 1 p1; Fig 5 p8; Fig 15 p31; Fig 16 p32 | CoInD |
| Shapes3D | CoInD | Matches the expected image in all three Figure 6 comparisons. | Its own failure panel shows a wrong wall colour, a sphere for a cylinder and a dissolved object. | wall colour mismatch; NOT composition; blue cylinder | Fig 6 p9; Fig 16 p32; Fig 18 p33 | CoInD |

### MultiDiffusion

| Dataset | Method as shown | Succeeds | Fails | On which | Where | Printed by |
|---|---|---|---|---|---|---|
| COCO validation | MultiDiffusion | Two bears and the separate objects in most layouts are clear. | It adds or swaps content: a second cat appears, a Longhorn becomes a patched dairy cow, and a dog's head overlaps its dish. | cat beside a laptop; Texas Longhorn; dog near a bowl; cat on a green cushion; stuffed bears on a bed | Fig 13 p15 | MultiDiffusion |
| Panoramas | MultiDiffusion | Long strips are continuous, with no visible join on any of the panoramas shown. | With no region mask the prompt repeats down a tall image, so one waterfall scene holds three separate waterfalls. | photo of a waterfall in a forest; anime and cartoon landscapes | Fig 3 p4; Fig 4 p6; Fig 9 p8; Fig 10 p13; Fig 12 p15 | MultiDiffusion |
| Region prompts | MultiDiffusion | With a mask per object, separate things appear in their regions, including elephant and cow, mouse on books, and squirrel with sign. | Objects can gain or lose detail: an unrequested rider appears, a robot dog has a box for a head, and a robot becomes a camera-like device. | dolomites robot and horse; moon scene; meadow elephant and cow | Fig 1 p2; Fig 5 p7; Fig 6 p7; Fig 7 p8; Fig 8 p8; Fig 9 p8; Fig 11 p14; Fig 12 p15; Fig 14 p16 | MultiDiffusion |
| Region prompts | MultiDiffusion without bootstrapping | A glass appears on the table. | The glass is a small tumbler that does not fill the tall stemmed mask. | glass of sparkling wine on a wooden table | Fig 6 p7 | MultiDiffusion |
| Satellite maps | MultiDiffusion | It produces a plausible blurred aerial terrain in the one sample shown, with no map printed to judge it against. | no failure shown in this paper's figures | one sample in Fig 3 | Fig 3 p5 | DomainExpansion |

### PiCo

| Dataset | Method as shown | Succeeds | Fails | On which | Where | Printed by |
|---|---|---|---|---|---|---|
| GPT-4o complex prompts | PiCo | Shows every asked-for object on 4 of the 5 complex prompts. | On blue candle, red shoes and a cat, the candle stands inside the shoe and the cat's body merges into it. | a blue candle and a red shoes and a cat | Fig 17 p15 | PiCo |
| PiCo conflict-elimination ablation | PiCo | Draws a realistic white dog in yellow sunglasses. | no failure shown in this paper's figures | none of the prompts shown: a white dog wearing a yellow sunglasses | Fig 8 p8 | PiCo |
| PiCo conflict-elimination ablation | PiCo without conflict elimination | Keeps the sunglasses on the dog. | The dog's body turns into a white cloth shape. | a white dog wearing a yellow sunglasses | Fig 8 p8 | PiCo |
| PiCo control-duration ablation | PiCo with control until step 15 | Separates the apple and banana. | Draws two brown dogs when one was asked for. | A green bench and a brown dog | Fig 3 p6 | PiCo |
| PiCo control-duration ablation | PiCo with control until step 25 | Separates apple and banana, and one brown dog from the bench. | no failure shown in this paper's figures | none of the prompts shown: A green apple and a yellow banana; A green bench and a brown dog | Fig 3 p6 | PiCo |
| PiCo control-duration ablation | PiCo with control until step 5 | Shows a dog in front of a green bench. | Draws two apples and no banana, and the dog is mostly white. | A green apple and a yellow banana; A green bench and a brown dog | Fig 3 p6 | PiCo |
| PiCo control-duration ablation | PiCo with control until step 50 | Keeps the apple, banana and bench apart. | The banana gets dark streaks and the dog degrades into a brown shape. | A green apple and a yellow banana; A green bench and a brown dog | Fig 3 p6 | PiCo |
| PiCo control-stage example | PiCo | Separates a sandwich from a bagel, and a pink cat from a sheep. | no failure shown in this paper's figures | none of the prompts shown: Bagel and sandwich (prompt not printed in the figure); Pink cat and brown sheep (prompt not printed in the figure) | Fig 10 p11; Fig 11 p11 | PiCo |
| PiCo control-stage example | PiCo with mask control from t=25 | Changes small local details of the uncontrolled image. | Keeps the fused bread and two sheep with no cat, and smears a sheep's face. | Bagel and sandwich (prompt not printed in the figure); Pink cat and brown sheep (prompt not printed in the figure) | Fig 10 p11; Fig 11 p11 | PiCo |
| PiCo control-stage example | PiCo with mask control from t=38 | Changes crust, sauce and wool details. | Keeps the fused bread and two sheep with no cat. | Bagel and sandwich (prompt not printed in the figure); Pink cat and brown sheep (prompt not printed in the figure) | Fig 10 p11; Fig 11 p11 | PiCo |
| PiCo cross-attention examples | PiCo | Separates the asked-for objects on all five prompts. | no failure shown in this paper's figures | none of the prompts shown: A pink cat and a brown sheep; A yellow cow and a brown dog; A blue backpack and a red car; A purple balloon floating above a red picnic blanket with yellow flowers; A turquoise chair next to a wooden table with a white vase and pink roses | Fig 18 p16 | PiCo |
| PiCo mask-strength ablation | PiCo with mask strength 1 | Keeps the same objects as the uncontrolled image. | The apple stays orange, extra birds appear, and no bear is blue. | A brown bird and a green apple; A blue bear and a brown boat | Fig 4 p6 | PiCo |
| PiCo mask-strength ablation | PiCo with mask strength 15 | Draws a brown bird and green apple, and one blue bear in a brown boat. | Adds an unrequested wooden block at the bow of the boat. | A brown bird and a green apple; A blue bear and a brown boat | Fig 4 p6 | PiCo |
| PiCo mask-strength ablation | PiCo with mask strength 50 | Keeps a recognisable boat. | Collapses bird and apple into one blob and dissolves the bear into speckles. | A brown bird and a green apple; A blue bear and a brown boat | Fig 4 p6 | PiCo |
| PiCo mask-type ablation | PiCo | Draws a blue apple and a separate green vase. | no failure shown in this paper's figures | none of the prompts shown: A blue apple and a green vase | Fig 7 p8 | PiCo |
| PiCo mask-type ablation | PiCo without the concept mask | Draws a green apple and a blue bowl. | Swaps the colours and draws no vase. | A blue apple and a green vase | Fig 7 p8 | PiCo |
| PiCo mask-type ablation | PiCo without the exclusive mask | Keeps the blue apple and green vase. | Adds a small cup nobody asked for. | A blue apple and a green vase | Fig 7 p8 | PiCo |
| PiCo mask-validation example | PiCo | Draws a blue cat sitting on a dark suitcase. | no failure shown in this paper's figures | none of the prompts shown: blue cat and brown suitcase (the two concept labels printed) | Fig 9 p9 | PiCo |
| PiCo mask-validation example | PiCo without mask validation | Draws a blue cat. | Leaves scraps at its feet and draws no suitcase. | blue cat and brown suitcase (the two concept labels printed) | Fig 9 p9 | PiCo |
| PiCo noise-score ablation | PiCo noise selection without the ITM score | Keeps a blue cat and yellow bowl. | Adds a yellow jug, and draws a red cup with the blue object cut by the frame. | A yellow bowl and a blue cat; A blue cup and a red vase | Fig 6 p8 | PiCo |
| PiCo noise-score ablation | PiCo noise selection without the concept score | Keeps the yellow plate and a blue bowl. | The cat is mostly grey-white and the red vase becomes a round form on a stem. | A yellow bowl and a blue cat; A blue cup and a red vase | Fig 6 p8 | PiCo |
| PiCo noise-score ablation | PiCo noise selection, both scores | Picks seeds showing a blue cat in a yellow bowl and a blue cup by a red vase. | Adds a small blue pot nobody asked for beside the cat. | A yellow bowl and a blue cat; A blue cup and a red vase | Fig 6 p8 | PiCo |
| PiCo noise-selection examples | PiCo noise selection, chosen seeds | Chosen seeds show both objects with the asked-for colours on all 4 prompts. | no failure shown in this paper's figures | none of the prompts shown: A brown cat and a gold clock; A blue cup and a red vase; A blue apple and a green vase; A red chair and a brown elephant | Fig 14 p13 | PiCo |
| PiCo noise-selection examples | PiCo noise selection, rejected seeds | Some rejected seeds are clean. | Others swap colours, drop an object, or put the cat inside the clock. | A brown cat and a gold clock; A blue cup and a red vase; A blue apple and a green vase; A red chair and a brown elephant | Fig 14 p13 | PiCo |
| PiCo noise-selection step count | PiCo noise selection after 1 step | No picture at this setting shows an object. | All three previews are coloured noise. | A green apple and a brown bird | Fig 15 p13 | PiCo |
| PiCo noise-selection step count | PiCo noise selection after 20 steps | The preview shows a green apple and a brown bird. | no failure shown in this paper's figures | none of the prompts shown: A green apple and a brown bird | Fig 15 p13 | PiCo |
| PiCo noise-selection step count | PiCo noise selection after 5 steps | The preview shows an apple and a bird apart. | no failure shown in this paper's figures | none of the prompts shown: A green apple and a brown bird | Fig 15 p13 | PiCo |
| SDXL, two prompts | PiCo | Separates a green apple from a mostly brown bird. | The plastic bottle comes out small and metallic. | a plastic bottle and a leather jacket | Fig 13 p12 | PiCo |
| T2I-CompBench | PiCo | Shows both objects with the asked-for attributes on 9 of the 10 prompts. | On a small bagel and a rectangular sandwich, one tile draws a bun with no hole, cut by the frame. | a small bagel and a rectangular sandwich | Fig 2 p4; Fig 16 p14 | PiCo |

### FreeFuse

| Dataset | Method as shown | Succeeds | Fails | On which | Where | Printed by |
|---|---|---|---|---|---|---|
| FreeFuse benchmark on FLUX.1-dev | FreeFuse | Keeps two to six reference subjects separate and recognisable in most tiles. | A snitch sits on a flat white disc pasted over the background, and one character's face is hidden. | Naruto reaching for the golden snitch; Leela, Greggs, Nezha and Ran | Fig 1 p2; Fig 7 p11; Fig 8 p12; Fig 16 p23 | FreeFuse |
| FreeFuse benchmark on FLUX.1-dev | FreeFuse with ControlNet | Two characters follow the two-person pose. | no failure shown in this paper's figures | none of the prompts shown: Two anime characters following a two-person pose | Fig 1 p2 | FreeFuse |
| FreeFuse benchmark on FLUX.1-dev | FreeFuse with IP-Adapter | Naruto is large and clear on the city background. | Alice is too small to confirm her likeness. | Naruto and Alice on a city background | Fig 1 p2 | FreeFuse |
| FreeFuse benchmark on FLUX.1-dev | FreeFuse with Redux | Conan and Pikachu stay separate on the style image. | no failure shown in this paper's figures | none of the prompts shown: Conan and Pikachu on a wave style image | Fig 1 p2 | FreeFuse |
| FreeFuse benchmark on FLUX.1-dev | FreeFuse with Style LoRA | Aobing and Nezha stay separate in the painted style. | no failure shown in this paper's figures | none of the prompts shown: Aobing and Nezha in a swirling night style | Fig 1 p2 | FreeFuse |
| Illustrious-XL anime characters | FreeFuse | Each of the 10 anime tiles shows two separate characters. | no failure shown in this paper's figures | none of the prompts shown: First row: five two-character anime prompts; Second row: five two-character anime prompts | Fig 12 p19 | FreeFuse |
| SDXL, one prompt | FreeFuse | Harry and Daiyu are distinct and keep their costumes. | no failure shown in this paper's figures | none of the prompts shown: Harry and Daiyu, faces close, autumn leaves | Fig 11 p19 | FreeFuse |

### DomainExpansion

| Dataset | Method as shown | Succeeds | Fails | On which | Where | Printed by |
|---|---|---|---|---|---|---|
| CLEVR | DomainExpansion | At 2 positions all 8 crosses are filled, and at 3 positions 26 of 27 are filled. | At 5 positions only 2 of 9 shown samples fill every cross (33 of 45 filled), and one sample fills only 1 of 5. | 5 positions in Fig B.2; one 3-position tile | Fig 2 p5; Fig B.1 p13; Fig B.2 p14 | DomainExpansion |
| Satellite maps | DomainExpansion | Roads, street grids and building blocks follow the map in most samples at 96, 128 and 256 px. | One green square on the map is drawn as a bare grey lot where the real image has a small park, and one park sample gains a bright blue patch the map lacks. | 128 px park map; 256 px grid with green square | Fig 3 p5; Fig B.5 p17; Fig B.6 p17; Fig B.7 p17 | DomainExpansion |

### Sculpting

| Dataset | Method as shown | Succeeds | Fails | On which | Where | Printed by |
|---|---|---|---|---|---|---|
| 2D grid (GFlowNet) | Sculpting | Harmonic means and contrasts move mass onto the named overlaps and away from the other model's region. | no failure shown in this paper's figures | two-model and three-model operations | Fig 3 p12 | Sculpting |
| MNIST digits 0-5 and 4-9 | Sculpting | The contrast toward digits 0-5 gives legible 0s to 3s. | The contrast toward 4-9 leaves at least 6 blobs and at least 4 stray 4s, and the harmonic mean admits a 9 and a 6. | harmonic mean and contrast toward 4-9 | Fig F.2 p31 | Sculpting |
| colored MNIST (chained operations) | Sculpting | Two of the three orderings of the three-way harmonic mean give 59 and 54 clean 0s of 64. | The ordering p1 with (p2 with p3) gives only 30 clean 0s of 64, though all three orderings should be the same distribution. | three-way harmonic mean in Fig F.5 | Fig F.5 p34 | Sculpting |
| colored MNIST (even odd and multiples of three) | Sculpting | Most composed grids show the expected digit sets and colours. | The even and multiples-of-three harmonic mean has three blank tiles and one faint ring, and contrasts admit stray blue 3s. | harmonic mean in Fig F.3; contrasts in Fig F.4 | Fig F.3 p32; Fig F.4 p32 | Sculpting |
| colored MNIST (three models) | Sculpting | Pairwise conditionals give the expected single digit and colour, with at most one stray in 64 tiles. | The y=1,2,3 grid has more than 20 of 64 open or broken zeros, and y=1,2,1 has thick lumpy strokes. | y=1,2,3 and y=1,2,1 | Fig 6 p14 | Sculpting |
| colored MNIST (two models) | Sculpting | The harmonic mean gives cyan 0s, and the contrasts give beige 0s and cyan 1s. | One beige contrast tile at row 6 column 8 reads closer to a 6. | contrast p1 with p2 | Fig F.1 p31 | Sculpting |
| molecules (GFlowNet) | Sculpting | The densest harmonic-mean contours sit where both rewards are high, and the contrasts stay apart. | At beta 32 the harmonic mean's outer contour stretches to SEH near 0.1, far from the high-SEH region. | harmonic mean at beta 32 | Fig 4 p12 | Sculpting |

### Catastrophic

This paper analyses composition and proposes no sampler of its own. What its figures show of FKC is under FKC, and its other methods are under "Other methods".

### TestTime

| Dataset | Method as shown | Succeeds | Fails | On which | Where | Printed by |
|---|---|---|---|---|---|---|
| CelebA | TestTime (LoRA) | Keeps the query face and rough attributes. | Faces blur and tints shift, and hair colour changes between samples. | all four CelebA queries | Fig 3 p8; Fig 5 p17 | TestTime |
| CelebA | TestTime (PoE) | Keeps the query face, smile and main hair length. | Samples are near-identical to each other with an orange cast, and gray-haired men barely smile. | all four CelebA queries | Fig 3 p8; Fig 5 p17 | TestTime |
| CelebA-HQ | TestTime (mode ascent) | Keeps the face and pose. | Skin is smoothed and the colours shift. | two face images | Fig 2 p4 | TestTime |
| ColorMNIST | TestTime (LoRA) | Keeps the digit and background on held-out 0, 2 and 7 compositions. | Strokes fade or turn ragged, and some cyan 2s turn green or green-teal. | digit 0 green on red; digit 2 cyan on brown; digit 7 pink on navy | Fig 3 p8; Fig 5 p17 | TestTime |
| ColorMNIST | TestTime (PoE) | Keeps the digit and background on 0 and 7 held-out compositions. | Two of five cyan 2s turn green, and a white background collapses all four samples into noise. | digit 2 cyan on brown; pink 3 on white; pink 3 on black | Fig 3 p8; Fig 5 p17; Fig 6 p18 | TestTime |
| ColorMNIST | TestTime (mode ascent) | Finds a pink 3 on brown among prototypes for the black-background query. | For a white background it finds a block, a 6 and a 0 on purple, with no white. | pink 3 on black; pink 3 on white | Fig 6 p18 | TestTime |
| Fashion-MNIST | TestTime (mode ascent) | Keeps garments recognisable at low noise. | At noise level 500 tiles become faint gray silhouettes. | noise levels 50 to 500 | Fig 4 p13 | TestTime |
| LSUN Churches | TestTime (mode ascent) | Keeps the outline of the church. | Colours turn flat and saturated and fine detail is lost. | two church images | Fig 2 p4 | TestTime |

### CO3

| Dataset | Method as shown | Succeeds | Fails | On which | Where | Printed by |
|---|---|---|---|---|---|---|
| Attend-and-Excite prompts | CO3 | It draws two separate recognisable subjects on most animal and animal-object prompts shown. | It fuses a brown seat into a white bench, adds extra objects and moves colors across objects on some prompts. | a brown chair and a white bench; a orange chair and a blue clock; a dog and a black apple; a blue crown and a red balloon; a gray crown and a white clock | Fig 3 p8; Fig 9 p23 | CO3 |
| Attend-and-Excite prompts on PixArt-Sigma | CO3 | It separates horse and monkey, bird and turtle, and lion and yellow clock. | It drops the crown, the car and the apple, turns the dog into a bear, and gives the apple the wrong color. | a blue crown and a red balloon; a turtle and a yellow car; a purple car and a pink apple; a dog and a bear; a dog and a black apple; a brown balloon and a pink car | Fig 6 p9; Fig 8 p22 | CO3 |
| Attend-and-Excite prompts with backgrounds added | CO3 | It keeps both subjects apart on all six prompts and adds the asked-for background. | It adds a second clock to the living room and a tiny extra animal beside the horse. | a lion and a yellow clock inside a living room; a horse and a monkey | Fig 7 p19 | CO3 |
| RareBench | CO3 | It keeps the main subjects of most rare prompts apart, including alligator with banana bottle and turtle with owl. | It misses rare attributes, leaves the lemon hornless, merges octopus with a bear face and gives a bird the lion's spots. | a hairless yak and a horned lemon; a horned pelican and a spotted lion; A hairy octopus; A thorny kangaroo; A giant spider | Fig 4 p8; Fig 11 p25 | CO3 |
| T2I-CompBench Complex | CO3 | It separates the named objects in most scenes and draws the batter hitting the ball. | It places the glass on the coaster rather than next to it and draws a floor flame instead of a torch. | The square coaster; The flickering torch | Fig 4 p8; Fig 10 p24 | CO3 |

### ToMe

| Dataset | Method as shown | Succeeds | Fails | On which | Where | Printed by |
|---|---|---|---|---|---|---|
| Attend-and-Excite prompts | ToMe | It draws a clear mouse on a clear turtle. | It drops the bow, adds a second turtle and leaves the crown as a clock finial. | a turtle and a mouse; a elephant with a bow; a gray crown and a white clock | Fig 3 p8 | CO3 |
| Binding prompts shown by the authors (set not stated) | ToMe | It binds each accessory to the right subject on most animal and person pairs and keeps the two subjects apart. | It sometimes changes the object asked for, such as hat to hood, sheep to goat, or loses earrings, and it adds a yellow scarf to the lion. | A bear with hat and a man with glasses; A cat with pink hat and a dog with blue sunglasses; A man wearing scarf and a girl wearing earrings; A lion with yellow crown and a sheep with white bandanas; A squirrel holding guns and a bear with hat | Fig 1 p2; Fig 5 p8; Fig 6 p9; Fig 12 p19; Fig 13 p19; Fig 14 p20 | ToMe |
| Binding prompts shown by the authors (set not stated) | ToMe with spliced prompts | It draws one clear animal with an accessory. | It loses the second subject in all three pictures. | a cat wearing glasses and a dog with hat | Fig 13 p19 | ToMe |
| Binding prompts shown by the authors (set not stated) | ToMe without end token substitution | It puts sunglasses on the corgi. | The boy also wears sunglasses. | a boy wearing hat and a dog wearing sunglasses | Fig 12 p19 | ToMe |
| RareBench | ToMe | It puts wool on the shark. | It drops the lion, octopus, owl and bottle and swaps the unicycle for a bicycle. | a horned pelican and a spotted lion; A hairy octopus; A knighted turtle; A wooly alligator; A thorny kangaroo | Fig 4 p8; Fig 11 p25 | CO3 |
| T2I-CompBench Complex | ToMe | It draws both objects in several scenes. | It turns object pairs into repeated wallpaper patterns and misreads crack as a crack in the ground. | The yellow pencil; The red apple; The colorful balloons; The soft fur coat; The sharp crisp crack | Fig 4 p8; Fig 10 p24 | CO3 |
| Token edit examples | ToMe token edit | It adds glasses, removes earrings and glasses, and turns the nurse male. | Removing the crown leaves a tall black headdress in its place, and removing earrings changes the background. | [queen-crown] with cats; girl with earrings removed | Fig 8 p9 | ToMe |

### AttendExcite

| Dataset | Method as shown | Succeeds | Fails | On which | Where | Printed by |
|---|---|---|---|---|---|---|
| Attend-and-Excite prompts | AttendExcite | It draws the elephant with a bow and shows both crown and clock. | It loses the mouse to a dark blur and makes the gray crown near black. | a turtle and a mouse; a gray crown and a white clock | Fig 3 p8 | CO3 |
| Attend-and-Excite prompts | AttendExcite | It shows both subjects as separate things on nearly every prompt, including the pairs that the other methods fuse. | Its own limitations figure draws a dog alone, a dog with a plush toy, and a bear head on a dog body, and four of eight yellow chairs are unrecognisable strips. | A dog and a bear; A bird and a yellow chair; A brown suitcase and a white bench; A yellow bow and a brown bench; A horse with a bow | Fig 1 p1; Fig 2 p2; Fig 4 p4; Fig 5 p5; Fig 9 p8; Fig 10 p11; Fig 11 p12; Fig 12 p12; Fig 13 p13; Fig 15 p16; Fig 16 p17; Fig 20 p20; Fig 21 p21; Fig 22 p22; Fig 23 p23; Fig 24 p24 | AttendExcite |
| Attend-and-Excite prompts | AttendExcite modifying all steps | Both subjects remain present in all pictures. | Updating at every step smears faces and leaves blotchy texture on objects. | A turtle and a yellow bowl; A cat and a dog; Yellow glasses and a gray bowl; A purple chair and a red bow | Fig 12 p12 | AttendExcite |
| Attend-and-Excite prompts | AttendExcite without Gaussian smoothing | It draws crowned frogs and some separate dogs and horses. | It loses the rabbit's crown, the dog in Fig 10, and the backpack, and it splits or pastes subjects. | A rabbit with a crown; A dog and squirrel; A gray backpack and yellow glasses; A dog and a horse; A dog and a squirrel | Fig 10 p11; Fig 11 p12 | AttendExcite |
| Attend-and-Excite prompts | AttendExcite without iterative refinement | It shows some clear pairs, such as a white dog beside a horse. | It drops crowns and fuses dog with horse and dog with squirrel. | A frog with a crown; A dog and a horse; A dog and a squirrel; A gray backpack and yellow glasses | Fig 11 p12 | AttendExcite |
| Free-form prompts shown by the authors | AttendExcite | It adds the missing subject or setting on nearly all free-form prompts, such as the salmon, library and kitchen. | A cat and dog share one jacket in a painting, and only one bird of four stands on the camera. | A painting of a cat and a dog in the style of Van Gogh; A bird perched on top of a vintage camera; An elephant with a sombrero | Fig 1 p1; Fig 6 p6; Fig 9 p8; Fig 17 p18; Fig 18 p19; Fig 20 p20 | AttendExcite |
| GPT-4o complex prompts | AttendExcite | Draws the cake with roses on a plate. | One tile drops the cat, one drops the shoes, and the frisbee is the wrong colour in both. | A black cat sitting on a wooden fence next to an orange pumpkin; A fuchsia umbrella shading a white bench next to a gray stone wall; A large brown dog chasing a yellow frisbee through a back yard; a blue candle and a red shoes and a cat | Fig 17 p15 | PiCo |
| StructureDiffusion prompts | AttendExcite | It draws the red sphere with blue cube, the egg over rice, and the black cord on the rug. | One elephant picture has no wall, the stone blocks do not read as benches, and one bench top looks like wood. | A cat is perched upon a stone bench; Two elephants walking by a green wall; A wooden building with a stone bench | Fig 7 p6; Fig 19 p19 | AttendExcite |
| T2I-CompBench | AttendExcite | Draws both objects on lighthouse and bench and on toy and bottle. | Loses the cactus, puts the wrong colours on apple and vase, and tangles the squirrel with the bird. | A blue apple and a green vase; a plastic bottle and a leather jacket; a small bagel and a rectangular sandwich; a squirrel and a bird and a yellow candle; A pink cat and a brown sheep; a green banana and a blue cake; rubber gloves and a metallic spoon; a tall palm tree and a short cactus | Fig 2 p4; Fig 16 p14 | PiCo |

### FlipConcept

| Dataset | Method as shown | Succeeds | Fails | On which | Where | Printed by |
|---|---|---|---|---|---|---|
| Custom Concept 101 and generated images | FlipConcept | All asked-for subjects are present, separate and close to their references on all 4 prompts. | no failure shown in this paper's figures | none of the prompts shown: A cat and a woman hugging each other, lighthouse background; A cat and a dog playing with a ball, mountain background; A plush teddybear and a plush panda playing with a ball, beach background; A guitar placed next to a cup, bar background; A man in the snow next to a big dog resting its front legs on a vintage suitcase, snow-covered house | Fig 1 p2; Fig 3 p5; Fig 4 p5 | FlipConcept |

### CFGRevisited

This paper re-evaluates guidance methods and proposes none of its own. Its figures of CFG, CFG++, APG and the rest are under "Other methods".

## Other methods, by the paper that printed them

### Printed by Liu

| Dataset | Method as shown | Succeeds | Fails | On which | Where | Printed by |
|---|---|---|---|---|---|---|
| Relational CLEVR | DALL-E 2 | It draws one sharp sphere in each picture. | It leaves the CLEVR scene and draws one sphere and a large slab, so the requested relations are absent. | both prompts in Fig 5 | Fig 5 p12 | Liu |
| CLEVR | EBM | All five positions are labelled in the two five-object layouts. | It adds unlabelled objects at five positions and labels only five and four of seven. | all four layouts in Fig 4 | Fig 4 p11 | Liu |
| FFHQ | EBM | Dark eyewear appears on the faces. | Faces are heavily smeared, so sex and expression cannot be read reliably. | NOT Smiling AND Glasses AND NOT Male | Fig 8 p20 | Liu |
| Relational CLEVR | EBM | Both arrangements are right. | Shapes are lumpy, and the cyan cylinder reads as a cube. | blue cube prompt in Fig 5 | Fig 5 p12 | Liu |
| Relational CLEVR | GLIDE | It draws sharp, clean shapes. | It repeats shapes and swaps types, with a yellow cube for the sphere and no gray cube. | both prompts in Fig 5 | Fig 5 p12 | Liu |
| Text prompts | GLIDE | The first-named subject is drawn cleanly, and dog and sky appear as two separate things. | Given one long sentence it drops a later object or puts a colour on the wrong thing: no polar bear, no aurora, no bungalows, no car, and a red tree where green was asked. | all six prompts in Fig 3; bear AND red tree | Fig 3 p10; Fig 9 p21 | Liu |
| CLEVR | LACE | It draws clean separate shapes. | It draws only two or three objects when five or seven are asked. | all four layouts in Fig 4 | Fig 4 p11 | Liu |
| FFHQ | LACE | All four faces are sharp, wear glasses and do not smile. | Two of the four are men although the prompt excludes men. | NOT Smiling AND Glasses AND NOT Male | Fig 8 p20 | Liu |
| Relational CLEVR | LACE | It draws clean blocks. | Neither prompt's objects or relations appear. | both prompts in Fig 5 | Fig 5 p12 | Liu |
| CLEVR | StyleGAN2 | It draws clean separate shapes. | It draws one to four objects at five requested positions or three labelled of seven. | all four layouts in Fig 4 | Fig 4 p11 | Liu |
| FFHQ | StyleGAN2 | Sharp women's faces appear. | No face wears glasses and two smile broadly. | NOT Smiling AND Glasses AND NOT Male | Fig 8 p20 | Liu |
| Relational CLEVR | StyleGAN2 | It draws clean shapes. | The requested objects are missing, so neither relation is drawn. | both prompts in Fig 5 | Fig 5 p12 | Liu |

### Printed by Du

| Dataset | Method as shown | Succeeds | Fails | On which | Where | Printed by |
|---|---|---|---|---|---|---|
| 2D densities | Reverse Diffusion | It finds the rough outline of a ring of clusters in a mixture. | Products keep mass along the ring, mixtures smear together and negation leaves the removed region filled. The paper states this baseline is the approach of Liu (p6). | ring times bar; three-cluster mixture; two-bar mixture; both negations; both disjoint products | Fig 2 p5; Fig A1 p17; Fig A2 p18; Fig A3 p18 | Du |
| CLEVR cube positions | Reverse Diffusion | One requested position is filled correctly. | One requested position stays empty at both 3 and 5 positions. The paper states this baseline is the approach of Liu (p6). | 3 positions; 5 positions | Fig 3 p6 | Du |
| ImageNet 128 px | Reverse Diffusion | Valley tiles are plausible. | Pizza, daisy and retriever tiles lose the object into texture or a malformed body. The paper states this baseline is the approach of Liu (p6). | Pizza; Daisy; Golden Retriever | Fig 4 p7 | Du |
| Text prompts | Single prompt | Plain beach and forest prompts are drawn correctly. | Colour goes to the wrong object and counts drift: the forest stays green, the lake turns purple instead of the trees, and two fruits become three or two different fruits. | red car in pink forest; lake with purple trees; two oranges; two of the same fruit | Fig 7 p8; Fig A4 p20; Fig A5 p21; Fig A6 p22; Fig A7 p23 | Du |

### Printed by SuperDiff

| Dataset | Method as shown | Succeeds | Fails | On which | Where | Printed by |
|---|---|---|---|---|---|---|
| Stable Diffusion AND pairs | Joint prompting | On 4 of 20 first-6-seed pairs most tiles carry both concepts: bottle cap and pawn, fireworks and dandelion, dog silhouette and mountains, teddy and panda. | One concept is the usual result on 14 of 20 first-6-seed rows, and 11 of those 20 rows fail outright. | all 20 AND pairs; weakest on a donut and a map, a sunflower and a lemon, a pineapple and a beehive | Fig 1 p2; Figs A5 to A24 p34 to p53 | SuperDiff |
| Stable Diffusion OR pairs | Joint prompting | Each tile is a recognisable object. | All 6 tiles pick the same concept on both pairs: candy canes only and pineapples only. | a candy cane or a flamingo; a pineapple or a beehive | Fig A25 p54 | SuperDiff |
| Stable Diffusion OR pairs | Prompt A or B (coin flip) | Each tile shows one clear concept, and both concepts appear for both pairs. | no failure shown in this paper's figures | a candy cane or a flamingo; a pineapple or a beehive | Fig A25 p54 | SuperDiff |

### Printed by FKC

| Dataset | Method as shown | Succeeds | Fails | On which | Where | Printed by |
|---|---|---|---|---|---|---|
| docking: P41594 and P23921 | DualDiff | Its 23-atom and 35-atom molecules print the strongest P23921 scores, -9.131 and -10.598. | It is strongest on both pockets at no molecule size. | molecule sizes 15 to 35 atoms | Fig 5 p8; Fig A6 p41 | FKC |
| ImageNet-512 | EDM2+CFG | The four samples show a recognisable dog, bird, jug and lemon. | The dog has a black lump behind its head, and the jug has a melted spout with a cup fused to its base. | dog and jug columns | Fig 3 p7 | FKC |
| docking: ATP1A1 and CPT2 | PoE without FKC | Its best P1 score, -8.870, is the strongest in the figure. | Its P2 scores, -8.133 and -9.291, are weaker than FKC's near -9.72. | two best molecules | Fig A8 p43 | FKC |
| SDXL two coloured objects | SDXL+CFG | The TV and toothbrush prompt and the handbag and mouse prompt both show their two objects. | Colours swap onto the wrong object, and the orange bench is replaced by a second refrigerator. | all three prompts in Fig A5 | Fig A5 p39 | FKC |
| mixture of 40 Gaussians | Target Score | Samples cover most of the 40 modes. | The mode near (-28, -37) is empty and two corner modes hold about one point each. | annealed target | Fig 2 p7 | FKC |
| docking: P41594 and P23921 | TargetDiff | Its 15-atom and 23-atom molecules print the strongest P41594 scores, -7.452 and -9.444. | It is weakest on both pockets at 19, 27 and 35 atoms. | molecule sizes 15 to 35 atoms | Fig 5 p8; Fig A6 p41 | FKC |
| mixture of 40 Gaussians | Tempered Noise | Samples cover the dense central modes. | Modes across the top left, top right and bottom right are empty. | annealed target | Fig 2 p7 | FKC |

### Printed by Bradley-Projective

| Dataset | Method as shown | Succeeds | Fails | On which | Where | Printed by |
|---|---|---|---|---|---|---|
| CLEVR location-conditioned | Bayes composition | Draws one object when asked for one. | With two or more locations it draws too few objects, and from four on most tiles are empty. | 2 to 9 composed locations | Fig 3 p3 | Bradley-Projective |
| Synthetic Gaussian mixtures | Bayes composition | Puts the stripes in the right columns. | The stripes stay dim on the empty-background data. | compose indices 0, 2, 4, 6 | Fig 12 p23 | Bradley-Projective |
| Synthetic Gaussian mixtures | Composition sampled with DDIM | Matches the exact composition on the aligned and orthogonal cases. | On the non-orthogonal case it misses patterns, and at a = 1 two of four clusters are nearly empty. | non-orthogonal transform; a = 1 | Fig 13 p28; Fig 15 p30 | Bradley-Projective |
| Synthetic Gaussian mixtures | Composition sampled with Langevin dynamics | Matches the exact composition in all three cases shown. | No failure shown in this paper's figures. | coordinate-aligned, orthogonal and non-orthogonal cases | Fig 13 p28 | Bradley-Projective |
| SDXL examples | Score composition on SDXL | Draws a dog in oil-painting style and a dog in a red hat. | Dog + horse gives one shaggy hybrid animal. | photo of a dog + photo of a horse | Fig 1 p1; Fig 7 p10 | Bradley-Projective |
| CLEVR colour-conditioned | Single-condition model | Draws objects of the asked colour and an empty background. | No failure shown in this paper's figures. | p0 to p7 and BG | Fig 5 p5; Fig 11 p19 | Bradley-Projective |
| CLEVR location-conditioned | Single-condition model | Draws one object at the conditioned place and an empty background. | No failure shown in this paper's figures. | p1 to p4 and pb | Fig 10 p18 | Bradley-Projective |
| CLEVR unconditional | Unconditional model | Draws clean scenes of separate objects. | No failure shown in this paper's figures. | 8 unconditional samples | Fig 9 p16 | Bradley-Projective |

### Printed by Bradley-Local

| Dataset | Method as shown | Succeeds | Fails | On which | Where | Printed by |
|---|---|---|---|---|---|---|
| CLEVR colour-conditioned | Color-conditioning | Shows separate objects in the asked colours up to four colours. | At eight colours, several samples miss at least one asked colour. | 1 to 8 colours | Fig 13 p30 | Bradley-Local |
| CIFAR-10 | EDM2 | Draws a recognisable white cat face for class 3. | The one sample shown is soft and blurred. | class 3 | Fig 11 p26 | Bradley-Local |
| CLEVR location-conditioned | Experiment 1 | Places one separate object per dot up to about seven dots. | Above about seven dots, objects crowd and cut into each other. | 1 to 12 conditioned locations | Fig 1 p2; Fig 7 p22; Fig 9 p24; Fig 10 p25 | Bradley-Local |
| CLEVR location-conditioned | Experiment 2 | Puts an object on a single conditioned dot. | Leaves most dots empty once more than one is asked for and adds objects nobody asked for. | 1 to 10 conditioned locations | Fig 1 p2; Fig 7 p22; Fig 9 p24 | Bradley-Local |
| CLEVR location-conditioned | Experiment 3 | Places one to three objects near the dots. | Stops adding objects past three dots and leaves most later dots empty. | 1 to 10 conditioned locations | Fig 1 p2; Fig 7 p22 | Bradley-Local |
| text prompts | MDM | Draws dogs biting croissants and cats with sushi and chopsticks. | The candlestick passes through the cow in some tiles and octopus bodies distort. | cow and candlestick; octopus in space; cat with sushi | Fig 17 p36 | Bradley-Local |
| text prompts | SDXL | Draws one subject with one object or style, such as a cat eating sushi or a dog in a hat. | None of the 10 seeds shows horse, dog and cat together, and cat with sushi and croissant fuses into a pattern. | horse, dog and cat prompts; cat eating sushi and a croissant; three-style prompt | Fig 4 p8; Fig 5 p10; Fig 14 p33; Fig 17 p36 | Bradley-Local |
| CLEVR location-conditioned | Standard architecture trained on 1 object | Draws one clean object. | Draws about one object whatever is asked, including for zero or ten dots. | 0 to 12 conditioned locations | Fig 8 p23; Fig 10 p25 | Bradley-Local |

### Printed by CoInD

| Dataset | Method as shown | Succeeds | Fails | On which | Where | Printed by |
|---|---|---|---|---|---|---|
| CelebA | LACE | Draws three clean unsmiling men at smile weight 0. | At smile weight 1 or more the faces patch, melt and grow long hair. | male AND smiling, smile weights 1, 2 and 6 | Fig 7 p10; Fig 11 p26 | CoInD |
| Colored MNIST | LACE | Draws clean 4s when asked for digit 4. | The four samples use only two colours, three of them green. | digit = 4, non-uniform support | Fig 5 p8 | CoInD |
| Shapes3D | LACE | Keeps the blue capsule recognisable on uniform support. | Adds ghost copies and noise, and draws a capsule where a cube was expected. | uniform AND; partial AND; partial NOT | Fig 6 p9 | CoInD |
| 2D Gaussian mixture | Vanilla diffusion objective | Keeps the three trained modes visible, though fainter. | Puts a spurious blob at the centre and misses the unseen mode. | unseen mode at (+1, +1) | Fig 14 p29 | CoInD |

### Printed by MultiDiffusion

| Dataset | Method as shown | Succeeds | Fails | On which | Where | Printed by |
|---|---|---|---|---|---|---|
| COCO validation | BLD | Bears, the bull, the laptop cat and the stuffed bears are clear. | A cat's head is covered by a lumpy grey mass and the dog's bowl barely reads as a bowl. | cat on a green cushion; dog near a bowl; wrestling-ring cake | Fig 13 p15 | MultiDiffusion |
| Panoramas | BLD | Content follows each prompt across the strip. | Content repeats and rectangular patches or hard joins show in the sky and ground. | skiers; forest; twilight mountains; anime; cartoon; city | Fig 4 p6; Fig 10 p13 | MultiDiffusion |
| Region prompts | BLD | Dogs, the bear and the elephant appear in the right places. | Objects go missing or warp: no snowy mountain, no bathroom mirror, a detached second squirrel tail and a torn plate. | red car and mountain; bathroom; squirrel; plate of cookies; avocado | Fig 5 p7; Fig 14 p16 | MultiDiffusion |
| Panoramas | Independent paths | Each crop is a plausible snowy village on its own. | Crops do not agree, leaving hard vertical joins and mismatched skies. | snowy village panorama | Fig 3 p4 | MultiDiffusion |
| Region prompts | Make-A-Scene | Both dogs, the car and mountain, and the cup and cookies are present. | Subjects blur or fuse: the squirrel merges into its sign, the elephant gains an extra shape, and the sign text is garbled. | squirrel with sign; black elephant; dogs; bathroom | Fig 5 p7; Fig 14 p16 | MultiDiffusion |
| COCO validation | SI | Cats, bears, the bull, the dog with a bowl and the stuffed bears are clear. | The wrestling-ring cake has garbled letters and no table, and the laptop is only a sliver. | cake like a wrestling ring; cat beside a laptop | Fig 13 p15 | MultiDiffusion |
| Panoramas | SI | Scenes follow the prompt and some ridgelines stay continuous. | Hard vertical joins and mismatched panels break several strips. | anime; forest; cartoon | Fig 4 p6; Fig 10 p13 | MultiDiffusion |
| Region prompts | SpaText | All seven layouts shown have every requested object, separate and clear. | no failure shown in this paper's figures | dogs; car and mountain; bathroom; squirrel; avocado and bear; cookies; elephant | Fig 5 p7; Fig 14 p16 | MultiDiffusion |

### Printed by PiCo

| Dataset | Method as shown | Succeeds | Fails | On which | Where | Printed by |
|---|---|---|---|---|---|---|
| GPT-4o complex prompts | BoxDiff | Shows a dog with a yellow frisbee, though the dog's body is warped. | Breaks the cake into smeared pieces, warps the cat, and grows the cat's head out of the red shoe. | A white cake with pink roses on silver plate; A black cat sitting on a wooden fence next to an orange pumpkin; A fuchsia umbrella shading a white bench next to a gray stone wall; A large brown dog chasing a yellow frisbee through a back yard; a blue candle and a red shoes and a cat | Fig 17 p15 | PiCo |
| T2I-CompBench | BoxDiff | Draws a separate glove and spoon. | Tears bodies, puts a cat head on a sheep body, sets a cactus on a palm trunk, and breaks the lighthouse scene into panels. | A blue apple and a green vase; a plastic bottle and a leather jacket; a small bagel and a rectangular sandwich; a tall lighthouse and a short bench; a squirrel and a bird and a yellow candle; A pink cat and a brown sheep; a green banana and a blue cake; a tall palm tree and a short cactus; a plastic toy and a glass bottle | Fig 2 p4; Fig 16 p14 | PiCo |
| PiCo noise-score ablation | Candidate noises not selected | Some candidates show the asked-for objects. | Others show a grey cat, an orange dish, or a red object cut off at the frame. | A yellow bowl and a blue cat; A blue cup and a red vase | Fig 6 p8 | PiCo |
| GPT-4o complex prompts | GORS | Draws the cake with roses on a plate. | Seats the cat on the pumpkin, puts the disc on the dog's head, and drops the cat in one candle tile. | A black cat sitting on a wooden fence next to an orange pumpkin; A fuchsia umbrella shading a white bench next to a gray stone wall; A large brown dog chasing a yellow frisbee through a back yard; a blue candle and a red shoes and a cat | Fig 17 p15 | PiCo |
| T2I-CompBench | GORS | One tile each on lighthouse and bench, banana and cake, and toy and bottle shows both objects. | Drops the bottle, spoon, cactus or bird, and grows a cat's face on a sheep. | A blue apple and a green vase; a plastic bottle and a leather jacket; a small bagel and a rectangular sandwich; a tall lighthouse and a short bench; a squirrel and a bird and a yellow candle; A pink cat and a brown sheep; a green banana and a blue cake; rubber gloves and a metallic spoon; a tall palm tree and a short cactus; a plastic toy and a glass bottle | Fig 2 p4; Fig 16 p14 | PiCo |
| PiCo CLIPSeg colour-bias examples | Generator not named in the figure | Draws one named object in each picture. | Replaces the other named object with a look-alike of the first class. | a fluffy white cat, a sleek black dog, and a gray rabbit; a yellow dog and a black cat; a purple eggplant and a pink rose; a red car and a blue bike | Fig 12 p12 | PiCo |
| GPT-4o complex prompts | Magnet | One tile of the cake prompt is right. | Replaces the cake with a tin, melts the cat into a pumpkin, turns the frisbee into a slab, and drops the cat in one tile. | A white cake with pink roses on silver plate; A black cat sitting on a wooden fence next to an orange pumpkin; A fuchsia umbrella shading a white bench next to a gray stone wall; A large brown dog chasing a yellow frisbee through a back yard; a blue candle and a red shoes and a cat | Fig 17 p15 | PiCo |
| T2I-CompBench | Magnet | Draws a lighthouse with a small separate bench. | Drops the bottle, spoon or bird, and puts a pink cat's face on a sheep. | A blue apple and a green vase; a plastic bottle and a leather jacket; a small bagel and a rectangular sandwich; a squirrel and a bird and a yellow candle; A pink cat and a brown sheep; a green banana and a blue cake; rubber gloves and a metallic spoon; a tall palm tree and a short cactus; a plastic toy and a glass bottle | Fig 2 p4; Fig 16 p14 | PiCo |
| PiCo control-duration ablation | No control | Keeps the apple recognisable. | A pear-like shape runs into the banana, and a dog's body lies along the bench. | A green apple and a yellow banana; A green bench and a brown dog | Fig 3 p6 | PiCo |
| PiCo mask-strength ablation | No control | Draws a bird, an apple, bears and a boat. | The apple stays orange and no bear is blue. | A brown bird and a green apple; A blue bear and a brown boat | Fig 4 p6 | PiCo |
| PiCo mask-type ablation | No control | Draws a green apple and a blue bowl. | Swaps the colours and draws no vase. | A blue apple and a green vase | Fig 7 p8 | PiCo |
| SDXL, two prompts | SDXL | Draws an apple and bird apart. | Turns the brown bird mostly green and draws no bottle. | a green apple and a brown bird; a plastic bottle and a leather jacket | Fig 13 p12 | PiCo |
| GPT-4o complex prompts | Stable Diffusion | Draws the cake and the umbrella scene correctly. | Seats the cat on a plank, paints the frisbee the wrong colour, and grows the candle out of the shoe. | A black cat sitting on a wooden fence next to an orange pumpkin; A large brown dog chasing a yellow frisbee through a back yard; a blue candle and a red shoes and a cat | Fig 17 p15 | PiCo |
| PiCo control-stage example | Stable Diffusion | Draws recognisable bread and sheep. | Fuses bagel and sandwich, and draws two sheep and no cat. | Bagel and sandwich (prompt not printed in the figure); Pink cat and brown sheep (prompt not printed in the figure) | Fig 10 p11; Fig 11 p11 | PiCo |
| PiCo cross-attention examples | Stable Diffusion | Both objects are visible for cat and sheep, cow and dog, and chair and table. | Draws a second backpack for the car, leaves the cow brown, and makes the vase pink. | A pink cat and a brown sheep; A yellow cow and a brown dog; A blue backpack and a red car; A purple balloon floating above a red picnic blanket with yellow flowers; A turquoise chair next to a wooden table with a white vase and pink roses | Fig 18 p16 | PiCo |
| T2I-CompBench | Stable Diffusion | Draws both objects on lighthouse and bench, gloves and spoon, palm and cactus, and toy and bottle. | Drops the cat or the bird, and fuses the bagel and sandwich into one round bread. | A blue apple and a green vase; a plastic bottle and a leather jacket; a small bagel and a rectangular sandwich; a squirrel and a bird and a yellow candle; A pink cat and a brown sheep; a green banana and a blue cake | Fig 2 p4; Fig 16 p14 | PiCo |

### Printed by FreeFuse

| Dataset | Method as shown | Succeeds | Fails | On which | Where | Printed by |
|---|---|---|---|---|---|---|
| SDXL, one prompt | CLoRA | Two figures appear. | Both wear the same robe and Harry's identity is lost. | Harry and Daiyu, faces close, autumn leaves | Fig 11 p19 | FreeFuse |
| SDXL, one prompt | Directly Sample | No tile keeps both identities. | Harry disappears into a second female face. | Harry and Daiyu, faces close, autumn leaves | Fig 11 p19 | FreeFuse |
| FreeFuse benchmark on FLUX.1-dev | Directly Sampling | No tile shows the asked-for characters apart. | Six characters collapse into blended faces. | Six reference characters side by side, including Naruto and Spider-Man | Fig 1 p2 | FreeFuse |
| FreeFuse benchmark on FLUX.1-dev | IP-Adapter | Places a woman and a separate car, and aims a pistol. | Merges or drops characters and loses reference faces. | Alcina, Alice, Greggs and Ran talking by a fountain pool; Alcina aiming a suppressed pistol; Alcina, Leela, Greggs and Conan in snow; Leela, Greggs, Nezha and Ran; Neytiri, Leela, Aobing and Greggs; Leela in front of the battle car | Fig 7 p11; Fig 16 p23 | FreeFuse |
| FreeFuse benchmark on FLUX.1-dev | LoRAShop | Keeps the four characters apart in the snow prompt. | Drops Nezha, doubles Leela, makes Ran and Alice realistic, and loses the pistol suppressor. | Alcina, Alice, Greggs and Ran talking by a fountain pool; Alcina aiming a suppressed pistol; Leela, Greggs, Nezha and Ran; Neytiri, Leela, Aobing and Greggs; Leela in front of the battle car | Fig 7 p11; Fig 16 p23 | FreeFuse |
| SDXL, one prompt | Mix-of-Show | Harry and Daiyu are both recognisable. | no failure shown in this paper's figures | none of the prompts shown: Harry and Daiyu, faces close, autumn leaves | Fig 11 p19 | FreeFuse |
| SDXL, one prompt | OMG | A separate couple appears. | Both wear generic coats and Daiyu's costume is gone. | Harry and Daiyu, faces close, autumn leaves | Fig 11 p19 | FreeFuse |
| FreeFuse benchmark on FLUX.1-dev | OmniGen | Keeps Leela and the battle car clear and separate. | Drops characters and moves clothing between them. | Alcina, Alice, Greggs and Ran talking by a fountain pool; Alcina aiming a suppressed pistol; Alcina, Leela, Greggs and Conan in snow; Leela, Greggs, Nezha and Ran; Neytiri, Leela, Aobing and Greggs | Fig 7 p11; Fig 16 p23 | FreeFuse |
| SDXL, one prompt | Orthogonal Adaptation | Harry and Daiyu are both recognisable. | no failure shown in this paper's figures | none of the prompts shown: Harry and Daiyu, faces close, autumn leaves | Fig 11 p19 | FreeFuse |
| FreeFuse benchmark on FLUX.1-dev | Redux | No tile keeps every asked-for subject. | Collapses group prompts to one portrait and grows the gun from a hat. | Alcina, Alice, Greggs and Ran talking by a fountain pool; Alcina aiming a suppressed pistol; Alcina, Leela, Greggs and Conan in snow; Leela, Greggs, Nezha and Ran; Neytiri, Leela, Aobing and Greggs; Leela in front of the battle car | Fig 7 p11; Fig 16 p23 | FreeFuse |
| FreeFuse benchmark on FLUX.1-dev | UMO | Keeps Alcina recognisable and the battle car clear. | Blends identities across characters and drops or replaces some of them. | Alcina, Alice, Greggs and Ran talking by a fountain pool; Alcina aiming a suppressed pistol; Alcina, Leela, Greggs and Conan in snow; Leela, Greggs, Nezha and Ran; Neytiri, Leela, Aobing and Greggs; Leela in front of the battle car | Fig 7 p11; Fig 16 p23 | FreeFuse |
| SDXL, one prompt | ZipLoRA | Two separate people appear and Harry has glasses. | Daiyu loses her ornaments and costume. | Harry and Daiyu, faces close, autumn leaves | Fig 11 p19 | FreeFuse |

### Printed by Sculpting

| Dataset | Method as shown | Succeeds | Fails | On which | Where | Printed by |
|---|---|---|---|---|---|---|
| 2D grid (GFlowNet) | base model | The base heatmaps show the lattice, triangle and circle targets. | no failure shown in this paper's figures | p1, p2 and p3 | Fig 3 p12 | Sculpting |
| MNIST digits 0-5 and 4-9 | base model | Both base grids show legible digits from their own sets. | no failure shown in this paper's figures | digits 0-5 and digits 4-9 | Fig F.2 p31 | Sculpting |
| colored MNIST (chained operations) | base model | Each base grid shows its digit set. | A few tiles are malformed blobs. | p1, p2 and p3 | Fig F.5 p34 | Sculpting |
| colored MNIST (even odd and multiples of three) | base model | The base grids show their digit sets. | A few tiles in the even-digit grid are thick blobs. | even digits | Fig F.3 p32; Fig F.4 p32 | Sculpting |
| colored MNIST (three models) | base model | Each base grid shows its digit set. | Two tiles in p3 are thick unreadable blobs. | p3 | Fig 6 p14 | Sculpting |
| colored MNIST (two models) | base model | Both base grids show their digit sets. | no failure shown in this paper's figures | p1 and p2 | Fig F.1 p31 | Sculpting |
| molecules (GFlowNet) | base model | The SEH and SA clouds each concentrate on high values of their own reward. | no failure shown in this paper's figures | SEH and SA at beta 32 and 96 | Fig 4 p12 | Sculpting |

### Printed by Catastrophic

| Dataset | Method as shown | Succeeds | Fails | On which | Where | Printed by |
|---|---|---|---|---|---|---|
| 2D Gaussians | conditional diffusion model | The learned single-condition clouds match the shapes of their ground-truth clouds. | no failure shown in this paper's figures | P^a1, P^a2 and P^0 | Fig 1 p9 | Catastrophic |
| rooms: couch and painting | naive denoising (K=1) | In distribution, most rooms show both objects (3 of 4 in Fig 2, 7 of 10 in Fig 4). | Rooms often drop one object, and out of distribution only 2 of 4 and 6 of 10 show both. | panels a and c of Figs 2 and 4 | Fig 2 p12; Fig 4 p33 | Catastrophic |
| rooms: couch and painting (separate models) | naive denoising (K=1) | One of four in-distribution rooms shows a couch and a painting. | Separate models almost always draw a couch alone, with 1 of 8 rooms showing a painting. | panels a and c of Fig 3 | Fig 3 p28 | Catastrophic |
| rooms: couch and table | naive denoising (K=1) | In distribution, a minority of rooms show a clear couch and table (1 of 4, 3 of 10). | Tables melt into couches or vanish, and out of distribution no room shows both. | panels b and d of Figs 2 and 4 | Fig 2 p12; Fig 4 p33 | Catastrophic |
| rooms: couch and table (separate models) | naive denoising (K=1) | Two of four in-distribution rooms show a couch with a small table against it. | Out of distribution, no room shows a separate table. | panels b and d of Fig 3 | Fig 3 p28 | Catastrophic |

### Printed by TestTime

| Dataset | Method as shown | Succeeds | Fails | On which | Where | Printed by |
|---|---|---|---|---|---|---|
| CelebA | Query-only | Keeps every visible attribute of the query. | Every sample is a grainy near-copy of the query. | all four CelebA queries | Fig 3 p8; Fig 5 p17 | TestTime |
| ColorMNIST | Query-only | Keeps the query digit, colour and background. | Every sample is a near-copy of the query with salt and pepper noise. | all four ColorMNIST queries | Fig 3 p8; Fig 5 p17 | TestTime |
| CelebA | Top-1 | Draws clean, varied, smiling faces. | Loses the asked attributes, drawing men for blond queries and women for the gray-haired man. | all four CelebA queries | Fig 3 p8; Fig 5 p17 | TestTime |
| ColorMNIST | Top-1 | Draws clear digits, with mostly 7s for the digit 7 query. | Draws the wrong digit or colour for the held-out composition, and on green 0 the background is also wrong. | all four ColorMNIST queries | Fig 3 p8; Fig 5 p17 | TestTime |
| CelebA | Top-3 | Draws smiling faces, some blurred. | Loses hair colour and sometimes gender across the queries. | all four CelebA queries | Fig 3 p8; Fig 5 p17 | TestTime |
| ColorMNIST | Top-3 | Draws digit-like shapes. | Draws mixed wrong digits and scribbles in the wrong colour. | all four ColorMNIST queries | Fig 3 p8; Fig 5 p17 | TestTime |

### Printed by CO3

| Dataset | Method as shown | Succeeds | Fails | On which | Where | Printed by |
|---|---|---|---|---|---|---|
| Attend-and-Excite prompts | Divide-Bind | It keeps both subjects on all three prompts shown and draws the bow and the crown clearly. | It crops subjects tightly and adds a small mouse-turtle hybrid to the turtle and mouse picture. | a turtle and a mouse; a elephant with a bow; a gray crown and a white clock | Fig 3 p8 | CO3 |
| Attend-and-Excite prompts on PixArt-Sigma | PixART | It draws a black dog with a black apple. | It usually drops one subject, drawing two horses, two turtles or two bears, or moves a color to the wrong object. | a horse and a monkey; a bird and a turtle; a dog and a bear; a blue crown and a red balloon; a turtle and a yellow car; a purple car and a pink apple | Fig 6 p9; Fig 8 p22 | CO3 |
| RareBench | R2F | It keeps the owl figures and the unicycle kangaroo. | It drops the lion, octopus and shark, and keeps banana and bottle as separate objects. | a horned pelican and a spotted lion; A hairy octopus; A wooly shark; A wooly alligator | Fig 4 p8; Fig 11 p25 | CO3 |
| T2I-CompBench Complex | R2F | It draws balloons with a silver airplane. | It leaves sharp claws unclear in the fur coat prompt. | The colorful balloons; The soft fur coat | Fig 4 p8 | CO3 |
| Attend-and-Excite prompts | SDXL | It draws two clear subjects on a few pairs such as dog and rabbit and cat with glasses. | It drops one subject, fuses subjects or moves a color to the wrong object on most prompts shown. | a turtle and a mouse; a bird and a lion; a horse and a green apple; a dog and a horse; a gray crown and a white clock | Fig 3 p8; Fig 5 p9; Fig 9 p23 | CO3 |
| RareBench | SDXL | It keeps some animals and objects together. | It drops the lion and the owl and ignores rare attributes such as hairless or horned. | a horned pelican and a spotted lion; A knighted turtle; a hairless yak; A wooly shark | Fig 4 p8; Fig 11 p25 | CO3 |
| T2I-CompBench Complex | SDXL | It draws both objects in most scenes. | It misses colors, relations and required objects, and shows no bat in the home-run prompt. | The colorful balloons; The soft fur coat; The triangular prism; The sharp crisp crack; The flickering torch | Fig 4 p8; Fig 10 p24 | CO3 |
| Attend-and-Excite prompts | SDXL+Corrector | It draws a bird with lions and a monkey with a bow tie. | It doubles the lion. | a bird and a lion; a monkey with a bow | Fig 5 p9 | CO3 |
| Attend-and-Excite prompts | SDXL+Corrector+weight-mod. | It draws a bird with lions and a monkey with a bow tie. | It doubles the lion and adds a second bird. | a bird and a lion; a monkey with a bow | Fig 5 p9 | CO3 |
| Attend-and-Excite prompts | SDXL+resampling | It adds the missing bird and the bow. | It doubles the lion. | a bird and a lion; a monkey with a bow | Fig 5 p9 | CO3 |
| Attend-and-Excite prompts | SDXL+resampling+Corrector | It keeps the bird and the bow tie. | It still doubles the lion. | a bird and a lion; a monkey with a bow | Fig 5 p9 | CO3 |
| Attend-and-Excite prompts | SDXL+resampling+Corrector+weight-mod. | It draws one bird with one lion and a monkey with a bow. | no failure shown in this paper's figures | a bird and a lion; a monkey with a bow | Fig 5 p9 | CO3 |
| Attend-and-Excite prompts | SynGen | It draws a clear turtle with clear mice. | It doubles the mouse, drops the bow and turns the crown into a clock finial. | a turtle and a mouse; a elephant with a bow; a gray crown and a white clock | Fig 3 p8 | CO3 |
| T2I-CompBench Complex | SynGen | It draws balloons with a silver airplane. | It drops the second object on the book and pencil prompts and blends the apple and pear colors. | The square book; The yellow pencil; The red apple; The triangular prism; The sharp crisp crack | Fig 10 p24 | CO3 |

### Printed by ToMe

| Dataset | Method as shown | Succeeds | Fails | On which | Where | Printed by |
|---|---|---|---|---|---|---|
| Binding prompts shown by the authors (set not stated) | Config A | It draws a clear cat and dog in the Fig 11 picture. | It repeats accessories and in Fig 6 draws two cats with no dog. | prompt not printed in Fig 6; a cat wearing sunglasses and a dog wearing hat | Fig 6 p9; Fig 11 p18 | ToMe |
| Binding prompts shown by the authors (set not stated) | Config B | It draws a clear cat and dog in the Fig 11 picture. | It draws two cats in Fig 6 and puts hats and sunglasses on both animals in Fig 11. | prompt not printed in Fig 6; a cat wearing sunglasses and a dog wearing hat | Fig 6 p9; Fig 11 p18 | ToMe |
| Binding prompts shown by the authors (set not stated) | Config C | It binds sunglasses to the cat and the hat to the dog in Fig 11. | In Fig 6 the second animal has a cat-like face. | prompt not printed in Fig 6; a cat wearing sunglasses and a dog wearing hat | Fig 6 p9; Fig 11 p18 | ToMe |
| Binding prompts shown by the authors (set not stated) | Config D | It draws a separate dog in a hat. | The cat also wears a cap. | prompt not printed in Fig 6 | Fig 6 p9 | ToMe |
| Binding prompts shown by the authors (set not stated) | Config E | It keeps two separate animals in Fig 11. | It swaps the accessories in Fig 11 and draws a cat-faced second animal in Fig 6. | prompt not printed in Fig 6; a cat wearing sunglasses and a dog wearing hat | Fig 6 p9; Fig 11 p18 | ToMe |
| Binding prompts shown by the authors (set not stated) | Config F | It draws a cat in sunglasses and a separate puppy in a hat. | It adds an unrequested pink object at the puppy's mouth. | prompt not printed in Fig 6 | Fig 6 p9 | ToMe |
| Binding prompts shown by the authors (set not stated) | ELLA 1.5 | It draws a yellow apple with a pear and a clear square box. | It gives the camera to the cat, repeats glasses across both people, and draws a plain disc in place of a round bag. | a squirrel holding camera and a cat wearing hat; a girl with a necklace and a boy with glasses; a cat with sunglasses and a dog wearing a hat; a round bag and a square box | Fig 5 p8 | ToMe |
| Binding prompts shown by the authors (set not stated) | PlayGround-v2 | It draws a clear squirrel, cat, boy and girl. | It repeats attributes across subjects and replaces one object with another of the same kind, such as two cats or two apples. | a dog with blue bow tie and a cat with red scarf; a cat with sunglasses and a dog wearing a hat; a yellow apple and a red pear; a round bag and a square box | Fig 1 p2; Fig 5 p8 | ToMe |
| Binding prompts shown by the authors (set not stated) | Ranni 1.5 | It binds the boy's glasses and the girl's necklace, and the apple and pear colors. | It merges animal, person and camera on the squirrel prompt and draws a flat card instead of a box. | a squirrel holding camera and a cat wearing hat; a cat with sunglasses and a dog wearing a hat; a round bag and a square box | Fig 5 p8 | ToMe |
| Binding prompts shown by the authors (set not stated) | SD3 | It draws a clear dog and a clear cat in both pictures. | It puts both a hat and sunglasses on both animals. | a dog wearing hat and a cat wearing sunglasses | Fig 1 p2 | ToMe |
| Binding prompts shown by the authors (set not stated) | SDXL | It draws both subjects on many two-subject prompts and binds some of the simplest attributes. | It often puts each accessory on both subjects, replaces one subject with a copy of the other, or fuses them. | a cat with scarf and a dog with tie; A white cat with glasses and a black dog with hat; A fox with sunglasses and a deer with crown; A bear with hat and a man with glasses | Fig 1 p2; Fig 2 p4; Fig 5 p8; Fig 12 p19; Fig 14 p20 | ToMe |
| Token arithmetic examples | SDXL | It draws each single-subject prompt clearly. | no failure shown in this paper's figures | a photo of a dog; doctor; king; queen | Fig 3 p5 | ToMe |
| Binding prompts shown by the authors (set not stated) | SDXL [EOT] tokens | It draws a dog and a cat from the end token alone. | Hats and sunglasses spread across both animals. | a cat wearing sunglasses and a dog wearing a hat | Fig 2 p4 | ToMe |
| Binding prompts shown by the authors (set not stated) | SDXL [dog] token | It draws a clear dog from the single token. | The dog also wears sunglasses that belong to the other subject. | a cat wearing sunglasses and a dog wearing a hat | Fig 2 p4 | ToMe |
| Binding prompts shown by the authors (set not stated) | SynGen XL | It binds the color and shape prompts with a red pear and a square parcel. | It drops the cat and repeats glasses and necklaces across the two people. | a squirrel holding camera and a cat wearing hat; a girl with a necklace and a boy with glasses; a cat with sunglasses and a dog wearing a hat | Fig 5 p8 | ToMe |
| Token arithmetic examples | Token addition | Adding or subtracting a token adds a hat, removes glasses or a crown, and turns a queen into a king. | no failure shown in this paper's figures | dog + hat; doctor - glasses; king - crown; queen - woman + man | Fig 3 p5 | ToMe |
| Token edit examples | Unedited prompt | It draws each base subject clearly. | no failure shown in this paper's figures | elephant; cat and dog; girl with earrings; man with glasses; queen with cats; nurse | Fig 8 p9 | ToMe |

### Printed by AttendExcite

| Dataset | Method as shown | Succeeds | Fails | On which | Where | Printed by |
|---|---|---|---|---|---|---|
| Attend-and-Excite prompts | Prompt-to-Prompt | It shows one clear frog on a pink bench. | It melts cat and dog bodies and drops the frog, the clock and usually the bowl. | A cat and a dog; A turtle and a yellow bowl; A frog and a pink bench; A red bench and a yellow clock | Fig 13 p13 | AttendExcite |
| Attend-and-Excite prompts | Stable Diffusion | It draws one subject cleanly and sometimes both, as in three of eight turtle and orange balloon pictures. | It usually drops one subject, and it sometimes fuses the two or moves a color to the wrong object. | A cat and a dog; A frog and a pink bench; A horse and a monkey; A bird and a bear; A green backpack and a brown suitcase | Fig 1 p1; Fig 2 p2; Fig 4 p4; Fig 5 p5; Fig 15 p16; Fig 16 p17; Fig 20 p20; Fig 21 p21; Fig 22 p22; Fig 24 p24 | AttendExcite |
| Free-form prompts shown by the authors | Stable Diffusion | It draws single scene subjects and one dog on a street clearly. | It drops the second subject or the setting on most prompts, such as the salmon, the library, the swan's kitchen and the juice. | A grizzly bear catching a salmon; The cloud gate sculpture in a library; A green apple and a red apple; A bird perched on top of a vintage camera; A blue dog and a pink cat | Fig 1 p1; Fig 6 p6; Fig 17 p18; Fig 18 p19; Fig 20 p20 | AttendExcite |
| Attend-and-Excite prompts | StructureDiffusion | It shows both subjects in a few pictures, such as one clear cat and dog and one frog on a pink bench. | It usually drops a subject and sometimes fuses subjects, as in a cat with frog feet. | A cat and a dog; A frog and a pink bench; A horse and a monkey; A rabbit with a crown; A cat and a frog; A mouse and a red car | Fig 5 p5; Fig 20 p20; Fig 21 p21; Fig 22 p22 | AttendExcite |
| Free-form prompts shown by the authors | StructureDiffusion | It draws clean single subjects. | It drops the dog and nearly always drops the cake. | A brown dog and a grey mouse; A red cake and a bouquet of yellow tulips | Fig 20 p20 | AttendExcite |
| StructureDiffusion prompts | StructureDiffusion | It shows both objects on several scene prompts. | It misses the blue cube and the black cord, and it swaps materials on the bench and building. | A red sphere and a blue cube; A green sphere and a red cube; A red cat sits on a rug with a black chord; A wooden building with a stone bench | Fig 7 p6; Fig 19 p19 | AttendExcite |

### Printed by FlipConcept

| Dataset | Method as shown | Succeeds | Fails | On which | Where | Printed by |
|---|---|---|---|---|---|---|
| Custom Concept 101 and generated images | Custom Diffusion | Keeps a cat and woman in one picture, and all three subjects in one tile of Fig 4. | Fuses cat and dog, turns the panda into a ball head, loses the references, and drops the man in one tile. | A cat and a woman hugging each other, lighthouse background; A cat and a dog playing with a ball, mountain background; A plush teddybear and a plush panda playing with a ball, beach background; A guitar placed next to a cup, bar background; A man in the snow next to a big dog resting its front legs on a vintage suitcase, snow-covered house | Fig 1 p2; Fig 3 p5; Fig 4 p5 | FlipConcept |
| Custom Concept 101 and generated images | Mix-of-Show | Keeps the lighthouse, the electric guitar and the teddy recognisable. | Drops the woman, dog and cup, and puts panda markings on the teddy and ball. | A cat and a woman hugging each other, lighthouse background; A cat and a dog playing with a ball, mountain background; A plush teddybear and a plush panda playing with a ball, beach background; A guitar placed next to a cup, bar background | Fig 1 p2; Fig 3 p5 | FlipConcept |
| Custom Concept 101 and generated images | Perfusion | Draws one recognisable subject per prompt. | Drops the cat and the cup, and fuses teddy and panda into one toy. | A cat and a woman hugging each other, lighthouse background; A cat and a dog playing with a ball, mountain background; A plush teddybear and a plush panda playing with a ball, beach background; A guitar placed next to a cup, bar background | Fig 1 p2; Fig 3 p5 | FlipConcept |

### Printed by CFGRevisited

| Dataset | Method as shown | Succeeds | Fails | On which | Where | Printed by |
|---|---|---|---|---|---|---|
| FLUX.2 [klein] 4B Base | APG | Right on Fig 1, cow/stop, 5 workers, 4 vases and 3 bats. | Draws 5 frisbees, shows 3 giraffes with 2 heads, drops the cup, and grows a disc from the motorcycle. | 4 frisbees; 3 giraffes; couch/cup; frisbee/moto | Fig 1 p4; Fig 2 p5 | CFGRevisited |
| Guidance-scale sweep, model not named in the figure | APG | Holds the horse and valley at all four scales and the portrait from 5 to 18. | Scale 1.5 distorts the nose and glasses in the portrait. | portrait prompt at guidance scales 1.5, 5, 12, 18 | Fig 3 p12; Fig 4 p12; Fig 5 p13 | CFGRevisited |
| SD3.5 Medium | APG | Clean portrait and valley. | The dog wears the tie, so the dog is not to its right. The rider cannot be confirmed as a woman. | a dog to the right of a tie; a woman riding a white horse through red poppies | Fig 1 p4 | CFGRevisited |
| FLUX.2 [klein] 4B Base | CFG | Right on all four Fig 1 prompts, and on cow/stop, 4 frisbees, 3 giraffes and 5 workers. | Drops the cup and the frisbee, and draws five tumblers for 4 vases and four bats for 3 bats. | couch/cup; frisbee/moto; 4 vases; 3 bats | Fig 1 p4; Fig 2 p5 | CFGRevisited |
| Guidance-scale sweep, model not named in the figure | CFG | Holds at scale 5 on all three prompts. | Scale 1.5 smears the portrait, and 12 to 18 burn colours and lose the library or forest to black. | white-horse prompt at guidance scales 1.5, 5, 12, 18; portrait prompt at guidance scales 1.5, 5, 12, 18; mountain-valley prompt at guidance scales 1.5, 5, 12, 18 | Fig 3 p12; Fig 4 p12; Fig 5 p13 | CFGRevisited |
| SD3.5 Medium | CFG | Clean portrait, valley and rider on a white horse. | The dog wears the tie, so the dog is not to its right. | a dog to the right of a tie | Fig 1 p4 | CFGRevisited |
| FLUX.2 [klein] 4B Base | CFG++ | Right on Fig 1, 4 frisbees, 3 giraffes, couch/cup and frisbee/moto. | Cuts the stop sign to STOH, draws 3 workers for 5, and draws 3 glass cylinders for 4 vases. | cow/stop; 5 workers; 4 vases; 3 bats | Fig 1 p4; Fig 2 p5 | CFGRevisited |
| SD3.5 Medium | CFG++ | Clean portrait. | The dog wears the tie, so the dog is not to its right. The valley lacks clear sunrise light and the rider cannot be confirmed as a woman. | a dog to the right of a tie; a misty mountain valley with a river at sunrise; a woman riding a white horse through red poppies | Fig 1 p4 | CFGRevisited |
| FLUX.2 [klein] 4B Base | CFG-Zero* | Right on all four Fig 1 prompts. | no failure shown in this paper's figures | none of the prompts shown: a portrait of a woman wearing round glasses in a library; a dog to the right of a tie; a misty mountain valley with a river at sunrise; a woman riding a white horse through red poppies | Fig 1 p4 | CFGRevisited |
| SD3.5 Medium | CFG-Zero* | Clean portrait. | The dog wears the tie, so the dog is not to its right. The sky has only faint warm light and the rider cannot be confirmed as a woman. | a dog to the right of a tie; a misty mountain valley with a river at sunrise; a woman riding a white horse through red poppies | Fig 1 p4 | CFGRevisited |
| FLUX.2 [klein] 4B Base | OSEG | Right on Fig 1, 3 giraffes, couch/cup and frisbee/moto. | Covers the cow's head with the sign and miscounts frisbees, workers, vases and bats. | cow/stop; 4 frisbees; 5 workers; 4 vases; 3 bats | Fig 1 p4; Fig 2 p5 | CFGRevisited |
| Guidance-scale sweep, model not named in the figure | OSEG | Holds the portrait at 1.5 and 5, and all three prompts at 5. | Scales 12 to 18 burn colours, and 18 loses the library and forest to black. | white-horse prompt at guidance scales 1.5, 5, 12, 18; portrait prompt at guidance scales 1.5, 5, 12, 18; mountain-valley prompt at guidance scales 1.5, 5, 12, 18 | Fig 3 p12; Fig 4 p12; Fig 5 p13 | CFGRevisited |
| SD3.5 Medium | OSEG | Clean portrait and valley. | The dog wears the tie, so the dog is not to its right. The rider sits the wrong way round on the horse. | a dog to the right of a tie; a woman riding a white horse through red poppies | Fig 1 p4 | CFGRevisited |
| FLUX.2 [klein] 4B Base | PAG | Right on Fig 1, 4 frisbees, 3 giraffes, couch/cup and 3 bats. | Misspells STOOP, draws 4 workers for 5, drops the frisbee, and draws glasses for vases. | cow/stop; 5 workers; frisbee/moto; 4 vases | Fig 1 p4; Fig 2 p5 | CFGRevisited |
| SD3.5 Medium | PAG | Portrait and rider are recognisable. | The dog wears the tie, so the dog is not to its right. The dog's face turns grey-brown and the river fades into haze. | a dog to the right of a tie; a misty mountain valley with a river at sunrise | Fig 1 p4 | CFGRevisited |
| FLUX.2 [klein] 4B Base | SAG | Right on Fig 1, cow/stop, 4 frisbees, 3 giraffes, couch/cup and 3 bats. | Draws 4 workers for 5, drops the frisbee, and draws 6 tumblers for 4 vases. | 5 workers; frisbee/moto; 4 vases | Fig 1 p4; Fig 2 p5 | CFGRevisited |
| Guidance-scale sweep, model not named in the figure | SAG | Holds at scale 5 on all three prompts. | Scale 1.5 smears the portrait, and 12 to 18 burn colours and lose the library or forest to black. | white-horse prompt at guidance scales 1.5, 5, 12, 18; portrait prompt at guidance scales 1.5, 5, 12, 18; mountain-valley prompt at guidance scales 1.5, 5, 12, 18 | Fig 3 p12; Fig 4 p12; Fig 5 p13 | CFGRevisited |
| SD3.5 Medium | SAG | Clean portrait, valley and woman on a white horse. | The dog wears the tie, so the dog is not to its right. | a dog to the right of a tie | Fig 1 p4 | CFGRevisited |
| FLUX.2 [klein] 4B Base | SEG | Right on all four Fig 1 prompts. | no failure shown in this paper's figures | none of the prompts shown: a portrait of a woman wearing round glasses in a library; a dog to the right of a tie; a misty mountain valley with a river at sunrise; a woman riding a white horse through red poppies | Fig 1 p4 | CFGRevisited |
| SD3.5 Medium | SEG | Clean portrait, valley and rider on a white horse. | The dog wears the tie, so the dog is not to its right. | a dog to the right of a tie | Fig 1 p4 | CFGRevisited |
| FLUX.2 [klein] 4B Base | TCFG | Right on all four Fig 1 prompts. | no failure shown in this paper's figures | none of the prompts shown: a portrait of a woman wearing round glasses in a library; a dog to the right of a tie; a misty mountain valley with a river at sunrise; a woman riding a white horse through red poppies | Fig 1 p4 | CFGRevisited |
| SD3.5 Medium | TCFG | Clean portrait, valley and rider on a white horse. | The dog wears the tie, so the dog is not to its right. | a dog to the right of a tie | Fig 1 p4 | CFGRevisited |
| FLUX.2 [klein] 4B Base | no-CFG | The woman in glasses and the valley are recognisable. | The dog becomes a print on patterned strips and the horse and rider dissolve. | a portrait of a woman wearing round glasses in a library; a dog to the right of a tie; a misty mountain valley with a river at sunrise; a woman riding a white horse through red poppies | Fig 1 p4 | CFGRevisited |
| SD3.5 Medium | no-CFG | The forest river is recognisable. | Hair covers the face, the dog merges with cloth, and no whole white horse appears. | a portrait of a woman wearing round glasses in a library; a dog to the right of a tie; a misty mountain valley with a river at sunrise; a woman riding a white horse through red poppies | Fig 1 p4 | CFGRevisited |
