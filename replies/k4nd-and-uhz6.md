# Replies to Reviewers k4nd and UHz6

Submission 56173. One section per reviewer point: the reviewer's words quoted, then the reply.
Equation, section and figure numbers refer to the revised paper
(`paper/iclr-56173/iclr2027_conference.tex`). k4nd's review is pages 4 and 5 of `Reviews.pdf`;
UHz6's is page 3. A reply marked pending has not been agreed yet.

## Reviewer k4nd

### W1: the residual does not establish a distinct representation of plurality

> "The proposed residual does not establish a distinct representation of plurality. … The paper
> does not establish that it specifically measures whether the named concepts appear as distinct
> objects."

We agree that $r_t$ in Equation 9 is the entire difference between the joint-prompt and
product-of-experts predictions, and that adding it in full recovers the joint prompt by
construction. We do not claim that $r_t$ represents plurality alone. A product of experts drops the
interaction the joint prompt carries, and that interaction can hold several kinds of meaning: a
relation between the concepts ("a dog on a cat"), an effect of one on the other ("a camel in the
rain" should be wet), and the layout and style of the scene.

Our study concerns one part of this interaction: plurality, whether two named concepts appear as two
objects rather than one hybrid. The training pairs were chosen for it, as pairs the product fuses
into a hybrid.

Within that scope, the evidence is that the correction is specific and early, not generic. A pair's
own $r_t$ restores the second object, while another pair's, another seed's, or a time-reordered one
does not (Figure 2). The first ten of 50 steps are enough (Figures 3 and 4). At partial strength
($\lambda$ of 0.6 to 0.65 in Figure 2) the second object appears while the layout and style still
look like the product's render.

We agree this does not separate plurality from everything else $r_t$ carries. Section 3 now states
this scope explicitly, and the contributions are rewritten to claim only what the experiments show.

### W2: methodological novelty

> "The methodological novelty is insufficiently established."

The method is deliberately simple: a standard adapter trained by prediction matching. Its
contribution is the target. Existing corrections of product-of-experts sampling keep the product as
the target and change how it is sampled (Du et al., 2023; Skreta et al., 2024, 2025; Dutta et al.,
2026). We instead adapt the experts so that their product reproduces the joint prompt's prediction,
and the joint prompt is never evaluated at inference. With this simple objective, plurality emerges on
concept pairs the adapter never trained on (Section 4.1, Figure 5). Section 2 now states this
distinction.

We agree that fusion under composition was identified before (Bradley et al., 2025), and we cite it
as such. What we add is a measurement of the missing term during sampling: it is specific to the pair
and seed, and concentrated in the first ten of 50 steps (Figures 2 to 4).

Tables 1 and 3 compare the adapter with PoE and CO3 on the same seeds, and Figure 5 adds a Langevin
corrector. The ablation the reviewer describes, the same joint-prompt supervision on pairs
the product does not fuse, would separate the effect of the supervision from that of the plurality
pairs; we have not run it and name it as future work.

### W3: no systematic quantitative evaluation of the adapter

> "The learned adapter lacks systematic quantitative evaluation."

We agree, and the revision adds this evaluation. The adapter, the seeds and the success bar were
all fixed before any held-out score was read (Section 4.1).

Table 1 scores every method on the 276 Attend-and-Excite prompts (66 animal pairs, 144
animal-object pairs, 66 object pairs) at seeds 9 to 12, with composition measured by BLIP-VQA and
image quality by CLIP score and ImageReward.

Table 3 scores the 53 held-out pairs described in Appendix C.2, grouped as hard, easy but unseen, and
not competing, at seeds 9 to 16, with 95% intervals from resampling pairs.

The training and held-out pairs, with every seed, are released with the code [TODO: anonymous
link]; Appendix C.2 describes the held-out groups.

The failure types the reviewer names (substituted identities, additional objects, style-dependent
loss) are counted per setting in Figure 10 (Appendix C.3).

### W4: method details poorly presented and sometimes inconsistent

**The training loss is not stated in Section 4**

> "Section 4 describes target matching and provides the teacher prediction and sampling rule, but
> it does not explicitly formulate the training loss. The squared-error objective appears in the
> training diagram in Appendix A, Figure 7."

Section 4 now states the training loss as Equation 12: the squared error between the adapted
composition and the joint-prompt prediction, over the cached states of every training pair and seed
at the first ten of the 50 sampling steps. The training diagram is removed, since Equation 12 now
states the objective in the main text.

**Evaluations described as returning the correction**

> "The main text describes individual adapted network evaluations as returning the correction,
> whereas Appendix B defines them as adapted denoising predictions whose composition induces the
> correction. Section 4 refers to 'Figure 10,' although the actual training diagram is Figure 7."

Section 4 now says each adapted evaluation returns a noise prediction, as Appendix B derives. At
inference we run the usual product-of-experts step on $c_1$, $c_2$ and the null prompt with the
adapter switched on: three predictions per step, and never the joint prompt. The joint prompt's
prediction is the product's prediction plus the correction $r_t$, so matching it trains the adapter
to add $r_t$. At inference, switching the adapter on adds it. The reference that printed as "Figure 10" is removed.

**Repetitive claims about learning an abstract concept**

> "The repetitive claims about learning an abstract concept occupy space that could instead
> clarify the objective, training data, evaluation protocol, and limitations."

We removed the repeated claims about learning an abstract concept from Sections 3 and 4 and the
Discussion. The introduction now states what the correction is and what it restores (see W1).
Section 4 now describes the training data and how pairs were chosen (Figure 7).
Appendix C.5 (Figure 13) and a new Discussion paragraph show the adapter on three concepts,
including where it fails: it separates the concepts in every case but stops at two animals.
Section 4.1 now opens with the evaluation protocol: one adapter chosen by a fixed rule, the same
seed for every method, the prompt sets and held-out pairs with their seeds, the scorers, and the renders
behind both tables in the appendix (Tables 1 and 3).

### W5: the qualitative baseline comparison and its starting noise

> "The Figure 5 caption states that CO3 uses different starting noise, while Section 4.1 states
> that all examples use the same initial seeds."

Every method now uses the same seed for each prompt, with every column of a row in Figure 5
starting from the same latent, and Section 4.1 says so. CO3's column is marked $\dagger$ because its later resampling noise comes from the
seed, as in its released code. Tables 1 and 3 make the comparison over 4 and 8 seeds, with the
images behind them in the appendix.

### Q1: evidence that the residual measures plurality

> "What evidence establishes that the residual in Equation (9) specifically measures plurality
> rather than the broader difference between joint and separate conditioning?"

Answered under W1 above.

### Q2: aggregate results over held-out concepts and seeds

> "Can the authors provide aggregate results for the learned adapter and baselines over documented
> held-out concepts and multiple seeds, including identity errors, additional objects, and
> style-dependent failures?"

Answered under k4nd W3: Tables 1 and 3. The identity errors, additional
objects and style-dependent failures are counted per setting in Figure 10 (Appendix C.3).

### Q3: pairs retained when only CO3 succeeds

> "Training pairs are retained when either joint prompting or CO3 succeeds, but supervision comes
> from the frozen joint-prompt prediction. How are pairs handled when only CO3 succeeds?"

We have addressed this in Section 4 and Figure 7. Every training target is the frozen model's
joint-prompt prediction; no CO3 picture is a target. CO3 was a second view during screening: a pair
entered the training set when the joint prompt drew both named things as separate objects. Figure 7
shows the joint prompt beside CO3 for five candidate pairs, including crow and raven, dropped
because the two birds are hard to tell apart. Where CO3 drew a kept pair more clearly on a seed
(seed 3 in Figure 7), that seed's target is still the joint prompt's picture.

### Q4: the objective in the main text, and the inconsistencies

> "Can the authors state the optimization objective explicitly in the main text and resolve the
> inconsistencies concerning network outputs, figure references, and baseline starting noise?"

The objective is Equation 12 in Section 4. The network outputs and the figure reference are
resolved as described under W4. Baseline starting noise is resolved as described under W5.

## Reviewer UHz6

### W1: the adapter is evaluated only qualitatively

> "The trained adapter, which serves as the main contribution of the paper is only evaluated
> qualitatively. … In addition to the examples being limited to similar entities (cat, dog), there
> are no standardized benchmark evaluations such as T2I-CompBench or GenEval counting."

The revision evaluates the adapter quantitatively on the 276 prompts of the Attend-and-Excite
benchmark, which cover animal pairs, animal-object pairs and object pairs (Table 1), and on 53
held-out pairs in three groups (Table 3). Table 2 reports GenEval's two-object and counting tasks and
T2I-CompBench's spatial and complex tasks with their official evaluators [TODO: slot R14]. Appendix
C.5 shows three concepts.

### W3: comparisons with samplers and attention-based methods, and scope

> "There are no quantitative comparisons with the compositional samplers discussed in Sec. 2.2, and
> CO3 appears only as one qualitative column. Attention-based methods that address missing or merged
> objects under joint prompts (e.g., Attend-and-Excite, CONFORM) target the same failure mode and are
> not discussed. The study is also limited to SDXL and to pairs of two concepts, mostly animals."

Tables 1 and 3 now compare quantitatively with CO3 on the same seeds, and Figure 5 adds a Langevin
corrector. Attend-and-Excite and CONFORM are now discussed in Section 2: they adjust
cross-attention under a single joint prompt, while our setting has no joint prompt at inference.
Beyond pairs of animals, Table 1 covers objects and Appendix C.5 three concepts.

### The training filter

> "In addition, the training filter keeps a pair if either the joint prompt or CO3 succeeds, but
> the target is always the joint prompt (Eq. 11). Pairs that pass only through CO3 would therefore
> supervise toward a target without plurality."

We have addressed this in Section 4 and Figure 7. Every training target is the frozen model's
joint-prompt prediction; no CO3 picture is a target. CO3 was a second view during screening: a pair
entered the training set when the joint prompt drew both named things as separate objects. Figure 7
shows the joint prompt beside CO3 for five candidate pairs, including crow and raven, dropped
because the two birds are hard to tell apart. Where CO3 drew a kept pair more clearly on a seed
(seed 3 in Figure 7), that seed's target is still the joint prompt's picture.

### Presentation issues

> "There are presentation issues: Sec. 4 refers to a Figure 10 that does not exist, and several
> sentences are broken (e.g., "inside create hybrid objects" in Sec. 5)."

The reference that printed as "Figure 10" in Section 4 is removed, and the Discussion now reads
"and instead create hybrid objects".

### Q1: the 48 training pairs, the held-out pairs, and whether cat and dog is trained on

> "Could the authors list the 48 training pairs and the held-out pairs, and clarify whether
> cat/dog is in the training set?"

The training set is defined by a criterion, not by the particular pairs. A pair is kept when the
frozen model's joint prompt draws both named concepts as separate objects, and dropped when the two
concepts cannot be told apart (Section 4, Figure 7). The same criterion applies whether it yields 48
pairs or a thousand, and 48 is where we stopped collecting. The method relies on the criterion, not on
any specific pair.

Cat and dog is not in the training set. It is held out, and no training pair contains a cat or a dog.
The held-out pairs and their groups are released with the code (Appendix C.2 describes the groups), and no pair in Figure 5 was trained on.

For reproducibility, the 48 training pairs are: (two animals, 28) badger and magpie, bear and boar,
bear and heron, beaver and raccoon, beaver and woodpecker, bison and weasel, buffalo and hippo, bull
and rooster, camel and pelican, chicken and rabbit, deer and heron, deer and wolf, donkey and duck,
donkey and rooster, emu and koala, giraffe and hyena, giraffe and ostrich, giraffe and zebra, gorilla
and chimpanzee, hedgehog and robin, horse and pig, horse and sheep, impala and jackal, langur and
peacock, lion and meerkat, lion and oryx, moose and chipmunk, peacock and pig; (an animal and an
object, 3) hamster and carrot, pony and red barn, snail and lantern; (two objects, 17) accordion and
scarecrow, bicycle and birdhouse, birdcage and hay bale, cactus and cooler box, coat rack and potted
plant, guitar and traffic cone, hay bale and ukulele, lighthouse and sailboat, mailbox and pumpkin,
microscope and test tube, rocking chair and cactus, soccer ball and park bench, suitcase and globe,
teddy bear and birthday cake, telescope and beach ball, typewriter and cactus, windmill and tractor.
They are also released with the code as the configuration file the trainer reads.

### Q2: success rate against PoE and CO3 over all held-out pairs and seeds

> "What is the success rate of the adapter compared with PoE and CO3 over all held-out pairs and
> multiple seeds?"

Answered under k4nd W3: Tables 1 and 3.

### Q3: restricting the adapter to the early steps

> "Given the finding in Sec. 3, does restricting the adapter to the early steps change the failure
> modes in Fig. 7?"

We tested this directly. Figure 11 switches the adapter on for the first 10, the first 25 or all 50
of the 50 steps. Compare its last two columns: the first 25 steps give essentially the same pictures
as all 50. The giraffe and lion row shows that the first 10 are too short for the learned correction,
although the measured correction needs only those ten (Section 3). The cow and horse row shows that
the species substitution of the submitted Fig. 7 (Figure 12 in the revision) persists at every
window, so it does not come from the late steps. Figure 10 counts each failure type across all 53
held-out pairs and eight seeds.

### Q4: pairs that do not compete for the same region

> "Does the adapter degrade PoE on pairs that do not compete for the same region, such as
> foreground/background pairs like lighthouse and sea?"

No. Table 3's not-competing group holds 21 held-out pairs whose concepts do not contend for one
region, including lighthouse and a stormy sea (Appendix C.2). On these, the adapter scores
0.73 ± 0.08 against the product's 0.74 ± 0.10 (BLIP-VQA, seeds 9 to 16, 95% intervals from
resampling pairs). The intervals overlap, so on pairs the product already composes, the adapter
neither helps nor harms on this measure. [TODO: CO3's row, slot R3.]
