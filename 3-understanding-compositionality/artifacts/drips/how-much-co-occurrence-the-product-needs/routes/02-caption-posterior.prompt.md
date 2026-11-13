# Route: deriving an expert from a caption posterior

Handed to `/submerge`. The sibling route covers Factorized Conditionals, local conditional scores
and CLEVR; this one covers the fourth word the walk used without a note behind it.

```text
/submerge I want to understand what it means to turn a caption into a generative model by Bayes,
and why doing that to two captions and multiplying the results is not the same operator as
multiplying two models each trained on its own images. I can follow the algebra when I see it but
I cannot derive it, and I have just been bitten by the difference.

Background

In my repo /home/molef/PhD/poe_repair_min-dog-and-rain I run an exact toy where each expert is a
mixture of Gaussians over 16x32 shape pictures, so every composed density and score is closed form.
The script is artifacts/notes/what-poe-loses-when-it-composes/toy_sampler_hybrids.py.

Two ways of building the two experts give completely different answers and I want to know why in
terms I can re-derive.

The first way puts the side-by-side pair picture inside each expert by hand at weight w, with the
rest of that expert's mass on its own single shapes. At w = 0.046 the exact product of the two
puts 300 of 300 draws on a clean circle-and-square pair.

The second way fixes one bank of seven pictures (three circles, three squares, one pair), scores
three captions against every picture, takes a softmax over the captions to get p(caption|picture),
and reads each expert off that by Bayes with a uniform picture prior. The pair picture's weight
inside the "circle" expert comes out at 0.046, the same number. But the exact product now puts
about 146 of 300 draws on a lone circle, about 150 on a lone square, and only 4 to 19 on a pair.
The numbers are in artifacts/results/how-much-co-occurrence-the-product-needs/toy-caption-competition.json.

My current explanation is that the Bayes-derived "circle" expert gives a lone square a weight of
0.107 (form 1) or 0.015 (form 2) rather than zero, and that this leakage is what the product keeps,
because a picture paired with itself pays no distance penalty while a circle paired with a square
pays exp(-2610). I want to know whether that explanation is right, where it is incomplete, and what
the general statement is.

Objectives

1. Teach me what a posterior over a latent image given a caption is, built up from a joint table I
   can compute by hand, so I can say what p(image|caption) proportional to p(caption|image) times a
   uniform prior means and why the prior being uniform matters.
2. Teach me what changes about a product of two densities when the two share support versus when
   their supports are disjoint, with the mixture-of-Gaussians pairing weight
   exp(-||mi - mj||^2 / (4 sigma^2)) as the worked instance, so I can predict which components
   survive before running anything.
3. Have me derive, on a bank of three or four pictures I can hold in my head, the exact product of
   two Bayes-derived caption experts, by hand, and show that its surviving components are the
   self-pairings weighted by the two posteriors multiplied together. Then have me predict the
   seven-picture case above and check the prediction against the JSON named in the background.
4. Teach me the condition under which the product of two caption-derived experts equals the product
   of two separately-trained experts, and what has to be true of the caption model for it to hold.
5. Connect this to the conditional independence assumption in Liu et al., "Compositional Visual
   Generation with Composable Diffusion Models" (arXiv 2206.01714). The assumption sits in the one
   sentence before their Equation 9. I want to be able to say, for my own toy, whether the dropped
   term is large or small and in which direction.

What to read

- /home/molef/PhD/poe_repair_min-dog-and-rain/artifacts/notes/what-poe-loses-when-it-composes/toy_sampler_hybrids.py,
  specifically the functions caption_match, caption_experts, hand_weighted and exact_samples. The
  pairing weight is the logw line inside exact_samples.
- /home/molef/PhD/poe_repair_min-dog-and-rain/artifacts/results/how-much-co-occurrence-the-product-needs/toy-caption-competition.json,
  the counts and both forms' implied weights.
- arXiv 2206.01714, section 4.2 up to and including Equation 9.
- The existing vault notes I should extend rather than duplicate:
  /home/molef/Obsidian/Mathematics/Statistics/Bayesian Inference/Posterior Distribution.md,
  /home/molef/Obsidian/Mathematics/Statistics/Probability/Conditional Independence.md, and
  /home/molef/Obsidian/Artificial Intelligence/Generative Modeling/Diffusion Models/Composition/Product, Mixture And Negation.md.

What is already known

- I am a visual, hands-on learner. Plain notation and a worked example first, the paper's symbols
  last with a mapping, one new idea per message, a picture for every rung.
- The product of two Gaussian mixtures is a mixture over every pairing of one component from each
  side, with the weight above. I understand this already and it does not need re-teaching, only
  using.
- In the seven-picture bank the squared distance between a single circle and a single square is
  104.4, and between a single circle and the pair picture 121.4, at sigma 0.1. So every pairing of
  unlike pictures is multiplied by about exp(-2610) or smaller.
- Liu et al. write no error term anywhere in the paper. The dropped factor, in their notation, is
  the gradient of log of p(c1,c2|x) over p(c1|x)p(c2|x). That is my derivation, not theirs, so do
  not attribute it to them.
- Code goes in /home/molef/playground/deep-learning/ and is disposable. Finished notes go to the
  Obsidian vault. Nothing is written into the poe_repair_min-dog-and-rain repo.

When something is missing

If a derivation needs a fact about the toy that is not in the two files named above, read the
script rather than assuming, and say what you read. If my explanation in the background turns out
to be wrong, say so plainly with the counterexample rather than teaching around it.

Rules

Show me the diagram set before writing any plan file. Do not write into
/home/molef/PhD/poe_repair_min-dog-and-rain. Do not re-teach the mixture pairing weight from
scratch.

What to hand back

- The journey folder under /home/molef/goal-setting/learning/deep-learning/diffusion-models/, its
  master plan, plan tree, gate file and sources, plus its row in /home/molef/goal-setting/learning/JOURNEYS.md.
- The hand-derivation and the prediction check under /home/molef/playground/deep-learning/, with
  the command that prints my predicted component weights beside the ones the script produces.
- The extensions to the three vault notes named above.

What you will see

Along the way: the diagram set first for a yes, then one plan at a time, each ending with
something I worked out myself.

At the end, in chat: what was asked in one sentence; every file written with its path; whether my
leakage explanation held, with the numbers that decided it; anything still open; at most one
question.

Done when I can predict, by hand and before running it, the share of the exact product that lands
on a pair for a caption bank I have not seen, and the prediction matches the script to within one
percentage point.
```
