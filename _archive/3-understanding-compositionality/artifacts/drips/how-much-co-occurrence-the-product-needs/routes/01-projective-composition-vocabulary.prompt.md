# Route: three words the composition literature uses that no note covers

Paste the block below into a fresh session. It is a `/submerge` brief.

```text
/submerge I keep meeting three words in the diffusion-composition literature and I cannot
derive any of them: factorized conditionals, local conditional scores, and CLEVR. I want to
understand them well enough to say, for a given pair of conditions and a given dataset, whether
score addition is exact, and to check that answer by hand on a small example I build myself.

Background

I work on product-of-experts composition for diffusion models, in the repo
/home/molef/PhD/poe_repair_min-dog-and-rain. Two papers by Arwen Bradley set the theory my work
argues against. "Mechanisms of Projective Composition of Diffusion Models" (arXiv 2502.04549,
ICML 2025) states the condition under which adding conditional scores is exact and calls it
Factorized Conditionals; its experiments are on CLEVR. "Local Mechanisms of Compositional
Generalization in Conditional Diffusion" (arXiv 2509.16447) proves that the same property is
equivalent to conditional scores being local, and fixes CLEVR composition by splitting the image
into patches conditioned only on what is inside them.

My own toy already measures the adjacent thing and does not touch these three. It is an exact
mixture of Gaussians over 16x32 shape pictures at
3-understanding-compositionality/artifacts/notes/what-poe-loses-when-it-composes/toy_sampler_hybrids.py, and it shows that the
product needs essentially no co-occurrence in its training images to prefer a composed picture,
while its samplers need about half the mass. That is the co-occurrence axis. Factorized
Conditionals is the coordinate-disjointness axis, and I cannot currently state it precisely or
test it.

Objectives

1. Teach me Factorized Conditionals from Bradley 2502.04549 Definition 5.2 and Theorem 5.3, in
   plain notation first and the paper's notation last, so I can state the condition in one
   sentence and say why score addition is exact under it.
2. Teach me the orthogonality heuristic of that paper's Lemma 8.1 and why Remark 5.5 says it is
   necessary and not sufficient, with a worked two-dimensional example where orthogonality holds
   and composition still fails.
3. Teach me local conditional scores from Bradley 2509.16447, and the equivalence it proves
   between locality and projective composition, so I can say what a score being local means for
   a text-conditioned model where no condition owns any pixels.
4. Teach me what CLEVR is as a dataset: how a scene is specified, what the conditions are in
   Bradley's experiments, and why its colour and position attributes make it the natural place to
   test coordinate disjointness.
5. Have me build one small example by hand, in a throwaway folder under
   /home/molef/playground/deep-learning/, where two conditions satisfy Factorized Conditionals
   exactly and score addition is exact to numerical tolerance, and a second where the condition
   is broken by one shared coordinate and the composition fails. Both checked by comparing the
   added score against the true joint score computed in closed form.

What to read

- arXiv 2502.04549, sections 3.2, 5.1 to 5.3, 8, and Figure 5. The repo already holds that
  figure at 3-understanding-compositionality/artifacts/notes/what-poe-loses-when-it-composes/figures/bradley-2025-projective-composition.png
  with its verbatim caption.
- arXiv 2509.16447, the locality equivalence and the patch-model experiment.
- /home/molef/PhD/poe_repair_min-dog-and-rain/plans/10-what-poe-loses-when-it-composes/what-poe-loses-when-it-composes.md
  sections 3.4 and 3.6, which already state both papers' results in this project's words and
  carry the reference list.
- /home/molef/PhD/poe_repair_min-dog-and-rain/plans/standing/literature/reading-register.md,
  the row for 2502.04549, which records what is on the abstract page and what is body-level, so
  nothing gets attributed to the paper that it does not say.
- /home/molef/PhD/poe_repair_min-dog-and-rain/artifacts/notes/what-poe-loses-when-it-composes/toy_sampler_hybrids.py,
  the toy whose exact product and three samplers the new example should resemble in style.

What is already known

- I am a visual, hands-on learner. Each rung derives in plain notation with a worked example and
  a picture, and prints the paper's symbols last with a mapping. One new idea per message.
- Lemma 5.7 of 2502.04549 is false as printed. A counterexample gives KL 2.883 against a bound of
  0.891, recorded at
  /home/molef/.claude/projects/-home-molef-PhD-poe-repair-min/memory/bradley-lemma-5-7-counterexample.md.
  Theorem K.4 misstates Delon and Desolneux Proposition 6, so the printed K.7 and K.8 lower bounds
  are also false. Do not teach either as a guarantee. Lemma 7.2 still holds, at tau at most 0.0891.
- The words "chimera" and "collapse" are this project's glosses for the paper's two failure modes
  and are not Bradley's terms. Do not attribute them to him.
- Section 5.3 of 2502.04549 concedes the conditions do not exactly hold in CLEVR and attaches no
  quantity to how far off they are. That missing quantity is the reason I am learning this.
- The product of two Gaussian mixtures is a mixture over all pairings, with the pairing of
  components at means mi and mj weighted by exp(-||mi - mj||^2 / (4 sigma^2)). The existing toy
  uses this and I already understand it.
- Learning journeys live under /home/molef/goal-setting/learning/ and are registered in
  JOURNEYS.md. Code is written in /home/molef/playground/<group>/ and is disposable. Finished
  notes go to the Obsidian vault under
  /home/molef/Obsidian/Artificial Intelligence/Generative Modeling/Diffusion Models/Composition/,
  where Composition.md, Composing Models With A Background.md and Product, Mixture And Negation.md
  already exist and should be extended rather than duplicated.

When something is missing

If a paper's PDF is not on disk, fetch it to the journey's sources folder and say so. If a
definition in the paper is ambiguous, state both readings with their consequence and ask me which
the rest of the journey should assume, and carry on with the one you recommend while I answer. If
a numerical check disagrees with the paper, show the disagreement with its numbers rather than
adjusting the check, and ask whether to pursue it.

Rules

Stop and show me the diagram set before writing any plan file. Nothing gets committed to the
learning tree until I have seen the pictures and said they match what I imagined. Never teach
Lemma 5.7 or Theorem K.4 as correct. Do not write anything into
/home/molef/PhD/poe_repair_min-dog-and-rain; this journey reads that repo and never writes to it.

What to hand back

- The journey folder under /home/molef/goal-setting/learning/deep-learning/diffusion-models/,
  with its master plan, its plan tree, its gate file naming all three roots, and its sources.
- Its row in /home/molef/goal-setting/learning/JOURNEYS.md.
- The approved diagram set, rendered, under the journey's artifacts/figures/.
- The hand-built example under /home/molef/playground/deep-learning/, with the command that
  prints the two score differences.

What you will see

Along the way: the diagram set first, for a yes or a correction, then one plan at a time.

At the end, in chat: what was asked in one sentence; every file written with its path; whether I
can now state Factorized Conditionals in one sentence and check it on my own example, with the
two printed score differences as the evidence; anything still open; at most one question.

Done when the hand-built example prints a score difference below 1e-6 in the factorized case and
a visibly larger one in the broken case, and I can say in my own words why.
```
