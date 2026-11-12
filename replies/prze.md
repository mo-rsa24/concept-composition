# Reply to Reviewer PrzE

Submission 56173. One section per reviewer point: the reviewer's words quoted, then the reply.
Equation, section and figure numbers refer to the revised paper
(`paper/iclr-56173/iclr2027_conference.tex`). PrzE's review is page 2 of `Reviews.pdf`.

### W1: lack of quantitative benchmarks

> "The paper does not report a single quantitative metric. Standard benchmarks such as T2I-Compench,
> GenEval and standard metrics such as CLIP, Imagereward are completely absent in this paper. The
> empirical results and a few qualitative results are insufficient for the claim in this paper."

The revision adds quantitative evaluation. Table 1 scores every method on the 276 Attend-and-Excite
prompts with BLIP-VQA (the T2I-CompBench composition metric), CLIP score and ImageReward. Table 2 reports GenEval's two-object, attribute-binding and position tasks
and T2I-CompBench's colour and shape sets, scored by their official evaluators.

### W2: narrow baselines and missing details

> "There is only CO3 baseline compared, the paper doesn't provide other corrected methods such as
> MCMC corrector. The paper only compares the baseline CO3 without the corrector they mention. The 48
> concept pairs are not listed so it's impossible to see what the adapter is trained on."

Figure 5 now includes a product of experts with a Langevin corrector (Du et al., 2023), the MCMC
correction the paper discusses, run at its published setting from the same starting latent, beside CO3,
ToMe and our adapter. Table 1 compares with CO3 on every image. Table 1 adds ToMe, run on SDXL from
the same starting latents. Section 4 now describes the training set and how pairs were
chosen (Figure 7); the 48 pairs are listed in our reply to Reviewer UHz6 and released with the code.

### Q1: quantitative benchmark

> "What are the evaluation scores for the proposed method? Can you show the results on some
> benchmarks and compare your method with other standard baselines?"

Answered under W1 and W2: Tables 1 and 2.

### Q2: multiple concepts

> "How does your method perform when composing more than two concepts? Can you show some results for
> more than two concepts?"

Appendix C.4 (Figure 16) shows one triple of each kind. The plain product fuses the three concepts or
drops one; the adapter, trained only on pairs, separates them in every case and draws all three named
things when the triple includes objects, but with two or three animals it draws only two. The
Discussion now addresses this.
