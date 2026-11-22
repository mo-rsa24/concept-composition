# Field scout brief

You are surveying published research as of 2026-11-22. Search the web, open the papers themselves, and report only what you read in them.

## The two research problems

**Problem 1: composing experts that never met.** Two diffusion models are trained separately, on disjoint data. Neither has seen an image containing both concepts, and no joint prompt or joint data exists. The goal is to sample images that contain both concepts as separate objects by combining the two trained models at sampling time, or with a small amount of extra machinery.

**Problem 2: restoring plurality in one model.** One text-to-image diffusion model has seen both concepts. Combining its predictions for two prompts as a product of experts tends to fuse them into one hybrid object. The goal is to make the composed sample contain two distinct objects, for example with a corrector, a guidance term, a classifier, a regional method or a better sampler.

## What to find

Find the papers a careful reviewer would expect either project to cite or compare against. Cover direct rivals, the theory of when a product of diffusion models is valid, region-based and layout-based composition, samplers and correctors for composition, attention-based multi-object methods, the benchmarks used to score them, and work from 2025 and 2026 that a slower search would miss.

These are already known, so include each one that matters and verify it:

Composable Diffusion (2206.01714), Reduce Reuse Recycle (2302.11552), SuperDiff (2412.17762), Feynman-Kac Correctors (2503.02819), MultiDiffusion (2302.08113), Projective Composition (2502.04549), Local Mechanisms of Compositional Generalization (2509.16447), CoInD (2503.01145), CO3 (2509.25940), ToMe (2411.07132), Attend-and-Excite (2301.13826), Compositional Sculpting (2309.16115), Catastrophic Compositional Generation (2606.23920), Test-Time Compositional Generalization (2605.07078), Diffusion Domain Expansion (2605.23275), PiCo (2505.03203), FreeFuse (2510.23515).

At least 10 of your rows must be papers **not** on that list.

## Rules

- Return at most 30 rows, ordered by how much each matters to the two problems.
- Open each paper's arXiv page, HTML version or PDF. If you cannot open it, keep the row, set `read` to `could not read`, and leave `evidence_quote` empty.
- `evidence_quote` must be one sentence copied word for word from the paper, supporting `idea`.
- Copy a number only if you saw it in the paper, and say where it is. Never estimate or fill a field from memory. Write `not stated` when the paper does not say.
- Check that each arXiv id matches the title.

## Output

Write a JSON list to the output path you were given, and nothing else to that file. Each row:

```json
{
  "title": "exact paper title",
  "arxiv_id": "2412.17762 or null",
  "year": 2024,
  "venue": "where it was published, or preprint",
  "serves": "problem 1, problem 2, or both",
  "relation": "direct rival, partial rival, theory, tool, or benchmark",
  "idea": "one plain sentence on what the method does",
  "needs_joint_data": "yes, no, or not stated: does it need images or a prompt showing both concepts?",
  "models": "the models it composes or uses",
  "datasets": "the datasets or prompt sets it reports on",
  "headline_result": "its main number with metric and dataset, or not stated",
  "where": "section, table or page of that number",
  "evidence_quote": "one sentence copied word for word",
  "url": "the page you read",
  "code_url": "repository URL, or not stated",
  "read": "full text, abstract only, or could not read"
}
```
