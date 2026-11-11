# GenEval evaluation prompts

`evaluation_metadata.jsonl` is GenEval's own prompt file (Ghosh et al., 2023), one JSON object per prompt with its task `tag`, the COCO classes and counts the detector must find, and the prompt text. It holds 553 prompts in six tasks; slot R14 uses two of them: `two_object` (99 prompts, "a photo of a bench and a sports ball") and `counting` (80 prompts, "a photo of two clocks").

Copied byte for byte from `prompts/evaluation_metadata.jsonl` of https://github.com/djghosh13/geneval at commit `af4902f`, the clone at `/datasets/mmolefe/repos/geneval` whose `evaluation/evaluate_images.py` scores the renders. GenEval's own generator draws 4 images per prompt (`generation/diffusers_generate.py`, `--n_samples 4`).

Five prompts are never rendered under `scripts/showcase/disallowed_subjects.py`: four `two_object` prompts and one `counting` prompt name a baseball bat, and five `two_object` prompts name a computer mouse, which the pair ledger has no decision for outside R1 and R5. `scripts/review_bench_sets.py --counts` prints what remains.
