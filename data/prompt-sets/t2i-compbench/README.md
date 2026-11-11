# T2I-CompBench validation prompts

`spatial_val.txt` (300 prompts, "a giraffe next to a lamp") and `complex_val.txt` (300 prompts, "The red hat was on top of the brown coat rack.") are T2I-CompBench's own validation sets for its 2D-spatial and complex categories (Huang et al., 2023), the sets its README names for metric calculation. Its README says 10 images are generated per prompt with a fixed seed across methods (`GORS_finetune/inference_eval.py`: seed 42 for the first image, 42 + n for the next).

Copied byte for byte from `examples/dataset/` of https://github.com/Karine-Huang/T2I-CompBench at commit `4aa4042`, the clone at `/datasets/mmolefe/repos/T2I-CompBench`. Its official scorers are UniDet (`UniDet_eval/2D_spatial_eval.py`) for spatial and the 3-in-1 score (`3_in_1_eval/3_in_1.py`, which reads UniDet's and BLIP-VQA's outputs) for complex.

Some prompts are never rendered under `scripts/showcase/disallowed_subjects.py`: three complex prompts name a spider or a bat, and 36 spatial and one complex prompt name a frog or a mouse, which the pair ledger has no decision for outside R1 and R5. `scripts/review_bench_sets.py --counts` prints what remains.
