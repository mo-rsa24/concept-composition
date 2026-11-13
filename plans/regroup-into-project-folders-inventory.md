# Regrouping inventory: every item in concept-composition and where it goes

Every scope, finding, walk, idea map, results folder, note, scene, experiment package and script
in the three checkouts of `concept-composition` (`main` is `/home/molef/PhD/concept-composition`,
`dog-and-rain` and `reviews` its two worktrees), placed against a project. Read by
[the regrouping plan](regroup-into-project-folders.md). Correct a row by changing its project
here; the move reads this file.

**How each row was placed**, in the "Why" column: `name` is a name rule (silicotb, x-ray, toy,
and-vector, inclusive, classifier); `cited Nx by P` counts mentions in that project's scopes,
routes, walk maps, idea maps, results cards and findings, and needs 70% of all mentions; `called
by` is a script called from that project's scripts; `judged by name` is a reading of the name
where nothing cites the item; `cited by nothing` sends it to the archive list.

**The "New name" column** is empty where the name already follows
`~/.claude/ARTIFACT_TREE_FORMAT.md` (lower kebab case, a results folder named by the question it
answers, no skill or process word, no private label). A filled cell is the name the move gives
it. File names inside each folder get the same check when its project is moved.

## Counts

| Project | scopes | report | drips | ideas | results | notes | scenes | experiments | scripts | total |
|---|---|---|---|---|---|---|---|---|---|---|
| 1-lora-correction | 13 | 15 | 17 | 6 | 30 | 2 | 5 | 16 | 95 | 199 |
| 2-classifier-guidance | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| 3-understanding-compositionality | 2 | 2 | 13 | 9 | 23 | 4 | 5 | 1 | 23 | 82 |
| 4-inclusive-captions | 2 | 0 | 1 | 2 | 3 | 0 | 1 | 1 | 2 | 12 |
| 5-text-embedding-third-expert | 0 | 1 | 5 | 2 | 4 | 0 | 0 | 0 | 20 | 32 |
| leaves for the medical portfolio | 3 | 0 | 3 | 2 | 4 | 0 | 0 | 0 | 3 | 15 |
| stays at the root, shared | 11 | 0 | 1 | 4 | 3 | 4 | 3 | 0 | 5 | 31 |
| _archive/, cited by nothing | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 141 | 143 |
| **all** | 31 | 18 | 43 | 25 | 67 | 10 | 14 | 18 | 289 | **515** |

## The names that change

| Kind | Now | New name | Project |
|---|---|---|---|
| results | `the-inclusive-captions-photo-pool` | `how-many-photos-the-inclusive-captions-pilot-can-train-on` | 4-inclusive-captions |
| results | `the-inclusive-captions-training-set` | moves to 4-inclusive-captions/data/training-set (a data set, not a result) | 4-inclusive-captions |
| results | `joint-against-poe-by-kind-of-composition` | `does-poe-match-the-joint-for-each-kind-of-composition` | 3-understanding-compositionality |
| drips | `answer-k4nd-and-uhz6` | `answer-reviewers-k4nd-and-uhz6` | 1-lora-correction |
| drips | `prze-10-wave-1` | `price-the-review-jobs-and-launch-the-first-renders` | 1-lora-correction |
| drips | `prze-2-langevin-corrector` | `prove-the-langevin-corrector-for-the-six-column-figure` | 1-lora-correction |
| report | `RESULTS_SUMMARY.md` | `results-summary.md` | 1-lora-correction |
| report | `instrument_smoke.md` | `instrument-smoke-test.md` | 1-lora-correction |
| report | `normalization_preregistration.md` | `normalization-preregistration.md` | 1-lora-correction |
| results | `co3-protocol-run-by-us` | `how-every-sdxl-method-compares-on-held-out-pairs` | 1-lora-correction |
| results | `every-adapter-across-ten-seeds` | `does-each-adapter-hold-across-ten-seeds` | 1-lora-correction |
| results | `four-pairs-four-methods` | `how-four-methods-compare-on-four-pairs` | 1-lora-correction |
| results | `four-pairs-one-adapter` | `does-one-adapter-compose-four-pairs` | 1-lora-correction |
| results | `residual-dynamics` | ? empty README; name decided after reading its two subfolders | 1-lora-correction |
| scopes | `prze-10-wave-1-renders.md` | `price-the-review-jobs-and-launch-the-first-renders.md` | 1-lora-correction |
| scopes | `prze-2-langevin-corrector.md` | `prove-the-langevin-corrector-for-the-six-column-figure.md` | 1-lora-correction |
| scopes | `prze-review-baselines.md` | `the-baselines-the-reviewers-asked-for.md` | 1-lora-correction |
| results | `overlap-regimes-on-real-films` | `where-each-overlap-regime-shows-on-real-films` | leaves for the medical portfolio |
| ideas | `codex-diagram-render-queue` | `diagrams-waiting-to-be-drawn` | stays at the root, shared |
| drips | `the-positional-and-vector` | `can-a-positional-and-vector-restore-composition` | 5-text-embedding-third-expert |
| results | `the-positional-and-vector` | `can-a-positional-and-vector-restore-composition` | 5-text-embedding-third-expert |

## 1-lora-correction

| Kind | Item | New name | In | Why |
|---|---|---|---|---|
| scopes | `01-showcase-the-trained-lora` |  | main, dog-and-rain, reviews | scope number |
| scopes | `02-can-we-trust-the-compose-rate` |  | main, dog-and-rain, reviews | scope number |
| scopes | `03-does-the-correction-cause-composition` |  | main, dog-and-rain, reviews | scope number |
| scopes | `04-does-the-fix-reach-unseen-pairs` |  | main, dog-and-rain, reviews | scope number |
| scopes | `05-when-does-the-outcome-lock-in` |  | main, dog-and-rain, reviews | scope number |
| scopes | `06-is-the-gap-the-samplers-or-the-models` |  | main, dog-and-rain, reviews | scope number |
| scopes | `07-writing-the-paper` |  | main, dog-and-rain, reviews | scope number |
| scopes | `08-improve-the-rank-32-pooled-adapter` |  | main, dog-and-rain, reviews | scope number |
| scopes | `09-designing-the-correction-loss` |  | main, dog-and-rain, reviews | scope number |
| scopes | `present-revision-to-supervisor.md` |  | main | loose plan file, judged by name |
| scopes | `prze-10-wave-1-renders.md` | price-the-review-jobs-and-launch-the-first-renders.md | main, reviews | loose plan file, judged by name |
| scopes | `prze-2-langevin-corrector.md` | prove-the-langevin-corrector-for-the-six-column-figure.md | main, reviews | loose plan file, judged by name |
| scopes | `prze-review-baselines.md` | the-baselines-the-reviewers-asked-for.md | main, reviews | loose plan file, judged by name |
| report | `RESULTS_SUMMARY.md` | results-summary.md | main, dog-and-rain, reviews | cited 6x by L cards, walks or findings |
| report | `decision-timeline.md` |  | main, dog-and-rain, reviews | cited 3x by L cards, walks or findings |
| report | `designing-the-correction-loss` |  | main, dog-and-rain, reviews | cited 222x by L |
| report | `diagram-prompts.md` |  | main, dog-and-rain, reviews | cited 297x by L |
| report | `diagrams` |  | main, dog-and-rain, reviews | cited 541x by L |
| report | `does-the-fix-reach-unseen-pairs` |  | main, dog-and-rain, reviews | cited 201x by L |
| report | `experiments-log.md` |  | main, dog-and-rain, reviews | cited 9x by L cards, walks or findings |
| report | `finding-and-explainer-template.md` |  | main, dog-and-rain, reviews | cited 3x by L |
| report | `instrument_smoke.md` | instrument-smoke-test.md | main, dog-and-rain, reviews | cited 9x by L |
| report | `is-the-adapter-just-copying-the-joint-prompt` |  | main, dog-and-rain, reviews | judged by name |
| report | `is-the-gap-the-samplers-or-the-models` |  | main, dog-and-rain, reviews | cited 561x by L |
| report | `normalization_preregistration.md` | normalization-preregistration.md | main, dog-and-rain, reviews | cited 15x by L |
| report | `paper-evidence-index.md` |  | main, dog-and-rain, reviews | cited 12x by L |
| report | `paper-review` |  | main, reviews | judged by name |
| report | `when-does-the-outcome-lock-in` |  | main, dog-and-rain, reviews | cited 281x by L |
| drips | `answer-iclr-reviews` |  | main, reviews | judged by name |
| drips | `answer-k4nd-and-uhz6` | answer-reviewers-k4nd-and-uhz6 | main | judged by name |
| drips | `consistency-model-basin-oracle` |  | main, dog-and-rain, reviews | cited 6x by L |
| drips | `does-a-clip-free-model-hybridise` |  | dog-and-rain | cited 62x by L |
| drips | `free-bound-model-share` |  | main, dog-and-rain, reviews | judged by name |
| drips | `how-big-the-correction-is` |  | main, dog-and-rain, reviews | judged by name |
| drips | `proofread-full-pass` |  | main, reviews | judged by name |
| drips | `proofread-section-3` |  | main, dog-and-rain, reviews | cited 3x by L |
| drips | `proofread-section-4` |  | main, reviews | judged by name |
| drips | `proofread-section-5` |  | main, dog-and-rain, reviews | judged by name |
| drips | `proofread-the-introduction` |  | main, dog-and-rain, reviews | judged by name |
| drips | `proofread-your-comments` |  | main, reviews | judged by name |
| drips | `prze-10-wave-1` | price-the-review-jobs-and-launch-the-first-renders | main, reviews | judged by name |
| drips | `prze-2-langevin-corrector` | prove-the-langevin-corrector-for-the-six-column-figure | main, reviews | judged by name |
| drips | `review-baselines` |  | main, reviews | judged by name |
| drips | `second-animal-species` |  | main, reviews | judged by name |
| drips | `seeing-the-correction-loss` |  | main, dog-and-rain, reviews | cited 3x by L |
| ideas | `designing-the-correction-loss` |  | main, dog-and-rain, reviews | cited 222x by L |
| ideas | `fixing-the-flat-composition` |  | main, dog-and-rain, reviews | judged by name |
| ideas | `improving-the-pooled-lora-run` |  | main, dog-and-rain, reviews | cited 39x by L |
| ideas | `parallel-sessions-on-poe-dynamics-and-fixes` |  | main, dog-and-rain, reviews | cited 6x by L |
| ideas | `the-shortest-path-to-a-clean-composing-render` |  | main, dog-and-rain, reviews | judged by name |
| ideas | `what-other-samplers-do-with-the-dropped-factor` |  | main, dog-and-rain, reviews | judged by name |
| results | `can-we-trust-the-compose-score` |  | main, dog-and-rain, reviews | cited 43x by L |
| results | `co3-protocol-run-by-us` | how-every-sdxl-method-compares-on-held-out-pairs | main | judged by name |
| results | `do-sampling-strategies-fix-the-adapter` |  | main | judged by name |
| results | `does-a-clip-free-model-hybridise` |  | dog-and-rain | cited 62x by L |
| results | `does-a-langevin-corrector-fix-the-product` |  | main, reviews | judged by name |
| results | `does-restricting-the-adapter-to-early-steps-change-the-failures` |  | main | judged by name |
| results | `does-sdxl-compose-at-pixarts-settings` |  | dog-and-rain | cited 23x by L |
| results | `does-the-adapter-compose-three-concepts` |  | main | judged by name |
| results | `does-the-fix-reach-unseen-pairs` |  | main, dog-and-rain, reviews | cited 201x by L |
| results | `does-the-interaction-term-cause-composition` |  | main, dog-and-rain, reviews | judged by name |
| results | `every-adapter-across-ten-seeds` | does-each-adapter-hold-across-ten-seeds | main | judged by name |
| results | `four-pairs-four-methods` | how-four-methods-compare-on-four-pairs | main | judged by name |
| results | `four-pairs-one-adapter` | does-one-adapter-compose-four-pairs | main | judged by name |
| results | `how-big-the-figure-text-prints` |  | main | judged by name |
| results | `how-big-the-strength-grid-should-be` |  | main | cited 4x by L cards, walks or findings |
| results | `how-far-the-corrected-run-separates-from-the-uncorrected-one` |  | main, dog-and-rain, reviews | judged by name |
| results | `how-much-correction-is-needed` |  | main, dog-and-rain, reviews | cited 3x by L cards, walks or findings |
| results | `how-the-adapter-scores-against-other-methods` |  | main | judged by name |
| results | `how-the-adapter-settings-change-cat-and-dog` |  | main | judged by name |
| results | `is-the-gap-the-samplers-or-the-models` |  | main, dog-and-rain, reviews | cited 561x by L |
| results | `residual-dynamics` | ? empty README; name decided after reading its two subfolders | main, dog-and-rain, reviews | cited 6x by L |
| results | `what-each-scorer-says-about-two-objects` |  | main | judged by name |
| results | `what-the-main-study-grids-show` |  | main | judged by name |
| results | `when-the-correction-must-arrive` |  | main, dog-and-rain, reviews | cited 6x by L cards, walks or findings |
| results | `which-adapter-composes-cat-and-dog` |  | main | judged by name |
| results | `which-adapter-composes-the-showcase-pairs` |  | main | judged by name |
| results | `which-fidelity-fix-wins` |  | main | cited 27x by L cards, walks or findings |
| results | `which-new-pairs-compose-cleanly` |  | main | judged by name |
| results | `which-way-the-correction-points` |  | main, dog-and-rain, reviews | cited 9x by L |
| results | `why-the-second-animal-loses-its-species` |  | main | judged by name |
| notes | `from-an-x-post-to-a-twisted-smc-baseline` |  | main, dog-and-rain, reviews | cited 8x by L cards, walks or findings |
| notes | `inference-time-scaling-as-a-search-over-noise` |  | main, dog-and-rain, reviews | cited 6x by L |
| scenes | `how-much-correction-is-needed` |  | main, dog-and-rain, reviews | cited 3x by L cards, walks or findings |
| scenes | `iclr-review-response` |  | main | judged by name |
| scenes | `noise-shell-and-basins` |  | main, dog-and-rain, reviews | cited 3x by L |
| scenes | `what-the-cached-runs-already-show` |  | main, dog-and-rain, reviews | cited 48x by L |
| scenes | `where-each-condition-lands` |  | main, dog-and-rain, reviews | cited 78x by L |
| experiments | `cfg_window_with_lora` |  | main, dog-and-rain, reviews | judged by name |
| experiments | `cfg_window_without_lora` |  | main, dog-and-rain, reviews | cited 3x by L |
| experiments | `compose_scorer_validation` |  | main, dog-and-rain, reviews | cited 39x by L |
| experiments | `correction_outside_the_unet` |  | main, dog-and-rain, reviews | judged by name |
| experiments | `cross_pair_lora_pooling` |  | main, dog-and-rain, reviews | cited 90x by L |
| experiments | `does_the_fix_reach_unseen_pairs` |  | main, dog-and-rain, reviews | cited 3x by L |
| experiments | `fk_steering` |  | main, dog-and-rain, reviews | cited 213x by L |
| experiments | `held_out_seeds` |  | main, dog-and-rain, reviews | cited 3x by L cards, walks or findings |
| experiments | `interaction_term` |  | main, dog-and-rain, reviews | cited 556x by L |
| experiments | `internal_force_failure` |  | main, dog-and-rain, reviews | judged by name |
| experiments | `mechanism_study` |  | main, dog-and-rain, reviews | cited 66x by L |
| experiments | `noise_search` |  | main, dog-and-rain, reviews | cited 210x by L |
| experiments | `noise_trajectory_search` |  | main, dog-and-rain, reviews | cited 78x by L |
| experiments | `one_pair_one_seed` |  | main, dog-and-rain, reviews | cited 57x by L |
| experiments | `residual_between_mono_and_poe` |  | main, dog-and-rain, reviews | cited 3x by L |
| experiments | `twisted_smc` |  | main, dog-and-rain, reviews | cited 96x by L |
| scripts | `adapter_transfers.py` |  | main, dog-and-rain, reviews | cited 24x by L |
| scripts | `adapter_vs_oracle_cost.py` |  | main, dog-and-rain, reviews | cited 12x by L |
| scripts | `animals_compose_transfer` |  | main, dog-and-rain, reviews | cited 18x by L |
| scripts | `build_cross_manifest.py` |  | main, dog-and-rain, reviews | called by 1 L script(s) |
| scripts | `build_eval_cache.py` |  | main, dog-and-rain, reviews | cited 6x by L cards, walks or findings |
| scripts | `build_group_cache.py` |  | main, dog-and-rain, reviews | called by 1 L script(s) |
| scripts | `build_lora_inspector_mds.py` |  | main, dog-and-rain, reviews | cited 6x by L |
| scripts | `build_lora_inspector_mds_semantic.py` |  | main, dog-and-rain, reviews | cited 6x by L |
| scripts | `build_lora_manifest.py` |  | main, dog-and-rain, reviews | cited 3x by L cards, walks or findings |
| scripts | `build_superdiff_cache.py` |  | main, dog-and-rain, reviews | cited 9x by L |
| scripts | `build_window_manifest.py` |  | main, dog-and-rain, reviews | cited 6x by L |
| scripts | `cache_smoke.py` |  | main, dog-and-rain, reviews | cited 12x by L |
| scripts | `check_fixed_schedule.py` |  | main, dog-and-rain, reviews | called by 1 L script(s) |
| scripts | `clean_tail_k_sweep.py` |  | main, dog-and-rain, reviews | cited 3x by L |
| scripts | `climb.py` |  | main, dog-and-rain, reviews | cited 120x by L |
| scripts | `cluster.sh` |  | main, dog-and-rain, reviews | cited 1472x by L |
| scripts | `commitment` |  | main, dog-and-rain, reviews | cited 204x by L |
| scripts | `commitment_step.py` |  | main, dog-and-rain, reviews | cited 9x by L cards, walks or findings |
| scripts | `commitment_step_frames.py` |  | main, dog-and-rain, reviews | cited 3x by L cards, walks or findings |
| scripts | `compose_rate_vs_correction.py` |  | main, dog-and-rain, reviews | cited 6x by L |
| scripts | `composition_scatter.py` |  | main, dog-and-rain, reviews | cited 9x by L |
| scripts | `concept_multi_poe.py` |  | dog-and-rain | cited 7x by L |
| scripts | `correction_loss_variants` |  | main, dog-and-rain, reviews | cited 27x by L |
| scripts | `correction_size_over_the_run.py` |  | main, dog-and-rain, reviews | cited 45x by L |
| scripts | `correction_size_vs_run_position.py` |  | main, dog-and-rain, reviews | cited 3x by L |
| scripts | `corrector_residual_curve.py` |  | main, dog-and-rain, reviews | cited 69x by L |
| scripts | `corrector_window_sweep.py` |  | main, dog-and-rain, reviews | cited 30x by L |
| scripts | `cross_pair_lora_pooling` |  | main, dog-and-rain, reviews | cited 90x by L |
| scripts | `cross_seed_lora_pooling` |  | main, dog-and-rain, reviews | cited 15x by L |
| scripts | `cross_tab.py` |  | main, dog-and-rain, reviews | called by 1 L script(s) |
| scripts | `decode_trajectory_frames.py` |  | main, dog-and-rain, reviews | called by 4 L script(s) |
| scripts | `dial_vs_three_instruments.py` |  | main, dog-and-rain, reviews | cited 3x by L |
| scripts | `direction_agreement_as_noise_is_moved.py` |  | main, dog-and-rain, reviews | cited 3x by L |
| scripts | `dose_strip.py` |  | main, dog-and-rain, reviews | cited 24x by L |
| scripts | `dose_trajectory_divergence.py` |  | main, dog-and-rain, reviews | called by 1 L script(s) |
| scripts | `fidelity_harness.py` |  | main, dog-and-rain, reviews | cited 12x by L cards, walks or findings |
| scripts | `figure_candidates.py` |  | main, dog-and-rain, reviews | cited 6x by L |
| scripts | `figure_candidates.sbatch` |  | main, dog-and-rain, reviews | cited 6x by L |
| scripts | `fk_steering` |  | main, dog-and-rain, reviews | cited 213x by L |
| scripts | `fork_curve.py` |  | main, dog-and-rain, reviews | cited 12x by L |
| scripts | `idea_probes` |  | main, dog-and-rain, reviews | cited 21x by L cards, walks or findings |
| scripts | `interaction_term_dose_matched.py` |  | main, dog-and-rain, reviews | cited 39x by L |
| scripts | `interaction_term_inject.py` |  | main, dog-and-rain, reviews | cited 18x by L |
| scripts | `interaction_term_window.py` |  | main, dog-and-rain, reviews | cited 63x by L |
| scripts | `joint_poe_adapter_grid.py` |  | main, dog-and-rain, reviews | called by 2 L script(s) |
| scripts | `later_start_grid.py` |  | main, dog-and-rain, reviews | cited 9x by L |
| scripts | `longer_correction_grid.py` |  | main, dog-and-rain, reviews | cited 9x by L |
| scripts | `lora_inspector.py` |  | main, dog-and-rain, reviews | cited 12x by L |
| scripts | `manifold` |  | main, dog-and-rain, reviews | cited 270x by L |
| scripts | `manifold_slide.py` |  | main, dog-and-rain, reviews | cited 18x by L |
| scripts | `mechanism_study` |  | main, dog-and-rain, reviews | cited 66x by L |
| scripts | `noise_search` |  | main, dog-and-rain, reviews | cited 210x by L |
| scripts | `noise_trajectory_search` |  | main, dog-and-rain, reviews | cited 78x by L |
| scripts | `normalization_candidates.py` |  | main, dog-and-rain, reviews | cited 3x by L |
| scripts | `overcorrection_grid.py` |  | main, dog-and-rain, reviews | called by 1 L script(s) |
| scripts | `pixart_multi_poe.py` |  | dog-and-rain | cited 10x by L |
| scripts | `plan_pulse.py` |  | main, dog-and-rain, reviews | cited 3x by L |
| scripts | `plausibility_climb.py` |  | main, dog-and-rain, reviews | cited 6x by L cards, walks or findings |
| scripts | `plot_dose_curves.py` |  | main, dog-and-rain, reviews | cited 33x by L |
| scripts | `plot_growing_window_curves.py` |  | main, dog-and-rain, reviews | cited 3x by L |
| scripts | `plot_window_curves.py` |  | main, dog-and-rain, reviews | cited 15x by L |
| scripts | `push_brightness_bias.py` |  | main, dog-and-rain, reviews | cited 3x by L |
| scripts | `quality_control.py` |  | main, dog-and-rain, reviews | cited 18x by L |
| scripts | `recover_growing_window_frames.py` |  | main, dog-and-rain, reviews | called by 2 L script(s) |
| scripts | `rt_correction_payoff.py` |  | main, dog-and-rain, reviews | cited 3x by L |
| scripts | `rt_direction_companions.py` |  | main, dog-and-rain, reviews | cited 6x by L |
| scripts | `rt_noise_interpolation.py` |  | main, dog-and-rain, reviews | cited 3x by L |
| scripts | `rt_reuse_figure.py` |  | main, dog-and-rain, reviews | cited 3x by L |
| scripts | `run_cross_sweep.py` |  | main, dog-and-rain, reviews | called by 3 L script(s) |
| scripts | `run_lora_inspector.sh` |  | main, dog-and-rain, reviews | cited 6x by L |
| scripts | `same_rule_four_models.py` |  | main, dog-and-rain, reviews | cited 32x by L |
| scripts | `same_rule_four_models.sbatch` |  | main, dog-and-rain, reviews | cited 32x by L |
| scripts | `score_seed_breadth.py` |  | main, dog-and-rain, reviews | called by 1 L script(s) |
| scripts | `sdxl_matched_multi_poe.py` |  | dog-and-rain | cited 3x by L |
| scripts | `showcase` |  | main, dog-and-rain, reviews | cited 1560x by L |
| scripts | `shrinkage_per_step.py` |  | main, dog-and-rain, reviews | cited 6x by L |
| scripts | `size_vs_timing.py` |  | main, dog-and-rain, reviews | cited 9x by L |
| scripts | `snr_collapse.py` |  | main, dog-and-rain, reviews | cited 33x by L |
| scripts | `spectrum.py` |  | main, dog-and-rain, reviews | cited 93x by L |
| scripts | `spectrum_windowed.py` |  | main, dog-and-rain, reviews | cited 12x by L |
| scripts | `storage` |  | main, dog-and-rain, reviews | cited 285x by L |
| scripts | `superdiff` |  | main, dog-and-rain, reviews | cited 603x by L |
| scripts | `timing_cliff_matched_dose.py` |  | main, dog-and-rain, reviews | cited 6x by L |
| scripts | `timing_vs_dose.py` |  | main, dog-and-rain, reviews | cited 6x by L |
| scripts | `trajectory_divergence.py` |  | main, dog-and-rain, reviews | cited 12x by L |
| scripts | `twisted_smc` |  | main, dog-and-rain, reviews | cited 96x by L |
| scripts | `what_the_adapter_changes.py` |  | main, dog-and-rain, reviews | cited 3x by L |
| scripts | `what_the_product_misses.py` |  | main, dog-and-rain, reviews | cited 12x by L |
| scripts | `window_map_all_pairs.py` |  | main, dog-and-rain, reviews | cited 6x by L |
| scripts | `window_position_grid.py` |  | main, dog-and-rain, reviews | cited 9x by L |
| scripts | `window_samples_over_map.py` |  | main, dog-and-rain, reviews | cited 3x by L |
| scripts | `window_strip.py` |  | main, dog-and-rain, reviews | cited 6x by L |
| scripts | `window_tab.py` |  | main, dog-and-rain, reviews | called by 1 L script(s) |
| scripts | `window_vs_commitment.py` |  | main, dog-and-rain, reviews | cited 6x by L cards, walks or findings |
| scripts | `window_width.py` |  | main, dog-and-rain, reviews | cited 9x by L |

## 2-classifier-guidance

| Kind | Item | New name | In | Why |
|---|---|---|---|---|
| drips | `can-a-classifier-supply-the-missing-term` |  | dog-and-rain | name |

## 3-understanding-compositionality

| Kind | Item | New name | In | Why |
|---|---|---|---|---|
| scopes | `10-does-roentgen-draw-two-findings` |  | main, reviews | scope number |
| scopes | `10-what-poe-loses-when-it-composes` |  | main, dog-and-rain, reviews | scope number |
| report | `does-training-longer-help-the-pooled-lora` |  | main, dog-and-rain, reviews | name |
| report | `what-the-adapter-ablations-show` |  | main, dog-and-rain, reviews | cited 2x by K cards, walks or findings |
| drips | `a-dog-and-rain` |  | main, dog-and-rain, reviews | name |
| drips | `camel-and-dog-in-the-training-data` |  | dog-and-rain | name |
| drips | `combine-kinds-of-composition` |  | dog-and-rain | name |
| drips | `forward-process-and-the-score` |  | dog-and-rain | name |
| drips | `how-much-co-occurrence-the-product-needs` |  | dog-and-rain | name |
| drips | `is-the-layout-the-missing-piece` |  | dog-and-rain | name |
| drips | `read-my-eye-my-brain` |  | main, reviews | name |
| drips | `reduce-reuse-recycle` |  | main, reviews | cited 18x by K |
| drips | `reduce-reuse-recycle-mscluster112` |  | main, reviews | cited 14x by K cards, walks or findings |
| drips | `showcase-the-trained-adapter` |  | main, dog-and-rain, reviews | name |
| drips | `simple-distributions-rebuild` |  | dog-and-rain | name |
| drips | `toy-failure-six-checkpoints` |  | dog-and-rain | name |
| drips | `what-poe-is-and-what-it-drops` |  | main, dog-and-rain, reviews | name |
| ideas | `disjoint-marginal-support-and-poe-chimeras` |  | main, dog-and-rain, reviews | judged by name |
| ideas | `earn-the-plurality-name` |  | main, dog-and-rain, reviews | judged by name |
| ideas | `is-poe-ill-posed-for-composition` |  | dog-and-rain | cited 6x by K |
| ideas | `one-training-recipe` |  | main, reviews | name |
| ideas | `poe-chimera-from-the-floor-up` |  | main, reviews | judged by name |
| ideas | `poe-overlap-dial` |  | dog-and-rain | cited 6x by K |
| ideas | `sdxl-shaped-hybrid-testbed` |  | dog-and-rain | cited 3x by K |
| ideas | `three-product-comparison` |  | main, dog-and-rain, reviews | judged by name |
| ideas | `which-variable-explains-what-poe-is-missing` |  | main, dog-and-rain, reviews | name |
| results | `can-a-score-see-species` |  | main | cited 8x by K cards, walks or findings |
| results | `does-a-camel-in-the-rain-come-out-wet` |  | dog-and-rain | name |
| results | `does-a-dog-in-the-rain-come-out-wet` |  | dog-and-rain | name |
| results | `does-an-effect-need-support` |  | dog-and-rain | name |
| results | `does-poe-of-cats-and-dogs-match-the-joint` |  | dog-and-rain | name |
| results | `does-the-cat-prompt-already-draw-dogs` |  | dog-and-rain | name |
| results | `does-the-hybrid-fit-both-captions` |  | dog-and-rain | name |
| results | `how-much-co-occurrence-the-product-needs` |  | dog-and-rain | name |
| results | `how-the-forward-process-dissolves-an-image` |  | dog-and-rain | name |
| results | `how-the-training-pairs-were-chosen` |  | main | name |
| results | `is-the-layout-the-missing-piece` |  | dog-and-rain | name |
| results | `joint-against-poe-by-kind-of-composition` | does-poe-match-the-joint-for-each-kind-of-composition | dog-and-rain | name |
| results | `what-co3-does-on-our-pairs` |  | main | cited 2x by K cards, walks or findings |
| results | `what-full-step-coverage-changes` |  | main | cited 2x by K cards, walks or findings |
| results | `what-the-adapter-ablations-show` |  | main | cited 2x by K cards, walks or findings |
| results | `what-the-fusion-is-not-caused-by` |  | dog-and-rain | name |
| results | `what-the-product-of-two-experts-draws-and-what-samplers-lose` |  | main | cited 3x by K cards, walks or findings |
| results | `what-the-training-captions-show` |  | dog-and-rain | name |
| results | `where-the-correction-lives-in-the-frame` |  | dog-and-rain | cited 1x by K cards, walks or findings |
| results | `which-checkpoint-to-take-each-figure-cell-from` |  | main | cited 6x by K cards, walks or findings |
| results | `which-pairs-trained-and-which-were-held-out` |  | main | name |
| results | `which-renders-show-it-working-and-failing` |  | main | cited 2x by K cards, walks or findings |
| results | `which-sampling-strategy-helps` |  | main | cited 2x by K cards, walks or findings |
| notes | `coind-conditional-independence-loss` |  | main, dog-and-rain, reviews | cited 6x by K |
| notes | `how-the-poe-toy-experiment-works` |  | main | name |
| notes | `interaction-term-as-pmi-gradient` |  | main, dog-and-rain, reviews | cited 3x by K cards, walks or findings |
| notes | `what-poe-loses-when-it-composes` |  | dog-and-rain | name |
| scenes | `how-the-caption-model-works` |  | dog-and-rain | cited 2x by K cards, walks or findings |
| scenes | `parts-world` |  | dog-and-rain | cited 6x by K |
| scenes | `three-ways-a-product-reaches-a-and-b` |  | dog-and-rain | judged by name |
| scenes | `what-an-expert-learns-cell-by-cell` |  | dog-and-rain | judged by name |
| scenes | `when-a-product-reaches-a-and-b` |  | dog-and-rain | judged by name |
| experiments | `reward_finetune` |  | main, dog-and-rain, reviews | cited 1x by K |
| scripts | `adapter_ablation_figures.py` |  | main, dog-and-rain, reviews | cited 7x by K cards, walks or findings |
| scripts | `attention_probe_read.py` |  | main, reviews | cited 2x by K cards, walks or findings |
| scripts | `build_training_cache.py` |  | main, dog-and-rain, reviews | name |
| scripts | `caption_readback.py` |  | main, dog-and-rain, reviews | name |
| scripts | `caption_wetness.py` |  | main, dog-and-rain, reviews | name |
| scripts | `caption_wetness.sbatch` |  | main, dog-and-rain, reviews | name |
| scripts | `census` |  | main, dog-and-rain, reviews | cited 184x by K |
| scripts | `concept_pair_pushes.py` |  | main, dog-and-rain, reviews | called by 1 K script(s) |
| scripts | `concept_pair_sheet.py` |  | main, dog-and-rain, reviews | cited 12x by K |
| scripts | `concept_pair_sheet.sbatch` |  | main, dog-and-rain, reviews | cited 12x by K |
| scripts | `frog_toad_late_window_boxes.py` |  | main, dog-and-rain, reviews | cited 1x by K |
| scripts | `hybrid_caption_fit` |  | dog-and-rain | name |
| scripts | `launch_recipe_run.sh` |  | main, reviews | cited 8x by K cards, walks or findings |
| scripts | `layout_from_joint_render.py` |  | main, dog-and-rain, reviews | name |
| scripts | `local_sdxl_poe.py` |  | dog-and-rain | cited 18x by K |
| scripts | `medical_pair_bar.py` |  | main, reviews | cited 2x by K |
| scripts | `medical_pair_score.py` |  | main, reviews | cited 2x by K |
| scripts | `medical_pair_table.py` |  | main, reviews | cited 2x by K |
| scripts | `render_joint_references.py` |  | main, dog-and-rain, reviews | called by 1 K script(s) |
| scripts | `render_repeat_and_ema.sbatch` |  | main, reviews | cited 2x by K cards, walks or findings |
| scripts | `species_judge.py` |  | main, reviews | cited 4x by K cards, walks or findings |
| scripts | `sweep_probe.sbatch` |  | main, dog-and-rain, reviews | cited 4x by K cards, walks or findings |
| scripts | `xhat0_readback.py` |  | main, dog-and-rain, reviews | cited equally by K and T; placed with K on 2026-11-13 |

## 4-inclusive-captions

| Kind | Item | New name | In | Why |
|---|---|---|---|---|
| scopes | `11-do-inclusive-captions-help-poe-compose` |  | main, dog-and-rain, reviews | scope number |
| scopes | `route-inclusive-captions-pilot.md` |  | main, dog-and-rain, reviews | route |
| drips | `inclusive-captions-pilot` |  | dog-and-rain | name |
| ideas | `ideas-that-tie-the-ladder-together.md` |  | dog-and-rain | cited 3x by I |
| ideas | `inclusive-captions-axis` |  | dog-and-rain | name |
| results | `does-any-caption-say-a-cat-and-a-dog` |  | dog-and-rain | judged by name |
| results | `the-inclusive-captions-photo-pool` | how-many-photos-the-inclusive-captions-pilot-can-train-on | dog-and-rain | name |
| results | `the-inclusive-captions-training-set` | moves to 4-inclusive-captions/data/training-set (a data set, not a result) | dog-and-rain | name |
| scenes | `how-the-inclusive-captions-pilot-works` |  | dog-and-rain | name |
| experiments | `inclusive_captions` |  | dog-and-rain | name |
| scripts | `inclusive_captions_both.sh` |  | dog-and-rain | name |
| scripts | `inclusive_captions_run.sh` |  | dog-and-rain | name |

## 5-text-embedding-third-expert

| Kind | Item | New name | In | Why |
|---|---|---|---|---|
| report | `can-a-text-space-third-expert-restore-composition.md` |  | dog-and-rain | name |
| drips | `a-learned-and-token` |  | dog-and-rain | name |
| drips | `does-one-prompt-already-say-one-animal` |  | dog-and-rain | name |
| drips | `separation-steering` |  | dog-and-rain | cited 5x by T cards, walks or findings |
| drips | `the-positional-and-vector` | can-a-positional-and-vector-restore-composition | dog-and-rain | name |
| drips | `today-the-third-concept` |  | dog-and-rain | name |
| ideas | `text-embedding-of-the-residual` |  | main, dog-and-rain, reviews | name |
| ideas | `the-and-vector-as-a-third-expert` |  | dog-and-rain | name |
| results | `does-one-prompt-already-say-one-animal` |  | dog-and-rain | name |
| results | `does-text-alone-predict-composition` |  | main, dog-and-rain, reviews | judged by name |
| results | `the-positional-and-vector` | can-a-positional-and-vector-restore-composition | dog-and-rain | name |
| results | `what-the-and-vector-is-made-of` |  | dog-and-rain | name |
| scripts | `and_vector_alignment_matrix.py` |  | dog-and-rain | name |
| scripts | `and_vector_fixed_x.py` |  | main, dog-and-rain, reviews | name |
| scripts | `and_vector_fixed_x.sbatch` |  | main, dog-and-rain, reviews | name |
| scripts | `and_vector_invert_x.py` |  | main, dog-and-rain, reviews | name |
| scripts | `and_vector_invert_x.sbatch` |  | main, dog-and-rain, reviews | name |
| scripts | `and_vector_norms.py` |  | dog-and-rain | name |
| scripts | `and_vector_per_step_x.py` |  | dog-and-rain | name |
| scripts | `and_vector_premise.py` |  | dog-and-rain | name |
| scripts | `and_vector_search.py` |  | dog-and-rain | name |
| scripts | `and_vector_search.sbatch` |  | dog-and-rain | name |
| scripts | `and_vector_sheet.py` |  | dog-and-rain | name |
| scripts | `and_vector_sheet.sh` |  | dog-and-rain | name |
| scripts | `and_vector_x_sheet.py` |  | dog-and-rain | name |
| scripts | `language_probes.py` |  | main, dog-and-rain, reviews | name |
| scripts | `learned_and_token.py` |  | dog-and-rain | name |
| scripts | `main_study_grids.py` |  | main, dog-and-rain, reviews | called by 3 T script(s) |
| scripts | `one_animal_axis.py` |  | dog-and-rain | name |
| scripts | `sd35_and_vector_render.py` |  | dog-and-rain | name |
| scripts | `separation_steering.py` |  | dog-and-rain | cited 5x by T cards, walks or findings |
| scripts | `separation_steering.sh` |  | dog-and-rain | cited 5x by T cards, walks or findings |

## leaves for the medical portfolio

| Kind | Item | New name | In | Why |
|---|---|---|---|---|
| scopes | `route-compose-x-rays-and-score-them.md` |  | main, dog-and-rain, reviews | route |
| scopes | `route-every-silicotb-chat-claim.md` |  | main, dog-and-rain, reviews | route |
| scopes | `route-scope-the-two-disease-problem.md` |  | dog-and-rain | route |
| drips | `does-roentgen-draw-two-findings` |  | main, reviews | name |
| drips | `silicotb-conversation` |  | main, dog-and-rain | name |
| drips | `what-two-diseases-do-on-one-film` |  | dog-and-rain | name |
| ideas | `silicotb-conversation` |  | dog-and-rain | name |
| ideas | `when-poe-reaches-a-and-b` |  | dog-and-rain | cited 18x by M |
| results | `does-roentgen-draw-two-findings` |  | main | name |
| results | `overlap-regimes-on-real-films` | where-each-overlap-regime-shows-on-real-films | dog-and-rain | name |
| results | `where-the-product-lands` |  | main | cited 1x by M cards, walks or findings |
| results | `which-findings-arrive-after-which` |  | dog-and-rain | cited 2x by M cards, walks or findings |
| scripts | `annotate_overlap_regimes.py` |  | dog-and-rain | name |
| scripts | `annotate_overlap_regimes_specs.py` |  | dog-and-rain | name |
| scripts | `where_poe_lands_on_the_manifold.py` |  | main, dog-and-rain, reviews | cited 1x by M cards, walks or findings |

## stays at the root, shared

| Kind | Item | New name | In | Why |
|---|---|---|---|---|
| scopes | `diagram-prompts.md` |  | main, dog-and-rain, reviews | loose plan file, judged by name |
| scopes | `diagrams` |  | main, dog-and-rain, reviews | shared plan folder |
| scopes | `plan-diagrams` |  | main, dog-and-rain, reviews | shared plan folder |
| scopes | `progress-map.md` |  | main, dog-and-rain, reviews | loose plan file, judged by name |
| scopes | `regroup-into-project-folders-inventory.md` |  | dog-and-rain | loose plan file, judged by name |
| scopes | `regroup-into-project-folders.md` |  | dog-and-rain | loose plan file, judged by name |
| scopes | `route-answer-the-iclr-reviews-references.md` |  | main | shared plan folder |
| scopes | `route-answer-the-iclr-reviews.md` |  | main, reviews | shared plan folder |
| scopes | `route-camel-in-the-rain.md` |  | main, reviews | shared plan folder |
| scopes | `split-into-idea-repos.md` |  | dog-and-rain | loose plan file, judged by name |
| scopes | `standing` |  | main, dog-and-rain, reviews | shared plan folder |
| drips | `present-revision-to-supervisor` |  | main | name |
| ideas | `codex-diagram-render-queue` | diagrams-waiting-to-be-drawn | main, dog-and-rain, reviews | name |
| ideas | `direction-gallery` |  | main, dog-and-rain, reviews | name |
| ideas | `merge-supervisor-structure` |  | main, dog-and-rain, reviews | name |
| ideas | `tidy-repo-loose-files` |  | main, dog-and-rain, reviews | name |
| results | `how-sdxl-turns-a-prompt-into-an-image` |  | main | name |
| results | `how-the-sdxl-vae-is-built` |  | main | name |
| results | `why-the-latent-is-scaled` |  | main | judged by name |
| notes | `batch-shape-nondeterminism` |  | main, dog-and-rain, reviews | name |
| notes | `session-merges` |  | main, dog-and-rain, reviews | name |
| notes | `supervisor-meetings` |  | dog-and-rain | name |
| notes | `the-mathematics-of-sampling-an-image` |  | dog-and-rain | name |
| scenes | `logsnr-explainer` |  | main, dog-and-rain, reviews | judged by name |
| scenes | `sdxl-vae-architecture-map` |  | main, dog-and-rain, reviews | name |
| scenes | `what-sampling-an-image-means` |  | dog-and-rain | name |
| scripts | `direction_agreement_by_shared_run_state.py` |  | main, dog-and-rain, reviews | name |
| scripts | `launch_sdlora.sh` |  | main, dog-and-rain, reviews | cited by I 6, L 3; shared launcher, kept at the root |
| scripts | `local` |  | main, dog-and-rain, reviews | cited by K 159, L 99, M 3; shared launcher, kept at the root |
| scripts | `plot_shared_plane_trajectories.py` |  | main, dog-and-rain, reviews | name |
| scripts | `rt_shared_component.py` |  | main, dog-and-rain, reviews | name |

## _archive/, cited by nothing

| Kind | Item | New name | In | Why |
|---|---|---|---|---|
| drips | `how-the-hypothesis-skills-chain` |  | main, dog-and-rain, reviews | judged by name |
| drips | `the-proposals-three-phases` |  | main, reviews | judged by name |
| scripts | `across_adapters.py` |  | main, dog-and-rain, reviews | cited by nothing: archive list |
| scripts | `across_adapters_by_family.py` |  | main, reviews | cited by nothing: archive list |
| scripts | `adapter_ablation_grids.py` |  | main, dog-and-rain, reviews | cited by nothing: archive list |
| scripts | `batch_node_probe.sh` |  | main | cited by nothing: archive list |
| scripts | `bench_claims.py` |  | main | cited by nothing: archive list |
| scripts | `bench_pod_worker.sh` |  | main | cited by nothing: archive list |
| scripts | `bench_worker.sh` |  | main | cited by nothing: archive list |
| scripts | `build_cwl_manifest.py` |  | main, dog-and-rain, reviews | cited by nothing: archive list |
| scripts | `celeba_attributes` |  | dog-and-rain | cited by nothing: archive list |
| scripts | `check_corrector_identity.py` |  | main, reviews | cited by nothing: archive list |
| scripts | `check_corrector_identity.sbatch` |  | main, reviews | cited by nothing: archive list |
| scripts | `check_langevin_order.py` |  | main, reviews | cited by nothing: archive list |
| scripts | `check_langevin_two_gaussians.py` |  | main, reviews | cited by nothing: archive list |
| scripts | `compose_rate_dissimilar_pair.py` |  | main, dog-and-rain, reviews | cited by nothing: archive list |
| scripts | `composing_unseen_pairs.py` |  | main, dog-and-rain, reviews | cited by nothing: archive list |
| scripts | `decode_expert_frames.py` |  | main, dog-and-rain, reviews | cited by nothing: archive list |
| scripts | `download_liver_cect.sh` |  | main, reviews | cited by nothing: archive list |
| scripts | `draw_mask_overlays.py` |  | main, dog-and-rain, reviews | cited by nothing: archive list |
| scripts | `failure_gallery.py` |  | main, dog-and-rain, reviews | cited by nothing: archive list |
| scripts | `figure7_limitations.py` |  | main, reviews | cited by nothing: archive list |
| scripts | `figure_text_audit.py` |  | main, reviews | cited by nothing: archive list |
| scripts | `figure_windows.sbatch` |  | main, dog-and-rain, reviews | cited by nothing: archive list |
| scripts | `fix_reduce_reuse_recycle_notebook.py` |  | main, reviews | cited by nothing: archive list |
| scripts | `gpu_probe.sh` |  | main, dog-and-rain, reviews | cited by nothing: archive list |
| scripts | `joint_prompt_through_adapter_figure.py` |  | main, reviews | cited by nothing: archive list |
| scripts | `later_start_strip.py` |  | main, dog-and-rain, reviews | cited by nothing: archive list |
| scripts | `launch_medical_pair_authors.sh` |  | main, reviews | cited by nothing: archive list |
| scripts | `launch_overdose_cells.sh` |  | main, dog-and-rain, reviews | cited by nothing: archive list |
| scripts | `launch_reward_kl.sh` |  | main, dog-and-rain, reviews | cited by nothing: archive list |
| scripts | `liver_cect_gallery.py` |  | main, reviews | cited by nothing: archive list |
| scripts | `longer_correction_strip.py` |  | main, dog-and-rain, reviews | cited by nothing: archive list |
| scripts | `make_main_row.py` |  | main, dog-and-rain, reviews | cited by nothing: archive list |
| scripts | `make_overleaf_bundle.sh` |  | main, dog-and-rain, reviews | cited by nothing: archive list |
| scripts | `medical_pair_check_authors.py` |  | main, reviews | cited by nothing: archive list |
| scripts | `medical_pair_generate.py` |  | main, reviews | cited by nothing: archive list |
| scripts | `medical_pair_generate_authors.py` |  | main, reviews | cited by nothing: archive list |
| scripts | `noise_slice.py` |  | main, dog-and-rain, reviews | cited by nothing: archive list |
| scripts | `overdose_cells.sbatch` |  | main, dog-and-rain, reviews | cited by nothing: archive list |
| scripts | `overleaf_sync.sh` |  | main, dog-and-rain, reviews | cited by nothing: archive list |
| scripts | `paper_font.py` |  | main, reviews | cited by nothing: archive list |
| scripts | `paper_main_grid.py` |  | main, reviews | cited by nothing: archive list |
| scripts | `paper_tile_figure.py` |  | main, reviews | cited by nothing: archive list |
| scripts | `probe_batch_capacity.py` |  | main, dog-and-rain, reviews | cited by nothing: archive list |
| scripts | `recap_landed` |  | main, dog-and-rain, reviews | cited by nothing: archive list |
| scripts | `recap_landed.py` |  | main, dog-and-rain, reviews | cited by nothing: archive list |
| scripts | `refresh_paper_slots.py` |  | main, reviews | cited by nothing: archive list |
| scripts | `regional_preview.py` |  | main, dog-and-rain, reviews | cited by nothing: archive list |
| scripts | `render_joint_failure_repair_cell.py` |  | main, dog-and-rain, reviews | cited by nothing: archive list |
| scripts | `render_svg_figure.py` |  | main, dog-and-rain, reviews | cited by nothing: archive list |
| scripts | `review_baseline_cell.py` |  | main, reviews | cited by nothing: archive list |
| scripts | `review_baselines_co3_tail.sh` |  | main | cited by nothing: archive list |
| scripts | `review_baselines_eval.sh` |  | main, reviews | cited by nothing: archive list |
| scripts | `review_baselines_fig5.sh` |  | main, reviews | cited by nothing: archive list |
| scripts | `review_baselines_grid.sh` |  | main, reviews | cited by nothing: archive list |
| scripts | `review_baselines_onecell.sh` |  | main, reviews | cited by nothing: archive list |
| scripts | `review_bench_co3.sbatch` |  | main | cited by nothing: archive list |
| scripts | `review_bench_geneval.py` |  | main | cited by nothing: archive list |
| scripts | `review_bench_joint.py` |  | main | cited by nothing: archive list |
| scripts | `review_bench_renders.sh` |  | main | cited by nothing: archive list |
| scripts | `review_bench_score.sh` |  | main | cited by nothing: archive list |
| scripts | `review_bench_sets.py` |  | main | cited by nothing: archive list |
| scripts | `review_bench_t2i.py` |  | main | cited by nothing: archive list |
| scripts | `review_bench_table.py` |  | main | cited by nothing: archive list |
| scripts | `review_benchmark_pairs.txt` |  | main, reviews | cited by nothing: archive list |
| scripts | `review_build_tables.py` |  | main | cited by nothing: archive list |
| scripts | `review_co3_selection.sh` |  | main, reviews | cited by nothing: archive list |
| scripts | `review_count_ours_nfe.py` |  | main | cited by nothing: archive list |
| scripts | `review_count_ours_nfe.sh` |  | main | cited by nothing: archive list |
| scripts | `review_held_out_pairs.txt` |  | main, reviews | cited by nothing: archive list |
| scripts | `review_install_geneval.sh` |  | main, reviews | cited by nothing: archive list |
| scripts | `review_install_geneval_finish.sh` |  | main, reviews | cited by nothing: archive list |
| scripts | `review_install_sd15_envs.sh` |  | main | cited by nothing: archive list |
| scripts | `review_install_t2icomp.sh` |  | main, reviews | cited by nothing: archive list |
| scripts | `review_install_t2icomp_unidet.sh` |  | main | cited by nothing: archive list |
| scripts | `review_judge_failures.py` |  | main | cited by nothing: archive list |
| scripts | `review_judge_prepare.py` |  | main | cited by nothing: archive list |
| scripts | `review_langevin_cells.py` |  | main | cited by nothing: archive list |
| scripts | `review_langevin_cells.sh` |  | main | cited by nothing: archive list |
| scripts | `review_langevin_subset.py` |  | main | cited by nothing: archive list |
| scripts | `review_langevin_subset.sh` |  | main | cited by nothing: archive list |
| scripts | `review_no_overlap_tables.py` |  | main | cited by nothing: archive list |
| scripts | `review_pick_by_blip.py` |  | main | cited by nothing: archive list |
| scripts | `review_r6_renders.py` |  | main, reviews | cited by nothing: archive list |
| scripts | `review_r6_renders.sbatch` |  | main, reviews | cited by nothing: archive list |
| scripts | `review_rule5_picks.py` |  | main | cited by nothing: archive list |
| scripts | `review_score_blip.sbatch` |  | main, reviews | cited by nothing: archive list |
| scripts | `review_score_blipvqa.py` |  | main, reviews | cited by nothing: archive list |
| scripts | `review_score_clip_ir.py` |  | main, reviews | cited by nothing: archive list |
| scripts | `review_score_clip_ir_cpu.sbatch` |  | main, reviews | cited by nothing: archive list |
| scripts | `review_score_comp.py` |  | main | cited by nothing: archive list |
| scripts | `review_score_comp_cpu.sbatch` |  | main | cited by nothing: archive list |
| scripts | `review_score_geneval.py` |  | main, reviews | cited by nothing: archive list |
| scripts | `review_score_rows.py` |  | main, reviews | cited by nothing: archive list |
| scripts | `review_sd15.sh` |  | main | cited by nothing: archive list |
| scripts | `review_sd15_cell.py` |  | main | cited by nothing: archive list |
| scripts | `review_sd15_prep.py` |  | main | cited by nothing: archive list |
| scripts | `review_seed_latents.py` |  | main, reviews | cited by nothing: archive list |
| scripts | `review_showcase_pick.py` |  | main, reviews | cited by nothing: archive list |
| scripts | `review_step4_census_pairs.txt` |  | main, reviews | cited by nothing: archive list |
| scripts | `review_step4_count.py` |  | main, reviews | cited by nothing: archive list |
| scripts | `review_step4_renders.sh` |  | main, reviews | cited by nothing: archive list |
| scripts | `review_syngen_indices.py` |  | main | cited by nothing: archive list |
| scripts | `review_syngen_sdxl.sh` |  | main | cited by nothing: archive list |
| scripts | `review_syngen_sdxl_cell.py` |  | main | cited by nothing: archive list |
| scripts | `review_three_concepts.py` |  | main | cited by nothing: archive list |
| scripts | `review_three_concepts.sbatch` |  | main | cited by nothing: archive list |
| scripts | `review_tome.sbatch` |  | main | cited by nothing: archive list |
| scripts | `review_tome_cell.py` |  | main, reviews | cited by nothing: archive list |
| scripts | `review_wave1_r1.sh` |  | main, reviews | cited by nothing: archive list |
| scripts | `review_wave1_r1_rev.sh` |  | main, reviews | cited by nothing: archive list |
| scripts | `review_wave1_renders.sh` |  | main, reviews | cited by nothing: archive list |
| scripts | `review_window_figure.py` |  | main | cited by nothing: archive list |
| scripts | `review_window_panels.py` |  | main | cited by nothing: archive list |
| scripts | `review_window_rows.py` |  | main | cited by nothing: archive list |
| scripts | `reward_kl.sbatch` |  | main, dog-and-rain, reviews | cited by nothing: archive list |
| scripts | `rt_direction_wall.py` |  | main, dog-and-rain, reviews | cited by nothing: archive list |
| scripts | `run-plan-tree.sh` |  | main, dog-and-rain, reviews | cited by nothing: archive list |
| scripts | `run_clean_tail_k_sweep.sh` |  | main, dog-and-rain, reviews | cited by nothing: archive list |
| scripts | `run_separation_over_time.py` |  | main, dog-and-rain, reviews | cited by nothing: archive list |
| scripts | `sampling_strategy_grid.py` |  | main, reviews | cited by nothing: archive list |
| scripts | `score_noise_slice.py` |  | main, dog-and-rain, reviews | cited by nothing: archive list |
| scripts | `score_overdose_cells.py` |  | main, dog-and-rain, reviews | cited by nothing: archive list |
| scripts | `sd35_x_sheet.py` |  | dog-and-rain | cited by nothing: archive list |
| scripts | `sdxl_shapes_poe.py` |  | dog-and-rain | cited by nothing: archive list |
| scripts | `seed1_across_adapters.py` |  | main, dog-and-rain, reviews | cited by nothing: archive list |
| scripts | `seed1_strength_dial.py` |  | main, dog-and-rain, reviews | cited by nothing: archive list |
| scripts | `seed1_two_adapters.py` |  | main, dog-and-rain, reviews | cited by nothing: archive list |
| scripts | `seed_clip_signature.py` |  | main, dog-and-rain, reviews | cited by nothing: archive list |
| scripts | `seed_curve_signature.py` |  | main, dog-and-rain, reviews | cited by nothing: archive list |
| scripts | `seed_method_grid.py` |  | main, reviews | cited by nothing: archive list |
| scripts | `seed_method_grid.sbatch` |  | main, reviews | cited by nothing: archive list |
| scripts | `show_pairs.py` |  | main, dog-and-rain, reviews | cited by nothing: archive list |
| scripts | `superdiff_with_adapter.py` |  | main, dog-and-rain, reviews | cited by nothing: archive list |
| scripts | `sweep_sheets.py` |  | main, dog-and-rain, reviews | cited by nothing: archive list |
| scripts | `text_orthogonality_probe.py` |  | main, dog-and-rain, reviews | cited by nothing: archive list |
| scripts | `timing_cliff_in_language.py` |  | main, dog-and-rain, reviews | cited by nothing: archive list |
| scripts | `timing_not_dose.py` |  | main, dog-and-rain, reviews | cited by nothing: archive list |
| scripts | `transpose_tile_spec.py` |  | main | cited by nothing: archive list |
| scripts | `watch_and_visualize.py` |  | main, dog-and-rain, reviews | cited by nothing: archive list |
| scripts | `when_the_correction_arrives.py` |  | main, dog-and-rain, reviews | cited by nothing: archive list |
| scripts | `where_the_correction_lands.py` |  | main, dog-and-rain, reviews | cited by nothing: archive list |
