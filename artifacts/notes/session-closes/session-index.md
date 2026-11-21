# Session index

The 73 Claude sessions in this project that have no closing note, one row each, with whether each is safe to archive. The other 11 sessions each have their own note in this folder.

**Which sessions these are**

Every transcript in the `concept-composition` and `replications-published-methods-on-our-pairs` Claude project folders whose first turn is before 2026-11-21 18:52 UTC, minus the 11 with closing notes.

**What each verdict means**

| Verdict | Sessions | Meaning |
|---|---:|---|
| `safe` | 29 | Not live, and every file it wrote exists and is committed, or sits outside git |
| `git-ignored outputs` | 3 | As safe, except at least one file exists only on disk because git ignores it |
| `moved or removed` | 17 | At least one file is not at the path written. Git history holds that path, and it may have moved or been removed |
| `uncommitted` | 16 | At least one file it wrote has uncommitted changes, in this repository or another |
| `missing` | 8 | At least one file it wrote is not on disk and not in this repository's git history |
| `live` | 0 | Its transcript changed in the last 30 minutes |

Only `safe` sessions qualify for archiving under the rule in [the session 1 prompt](../progress/prompts/session-1-records.md).

**What this check can and cannot see**

File states were read at 2026-11-21 22:32 on commit `25dbff30`. The check counts each session's Write and Edit calls, its subagents' calls included, and follows the paths in `RENAMES.md`, the old `poe_repair_min` names and the numbered project folders. It cannot see files made by shell commands, so a `safe` row means every file the transcript names as written is accounted for. Temporary scratch files are left out of the file counts.

## The 73 sessions

| Session | Last active | Title | Branch | Files written | Verdict |
|---|---|---|---|---|---|
| `b65be185` | 2026-10-09 | 🌪️ Artifact: Understand the nature of composition | `exec/correction-loss-01` | 97: 84 committed, 4 git-ignored, 6 missing, 3 no git | `missing` |
| `6b087717` | 2026-10-09 | EDM2 projective composition on CLEVR | `exec/correction-loss-01` | 4: 4 committed | `safe` |
| `dc7b177c` | 2026-10-09 | Car tracking and dashcam research for South Africa | `exec/correction-loss-01` | 1: 1 no git | `safe` |
| `e557e527` | 2026-10-22 | Testing my idea (fork) | `exec/dog-and-rain` | 4: 2 committed, 1 moved or removed, 1 missing | `missing` |
| `6b5a3001` | 2026-10-27 | Testing my idea | `exec/dog-and-rain` | 16: 15 committed, 1 missing | `missing` |
| `52d6ec28` | 2026-10-22 | 🎱Getting the third concept | `exec/dog-and-rain` | 5: 5 committed | `safe` |
| `91603601` | 2026-10-22 | 🐪Finding wet camels in the training data | `exec/dog-and-rain` | 4: 3 committed, 1 moved or removed | `moved or removed` |
| `7d7e9267` | 2026-10-28 | ❌ Toy failure in six checkpoints | `exec/dog-and-rain` | 99: 13 committed, 1 git-ignored, 85 moved or removed | `moved or removed` |
| `04901db4` | 2026-10-28 | ❌ Toy failure in six checkpoints (fork) | `exec/dog-and-rain` | 12: 6 committed, 1 git-ignored, 5 moved or removed | `moved or removed` |
| `0ae9cc8b` | 2026-10-27 | Project setup boilerplate skill | `exec/dog-and-rain` | none | `safe` |
| `72978890` | 2026-10-28 | Browser automation setup walkthrough | `exec/dog-and-rain` | 10: 8 committed, 1 git-ignored, 1 missing | `missing` |
| `6b92d9c1` | 2026-10-28 | Claude in Chrome browser automation | `main` | 18: 18 committed | `safe` |
| `2875e4f6` | 2026-10-28 | Claude in Chrome browser automation | `exec/dog-and-rain` | 3: 3 committed | `safe` |
| `ac09f6b4` | 2026-10-29 | 🩻 Medical interpretation of composition | `exec/dog-and-rain` | 3: 2 committed, 1 git-ignored | `git-ignored outputs` |
| `0e9d0d0f` | 2026-10-29 | LoRA training scope and defaults | `exec/dog-and-rain` | 7: 5 committed, 1 git-ignored, 1 moved or removed | `moved or removed` |
| `b9fb3e38` | 2026-10-28 | Claude Code stopped responding notice | `exec/dog-and-rain` | none | `safe` |
| `cc6dbbca` | 2026-10-29 | Cluster follow-up question | `exec/dog-and-rain` | none | `safe` |
| `e7a90753` | 2026-11-04 | Session f2a72770-e542-440a-8192-ad9524414ae8 lookup | `exec/inclusive-captions-pilot` | 1: 1 moved or removed | `moved or removed` |
| `b4e1ea0a` | 2026-11-04 | Silicotb conversation claims | `exec/how-much-co-occurrence-the-product-needs` | 1: 1 moved or removed | `moved or removed` |
| `7b180fb3` | 2026-11-05 | Inclusive captions axis drip-idea walk | `exec/how-much-co-occurrence-the-product-needs` | 15: 14 committed, 1 missing | `missing` |
| `6dac739e` | 2026-11-04 | PoE ill-posedness in compositional diffusion | `exec/does-a-clip-free-model-hybridise` | 1: 1 committed | `safe` |
| `db441e4f` | 2026-11-05 | Positional x_pos third expert for PoE | `exec/how-much-co-occurrence-the-product-needs` | 17: 10 committed, 2 git-ignored, 3 moved or removed, 2 missing | `missing` |
| `553b6a70` | 2026-11-04 | Hybrid vs photo CLIP cat-dog similarity | `exec/does-one-prompt-already-say-one-animal` | 4: 4 committed | `safe` |
| `7a6480ab` | 2026-11-04 | PixArt-alpha CLIP-free hybrid composition | `exec/how-much-co-occurrence-the-product-needs` | 5: 4 committed, 1 moved or removed | `moved or removed` |
| `14ac0b63` | 2026-11-05 | Pair-image weight sweep for PoE toy (fork) | `exec/how-much-co-occurrence-the-product-needs` | 3: 2 committed, 1 moved or removed | `moved or removed` |
| `41c11448` | 2026-11-05 | Pair-image weight sweep for PoE toy | `exec/how-much-co-occurrence-the-product-needs` | 4: 3 committed, 1 moved or removed | `moved or removed` |
| `01bb3d05` | 2026-11-05 | Classifier-guided cat-dog composition | `exec/how-much-co-occurrence-the-product-needs` | 5: 3 committed, 2 git-ignored | `git-ignored outputs` |
| `0ad1441b` | 2026-11-04 | Oracle layout for cat-by-dog product | `exec/does-one-prompt-already-say-one-animal` | 7: 7 committed | `safe` |
| `bd0aa838` | 2026-11-04 | PixArt-alpha sampling loop | `main` | none | `safe` |
| `e97868d1` | 2026-11-04 | Language-space composition claims for PoE repair | `exec/does-one-prompt-already-say-one-animal` | none | `safe` |
| `a7ec4286` | 2026-11-04 | Chest disease pairs: four arrangements on one film | `exec/how-much-co-occurrence-the-product-needs` | 1: 1 moved or removed | `moved or removed` |
| `b7116f56` | 2026-11-10 | Mac machine sync script | `exec/correction-loss-01` | 5: 5 committed | `safe` |
| `1cd40a85` | 2026-11-10 | Pulling Mac's updates | `exec/correction-loss-01` | none | `safe` |
| `34b8ea9f` | 2026-11-11 | Reviewer PrzE rebuttal ICLR 2027 | `exec/correction-loss-01` | 17: 16 committed, 1 moved or removed | `moved or removed` |
| `17925940` | 2026-11-12 | ICLR eleven evaluation tools and rival methods | `exec/correction-loss-01` | 34: 30 committed, 4 no git | `safe` |
| `9b0c49df` | 2026-11-10 | Ground check for ICLR 56173 reviews | `exec/correction-loss-01` | 11: 8 committed, 3 moved or removed | `moved or removed` |
| `66237ec2` | 2026-11-11 | Second animal species fidelity in adapter | `exec/correction-loss-01` | 17: 4 committed, 1 git-ignored, 2 moved or removed, 10 missing | `missing` |
| `2aaf06ce` | 2026-11-11 | ICLR 2027 review response protocol | `exec/correction-loss-01` | 23: 17 committed, 1 git-ignored, 5 moved or removed | `moved or removed` |
| `a4d17e2c` | 2026-11-11 | ICLR review compute plan and wave 1 launch | `exec/correction-loss-01` | 8: 6 committed, 2 moved or removed | `moved or removed` |
| `afd3106a` | 2026-11-11 | Langevin corrector Gaussian proof and figure | `exec/correction-loss-01` | 13: 10 committed, 3 moved or removed | `moved or removed` |
| `ece09d30` | 2026-11-11 | CO3, ToMe and SuperDiff review baselines | `exec/correction-loss-01` | 12: 10 committed, 2 moved or removed | `moved or removed` |
| `47281f64` | 2026-11-11 | PRZE-1 and PRZE-3b metric scoring | `exec/correction-loss-01` | 10: 8 committed, 1 git-ignored, 1 moved or removed | `moved or removed` |
| `1a9eecdc` | 2026-11-12 | ICLR 2027 submission 56173 reviewer replies | `exec/correction-loss-01` | 38: 32 committed, 4 moved or removed, 2 missing | `missing` |
| `a40d70e8` | 2026-11-12 | ICLR 2027 review response figures | `exec/correction-loss-01` | 14: 14 committed | `safe` |
| `96716791` | 2026-11-13 | Drip-skill-route for toy animal experiments | `exec/correction-loss-01` | 1: 1 committed | `safe` |
| `804ea312` | 2026-11-13 | Classifier-guided PoE route, toy to SDXL | `exec/correction-loss-01` | 25: 20 committed, 1 uncommitted, 3 git-ignored, 1 no git | `uncommitted` |
| `cb5e6071` | 2026-11-12 | 2211.17091 discriminator guidance | `exec/how-much-co-occurrence-the-product-needs` | 1: 1 committed | `safe` |
| `b3cb8672` | 2026-11-13 | Cluster PhD folder organogram regrouping | `exec/correction-loss-01` | 5: 2 committed, 3 no git | `safe` |
| `8320ea28` | 2026-11-13 | Recovering poe_repair_min_dog Claude sessions | `exec/correction-loss-01` | none | `safe` |
| `2c17d1cd` | 2026-11-16 | Porting fourteen composition papers to product of experts | `exec/triage-kit` | 5: 4 committed, 1 uncommitted | `uncommitted` |
| `9c235d67` | 2026-11-16 | Shared kit for composition paper ports | `exec/triage-kit` | 12: 5 committed, 7 uncommitted | `uncommitted` |
| `60eca943` | 2026-11-16 | Fkc-diffusion repo clone and run | `exec/triage-kit` | 19: 6 committed, 12 uncommitted, 1 no git | `uncommitted` |
| `5f3f3321` | 2026-11-16 | Drips we were busy with | `exec/correction-loss-01` | 2: 1 committed, 1 uncommitted | `uncommitted` |
| `027fd4e9` | 2026-11-16 | FKC product sampler plumbing at β = 1 | `exec/triage-kit` | 7: 1 committed, 3 uncommitted, 3 no git | `uncommitted` |
| `5d69581b` | 2026-11-17 | FKC diffusion product loop code walk | `exec/triage-kit` | 21: 9 committed, 10 uncommitted, 2 missing | `uncommitted` |
| `db7ff71d` | 2026-11-16 | Classifier guidance box on PoE HLD | `exec/correction-loss-01` | 3: 3 uncommitted | `uncommitted` |
| `b1016d99` | 2026-11-17 | Classifier guidance in diffusion models | `exec/classifier-guidance-ladder` | 26: 25 committed, 1 uncommitted | `uncommitted` |
| `140e9aa0` | 2026-11-17 | Kim et al. discriminator guidance read | `exec/classifier-guidance-ladder` | 1: 1 git-ignored | `git-ignored outputs` |
| `ca7fdfd2` | 2026-11-17 | ACE repo replication port | `exec/triage-kit` | 3: 3 uncommitted | `uncommitted` |
| `38f261c3` | 2026-11-18 | Guided-diffusion classifier guidance reproduction | `exec/classifier-guidance-ladder` | 154: 115 committed, 1 uncommitted, 38 git-ignored | `uncommitted` |
| `2a37400a` | 2026-11-17 | PDF to editable DOCX conversion | `exec/classifier-guidance-ladder` | none kept (1 scratch) | `safe` |
| `0bfb147d` | 2026-11-17 | SDXL product sampler classifier push repair | `exec/classifier-guidance-ladder` | 9: 4 committed, 1 uncommitted, 4 git-ignored | `uncommitted` |
| `f1fcac6c` | 2026-11-17 | Scale-up ladder for PoE guidance route | `exec/classifier-guidance-ladder` | 2: 2 committed | `safe` |
| `ba678e5d` | 2026-11-17 | Classifier-guided product of experts diagrams | `exec/classifier-guidance-ladder` | 2: 2 committed | `safe` |
| `fe46afa6` | 2026-11-17 | Classifier guidance on exact toy | `exec/classifier-guidance-ladder` | 9: 6 committed, 1 uncommitted, 1 git-ignored, 1 moved or removed | `uncommitted` |
| `fc439ed4` | 2026-11-17 | Cat × dog discriminator test on SDXL | `exec/classifier-guidance-ladder` | 4: 3 committed, 1 uncommitted | `uncommitted` |
| `4ab26246` | 2026-11-18 | VLM gradient for POE CFG plurality | `exec/classifier-guidance-ladder` | 2: 1 uncommitted, 1 git-ignored | `uncommitted` |
| `e96b9dd9` | 2026-11-18 | Adapter plus checker push on cat-dog | `exec/classifier-guidance-ladder` | 3: 3 committed | `safe` |
| `5743058a` | 2026-11-18 | Chest X-ray composition route plan | `exec/classifier-guidance-ladder` | 4: 3 uncommitted, 1 missing | `uncommitted` |
| `2431f9b0` | 2026-11-19 | ICLR revised paper materials | `exec/classifier-guidance-ladder` | 2: 2 committed | `safe` |
| `a9f3f351` | 2026-11-19 | ICLR Revised Comparison PDF annotations | `exec/classifier-guidance-ladder` | none | `safe` |
| `158ae1de` | 2026-11-19 | CO3 paper discussion | `exec/classifier-guidance-ladder` | none | `safe` |
| `3146bf79` | 2026-11-20 | Linked learning reply style setup | `exec/classifier-guidance-ladder` | 1: 1 committed | `safe` |

## What keeps a session from safe

At most five paths per session, chosen in the order uncommitted, missing, moved or removed, then git-ignored.

| Session | Verdict | Files |
|---|---|---|
| `b65be185` | `missing` | `.playwright-mcp/ch02-sweep.js`<br/>`.playwright-mcp/ch13-shoot.js`<br/>`.playwright-mcp/ch7-drive.js`<br/>`~/Obsidian/Artificial Intelligence/Generative Modeling/Diffusion Models.md`<br/>`~/Obsidian/Mathematics/Statistics/Distributions/Multivariate Normal Distribution.md`<br/>and 5 more |
| `e557e527` | `missing` | `artifacts/results/does-a-camel-in-the-rain-come-out-wet/caption-counts.md`<br/>`artifacts/drips/a-dog-and-rain/EXECUTE_WALK.md` |
| `6b5a3001` | `missing` | `artifacts/results/does-a-camel-in-the-rain-come-out-wet/caption-counts.md` |
| `91603601` | `moved or removed` | `artifacts/drips/camel-and-dog-in-the-training-data/EXECUTE_WALK.md` |
| `7d7e9267` | `moved or removed` | `artifacts/drips/simple-distributions-rebuild/ladder/05-the-overlap-dial.md`<br/>`artifacts/drips/simple-distributions-rebuild/01-one-expert-from-scratch.md`<br/>`artifacts/drips/simple-distributions-rebuild/PROMPTS.md`<br/>`artifacts/drips/simple-distributions-rebuild/REBUILD_WALK.md`<br/>`artifacts/drips/simple-distributions-rebuild/ladder/00-setup.md`<br/>and 81 more |
| `04901db4` | `moved or removed` | `artifacts/drips/simple-distributions-rebuild/ladder/05-the-overlap-dial.md`<br/>`artifacts/drips/simple-distributions-rebuild/PROMPTS.md`<br/>`artifacts/drips/simple-distributions-rebuild/REBUILD_WALK.md`<br/>`artifacts/drips/simple-distributions-rebuild/ladder/10-latent-space-and-vaes.md`<br/>`artifacts/drips/simple-distributions-rebuild/ladder/11-the-text-conditioned-testbed.md`<br/>and 1 more |
| `72978890` | `missing` | `artifacts/results/what-the-and-vector-is-made-of/README.md`<br/>`~/.claude/projects/-home-molef-PhD-poe-repair-min/memory/composition-calls-overcall-on-downscaled-sheets.md` |
| `ac09f6b4` | `git-ignored outputs` | `~/.claude/projects/-home-molef-PhD-poe-repair-min/memory/paper-scout-profile.md` |
| `0e9d0d0f` | `moved or removed` | `supervisor-meeting/README.md`<br/>`~/.claude/projects/-home-molef-PhD-poe-repair-min/memory/paper-scout-profile.md` |
| `e7a90753` | `moved or removed` | `scripts/annotate_overlap_regimes.py` |
| `b4e1ea0a` | `moved or removed` | `artifacts/ideas/silicotb-conversation/routes/01-annotated-radiographs-for-each-regime.prompt.md` |
| `7b180fb3` | `missing` | `artifacts/results/how-the-inclusive-captions-pilot-works/README.md` |
| `db441e4f` | `missing` | `artifacts/results/the-positional-and-vector/README.md`<br/>`artifacts/results/the-positional-and-vector/calls.json`<br/>`artifacts/drips/the-positional-and-vector/EXECUTE_WALK.md`<br/>`supervisor-meeting/README.md`<br/>`plans/10-what-poe-loses-when-it-composes/today-the-positional-and-vector.md`<br/>and 2 more |
| `7a6480ab` | `moved or removed` | `artifacts/drips/does-a-clip-free-model-hybridise/EXECUTE_WALK.md` |
| `14ac0b63` | `moved or removed` | `artifacts/results/how-much-co-occurrence-the-product-needs/README.md` |
| `41c11448` | `moved or removed` | `artifacts/results/how-much-co-occurrence-the-product-needs/README.md` |
| `01bb3d05` | `git-ignored outputs` | `~/.claude/projects/-home-molef-PhD-poe-repair-min/memory/clip-ranks-but-cannot-steer.md`<br/>`~/.claude/projects/-home-molef-PhD-poe-repair-min/memory/co-occurrence-means-two-things-in-frame.md` |
| `a7ec4286` | `moved or removed` | `artifacts/drips/what-two-diseases-do-on-one-film/COMBINE_MAP.md` |
| `34b8ea9f` | `moved or removed` | `artifacts/drips/answer-iclr-reviews/ROUTE_WALK.md` |
| `9b0c49df` | `moved or removed` | `artifacts/drips/answer-iclr-reviews/EXECUTE_WALK.md`<br/>`artifacts/drips/answer-iclr-reviews/2026-11-10-ground-check-retro.md`<br/>`artifacts/drips/answer-iclr-reviews/ground-check.md` |
| `66237ec2` | `missing` | `artifacts/results/why-the-second-animal-loses-its-species/attention_maps.py`<br/>`artifacts/results/why-the-second-animal-loses-its-species/conform_on_adapter.py`<br/>`artifacts/results/why-the-second-animal-loses-its-species/lion_partner_sweep.py`<br/>`artifacts/results/why-the-second-animal-loses-its-species/live_split.py`<br/>`artifacts/results/why-the-second-animal-loses-its-species/live_split_controls.py`<br/>and 8 more |
| `2aaf06ce` | `moved or removed` | `artifacts/drips/answer-iclr-reviews/2026-11-10-frozen-set-and-adapter-retro.md`<br/>`artifacts/drips/answer-iclr-reviews/2026-11-10-protocol-and-slots-retro.md`<br/>`artifacts/drips/answer-iclr-reviews/2026-11-11-presentation-retro.md`<br/>`artifacts/drips/answer-iclr-reviews/EXECUTE_WALK-step-2.md`<br/>`artifacts/drips/answer-iclr-reviews/EXECUTE_WALK-step-4.md`<br/>and 1 more |
| `a4d17e2c` | `moved or removed` | `artifacts/drips/prze-10-wave-1/EXECUTE_WALK.md`<br/>`plans/prze-10-wave-1-renders.md` |
| `afd3106a` | `moved or removed` | `artifacts/drips/prze-2-langevin-corrector/EXECUTE_WALK.md`<br/>`artifacts/results/does-a-langevin-corrector-fix-the-product/README.md`<br/>`plans/prze-2-langevin-corrector.md` |
| `ece09d30` | `moved or removed` | `artifacts/drips/review-baselines/EXECUTE_WALK.md`<br/>`plans/prze-review-baselines.md` |
| `47281f64` | `moved or removed` | `artifacts/drips/answer-iclr-reviews/EXECUTE_WALK-step-9.md`<br/>`~/.claude/projects/-home-molef-PhD-poe-repair-min/memory/no-prediction-gates.md` |
| `1a9eecdc` | `missing` | `replies/k4nd.md`<br/>`replies/uhz6.md`<br/>`paper/iclr-56173/iclr2027_conference.tex`<br/>`paper/iclr-56173/CHANGES_FOR_SUPERVISORS.md`<br/>`replies/k4nd-and-uhz6.md`<br/>and 1 more |
| `804ea312` | `uncommitted` | `2-classifier-guidance/plans/route-classifier-guided-poe-from-toy-to-animals.md`<br/>`~/.claude/projects/-home-molef-PhD-concept-composition/memory/control-line-is-a-table.md`<br/>`~/.claude/projects/-home-molef-PhD-concept-composition/memory/repo-renamed-data-folder-kept.md`<br/>`~/.claude/projects/-home-molef-PhD-poe-repair-min/memory/phd-organogram.md` |
| `2c17d1cd` | `uncommitted` | `~/PhD/replications/published-methods-on-our-pairs/plans/route-port-and-read-the-composition-papers.md` |
| `9c235d67` | `uncommitted` | `~/PhD/replications/published-methods-on-our-pairs/artifacts/results/does-a-published-method-separate-cat-and-dog/README.md`<br/>`~/PhD/replications/published-methods-on-our-pairs/plans/route-port-and-read-the-composition-papers.md`<br/>`~/PhD/replications/published-methods-on-our-pairs/plans/triage-the-fourteen-papers-on-cat-and-dog.md`<br/>`~/PhD/replications/published-methods-on-our-pairs/report/composition-papers-on-our-pairs/w1-which-papers-go-deep.md`<br/>`~/PhD/replications/published-methods-on-our-pairs/runbook/rendering-baseline-rows-on-biggpu.md`<br/>and 2 more |
| `60eca943` | `uncommitted` | `~/PhD/replications/fkc-diffusion/walk/row2_exact_scores.py`<br/>`~/PhD/replications/published-methods-on-our-pairs/artifacts/drips/fkc-diffusion/REPO_WALK.md`<br/>`~/PhD/replications/published-methods-on-our-pairs/artifacts/drips/fkc-diffusion/routes/fkc-loop-codewalk.md`<br/>`~/PhD/replications/published-methods-on-our-pairs/artifacts/drips/port-feynman-kac-correctors/EXECUTE_WALK.md`<br/>`~/PhD/replications/published-methods-on-our-pairs/artifacts/notes/session-merges/2026-11-16-port-feynman-kac-correctors-onto-our-product-retro.md`<br/>and 7 more |
| `5f3f3321` | `uncommitted` | `2-classifier-guidance/plans/route-classifier-guided-poe-from-toy-to-animals.md` |
| `027fd4e9` | `uncommitted` | `~/PhD/replications/published-methods-on-our-pairs/.vscode/launch.json`<br/>`~/PhD/replications/published-methods-on-our-pairs/artifacts/results/what-resampling-does-in-the-fkc-product-sampler/README.md`<br/>`~/PhD/replications/published-methods-on-our-pairs/artifacts/results/where-each-method-touches-the-noise-prediction/README.md` |
| `5d69581b` | `uncommitted` | `~/PhD/replications/fkc-diffusion/walk/animate_three_ways.py`<br/>`~/PhD/replications/fkc-diffusion/walk/concept_pictures.py`<br/>`~/PhD/replications/fkc-diffusion/walk/equation_pictures.py`<br/>`~/PhD/replications/published-methods-on-our-pairs/artifacts/results/what-each-line-of-the-fkc-loop-computes/README.md`<br/>`~/PhD/replications/published-methods-on-our-pairs/artifacts/results/what-each-line-of-the-fkc-loop-computes/diagram-prompts.md`<br/>and 7 more |
| `db7ff71d` | `uncommitted` | `2-classifier-guidance/plans/route-classifier-guided-poe-from-toy-to-animals.md`<br/>`~/PhD/replications/published-methods-on-our-pairs/artifacts/results/where-each-method-touches-the-noise-prediction/README.md`<br/>`~/PhD/replications/published-methods-on-our-pairs/scripts/draw_hld_gap_pictures.py` |
| `b1016d99` | `uncommitted` | `2-classifier-guidance/plans/route-classifier-guided-poe-from-toy-to-animals.md` |
| `140e9aa0` | `git-ignored outputs` | `~/.claude/projects/-home-molef-PhD-concept-composition/memory/discriminator-guidance-poe-caveats.md` |
| `ca7fdfd2` | `uncommitted` | `~/PhD/replications/published-methods-on-our-pairs/artifacts/drips/ace/REPO_WALK.md`<br/>`~/PhD/replications/published-methods-on-our-pairs/artifacts/drips/ace/run_r3_figure_1.py`<br/>`~/PhD/replications/published-methods-on-our-pairs/artifacts/drips/ace/run_r3b_counts.py` |
| `38f261c3` | `uncommitted` | `2-classifier-guidance/plans/route-classifier-guided-poe-from-toy-to-animals.md`<br/>`2-classifier-guidance/artifacts/results/does-classifier-guidance-work-on-the-released-weights/README.md`<br/>`2-classifier-guidance/artifacts/results/what-each-line-of-classifier-guidance-computes/README.md`<br/>`2-classifier-guidance/artifacts/results/what-each-line-of-classifier-guidance-computes/classifier-guidance-loop-high-level-design-explainer.json`<br/>`2-classifier-guidance/artifacts/results/what-each-line-of-classifier-guidance-computes/equation_pictures.py`<br/>and 34 more |
| `0bfb147d` | `uncommitted` | `2-classifier-guidance/plans/route-classifier-guided-poe-from-toy-to-animals.md`<br/>`3-understanding-compositionality/artifacts/results/the-exact-toy/can-a-classifier-push-fix-the-product/README.md`<br/>`3-understanding-compositionality/artifacts/results/the-exact-toy/can-a-classifier-push-fix-the-product/the-side-by-side-discriminator-d.py`<br/>`3-understanding-compositionality/artifacts/results/the-exact-toy/can-a-classifier-push-fix-the-product/the-side-by-side-discriminator-inputs.py`<br/>`3-understanding-compositionality/artifacts/results/the-exact-toy/can-a-classifier-push-fix-the-product/the-side-by-side-discriminator.json` |
| `fe46afa6` | `uncommitted` | `2-classifier-guidance/plans/route-classifier-guided-poe-from-toy-to-animals.md`<br/>`2-classifier-guidance/plans/19-classifier-push-on-the-exact-toy.md`<br/>`3-understanding-compositionality/artifacts/results/the-exact-toy/can-a-classifier-push-fix-the-product/the-side-by-side-discriminator-result-inputs.py` |
| `fc439ed4` | `uncommitted` | `2-classifier-guidance/plans/route-classifier-guided-poe-from-toy-to-animals.md` |
| `4ab26246` | `uncommitted` | `2-classifier-guidance/plans/24-mnist-digits-on-a-canvas.md`<br/>`~/.claude/projects/-home-molef-PhD-concept-composition/memory/discriminator-guidance-set-aside.md` |
| `5743058a` | `uncommitted` | `~/PhD/medical-imaging/compositional_diffusion/artifacts/results/what-the-papers-report/what-the-papers-near-factored-cfg-leave-open/README.md`<br/>`~/PhD/medical-imaging/compositional_diffusion/artifacts/results/what-the-papers-report/what-the-papers-near-factored-cfg-leave-open/open-problems-near-factored-cfg.md`<br/>`~/PhD/medical-imaging/compositional_diffusion/artifacts/results/what-the-papers-report/which-finding-pairs-the-papers-back/README.md`<br/>`~/PhD/medical-imaging/compositional_diffusion/artifacts/drips/an-open-chest-xray-problem-for-iclr/ROUTE_WALK.md` |
