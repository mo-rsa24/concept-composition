"""Where each loop item should land. First matching prefix wins. Read by census.py --markdown."""

P = "artifacts/notes/progress/"
CG = "2-classifier-guidance/"
M = "context/methods/"
R2 = CG + "report/"
WAIT = "waits for a verdict: the step is still running"
PROC = "none: a picture of the route's own plan, which stays beside the plan it explains"

RULES = [
    # Loop 4
    (P + "targets/one-network-against-separate-experts", M + "01-the-product-of-experts/01-overview.md", ""),
    (P + "targets/binding-each-body-ladder", R2 + "the-method-search/does-giving-each-subject-its-own-region-fix-identity.md", "the design ladder for M9"),
    (P + "targets/how-far-sample-and-paste-reaches", R2 + "the-method-search/does-composing-two-digit-models-beat-sampling-and-pasting.md", "context for M1's baseline"),
    (P + "targets/rivals-by-dataset", "1-lora-correction/report/against-other-methods/do-the-rivals-reproduce-their-printed-scores.md", "which rival reports on which benchmark"),
    (P + "targets/reads/how-a-number-becomes-a-target", "1-lora-correction/report/against-other-methods/do-the-rivals-reproduce-their-printed-scores.md", "how a printed number becomes a bar"),
    (P + "targets/", "", PROC),
    # Loop 5
    (P + "reproductions/RTD/", "1-lora-correction/report/against-other-methods/do-the-rivals-reproduce-their-printed-scores.md", ""),
    (P + "reproductions/FKC/", "1-lora-correction/report/against-other-methods/do-the-rivals-reproduce-their-printed-scores.md", ""),
    (P + "reproductions/which-rival-reruns-set-a-bar", "1-lora-correction/report/against-other-methods/do-the-rivals-reproduce-their-printed-scores.md", ""),
    (P + "reproductions/prompts-the-guard-keeps-out", "1-lora-correction/report/against-other-methods/do-the-rivals-reproduce-their-printed-scores.md", "the prompts dropped by the subject guard"),
    (P + "reproductions/CoInD/", "", "no finding: CelebA was made context by the owner's answer to board question 35, so no generated face was scored"),
    (P + "reproductions/CO3/", "", WAIT),
    (P + "reproductions/CoMat/", "", WAIT),
    (P + "reproductions/ToMe/", "", WAIT),
    (P + "reproductions/inclusive-finetune-and-push/", "", WAIT + " (Loop 6's inclusive fine-tune)"),
    (P + "reproductions/", "", PROC),
    # Loop 6
    (P + "method-search/01-beat-the-baseline/what-the-push-follows", M + "02-the-readout-checker/01-overview.md", ""),
    (P + "method-search/01-beat-the-baseline/paper-2", R2 + "the-method-search/does-adding-the-kept-push-to-co3-beat-co3-alone.md", ""),
    (P + "method-search/01-beat-the-baseline/smoke-puma", R2 + "the-method-search/does-adding-the-kept-push-to-co3-beat-co3-alone.md", ""),
    (P + "method-search/01-beat-the-baseline/", R2 + "the-method-search/does-composing-two-digit-models-beat-sampling-and-pasting.md", "the digit half; the SDXL half is does-adding-the-kept-push-to-co3-beat-co3-alone.md"),
    (P + "method-search/01c-clevr-paste-first/", "", WAIT + " (M7)"),
    (P + "method-search/04-bind-each-body/", R2 + "the-method-search/does-giving-each-subject-its-own-region-fix-identity.md", ""),
    (P + "method-search/abstract-checker-idea", M + "03-the-digit-map-checkers/01-overview.md", "the idea before Loop 8 tested it"),
    (P + "method-search/how-the-checker-could-scale", M + "02-the-readout-checker/01-overview.md", ""),
    (P + "method-search/", "", PROC),
    # Loop 7 reading
    (P + "abstract-checkers/01-reading/style-annotated-explainer", "", "none: a style sample for drawing, not a result or a method"),
    (P + "abstract-checkers/01-reading/glide-clip-score-against-people", M + "04-the-sentence-score-push/01-overview.md", "a figure reproduced from the GLIDE paper"),
    (P + "abstract-checkers/01-reading/what-the-papers-cover", "", PROC),
    (P + "abstract-checkers/01-reading/what-the-closer-papers-read", "", PROC),
    (P + "abstract-checkers/01-reading/", M + "03-the-digit-map-checkers/01-overview.md", "background from the reading"),
    # Loop 8
    (P + "abstract-checkers/04-environment/", R2 + "abstract-checkers-on-digits/is-the-small-environment-repeatable.md", ""),
    (P + "abstract-checkers/05-does-it-learn/", R2 + "abstract-checkers-on-digits/does-a-checker-learn-plurality-on-unseen-digits.md", ""),
    (P + "abstract-checkers/06-does-it-compose/", R2 + "abstract-checkers-on-digits/does-the-count-checkers-push-compose-unseen-digits.md", ""),
    (P + "abstract-checkers/08-remove-one-checker/", R2 + "abstract-checkers-on-digits/does-removing-one-checker-remove-its-property.md", ""),
    (P + "abstract-checkers/remove-one-checker", R2 + "abstract-checkers-on-digits/does-removing-one-checker-remove-its-property.md", "the design picture for V5"),
    (P + "abstract-checkers/", "", PROC),
    # Loop 9
    (P + "noise-text-critic/02-can-it-read/", R2 + "the-sentence-score/can-a-sentence-score-read-the-five-properties.md", ""),
    (P + "noise-text-critic/02b-linear-probe/", R2 + "the-sentence-score/does-a-linear-probe-on-frozen-image-vectors-read-unseen-pairs.md", ""),
    (P + "noise-text-critic/04-train-our-own/", R2 + "the-sentence-score/does-our-own-noise-aware-model-read-unseen-digit-pairs.md", ""),
    (P + "noise-text-critic/02c-fair-test/", R2 + "the-sentence-score/is-the-sentence-score-test-fair.md", ""),
    (P + "noise-text-critic/01-reading/latent-clip", M + "05-latent-clip/01-overview.md", ""),
    (P + "noise-text-critic/way-2", M + "05-latent-clip/01-overview.md", ""),
    (P + "noise-text-critic/01-reading/sentence-score-push", M + "04-the-sentence-score-push/01-overview.md", ""),
    (P + "noise-text-critic/way-1", M + "04-the-sentence-score-push/01-overview.md", ""),
    (P + "noise-text-critic/way-3", M + "04-the-sentence-score-push/01-overview.md", "the third way, training our own, as a section of the overview"),
    (P + "noise-text-critic/three-way-space-idea", M + "07-the-score-path/01-overview.md", ""),
    (P + "noise-text-critic/", "", PROC),
    # Loop 10
    (P + "score-space/05-signal/", R2 + "does-the-score-path-carry-which-digits-are-present.md", ""),
    (P + "score-space/00-ladder/style-sample", "", "none: a style sample for drawing, not a result or a method"),
    (P + "score-space/00-ladder/2-miniature/", M + "07-the-score-path/03-the-miniature.md", ""),
    (P + "score-space/00-ladder/0.5-network-view/", M + "07-the-score-path/02-the-network-view.md", ""),
    (P + "score-space/", M + "07-the-score-path/01-overview.md", ""),
    # Loop 12, started 2026-11-24
    (P + "leakage/", "", "Loop 12 started on 2026-11-24; its landing page is set when its first verdict lands"),
    # Scenes
    (CG + "artifacts/scenes/annotated-overviews/loop-4", M + "08-the-blip-vqa-score/01-overview.md", ""),
    (CG + "artifacts/scenes/annotated-overviews/loop-6", M + "02-the-readout-checker/01-overview.md", ""),
    (CG + "artifacts/scenes/annotated-overviews/loop-8", M + "03-the-digit-map-checkers/01-overview.md", ""),
    (CG + "artifacts/scenes/product-of-experts-interaction-term/", M + "01-the-product-of-experts/", "one page per rung, overview first"),
    (CG + "artifacts/scenes/digit-map-checkers/miniature-page/", M + "03-the-digit-map-checkers/03-the-miniature.md", "frames of one interactive page; the page is linked, not each frame"),
    (CG + "artifacts/scenes/digit-map-checkers/", M + "03-the-digit-map-checkers/", "one page per rung, overview first"),
    (CG + "artifacts/scenes/noisyclip-sentence-score/style-sample", "", "none: a style sample for drawing, not a result or a method"),
    (CG + "artifacts/scenes/noisyclip-sentence-score/reference", "", "none: style references for drawing"),
    (CG + "artifacts/scenes/noisyclip-sentence-score/", M + "04-the-sentence-score-push/", "one page per rung, overview first; Latent CLIP's overview goes to 05-latent-clip"),
    (CG + "artifacts/scenes/word-map-checker/", M + "06-the-word-map-checker/", "one page per rung, overview first"),
    (CG + "artifacts/scenes/", "", "made before the loops; not part of this drain"),
    (CG + "artifacts/results/", "", "made before the loops; not part of this drain"),
    ("1-lora-correction/artifacts/results/", "", "made before the loops and moved into place on 2026-11-22; not part of this drain"),
]


def landing(path: str):
    for prefix, page, note in RULES:
        if path.startswith(prefix):
            return page, note
    return "", "no rule yet"
