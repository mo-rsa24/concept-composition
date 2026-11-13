# Classifier guidance

**The question:** can a small classifier's gradient, added at each sampling step with SDXL frozen,
stand in for the trained LoRA?

Start at [the route](plans/route-classifier-guided-poe-from-toy-to-animals.md), then
[the outcome board](report/classifier-guided-poe/board.md). Plans 15, 17 and 18 are in `plans/`
with plan 15's verdict file in `plans/review/`. The sampler, composers and heads are in the
shared `poe_repair` package; the runners are in `scripts/`.

## Results

Every result, grouped by the question it answers, with a picture and one line each: [the results index](artifacts/results/README.md).
