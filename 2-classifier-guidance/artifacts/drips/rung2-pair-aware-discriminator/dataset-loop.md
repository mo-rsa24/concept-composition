# Rung 2 training set: the loop until it is clean

Read by each pass of the `/loop` the reader started on 2026-11-17. Each pass reads this file,
does the next step, and rewrites "Where the loop is" and "Next batch".

## Clean means

1. At least 15 positives: tiles from the three pair wordings called one cat and one dog, two
   separate bodies, an intact photo.
2. Every negative is an intact photo: one lone cat, one lone dog and one plain product per seed
   1 to 8. A damaged negative is replaced by a fresh-noise draw of the same column under the same
   seed, never deleted.
3. No tile from seeds 9 to 16 anywhere.

The loop stops when all three hold, then hands the dataset to the reader before round 3.

## How a pass works

1. Read the calls file: `2-classifier-guidance/artifacts/results/does-a-small-discriminator-steer-cat-x-dog/training-set/calls.csv`.
2. If clean, stop the loop and report.
3. If a batch is running (`pgrep -af render_training_set.py` on `mscluster107`), wait.
4. If a batch landed: rsync tiles back (excluding `latents/` and `raw/`), build full-resolution
   strips, call each new tile by eye at full resolution, append the calls (`called_by` "claude,
   delegated by the reader 2026-11-17 (veto after)"), append every failed pair-wording tile to
   `problem-seeds.csv`, commit the walk files.
5. Otherwise launch the next batch on `mscluster107` GPU 0 with `scripts/cluster.sh run` after
   checking the card is free and the tree is committed.

Draw numbers continue per seed; generator number is 10,000 + 100 × seed + draw. Draws 1 and 2 of
seeds 5 to 8 are used by the pair wordings.

## Where the loop is

Pass 0: 64 tiles called, 17 positives (condition 1 holds). Five negatives are damaged
(condition 2 fails): seed 1 lone cat (a drawn face), seed 5 lone cat, lone dog and product
(engraving style), seed 6 lone cat (a cut-out portrait).

## Next batch

```
cd /home-mscluster/mmolefe/Playground/PhD/concept-composition && PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python && S=2-classifier-guidance/scripts/rung2/render_training_set.py && $PY $S --seeds 1 5 6 --extra-draws 1 --draw-start 3 --extra-columns lone_cat --tag neg-a && $PY $S --seeds 5 --extra-draws 1 --draw-start 4 --extra-columns lone_dog product --tag neg-b
```

Log: `/datasets/mmolefe/poe_repair_min/outputs/classifier_guidance_ladder/rung2/logs/negatives.log`.

## The problem-seed list

`training-set/problem-seeds.csv`: every pair-wording tile that did not give one cat and one dog
apart and intact, with the noise it came from. For the full-scale comparison (our method against
the joint prompt and CO3), these seeds' noise also trains the rung 2 discriminator, so that
comparison is drawn on fresh held-out seeds collected the same way.
