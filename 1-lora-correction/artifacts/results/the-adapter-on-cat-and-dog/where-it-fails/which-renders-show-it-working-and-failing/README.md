# Which renders show it working and failing

<!-- cluster:start -->

## On the cluster

The other 173 files of this result (169.3 MB) sit at `/datasets/mmolefe/poe_repair_min/artifacts-archive/1-lora-correction/artifacts/results/the-adapter-on-cat-and-dog/where-it-fails/which-renders-show-it-working-and-failing/`. [The file list](on-the-cluster.tsv) gives each file's size and sha256 checksum.

| Run | Files | Size | Kept on the laptop | Cluster folder |
|---|---|---|---|---|
| `(files at this level)` | 1 | 4.9 MB | no picture | `/datasets/mmolefe/poe_repair_min/artifacts-archive/1-lora-correction/artifacts/results/the-adapter-on-cat-and-dog/where-it-fails/which-renders-show-it-working-and-failing/` |
| `adapter-switch-off` | 2 | 7.3 MB | [a_tiger__x__a_parrot.png](adapter-switch-off/a_tiger__x__a_parrot.png) | `/datasets/mmolefe/poe_repair_min/artifacts-archive/1-lora-correction/artifacts/results/the-adapter-on-cat-and-dog/where-it-fails/which-renders-show-it-working-and-failing/adapter-switch-off/` |
| `contrast-corrector` | 26 | 19.0 MB | [ours_run9c6_w50.png](contrast-corrector/a_goat__x__a_cat/seed11/ours_run9c6_w50.png) | `/datasets/mmolefe/poe_repair_min/artifacts-archive/1-lora-correction/artifacts/results/the-adapter-on-cat-and-dog/where-it-fails/which-renders-show-it-working-and-failing/contrast-corrector/` |
| `elephant-and-penguin` | 95 | 69.4 MB | no picture | `/datasets/mmolefe/poe_repair_min/artifacts-archive/1-lora-correction/artifacts/results/the-adapter-on-cat-and-dog/where-it-fails/which-renders-show-it-working-and-failing/elephant-and-penguin/` |
| `existing` | 37 | 42.9 MB | [out_out__a_cat__x__a_dog__seed09.png](existing/x0-seeds-step-45000/out_out__a_cat__x__a_dog__seed09.png) | `/datasets/mmolefe/poe_repair_min/artifacts-archive/1-lora-correction/artifacts/results/the-adapter-on-cat-and-dog/where-it-fails/which-renders-show-it-working-and-failing/existing/` |
| `held-out-pair-candidates` | 10 | 19.7 MB | no picture | `/datasets/mmolefe/poe_repair_min/artifacts-archive/1-lora-correction/artifacts/results/the-adapter-on-cat-and-dog/where-it-fails/which-renders-show-it-working-and-failing/held-out-pair-candidates/` |
| `spread-and-handover` | 2 | 6.1 MB | [a_tiger__x__a_parrot.png](spread-and-handover/a_tiger__x__a_parrot.png) | `/datasets/mmolefe/poe_repair_min/artifacts-archive/1-lora-correction/artifacts/results/the-adapter-on-cat-and-dog/where-it-fails/which-renders-show-it-working-and-failing/spread-and-handover/` |

Bring one run back to the laptop from this folder:

```bash
rsync -a mscluster84:/datasets/mmolefe/poe_repair_min/artifacts-archive/1-lora-correction/artifacts/results/the-adapter-on-cat-and-dog/where-it-fails/which-renders-show-it-working-and-failing/<run>/ <run>/
```

<!-- cluster:end -->
