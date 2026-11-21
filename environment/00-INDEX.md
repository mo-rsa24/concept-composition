# 🗺️ Environment: concept-composition

What is true about the system this work runs on.

Read [the overview](overview.md) first, always. Then the row below matching what you are about
to touch.

## Table of contents

- [Before I...](#before-i)
- [The systems](#the-systems)
- [Still open](#still-open)

## Before I...

Navigation: 📋 [TOC](#table-of-contents) | [Next](#the-systems) ➡️

| The moment | Read this | Because |
|---|---|---|
| submit a job, or write an `sbatch` header | [hpc/nodes.md](hpc/nodes.md) | partitions differ in GPU model and time limit, and `sinfo` cannot tell you which node has a GPU: asking for the wrong one queues silently |
| estimate how long a training run will take, before launching it | [hpc/throughput.md](hpc/throughput.md) | real measured step time, checkpoint-save time, and eval-pass time, by GPU model and rank, because a guess here has cost real GPU-hours before |
| decide whether to submit or SSH into a shared device | [hpc/execution-protocol.md](hpc/execution-protocol.md) | "run an experiment" is a six-step decision here, not one action, and the shared-device path has mandatory safety rules |
| launch anything over SSH onto a node this session is not on | [hpc/execution-protocol.md](hpc/execution-protocol.md) | a relative path in the launch line resolves against `$HOME`, not the repo, and the failure is silent (`poe-launch-001`) |
| read or write project data, or add a disk guard to a job script | [storage.md](storage.md) | two filesystems, different quotas, no automatic mirroring, and a guard that checks the wrong mount has already caused a real loss (`poe-disk-001`) |
| run the authors' Reduce, Reuse, Recycle code (the 2D notebook or `inf_sample.py`) | [reduce-reuse-recycle-code.md](reduce-reuse-recycle-code.md) | the notebook fails on current jax until one argument is renamed, the CLEVR script borrows another venv's CUDA torch, and the released U-Net cannot run a forward pass as shipped |
| build or edit the paper | [paper.md](paper.md) | no system LaTeX exists here; the extension's default recipe fails outright, and a recent config change to fix that is unverified |
| hit an error that looks like it has happened before | [known-failures.md](known-failures.md) | 18 catalogued patterns with symptom, root cause, and solution, so a failure is not re-diagnosed from scratch |
| want to know how sure a fact in this folder actually is | [provenance.md](provenance.md) | every fact is marked verified, stated, or inferred, and several were re-checked live on 2026-08-24 |

## The systems

Navigation: ⬅️ [Before I...](#before-i) | 📋 [TOC](#table-of-contents) | [Next](#still-open) ➡️

| Folder | What you operate there | Leaves |
|---|---|---|
| [hpc/](hpc) | the cluster: partitions, nodes, GPUs, the launch protocol, measured throughput | 3 |
| [storage.md](storage.md) | the map of every folder on the two filesystems, their sizes, and the disk-guard rule | 1 (flat, no second leaf yet) |
| [reduce-reuse-recycle-code.md](reduce-reuse-recycle-code.md) | the authors' code for Du et al. 2023, cloned at `~/PhD/replications/reduce-reuse-recycle` | 1 (flat, no second leaf yet) |
| [paper.md](paper.md) | the LaTeX build for `writing/iclr/` | 1 (flat, no second leaf yet) |

How every fact here was established is in [provenance.md](provenance.md). The project failure
catalog is [known-failures.md](known-failures.md).

## Still open

Navigation: ⬅️ [The systems](#the-systems) | 📋 [TOC](#table-of-contents)

- **The [storage map](storage.md#the-storage-map) is unverified.** `mscluster84` did not answer SSH
  on 2026-11-21, so its folders come from this folder's records and the code. The map's warning
  holds the command that finishes it.
- **Whether `.vscode/settings.json` (added 2026-08-18, commit `2c4d0b6`) makes VS Code's plain
  "Build LaTeX project" work without manually picking the `tectonic` recipe.** Testing it means
  running a build. See [the paper build page](paper.md).
- **Whether every job script's disk guard reads the same resolved root as its output path,**
  after the `paths.resolve()` commits. See `poe-disk-001` in the
  [failure catalog](known-failures.md).
- **One script still names the retired `docs/ENVIRONMENT.md`:** a comment at line 9 of
  `1-lora-correction/scripts/animals_compose_transfer/smoke_live_curves.sh`. That file's last
  version is in git history at commit `94069341`.
- **No `reference`-type memory entry names the `environment/` folder.** The closest entry,
  `hpc-and-execution-conventions` (`type: project`), repeats several facts and points at the
  retired `docs/ENVIRONMENT.md`.

<!-- built-to: PAGE_LOOK_FORMAT v1 -->
