<a id="provenance"></a>

# 🗺️ Environment: how each fact in this folder was established

Navigation: 📋 [Index](00-INDEX.md) | [Overview](overview.md#provenance)

One row per fact, wherever in this folder the fact lives. `verified` means someone ran the check
live on the date given and saw the result. `stated` means a person said it and nobody checked it.
`inferred` means it was read out of the repository or a document and confirmed by neither a
person nor a run this sitting.

| Fact | Where it lives | Mark | How it was established |
|---|---|---|---|
| Cluster session node was `mscluster85`, `bigbatch`, RTX 3090 24GB, 123GB RAM, 28 cores | `hpc/nodes.md` | verified | `nvidia-smi` on 2026-08-04 |
| Cluster session node's partition and state (`bigbatch`, `alloc`) | `hpc/nodes.md` | verified | `sinfo -N -o "%N %P %t"`, live, 2026-08-24 |
| The laptop checkout has no `squeue`, `sinfo` or `sbatch` and reaches the cluster through `scripts/cluster.sh` | `overview.md`, `hpc/nodes.md` | verified | `which squeue sinfo sbatch` printed nothing on the laptop, 2026-11-22 |
| GPU work launches through `sbatch` inside caps of one `biggpu` job and six `bigbatch` jobs per user, a denied `sbatch` waits, and a `nohup` launch over SSH outside the session's own Slurm job needs the owner's yes | `hpc/execution-protocol.md`, `overview.md`, `hpc/nodes.md` | stated | the owner's decision, recorded in the shared session rules on 2026-11-22 |
| Partition table (nodes, time limits) for `biggpu`, `bigbatch`, `batch`, `stampede`, `gpuexpress` | `hpc/nodes.md` | verified | `sinfo -o "%P %a %l %D"`, live, 2026-08-24 (re-confirms a 2026-08-04 check) |
| `sinfo` GRES column reads `(null)` for every partition | `hpc/nodes.md`, `overview.md` | verified | `sinfo -o "%P %N %G"`, live, 2026-08-24 |
| `mscluster106`: 2x Quadro RTX 8000, 49GB each | `hpc/nodes.md` | verified | `nvidia-smi` over SSH, 2026-08-19 |
| `mscluster108`: 2x Quadro RTX 8000, 49GB each; `mscluster109`: 2x RTX A6000, 49GB each; `mscluster110`/`111`/`112`: 1x RTX PRO 6000 Blackwell, ~96GB each | `hpc/nodes.md` | verified | `nvidia-smi` over SSH on all six live nodes, 2026-09-01 |
| `co3` python resolves to `python3.10` at the stated absolute path | `hpc/nodes.md`, `overview.md` | verified | `ls -la`, live, 2026-08-24 |
| Env list under `~/miniforge3/envs/` (13 environments including `co3`, `co3_bw`, `superdiff`) | `hpc/nodes.md` | verified | `ls`, live, 2026-08-24 |
| `co3_bw`'s exact purpose versus `co3` | `hpc/nodes.md` | verified | live, 2026-09-05: a `co3` launch on `mscluster110` (Blackwell) died with a CUDA error inside a minute; the same script under `co3_bw` rendered 24 images in 10 minutes |
| `mscluster111` and `mscluster112` cards work under `co3_bw`, and each has a fault form it has shown before | `hpc/nodes.md`, `known-failures.md` | verified | live over SSH, 2026-11-17: `torch.cuda.is_available()` True and a 1000×1000 ones matmul summing to 1e9 on both; the fault forms seen 2026-09-05 on 111 (one 16-second sampling took 15 minutes on the CPU) and 2026-10-13 on 112 (`CUDA unknown error`) |
| `superdiff` exists for SuperDiff work if `co3` dependencies conflict | `hpc/nodes.md` | stated | not re-verified this sitting |
| Shared-device path and its safety rules | `hpc/execution-protocol.md` | verified | live over SSH on `mscluster106`, 2026-08-19: GPU 1 at 1MiB/0% while GPU 0 carried another user's 8GB process; a torch matmul from `co3_bw` on GPU 1 succeeded without touching GPU 0 |
| SSH launch line must use absolute paths (`poe-launch-001`) | `hpc/execution-protocol.md`, `known-failures.md` | verified | reproduced three times, 2026-08-19, launching the first short run on `mscluster106` |
| `/datasets` and `/home-mscluster` sizes, used, free | `storage.md` | verified | `df -h /datasets /home-mscluster`, live, 2026-11-21: `/datasets` 377T, 200T used (54%), 177T free; `/home-mscluster` 73T, 42T used (58%), 32T free |
| Storage map: folders, links, sizes and disk use | `storage.md` | verified | `ls`, `du -sk` (60-second cap per folder) and `df -h` through `mscluster84`, live, 2026-11-21 |
| Folders the storage map marks "not measured" | `storage.md` | verified | `du -sk` hit its 60-second cap on them, live, 2026-11-21, so each total on that page is a lower bound |
| How often the code names each storage folder | `storage.md` | verified | `grep` over `.py`, `.sh`, `.sbatch`, `.yaml` and `.yml` files in the repository, 2026-11-21 |
| Cluster checkout at `ccf76f13` with 132 tracked and 56 untracked files in `paper/iclr/figures/` | `paper.md` | verified | `git ls-files` on the cluster checkout and `git check-ignore` against the current `.gitignore`, 2026-11-21 |
| No per-user `quota` command on these nodes | `storage.md` | verified | `which quota`, live, 2026-08-24 |
| `/home-mscluster` hit 100% once and killed checkpointing | `storage.md`, `overview.md` | stated | hard-won operational history, carried from the project's earlier cluster notes; no run reproduces this on demand |
| Disk guard on `run_dose_sweep.sh` checked the wrong mount, 3.4GB landed on `/home-mscluster` | `storage.md`, `known-failures.md` | stated | recorded in the project's own `CLAUDE.md`; not re-run this sitting |
| Cached residual trajectory contents (18 train + 58 heldout pairs, ~12 seeds, 50 steps) | `storage.md` | verified | direct filesystem inspection and a loaded residual file, 2026-08-04 |
| W&B API key present in `~/.netrc`; `WANDB_API_KEY` set in-session | `overview.md` | verified | file existence checked (not read) and env var presence checked, live, 2026-08-24 |
| `pdflatex`, `xelatex`, `latexmk`, `bibtex` absent from PATH | `paper.md`, `overview.md` | verified | `which` for each, live, 2026-08-24 (re-confirms a 2026-08-05 check) |
| `tectonic` present and executable at the stated absolute path | `paper.md` | verified | `ls -la` and `which`, live, 2026-08-24 |
| `tectonic` builds the stock template successfully | `paper.md` | stated | carried from the 2026-08-05 note; not rebuilt this sitting (a build would write a PDF outside this folder's scope) |
| LaTeX Workshop extension present, version `10.15.2`, server-side | `paper.md` | verified | `ls ~/.vscode-server/extensions/`, live, 2026-08-24 (re-confirms a 2026-08-05 check) |
| Extension's 9 default recipes, `latexmk` first, `tectonic` last | `paper.md` | inferred | read from `package.json` on 2026-08-05; not re-read this sitting |
| `.vscode/settings.json` exists, tracked, defines a custom `tectonic` recipe, no `recipe.default` key | `paper.md`, `overview.md` | verified | file read directly, live, 2026-08-24; `git log` shows it added in commit `2c4d0b6` |
| Whether the new `.vscode/settings.json` makes plain "Build LaTeX project" work without picking a recipe | `paper.md` | **open** | deliberately not tested this sitting; testing means running a build, which writes outside this folder's scope |
| VS Code keystroke build route (`Ctrl+Shift+P` -> Build with recipe -> tectonic) | `paper.md` | stated | user-stated 2026-08-05; the command palette cannot be driven from a shell, so this has never been machine-verified |
| The repository-wide `*.pdf` rule is commented out at `.gitignore:43`; only `writing/restoring-plurality-in-compositional-diffusion_overleaf/**/*.pdf` (`.gitignore:72`) keeps PDFs out | `paper.md` | verified | `.gitignore` read and `git check-ignore -v` run on a PDF under that folder and on `foo.pdf` (not ignored), 2026-11-22 |
| Fraction-of-distance-reached, direction-cosine, and scorer failure patterns (`poe-score-001/002`, `poe-lora-001/002`) | `known-failures.md` | stated | carried verbatim from `environment/known-failures.md`, first discovered at step-09; not re-run this sitting |
| `batch` partition: `mscluster[120-219]`, 100 nodes, 1-day limit, one RTX 3060 12 GB per node; a single SDXL 1024 render took 204 s on `mscluster160` | `hpc/nodes.md` | verified | over SSH on `mscluster134`, `160`, `200` and `217`, 2026-11-04 |
| Idle `bigbatch` probe of 2026-11-19: `mscluster68`, `75`, `76`, `82` faulted, `mscluster83` healthy, `mscluster71` died mid-job (job 90386) and drained jobs 90390 to 90395 | `hpc/nodes.md` | stated | recorded on the page as a 2026-11-19 probe; the page does not say it was re-run |
| `/datasets` invisible from inside a Slurm job on `mscluster72` and `73` while visible over SSH; eight jobs died | `hpc/nodes.md` | stated | recorded on the page for 2026-09-12; not re-run |
| `mscluster88` works under `co3_bw`: RTX 3090 24 GB, matmul sums to 1e9, PyPI at 13.5 MB/s | `hpc/nodes.md` | verified | live, 2026-10-13 |
| Compute nodes cannot reach the VS Code extension marketplace; the laptop-download setting is the fix | `hpc/nodes.md` | verified | from `mscluster88`, 2026-10-13: downloads returned an 875-byte "The request is blocked." page; `ms-python.python` 2026.8.0 and `ms-toolsai.jupyter` 2025.9.1 installed with `remote.downloadExtensionsLocally` |
| The 2D notebook needs `a_min=` renamed to `min=` on jax 0.11.2 | `reduce-reuse-recycle-code.md` | verified | 2026-10-13 |
| The 2D notebook's environment is `.venv-2d` (Python 3.12, jax 0.11.2 on CPU) | `reduce-reuse-recycle-code.md` | verified | 2026-10-13 |
| The notebook's `tensorflow`, `tensorflow_datasets` and `pandas` imports are unused | `reduce-reuse-recycle-code.md` | verified | 2026-10-13 |
| `inf_sample.py` calls `model1.to("cuda")` unconditionally | `reduce-reuse-recycle-code.md` | verified | 2026-10-13, by reading the code |
| The CLEVR script borrows the playground torch `2.14.1+cu130` interpreter plus `.clevr-extras` | `reduce-reuse-recycle-code.md` | verified | 2026-10-13 |
| The released energy U-Net never assigns `h` before line 1386 of `unet.py` | `reduce-reuse-recycle-code.md` | inferred | 2026-10-13, from reading the code; not run |
| Checkpoint `ebm-49x1874.pt`: 500,704,677 bytes, 542 tensors, 125.1 M parameters | `reduce-reuse-recycle-code.md` | verified | 2026-10-13 |
| Cluster clone at `/home-mscluster/mmolefe/PhD/reduce-reuse-recycle`, branch `walk` at `81177d9` | `reduce-reuse-recycle-code.md` | verified | 2026-10-13 |
| Cluster 2D environment is a uv venv on `/datasets` with jax 0.11.2 and the `cuda12` plugin; no mamba env holds a usable jax | `reduce-reuse-recycle-code.md` | verified | 2026-10-13: `jax.devices()` prints `[CudaDevice(id=0)]`, matmul sums to 1e9, install took 5 min 15 s |
| CelebA parquet copy at `/datasets/mmolefe/celeba/flwrlabs-celeba/img_align+identity+attr/`: train 162,770 faces (1,387 blond men), valid 19,867 (182), test 19,962 (180) | `storage.md` | stated | the page records the 2026-11-04 pull and the counts from the `Male` and `Blond_Hair` columns without a mark |
| MNIST raw files at `/datasets/mmolefe/raw/mnist/`, copied from the laptop | `storage.md` | stated | the page records the 2026-11-19 copy with matching sha256 on all four files, without a mark |
| `/datasets/mmolefe` layout since 2026-11-13: `raw/`, `projects/<project>/`, `_archive/`, symlinks at old names | `storage.md` | stated | recorded on the page from `/datasets/mmolefe/README.md`, without a mark |
| `/datasets` outage: `Input/output error` from about 13:10 UTC on 2026-11-17, still failing at 05:40 UTC on 2026-11-18, reading normally on 2026-11-21 | `storage.md` | verified | `ls /datasets/mmolefe` and `df -h /datasets` over `scripts/cluster.sh sh` on 2026-11-17 and 2026-11-18; the 2026-11-21 reading is the `df -h` in the size row above |
| Rank 8 step time, steady state: 0.706 s/step | `hpc/throughput.md` | measured | the page's own mark, "measured": `mscluster109`, RTX A6000, 2026-09-01 |
| Rank 16 step time, steady state: 1.236 s/step | `hpc/throughput.md` | measured | the page's own mark, "measured": `mscluster106`, RTX 8000, 2026-09-01 |
| Rank 32 step time, steady state: 1.144 s/step | `hpc/throughput.md` | measured | the page's own mark, "measured": `mscluster108`, RTX 8000, 2026-09-01 |
| Rank 32 total wall time, 1,000 steps: 1,213 s | `hpc/throughput.md` | measured | the page's own mark, "measured": same run, 2026-09-01 |
| Rank 16 total wall time, 1,000 steps: 1,256 s | `hpc/throughput.md` | measured | the page's own mark, "measured": same run, 2026-09-01 |
| Checkpoint save time: 0.7 s | `hpc/throughput.md` | measured | the page's own mark, "measured": `timing_smoke_r8`, RTX A6000, 2026-09-01 |
| Tracking-set eval pass: 163.1 s (40.8 s/cell) | `hpc/throughput.md` | measured | the page's own mark, "measured": `timing_smoke_r8`, RTX A6000, epoch 200, 2026-09-01 |
| Rank 8 training loop, 10,000 steps: 7,275 s (2h1m) | `hpc/throughput.md` | measured | the page's own mark, "measured": `timing_smoke_r8`, 2026-09-01 |
| Rank 8 total wall time, 10,000 steps: 7,583 s (2h6m) | `hpc/throughput.md` | measured | the page's own mark, "measured": `timing_smoke_r8`, 2026-09-01 |
| Rank 8 step time on Blackwell: 0.236 s/step | `hpc/throughput.md` | measured | the page's own mark, "measured": `mscluster110`, run `8vl2uzak`, 2026-09-01 |
| Checkpoint save time on Blackwell: 0.3 s | `hpc/throughput.md` | measured | the page's own mark, "measured": run `8vl2uzak`, 2026-09-01/02 |
| Tracking-set eval pass on Blackwell: 64.1 s (16.0 s/cell), about 45 s on later passes | `hpc/throughput.md` | measured | the page's own mark, "measured": run `8vl2uzak`, 2026-09-01/02 |
| Tracking-set eval pass on RTX 8000: 206.0 s (51.5 s/cell) | `hpc/throughput.md` | measured | the page's own mark, "measured": `mscluster106`, run `jii2mtb0`, 2026-09-02 |

<!-- built-to: PAGE_LOOK_FORMAT v1 -->
