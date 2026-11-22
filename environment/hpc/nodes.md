# 🗺️ Environment: cluster partitions, nodes and GPUs

Navigation: 📋 [Index](../00-INDEX.md) | [Overview](../overview.md#compute--runtime)

A session on the cluster (VS Code + Claude) runs ON a compute node, not the login node. Claude
never runs on the login node. A session on the laptop has no Slurm commands and reaches the cluster
through `scripts/cluster.sh`. So the real execution split this project cares about is
"in-session on the current node" versus "submitted or interactive job on a bigger GPU," not
"login vs compute."

**The cluster session node, when last checked, is `mscluster85`, on `bigbatch`** (RTX 3090, 24GB VRAM, 123GB
RAM, 28 cores). Verified live via `nvidia-smi` on 2026-08-04, and its partition re-verified
live via `sinfo -N -o "%N %P %t"` on 2026-08-24 (state: `alloc`).

## Table of contents

- [Partitions](#partitions)
- [Python environments](#python-environments)
- [VS Code extensions on the compute nodes](#vs-code-extensions-on-the-compute-nodes)
- [🖼️ In pictures, code and further reading](#in-pictures-code-and-further-reading)
- [Cross-references](#cross-references)

## Partitions

Navigation: 📋 [TOC](#table-of-contents) | [Next](#python-environments) ➡️

Verified live via `sinfo -o "%P %a %l %D"` on 2026-08-24 (columns: partition, availability,
time limit, node count).

| Partition | Nodes | Time limit | Notes |
|---|---|---|---|
| `biggpu` | mscluster[106-112], 7 nodes | 3 days | First choice for GPU jobs. `mscluster106`: 2x Quadro RTX 8000, 49GB each. `mscluster108`: 2x Quadro RTX 8000, 49GB each. `mscluster109`: 2x RTX A6000, 49GB each. `mscluster110`, `111`, `112`: 1x NVIDIA RTX PRO 6000 Blackwell Workstation Edition each, ~96GB VRAM (Blackwell generation, newer and larger than every other `biggpu` card). `mscluster107`: 2x Quadro RTX 8000, 49GB each, both answering `nvidia-smi` and both idle on 2026-11-04. With the owner's yes, a second job can run beside someone else's on these nodes: pin the free card with `CUDA_VISIBLE_DEVICES`, since device 0 is the one other users take first. `mscluster110`, `111` and `112` cannot be driven by `co3` at all, and this is the reason the 2026-09-05 `mscluster111` report has: `co3` carries torch 2.5.1+cu118, and a Blackwell card needs CUDA 12.8 or later, so `torch.cuda.is_available()` returns False and a launch runs on the CPU with no error and no warning. Measured again on 2026-11-04 on `mscluster111`: `nvidia-smi` reports the card present with 2 MiB used and no compute apps while a job is running, and a 20-step SDXL LoRA training step took 11 seconds against well under a second on a working GPU. Every launcher asserts `torch.cuda.is_available()` after pinning the device, but a script written without that assert will complete and report plausible numbers. Check `nvidia-smi --query-compute-apps` from another shell: a GPU job that uses no GPU memory is on the CPU. On 2026-09-05 `mscluster111` also read pstate `[GPU requires reset]` and utilisation `[N/A]` under `co3_bw`, and a 320-render job crawled there at 14 GB RSS for ten minutes before it was caught. All others verified live via `nvidia-smi` on 2026-09-01. `sinfo` GRES reads `(null)` for every partition, so per-node GPU models are only discoverable by SSH + `nvidia-smi`, not by querying Slurm. |
| `bigbatch` | mscluster[42-89], 48 nodes | 3 days | Fallback when no `biggpu` node is idle. `mscluster50` and `mscluster51` sit `idle` in `sinfo` but their GPU is [hardware-faulted](../known-failures.md#entry-id-poe-launch-002) (`nvidia-smi`: "Unable to determine the device handle for GPU0: 0000:17:00.0: Unknown Error"), so a job lands there and the [GPU guard](execution-protocol.md) aborts it in under a second (jobs 49277 and 49278, 2026-09-03). A probe of every idle `bigbatch` node over SSH (`ssh <node> nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv`) finds the same fault on more nodes than Slurm reports. On 2026-09-10 the faulted set was `mscluster44`, `45`, `50`, `57`, `65` and `76`, all reporting `Unable to determine the device handle for GPU0: 0000:17:00.0`; job 52566 landed on `mscluster76` and the guard refused it in under a second. `mscluster74` reads utilisation `[N/A]`, which the launchers treat as faulted. The healthy idle cards that day were RTX 3090 24 GB on `mscluster46`, `49`, `72`, `73`, `75` and `88`. On 2026-09-24 every idle `bigbatch` node answered `nvidia-smi` with "No devices were found": `mscluster44`, `57`, `59`, `62`, `65`, `75` and `83`, with `mscluster45` not answering at all, so job 59136 died on `mscluster59` in seconds. `mscluster59`, `62` and `75` were healthy at the previous probe, and `75` was one of the six the 2026-09-10 probe named as good. Idle on this partition is evidence of a dead card rather than a free one. The fault set grows between probes, so prefer pinning: run the probe over `sinfo -p bigbatch -N -h -o "%N %t" | awk '$2=="idle"'` and pass `--nodelist=<healthy node>`. Excluding works too but goes stale, and the current list is `--exclude=mscluster44,mscluster45,mscluster50,mscluster51,mscluster57,mscluster65,mscluster68,mscluster71,mscluster74,mscluster75,mscluster76,mscluster82`. On 2026-11-19 a probe of the idle nodes found `mscluster68`, `75`, `76` and `82` with the device-handle fault and `mscluster83` healthy (it ran a job), and `mscluster71`'s card died mid-job (job 90386 trained, then `CUDA error: unspecified launch failure` while drawing, then `nvidia-smi` failed); the six jobs queued behind it were each sent to `mscluster71` and refused by the GPU guard in one second (jobs 90390 to 90395), so a card that dies drains the queue into itself until it is excluded. `sinfo` lists the session node `mscluster85` as idle while it carries this project's own runs, so idle is not free there either. A second fault is invisible to every GPU probe: on 2026-09-12 `mscluster72` and `mscluster73` could not see `/datasets` **from inside a Slurm job**, while `ssh <node> test -d /datasets/...` from the session node succeeded on both, so an SSH probe passes them as healthy and the job then writes nowhere. Eight jobs (53301, 53302, 53312, 53313, 53316, 53317, 53319, 53320) died on those two nodes and none on `mscluster46`, `49`, `52`-`56`, `58`, `59` or `75`, which all probed `datasets=OK` under Slurm the same hour. Probe the filesystem the job will write to with a two-minute `sbatch`, never with `ssh`, and keep the launcher's own directory guard: before it existed the `df` check ran against a missing path, left the usage string empty, and the `>= 90` test passed silently. |
| `batch` | mscluster[120-219], 100 nodes | 1 day | Short jobs only. One NVIDIA GeForce RTX 3060, 12 GB, per node, verified over SSH on 2026-11-04 on `mscluster134`, `160`, `200` and `217`; on that day 72 were idle (`mscluster[134-143,156-215,217-218]`) and `mscluster134` already held 11.5 GB. `/datasets` is visible from those four over SSH. One SDXL 1024 render with the `co3` interpreter fits: a single-picture job on `mscluster160` took 204 s end to end, almost all of it loading SDXL from `/datasets`, so a node pays back only on jobs of many pictures. Too small for two SDXL UNets with gradients, so classifier training stays on a 49 GB card. Reachable over SSH. This project launches on `bigbatch` and `biggpu`, so running work here needs the owner's yes. |
| `stampede` | mscluster[22-41], 20 nodes | 3 days | Rarely used for this project. |
| `gpuexpress` | mscluster116, 1 node | 1 hour | Short wiring checks only. |

**`sinfo` never shows GPU GRES on this cluster.** Confirmed again live on 2026-08-24
(`sinfo -o "%P %N %G"`): every partition reads `(null)` in the GRES column. A job script that
trusts Slurm to pick a GPU-capable node will not get one; node capability is tribal knowledge,
recorded in the table above, not queryable. GRES (generic resource) is Slurm's name for a
countable extra on a node, such as a GPU; `(null)` means the cluster declares none.

**Many nodes show down or drain at any given time.** An idle-looking partition (available in
the `%a` column) can still queue a job, because individual nodes inside it are unavailable.
Check node STATE (`sinfo -N -o "%N %P %t"`), not just partition-level availability, before
assuming a submission will start promptly.

## Python environments

Navigation: ⬅️ [Previous](#partitions) | 📋 [TOC](#table-of-contents) | [Next](#vs-code-extensions-on-the-compute-nodes) ➡️

`/home-mscluster/mmolefe/miniforge3/envs/` (mamba/miniforge). Verified live via `ls` on
2026-08-24: `co3`, `co3_bw`, `compvis_ldm`, `cxr`, `jax`, `jax115`, `ldm`, `mamba`, `rosalia`,
`score-sde`, `superdiff`, `superdiffusion`, `tex`.

**`co3` is the project environment.** Absolute path:
`/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python` (a symlink to `python3.10`, verified
live via `ls -la` on 2026-08-24). Every job script must use this absolute path, never a bare
`python`, because nothing in the [execution model](../overview.md#execution-model) activates an environment implicitly.

**`co3_bw` is the environment for the Blackwell nodes (mscluster110, 111, 112).** Measured
2026-09-03 on mscluster112: `co3_bw` is torch 2.11.0+cu128 with kernels for `sm_100` and
`sm_120`, and the same diffusers 0.29.2 and transformers 4.44.1 as `co3`, plus peft 0.19.1.
`co3` is torch 2.5.1+cu118 with kernels up to `sm_90`, so on an RTX PRO 6000 Blackwell a
`co3` CUDA op produces no output at all rather than an error. Rule: `co3` on the RTX 8000 and
A6000 nodes (106, 108, 109) and on the session node; `co3_bw` on 110 to 112. Launchers take
the python path as `PY=` for this reason (`scripts/launch_sdlora.sh`,
`1-lora-correction/scripts/animals_compose_transfer/smoke_live_curves.sh`).

**`mscluster111`'s card works under `co3_bw`, and has faulted before (verified 2026-11-17).**
Over SSH it read 2 MiB used, 0% utilisation and 25 °C; `torch.cuda.is_available()` under `co3_bw`
returned `True` (torch 2.11.0+cu128), a 1000×1000 ones matmul on the card summed to 1e9, and
`/datasets` was visible. The fault it shows when it fails: `nvidia-smi` lists the card with 2 MiB
used and `utilization.gpu=[N/A]`, `torch.cuda.is_available()` returns `False`, and a sampler runs on
the CPU at about 40 times the GPU time with no error in its log (seen 2026-09-05). Run the CUDA probe
below before any GPU work here; the shared-device launch script refuses a device whose utilisation
reads `[N/A]` or whose CUDA probe fails, per
[poe-launch-002](../known-failures.md#entry-id-poe-launch-002).

**`mscluster112`'s card works under `co3_bw`, and has faulted before (verified 2026-11-17).** The
same probe passed: 2 MiB used, 0%, 26 °C, `torch.cuda.is_available()` `True`, the matmul summing to
1e9, `/datasets` visible. The fault it shows when it fails is the one the `bigbatch` nodes above
show: `nvidia-smi` prints `Unable to determine the device handle for GPU0: 0000:01:00.0: Unknown
Error` and then `No devices were found`, and `torch.cuda.get_device_name(0)` under `co3_bw` raises
`RuntimeError: CUDA unknown error` (seen 2026-10-13, node uptime 54 days). It shows up even with the
node allocated and a VS Code remote session open on it, so holding the node says nothing about its
card. A faulted card needs an administrator reset or a reboot.

The probe for either card, from the laptop:
`scripts/cluster.sh sh 'ssh <node> "/home-mscluster/mmolefe/miniforge3/envs/co3_bw/bin/python -c \"import torch; print(torch.cuda.is_available()); x=torch.ones(1000,1000,device=\\\"cuda\\\"); print((x@x).sum().item())\""'`.
It prints `True` and `1000000000.0` on a working card.

**`mscluster88` works under `co3_bw` (verified 2026-10-13).** RTX 3090, 24 GB, driver 595.71.05;
`torch.cuda.is_available()` is True and a 1000×1000 ones matmul on the card sums to 1e9. From
this node PyPI downloaded at 13.5 MB/s (30 MB of the `nvidia-cublas-cu12` wheel in 2.3 s), so a
[2.3 GB CUDA install](../reduce-reuse-recycle-code.md#on-the-cluster-mscluster88) takes about three minutes here.


`superdiff` exists for SuperDiff-related work if its dependencies conflict with `co3`
(stated, not re-verified this sitting).

## VS Code extensions on the compute nodes

Navigation: ⬅️ [Previous](#python-environments) | 📋 [TOC](#table-of-contents) | [Next](#in-pictures-code-and-further-reading) ➡️

**The compute nodes cannot reach the VS Code extension marketplace.** From `mscluster88`, every
extension download returns an 875-byte Microsoft page reading "The request is blocked." Both
`code-server --install-extension` and the Extensions panel's install fail with `Extract: End of
central directory record signature not found`. Install from VSIX also fails: the main extension
unpacks, but the members of its extension pack (for `ms-python.python`: Pylance, debugpy, Python
Environments) are fetched from the node, those downloads are blocked, and the install rolls back.
The truncated downloads sit in `~/.vscode-server/data/CachedExtensionVSIXs/`. (verified 2026-10-13)

**The fix is to let the laptop download remote extensions.** On the laptop, tick **Remote: Download
Extensions Locally** (`remote.downloadExtensionsLocally`) in User settings, reload the window, then
click Install in SSH: mscluster88 in the Extensions panel. VS Code fetches each extension and its
pack members on the laptop and copies them over. `ms-python.python` 2026.8.0 and
`ms-toolsai.jupyter` 2025.9.1 installed this way. (verified 2026-10-13)

<!-- related:start (drain) -->
<a id="in-pictures-code-and-further-reading"></a>

## 🖼️ In pictures, code and further reading

Navigation: ⬅️ [Previous](#vs-code-extensions-on-the-compute-nodes) | 📋 [TOC](#table-of-contents) | [Next](#cross-references) ➡️

No picture of the cluster exists under `artifacts/`. The block gives the probe that reads per-card state, which `sinfo` cannot, and the faults the page records, one row each.

### The code behind it

`scripts/gpu_probe.sh` (lines 7 to 12, named here, not linked) prints one line per card with utilisation in the last column, so a card reading `[N/A]` shows it there.

```bash
n=$(hostname -s)
declare -A IDX
while IFS=', ' read -r i u m t z; do
  IDX[$u]=$i
  echo "GPU $n $i $m $t $z"
done < <(nvidia-smi --query-gpu=index,uuid,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits)
```

- The query asks for index, uuid, memory used, memory total and utilisation. The `bigbatch` probe in the partition table asks for name, memory used and utilisation, so the two share memory used and utilisation.
- The file's own header says `[N/A]` marks a faulted card, which agrees with the page.

### What comes out

Seven of the faults the page records, in date order, each with the card or node it names (all `stated`, from the table row and sections above):

| Date | Node | What was seen |
|---|---|---|
| 2026-09-03 | `mscluster50`, `51` | idle in `sinfo`, device-handle fault, jobs 49277 and 49278 refused by the guard |
| 2026-09-05 | `mscluster111` | `[GPU requires reset]`, utilisation `[N/A]` under `co3_bw` |
| 2026-09-10 | six `bigbatch` nodes | device-handle fault; job 52566 refused on `mscluster76` |
| 2026-09-12 | `mscluster72`, `73` | `/datasets` invisible inside a Slurm job, visible over SSH; 8 jobs died |
| 2026-09-24 | seven idle `bigbatch` nodes | "No devices were found"; job 59136 died on `mscluster59` |
| 2026-10-13 | `mscluster112` | `No devices were found`, `CUDA unknown error` |
| 2026-11-19 | `mscluster71` | card died mid-job (job 90386); jobs 90390 to 90395 refused |

### Go deeper

| What | Link |
|---|---|
| How a run is launched across these nodes | [execution protocol](execution-protocol.md) |
| Timings per card | [training run timings](throughput.md) |
| The fault entry | [poe-launch-002](../known-failures.md#entry-id-poe-launch-002) |
| Storage the jobs write to | [storage](../storage.md) |

### Still open

No screenshot or filed `nvidia-smi` capture of a faulted card is under `artifacts/`; the readings live only as text on this page.
<!-- related:end -->

## Cross-references

Navigation: ⬅️ [Previous](#in-pictures-code-and-further-reading) | 📋 [TOC](#table-of-contents)

- The cross-device spread named in [the finding "does a Langevin corrector remove part of the correction"](../../1-lora-correction/report/is-the-gap-the-samplers-or-the-models/does-a-langevin-corrector-remove-part-of-the-correction.md): one uncorrected cell reads 0.153, 0.169 and 0.180 on the RTX 3090, the RTX PRO 6000 Blackwell and the Quadro RTX 8000, so a comparison across corrector counts stays on one device.

<!-- built-to: PAGE_LOOK_FORMAT v1 -->
