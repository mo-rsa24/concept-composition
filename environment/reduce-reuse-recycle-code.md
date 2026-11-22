# 🗺️ Environment: the Reduce, Reuse, Recycle reference code

Navigation: 📋 [Index](00-INDEX.md) | [Overview](overview.md)

What is true about running the authors' code for Du et al. 2023 (arXiv 2302.11552) on the laptop.
The clone is `/home/molef/PhD/replications/reduce-reuse-recycle`, local branch `walk` at upstream commit
`81177d9`. It has two halves with separate environments: a JAX notebook for the 2D results, and a
PyTorch script for the CLEVR results. Where the code differs from the paper is not an environment
fact; that audit is
[the code-against-paper audit](/home/molef/goal-setting/learning/deep-learning/diffusion-models/02-sampler-correctors-for-composition/drips/read-reduce-reuse-recycle/routes/code-vs-paper.md).

## Table of contents

- [The 2D notebook](#the-2d-notebook)
- [The CLEVR script](#the-clevr-script)
- [On the cluster (mscluster88)](#on-the-cluster-mscluster88)
- [🖼️ In pictures, code and further reading](#in-pictures-code-and-further-reading)
- [Cross-references](#cross-references)

## The 2D notebook

Navigation: 📋 [TOC](#table-of-contents) | [Next](#the-clevr-script) ➡️

**`notebooks/simple_distributions.ipynb` runs on jax 0.11.2 only after one rename.** The trainer
cell (the one starting `# Simple diffusion model training`) calls
`jnp.clip(self._posterior_variance, a_min=...)`. jax 0.11 accepts only `min=` and `max=`, so
building any model fails with `TypeError: clip() got an unexpected keyword argument 'a_min'`, raised
at the first `hk.multi_transform(...)` call, not in the trainer cell itself. Change `a_min=` to
`min=` and rerun the trainer cell. It is the only `a_min` or `a_max` in the notebook.
(verified 2026-10-13)

**Its environment is `/home/molef/PhD/replications/reduce-reuse-recycle/.venv-2d`**, Python 3.12, with jax
0.11.2 on CPU, dm-haiku, optax, distrax and chex from PyPI unpinned. jax prints "a CUDA-enabled
jaxlib is not installed. Falling back to cpu" on first use; that is expected, and the notebook's
models are small enough for CPU. (verified 2026-10-13)

**The notebook's `tensorflow`, `tensorflow_datasets` and `pandas` imports are dead.** No cell uses
`tf.`, `tfds.` or `pd.`. Deleting those import lines saves installing TensorFlow. The `!pip
install` cell is also unneeded with the venv. (verified 2026-10-13)

## The CLEVR script

Navigation: ⬅️ [Previous](#the-2d-notebook) | 📋 [TOC](#table-of-contents) | [Next](#on-the-cluster-mscluster88) ➡️

**`inf_sample.py` needs a GPU, with no fallback.** It calls `model1.to("cuda")` unconditionally, so
it cannot run on a CPU-only machine. (verified 2026-10-13, by reading)

**Its interpreter is borrowed, read-only.** The laptop's only working CUDA torch is in
`/home/molef/playground/deep-learning/my-eye-my-brain/.venv` (torch `2.14.1+cu130`,
`torch.cuda.is_available()` True on the RTX 4070 Laptop, 8,188 MiB). It lacks matplotlib, which
sits in `<clone>/.clevr-extras` and is reached with `PYTHONPATH=<clone>/.clevr-extras`. That folder
also carries numpy 2.5.4, which shadows the venv's own. If the playground venv is deleted, CLEVR
has no interpreter until a CUDA torch is downloaded again (about 2.5 GB). The authors pin torch
2.0.0 with CUDA 11.7; Docker with the matching image would need the NVIDIA Container Toolkit,
which is not installed (`docker info` lists only the `runc` runtime). (verified 2026-10-13)

**The released energy U-Net cannot run a forward pass.** `composable_diffusion/unet.py` never
assigns `h` before `h = module(h, emb)` at line 1386, so any call raises `UnboundLocalError`. The
upstream guided-diffusion line `h = x.type(self.dtype)` is missing. (inferred 2026-10-13, from
reading the code; audit row R59 says the same; not yet run)

**The checkpoint `ebm-49x1874.pt` is 500,704,677 bytes** (sha256 starting `9e9aa1326eecdcff`),
542 tensors and 125.1 M parameters, from the Google Drive link at README line 59. Google serves a
virus-scan page for a file this size, so a plain `curl` fetches HTML, not the weights; the download
needs the page's `confirm` and `uuid` values. (verified 2026-10-13)

## On the cluster (mscluster88)

Navigation: ⬅️ [Previous](#the-clevr-script) | 📋 [TOC](#table-of-contents) | [Next](#in-pictures-code-and-further-reading) ➡️

**The cluster clone is `/home-mscluster/mmolefe/PhD/reduce-reuse-recycle`**, branch `walk` at
`81177d9`, with the same notebook fixes as the laptop. The walk runs on `mscluster88` (`bigbatch`,
RTX 3090, 24 GB) because `mscluster112`'s card is faulted; see the
[node table](hpc/nodes.md). (verified 2026-10-13)

**The 2D notebook's environment is a uv venv on `/datasets`**, at
`/datasets/mmolefe/reduce-reuse-recycle/venvs/rrr-jax`, linked into the clone as `.venv`. Python
3.12.10, jax 0.11.2 with the `cuda12` plugin, dm-haiku 0.0.17, optax 0.2.8, distrax 0.1.9, chex
0.1.92. Its uv cache is `/datasets/mmolefe/reduce-reuse-recycle/uv-cache`, so the 2.3 GB of CUDA
wheels never land under `/home-mscluster`. `jax.devices()` prints `[CudaDevice(id=0)]` and a
1000×1000 ones matmul sums to 1e9. The install took 5 min 15 s from PyPI. (verified 2026-10-13)

**No existing mamba env holds a usable jax.** `jax` has none installed; `jax115` has jax 0.4.30
with a CPU-only jaxlib and no haiku. (verified 2026-10-13)

<!-- related:start (drain) -->
<a id="in-pictures-code-and-further-reading"></a>

## 🖼️ In pictures, code and further reading

Navigation: ⬅️ [Previous](#on-the-cluster-mscluster88) | 📋 [TOC](#table-of-contents) | [Next](#cross-references) ➡️

A comparison table in `artifacts/`, [the toy beside Du et al.](../4-inclusive-captions/artifacts/scenes/how-the-toy-relabelling-experiment-works/network-views/15-du-et-al-side-by-side.png), quotes this code's files and line numbers. The one excerpt below is the line the page says breaks the released U-Net, read from the clone the page names (outside this project, read only).

### The code behind it

Excerpt from `composable_diffusion/unet.py` in the clone `/home/molef/PhD/replications/reduce-reuse-recycle` (lines 1385 to 1387):

```python
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
```

- Line 1386 reads `h` before the function has assigned it, which is the `UnboundLocalError` this page lists under the CLEVR script.
- The upstream line `h = x.type(self.dtype)` does not appear anywhere in this file: a search for `x.type(self.dtype)` returned nothing.
- This is reading only. The page marks the failure `inferred` and nothing was run.

### Go deeper

| What | Link |
|---|---|
| The recipe for setting this code up | [setting up the reference code](../runbook/setting-up-the-reduce-reuse-recycle-code.md) |
| Cluster nodes the walk runs on | [HPC nodes](hpc/nodes.md) |
| The route that uses this code | [route for classifier-guided PoE](../2-classifier-guidance/plans/route-classifier-guided-poe-from-toy-to-animals.md) |

### Still open

- No recorded run of `inf_sample.py` exists in `artifacts/`. For the notebook, that comparison table and its card record that a laptop walk ran the definitions, the samplers and the bar expert's training, and did not run the product and mixture sampling cells.

<!-- related:end -->

## Cross-references

Navigation: ⬅️ [Previous](#in-pictures-code-and-further-reading) | 📋 [TOC](#table-of-contents)

- The mention of **2.3 GB CUDA install**, timed from the PyPI download rate measured on `mscluster88`, in [the HPC nodes fact, under Python environments](hpc/nodes.md#python-environments).
- The mention of **the Reduce, Reuse, Recycle copy** of the guided-diffusion U-Net, missing the upstream line `h = x.type(self.dtype)`, in [the route for classifier-guided PoE](../2-classifier-guidance/plans/route-classifier-guided-poe-from-toy-to-animals.md#what-we-know-so-far).

<!-- built-to: PAGE_LOOK_FORMAT v1 -->
