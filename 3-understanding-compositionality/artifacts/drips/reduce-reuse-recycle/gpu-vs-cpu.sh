#!/bin/bash
# Train the bar expert (notebook cell 33 at 81177d9) on the GPU kernel, then on the CPU kernel,
# one after the other so neither shares the machine with the other.
W=/home/molef/PhD/poe_repair_min/3-understanding-compositionality/artifacts/drips/reduce-reuse-recycle
V=/home/molef/PhD/reduce-reuse-recycle/.venv-2d/bin/python
cd $W
COMPAT='import chex
chex.assert_type = lambda *args, **kwargs: None
jax.tree_leaves = jax.tree_util.tree_leaves
jax.tree_map = jax.tree_util.tree_map
print("devices:", jax.devices())'
echo "== GPU kernel start $(date +%T)"
nohup $V -m ipykernel_launcher -f $W/kernel-gpu.json > $W/kernel-gpu.log 2>&1 &
sleep 5
export KERNEL=gpu
$V kernel.py start
$V kernel.py run 1 2 3 | grep wall
$V kernel.py code "$COMPAT"
$V kernel.py run 4 5 6 7 8 9 10 12 14 15 16 17 18 19 20 22 23 24 25 26 27 29 30 | grep -E 'wall|error'
echo "== GPU bar training $(date +%T)"
$V kernel.py run 33 | grep -vE 'CUDA-enabled|^[0-9]+ ' | grep -E 'wall|TEST|error|devices'
echo "== done $(date +%T)"
