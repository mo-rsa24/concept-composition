#!/usr/bin/env python3
"""Step M10's quarter check on a pod: when the full run saves a quarter's checkpoint, sample it on the CPU, count, and record the loss.

For each quarter (500,224, 1,000,448, 1,500,672 and 2,000,896 images seen, named 500, 1000, 1500 and 2000) it samples the 0.100 EMA snapshot with 16 seeds (0 to 15) per count label
in 32 steps on the CPU, scores the 48 samples with blob_count.py, takes the mean training loss over the quarter from stats.jsonl, and writes
quarters.jsonl and a grid png in <run>/quarters/. It decides nothing; the stopping rule is in M10's design.
Usage: python quarter_watch.py <run dir> <r|b>
"""
import glob, json, os, sys, time
import numpy as np
import torch
from PIL import Image

run, colour = sys.argv[1], sys.argv[2]
W = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, W); sys.path.insert(0, os.path.join(W, '..', 'edm2'))
from blob_count import count
import generate_images as gi
torch.set_num_threads(32)
Q = [500, 1000, 1500, 2000]
out = os.path.join(run, 'quarters'); os.makedirs(out, exist_ok=True)
done = {json.loads(l)['kimg'] for l in open(os.path.join(out, 'quarters.jsonl'))} if os.path.exists(os.path.join(out, 'quarters.jsonl')) else set()
prev = 0
for q in Q:
    if q in done:
        prev = q; continue
    pat = os.path.join(run, f'network-snapshot-{q:07d}-0.100.pkl')
    saved = f'network-snapshot-{q:07d}-0.100.pkl ... done'
    while not (glob.glob(pat) and os.path.exists(os.path.join(run, 'train.log')) and saved in open(os.path.join(run, 'train.log'), errors='ignore').read()):
        time.sleep(30)
    net = glob.glob(pat)[0]; t0 = time.time()
    rows, per = [], {}
    for k in (1, 2, 3):
        d = os.path.join(out, f'{q}', f'count-{k}')
        gi.generate_images(net=net, outdir=d, seeds=range(16), class_idx=k - 1, num_steps=32, max_batch_size=16, device=torch.device('cpu'),
                           verbose=False)
        ims = [Image.open(p).convert('RGB') for p in sorted(glob.glob(os.path.join(d, '*.png')))]
        per[k] = float(np.mean([count(im, colour) == k for im in ims])); rows.append(ims)
    losses = [s['Loss/loss'] for s in map(json.loads, open(os.path.join(run, 'stats.jsonl'))) if 'Loss/loss' in s and prev < s['Progress/kimg'] <= q + 1]
    rec = {'kimg': q, 'net': os.path.basename(net), 'right_count_share': round(float(np.mean(list(per.values()))), 3),
           'per_count': {k: round(v, 3) for k, v in per.items()}, 'mean_loss_this_quarter': round(float(np.mean(losses)), 4) if losses else None,
           'sample_seconds': round(time.time() - t0, 1)}
    grid = Image.new('RGB', (16 * 64, 3 * 64), 'white')
    for i, ims in enumerate(rows):
        for j, im in enumerate(ims): grid.paste(im.resize((64, 64)), (j * 64, i * 64))
    grid.save(os.path.join(out, f'grid-{q:04d}.png'))
    with open(os.path.join(out, 'quarters.jsonl'), 'a') as f: f.write(json.dumps(rec) + '\n')
    print(json.dumps(rec), flush=True); prev = q
