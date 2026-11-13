#!/usr/bin/env python
"""Learn private token rows for PoE(dog, cat, x), with every model weight frozen.

Unlike separation_steering.py (which pushed the latent with a roughly 2% update),
this optimises x through its effect on subsequent latents and hence subsequent
expert attention. At a fixed latent the dog/cat maps have NO dependence on x.
We use truncated backpropagation through --unroll DDIM updates, following a
no-grad prefix from the cached noise, and measure attention at the next step.
This is a biased, memory-bounded surrogate gradient, not full-trajectory BPTT.
The sampling schedule remains 50 DDIM steps, 7.5 guidance, 1024 square.

K private input embedding rows per encoder replace evenly spaced content slots
in 'a dog next to a cat'. Initially this is exactly the real phrase. No tokenizer
or original vocabulary row is changed. Frozen encoders are differentiable with
respect to these rows; their output is NOT a free 77x2048 parameter. An anchor
penalty in input-row and encoded-prompt space discourages off-manifold drift;
passing through a frozen encoder alone does not guarantee a natural prompt.

The detector measures plurality, NOT species identity: two dogs count as success.
Seed 8 is rendered but excluded from the predeclared 6/7 success criterion.
Training seeds default to 1..7; monitoring uses seed 1, so final seeds are not
an untouched validation set. No checkpoint is selected by evaluation counts.

python scripts/and_vector_invert_x.py
python scripts/and_vector_invert_x.py --evaluate --x /datasets/mmolefe/poe_repair_min/outputs/invert_x/x.pt --seeds 1 2 3 4 5 6 7 8
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

REQUIRED_SUCCESSES = 6
JUDGEABLE_SEEDS = tuple(range(1, 8))
NOTICE = 'Animal counts do not identify species; two dogs count as success.'
PHRASE = 'a dog next to a cat'
DEFAULT_OUT = Path('/datasets/mmolefe/poe_repair_min/outputs/invert_x')
REPO = Path(os.environ.get('POE_REPO', Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'scripts'))


def arguments():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--evaluate', action='store_true')
    p.add_argument('--x', type=Path)
    p.add_argument('--out', type=Path, default=DEFAULT_OUT)
    p.add_argument('--seeds', nargs='+', type=int, default=list(range(1, 9)))
    p.add_argument('--train-seeds', nargs='+', type=int, default=list(range(1, 8)))
    p.add_argument('--tokens', type=int, default=4)
    p.add_argument('--iters', type=int, default=100)
    p.add_argument('--lr', type=float, default=0.001)
    p.add_argument('--anchor-weight', type=float, default=0.1)
    p.add_argument('--unroll', type=int, default=1)
    p.add_argument('--check-every', type=int, default=10)
    p.add_argument('--monitor-seed', type=int, default=1)
    a = p.parse_args()
    if a.evaluate and a.x is None:
        p.error('--evaluate requires --x')
    if not (1 <= a.tokens <= 6 and 1 <= a.unroll <= 19):
        p.error('--tokens must be 1..6; --unroll must be 1..19')
    if min(a.iters, a.check_every) < 1 or a.lr <= 0 or a.anchor_weight <= 0:
        p.error('iterations, check interval, learning rate and anchor weight must be positive')
    if not a.out.resolve().is_relative_to(Path('/datasets')):
        p.error('large outputs must be under /datasets')
    return a


class AttentionReadout:
    """Observe all attn2 layers without changing their original processor output.

    Recompute only the readout probabilities, with the original processor still
    producing the denoiser output. Resize each head-averaged map to 32 square,
    average equally across layers, then normalise spatial mass. Standard SDXL
    attn2 has no spatial/group/qk norm or residual connection; fail if it differs.
    """
    def __init__(self, unet):
        self.active = False
        self.maps = []
        processors = {}
        for name, original in unet.attn_processors.items():
            if name.endswith('attn2.processor'):
                processors[name] = self.wrap(original)
            else:
                processors[name] = original
        unet.set_attn_processor(processors)

    def wrap(self, original):
        owner = self

        class Observer:
            def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                         attention_mask=None, temb=None, **kwargs):
                import torch
                import torch.nn.functional as F
                if owner.active:
                    assert hidden_states.ndim == 3 and attention_mask is None
                    assert attn.spatial_norm is None and attn.group_norm is None
                    assert getattr(attn, 'norm_q', None) is None
                    assert getattr(attn, 'norm_k', None) is None
                    context = encoder_hidden_states
                    if attn.norm_cross:
                        context = attn.norm_encoder_hidden_states(context)
                    b, n, _ = hidden_states.shape
                    side = int(n ** 0.5)
                    assert side * side == n
                    q = attn.to_q(hidden_states).view(b, n, attn.heads, -1).transpose(1, 2)
                    k = attn.to_k(context).view(b, -1, attn.heads, q.shape[-1]).transpose(1, 2)
                    probs = (q.float() @ k.float().transpose(-1, -2) * attn.scale).softmax(-1)
                    # dog and cat are each exactly one content token (checked at startup).
                    m = probs[..., 1].mean(1).reshape(b, 1, side, side)
                    owner.maps.append(F.interpolate(m, (32, 32), mode='bilinear', align_corners=False).flatten(1))
                return original(attn, hidden_states, encoder_hidden_states, attention_mask, temb, **kwargs)
        return Observer()


def main():
    args = arguments()
    import torch
    from torch.utils.checkpoint import checkpoint
    from concept_pair_sheet import NOISE_PAIR
    from poe_repair.run import make_ctx
    from poe_repair.runtime import encode_prompt_sdxl, decode_latents, tweedie_mean, ddim_prev_from_x0_eps
    from poe_repair.composers._helpers import init_latents_for_cell
    from poe_repair._sdxl.sdipc_utils import canonicalize_grid_asset_payload
    from poe_repair.experiments.interaction_term.cell import cell_from_slug
    from poe_repair.methods._sampling import add_time_ids, write_decoded_image
    from poe_repair.experiments.compose_scorer_validation.detection_scorer import count_instances

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required')
    args.out.mkdir(parents=True, exist_ok=True)
    print(NOTICE, flush=True)
    # bfloat16 avoids fp16 underflow in the multi-step gradient; no weights train.
    ctx = make_ctx(num_inference_steps=50, guidance_scale=7.5, dtype=torch.bfloat16)
    unet, sched = ctx.models['unet'], ctx.scheduler
    for model in ctx.models.values():
        if isinstance(model, torch.nn.Module):
            model.requires_grad_(False).eval()
    def encode(prompt):
        return encode_prompt_sdxl(prompt, models=ctx.models, device=ctx.device, dtype=ctx.dtype)
    with torch.no_grad():
        fixed = {p: encode(p) for p in ('dog', 'cat', '', PHRASE)}
    saved = torch.load(args.x, map_location='cpu', weights_only=True) if args.x else None
    if saved:
        if saved['required_successes'] != REQUIRED_SUCCESSES or saved['phrase'] != PHRASE:
            raise ValueError('incompatible checkpoint protocol')
        args.tokens = saved['tokens']
    rows, anchors, positions, ids_list = [], [], [], []
    encoders = [ctx.models[k] for k in ('text_encoder', 'text_encoder_2')]
    for index, (tokname, encoder) in enumerate(zip(('tokenizer', 'tokenizer_2'), encoders)):
        tok = ctx.models[tokname]
        assert all(len(tok(p).input_ids) == 3 for p in ('dog', 'cat'))
        content = tok(PHRASE, add_special_tokens=False).input_ids
        if args.tokens > len(content):
            raise ValueError('more private rows than phrase content slots')
        pos = torch.linspace(1, len(content), args.tokens).round().long().to(ctx.device)
        ids = tok([PHRASE], padding='max_length', max_length=tok.model_max_length,
                  truncation=True, return_tensors='pt').input_ids.to(ctx.device)
        anchor = encoder.get_input_embeddings()(ids)[0, pos].detach().float()
        value = saved['rows'][index].to(ctx.device) if saved else anchor.clone()
        if value.shape != anchor.shape:
            raise ValueError('checkpoint row shape mismatch')
        rows.append(torch.nn.Parameter(value.float()))
        anchors.append(anchor)
        positions.append(pos)
        ids_list.append(ids)

    def encode_x():
        outputs = []
        for encoder, row, pos, ids in zip(encoders, rows, positions, ids_list):
            # Functional private rows: original embedding table never changes.
            def replace(module, inputs, output, row=row, pos=pos):
                return output.index_copy(1, pos, row.to(output.dtype).unsqueeze(0))
            hook = encoder.get_input_embeddings().register_forward_hook(replace)
            try:
                outputs.append(encoder(ids, output_hidden_states=True))
            finally:
                hook.remove()
        return (torch.cat([o.hidden_states[-2] for o in outputs], dim=-1), outputs[1].text_embeds)

    if not saved:
        with torch.no_grad():
            assert all(torch.equal(a, b) for a, b in zip(encode_x(), fixed[PHRASE])), 'initialisation must reproduce phrase'
    tap = AttentionReadout(unet)

    def predict(lat, t, embedding, capture=False):
        seq, pool = embedding
        tap.active, tap.maps = capture, []
        try:
            eps = unet(sched.scale_model_input(lat, t), t, encoder_hidden_states=seq,
                       added_cond_kwargs={'text_embeds': pool, 'time_ids': add_time_ids(
                           height=1024, width=1024, batch_size=1, device=ctx.device, dtype=ctx.dtype)},
                       timestep_cond=None).sample
            m = torch.stack(tap.maps).mean(0) if capture else lat.new_zeros((1, 1024))
            m = m.float() / m.float().sum(-1, keepdim=True).clamp_min(1e-12) if capture else m
            return eps, m
        finally:
            tap.active, tap.maps = False, []

    def forward(lat, t, emb, capture=False):
        if torch.is_grad_enabled():
            return checkpoint(lambda z, s, p: predict(z, t, (s, p), capture),
                              lat, *emb, use_reentrant=False)
        return predict(lat, t, emb, capture)

    def step(lat, i, x, capture=False):
        t = sched.timesteps[i]
        ea, ma = forward(lat, t, fixed['dog'], capture)
        eb, mb = forward(lat, t, fixed['cat'], capture)
        e0, _ = forward(lat, t, fixed[''])
        ex, _ = forward(lat, t, x)
        eps = (e0.float() + 7.5 * ((ea.float() - e0.float()) +
               (eb.float() - e0.float()) + (ex.float() - e0.float()))).to(ctx.dtype)
        abar = sched.alphas_cumprod[int(t)].to(device=ctx.device, dtype=ctx.dtype)
        nxt = ddim_prev_from_x0_eps(scheduler=sched, timestep=t, step_index=i,
                                   x0=tweedie_mean(lat, abar, eps), eps=eps)
        return nxt, torch.minimum(ma, mb).sum() if capture else None

    def initial(seed):
        cell = cell_from_slug(NOISE_PAIR, seed)
        assert (cell.height, cell.width) == (1024, 1024)
        # The helper allows random fallback; this experiment requires cached noise.
        canonicalize_grid_asset_payload(pair_dir=cell.pair_dir, require_base_flats=True)
        init, sigma = init_latents_for_cell(cell, ctx)
        sched.set_timesteps(50)
        return (init / sigma).to(device=ctx.device, dtype=ctx.dtype)

    def render(seed, directory):
        directory.mkdir(parents=True, exist_ok=True)
        with torch.no_grad():
            x, lat, overlaps = encode_x(), initial(seed), []
            for i in range(50):
                lat, overlap = step(lat, i, x, capture=True)
                overlaps.append(float(overlap))
            path = directory / f'seed{seed:02d}.png'
            write_decoded_image(decode_latents(ctx.models, lat).cpu(), path)
        # CPU detector avoids keeping another large model on the training GPU.
        count, boxes = count_instances(path, device=torch.device('cpu'))
        rec = dict(seed=seed, count=count, boxes=boxes, success=count >= 2,
                   judgeable=seed in JUDGEABLE_SEEDS, attention_overlap=sum(overlaps)/50,
                   overlap_per_step=overlaps, steps=50, guidance=7.5, noise_from=NOISE_PAIR,
                   notice=NOTICE)
        path.with_suffix('.json').write_text(json.dumps(rec, indent=2))
        print(f'seed {seed}: {count} animals; {NOTICE}', flush=True)
        return rec

    def save(iteration):
        dest = args.out / 'x.pt'
        torch.save(dict(rows=[r.detach().cpu() for r in rows], tokens=args.tokens,
                        phrase=PHRASE, required_successes=REQUIRED_SUCCESSES,
                        iteration=iteration, args={k: str(v) if isinstance(v, Path) else v
                                                  for k, v in vars(args).items()}), dest.with_suffix('.tmp'))
        dest.with_suffix('.tmp').replace(dest)

    if not args.evaluate:
        opt = torch.optim.Adam(rows, lr=args.lr)
        monitors = [dict(iteration=0, **render(args.monitor_seed, args.out / 'monitor/step0000'))]
        save(0)
        with (args.out / 'training.jsonl').open('w') as log:
            for iteration in range(1, args.iters + 1):
                seed = args.train_seeds[(iteration - 1) % len(args.train_seeds)]
                target_step = args.unroll + (iteration - 1) % (20 - args.unroll)
                with torch.no_grad():
                    lat, old_x = initial(seed), encode_x()
                    for i in range(target_step - args.unroll):
                        lat, _ = step(lat, i, old_x)
                opt.zero_grad(set_to_none=True)
                x = encode_x()
                for i in range(target_step - args.unroll, target_step):
                    lat, _ = step(lat, i, x)
                # Read the NEXT experts: only here does x have a gradient path.
                _, ma = forward(lat, sched.timesteps[target_step], fixed['dog'], True)
                _, mb = forward(lat, sched.timesteps[target_step], fixed['cat'], True)
                overlap = torch.minimum(ma, mb).sum()
                anchor = sum((r-a).square().mean() / a.square().mean().clamp_min(1e-12)
                             for r, a in zip(rows, anchors))
                anchor = anchor + sum((v.float()-a.float()).square().mean() /
                                     a.float().square().mean().clamp_min(1e-12)
                                     for v, a in zip(x, fixed[PHRASE]))
                distance = sum((v.float()-e.float()).square().sum() for v, e in zip(x, fixed[''])).sqrt()
                loss = overlap + args.anchor_weight * anchor
                loss.backward()
                gradnorm = torch.nn.utils.clip_grad_norm_(rows, 1.0, error_if_nonfinite=True)
                if not torch.isfinite(loss) or float(gradnorm) == 0:
                    raise RuntimeError('nonfinite loss or zero gradient to private rows')
                opt.step()
                rec = dict(iteration=iteration, seed=seed, target_step=target_step,
                           overlap=float(overlap.detach()), empty_distance=float(distance.detach()),
                           anchor=float(anchor.detach()), grad_norm=float(gradnorm))
                print(json.dumps(rec), flush=True)
                log.write(json.dumps(rec)+'\n')
                log.flush()
                # Drop the training graph before full rendering.
                del loss, overlap, anchor, distance, x, lat, ma, mb
                save(iteration)
                if iteration % args.check_every == 0 or iteration == args.iters:
                    monitors.append(dict(iteration=iteration, **render(args.monitor_seed,
                                    args.out / f'monitor/step{iteration:04d}')))
                    (args.out / 'monitor.json').write_text(json.dumps(monitors, indent=2))
        first, last = monitors[0], monitors[-1]
        trend = ('overlap fell but animal count did not rise: surrogate diverged' if
                 last['attention_overlap'] < first['attention_overlap'] and last['count'] <= first['count'] else
                 'overlap fell and count rose: consistent on the monitored seed' if
                 last['attention_overlap'] < first['attention_overlap'] and last['count'] > first['count'] else
                 'no joint improvement established on the monitored seed')
        print(f'{trend}. {NOTICE}', flush=True)
        (args.out / 'surrogate-result.json').write_text(json.dumps(dict(verdict=trend, notice=NOTICE), indent=2))

    results = [render(seed, args.out / 'renders') for seed in dict.fromkeys(args.seeds)]
    successes = sum(r['success'] for r in results if r['judgeable'])
    complete = set(JUDGEABLE_SEEDS).issubset(args.seeds)
    verdict = ('PASS' if successes >= REQUIRED_SUCCESSES else 'FAIL') if complete else 'INCOMPLETE'
    summary = dict(results=results, successes=successes, required=REQUIRED_SUCCESSES,
                   denominator=7, verdict=verdict, notice=NOTICE)
    (args.out / 'counts.json').write_text(json.dumps(summary, indent=2))
    print(f'{verdict}: {successes}/7 judgeable seeds; required {REQUIRED_SUCCESSES}/7. {NOTICE}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
