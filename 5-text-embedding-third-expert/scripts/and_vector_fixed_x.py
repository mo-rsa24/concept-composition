#!/usr/bin/env python
"""Learn ONE cat-dog text condition across all timesteps; log W&B comparison grids.

Only K private input-token rows in each frozen SDXL text encoder train. Original
vocabulary rows, cat/dog conditions, UNet, scheduler, 7.5 CFG and 50 DDIM steps stay
fixed. The target at a shared latent is eps_joint - eps_cat - eps_dog + 2 eps_empty.
This tests joint-prompt recovery, unlike and_vector_invert_x.py's attention loss.

Teacher trajectories and periodically refreshed PoE(cat,dog,x) trajectories supply
DETACHED training states. The loss backpropagates through eps_x and both frozen
encoders, not through sampling. x is shared across every seed and timestep.
Default: 5000 Adam updates, grids at 0, every 500, and the final step. Seeds 1..4
train; grids show 1..8, labelled by split. Held-out seeds never enter the optimizer.
The exact joint prompt is 'a cat and dog' (not earlier experiments' 'dog and cat').
Animal counts, when enabled, measure plurality only: two dogs count as success.

All large outputs live on /datasets. Online W&B is required by default; errors
propagate instead of silently turning the requested live experiment into offline.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import sys
import uuid

REPO = Path(os.environ.get('POE_REPO', Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'scripts'))

PHRASE = 'a dog next to a cat'
JOINT = 'a cat and dog'
GUIDANCE = 7.5
SAMPLING_STEPS = 50
COLUMNS = [('cat', ['cat', 'SDXL']), ('dog', ['dog', 'SDXL']),
           ('joint', [JOINT, 'SDXL joint target']), ('x', ['x only', 'SDXL learned condition']),
           ('poe_x', ['PoE(cat, dog, x)', 'learned third expert']),
           ('poe', ['PoE(cat, dog)', 'unchanged baseline'])]


def correction_target(cat, dog, joint, empty):
    return joint.float() - cat.float() - dog.float() + 2 * empty.float()


def guided_sum(empty, conditions):
    return empty.float() + GUIDANCE * sum(e.float() - empty.float() for e in conditions)


def checked_start(cell, ctx, first_timestep):
    """Use the project's fp16 starting noise, verified against its disk cache.

    The cluster has residual caches for seeds 1..8, but no pilot grid assets for
    those seeds. Validate the helper's deterministic fp16 fallback against the
    actual cached first latent before converting to the run's arithmetic dtype.
    """
    import torch
    from dataclasses import replace
    from poe_repair.composers._helpers import init_latents_for_cell
    from poe_repair._sdxl.sdipc_utils import canonicalize_grid_asset_payload
    from poe_repair.experiments.interaction_term.cache import cell_dir
    noise_ctx = replace(ctx, dtype=torch.float16)
    try:
        canonicalize_grid_asset_payload(pair_dir=cell.pair_dir, require_base_flats=True)
        source = cell.pair_dir / 'grid_assets.json'
        init, sigma = init_latents_for_cell(cell, noise_ctx)
        lat = init / sigma
    except FileNotFoundError:
        source = cell_dir(cell.pair_slug, cell.seed) / 'residuals' / 'step_000.pt'
        cached = torch.load(source, map_location='cpu', weights_only=True)
        if int(cached['timestep']) != int(first_timestep):
            raise ValueError(f'cached initial timestep mismatch: {source}')
        init, sigma = init_latents_for_cell(cell, noise_ctx)
        lat = cached['x_t']
        if not torch.equal((init / sigma).cpu().float(), lat.float()):
            raise ValueError(f'helper starting noise differs from cached first latent: {source}')
    return lat.to(device=ctx.device, dtype=ctx.dtype), str(source)


class PrivatePrompt:
    """Private rows replace phrase slots during x encoding only; no vocabulary edits."""
    def __init__(self, models, tokens=4):
        import torch
        self.encoders = [models[k] for k in ('text_encoder', 'text_encoder_2')]
        self.rows = torch.nn.ParameterList()
        self.anchors, self.ids, self.positions = [], [], []
        for tok_key, encoder in zip(('tokenizer', 'tokenizer_2'), self.encoders):
            tok = models[tok_key]
            ids = tok([PHRASE], padding='max_length', max_length=tok.model_max_length,
                      truncation=True, return_tensors='pt').input_ids.to(encoder.device)
            n = len(tok(PHRASE, add_special_tokens=False).input_ids)
            if not 1 <= tokens <= n:
                raise ValueError(f'tokens must be between 1 and {n}')
            pos = torch.linspace(1, n, tokens).round().long().to(encoder.device)
            anchor = encoder.get_input_embeddings()(ids)[0, pos].detach().float()
            self.rows.append(torch.nn.Parameter(anchor.clone()))
            self.anchors.append(anchor)
            self.ids.append(ids)
            self.positions.append(pos)

    def encode(self):
        import torch
        outs = []
        for encoder, row, pos, ids in zip(self.encoders, self.rows, self.positions, self.ids):
            def replace(module, inputs, output, row=row, pos=pos):
                return output.index_copy(1, pos, row.to(output.dtype).unsqueeze(0))
            hook = encoder.get_input_embeddings().register_forward_hook(replace)
            try:
                outs.append(encoder(ids, output_hidden_states=True))
            finally:
                hook.remove()
        return torch.cat([o.hidden_states[-2] for o in outs], -1), outs[1].text_embeds

    def anchor_loss(self):
        return sum((r-a).square().mean() / a.square().mean().clamp_min(1e-12)
                   for r, a in zip(self.rows, self.anchors))


def arguments():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--steps', type=int, default=5000)
    p.add_argument('--eval-every', type=int, default=500)
    p.add_argument('--train-seeds', nargs='+', type=int, default=[1, 2, 3, 4])
    p.add_argument('--seeds', nargs='+', type=int, default=list(range(1, 9)))
    p.add_argument('--tokens', type=int, default=4)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--anchor-weight', type=float, default=1e-3)
    p.add_argument('--on-policy-fraction', type=float, default=0.5)
    p.add_argument('--random-seed', type=int, default=2026)
    p.add_argument('--name', default='cat-dog-fixed-x')
    p.add_argument('--run-id', default=None)
    p.add_argument('--entity', default='prime_lab')
    p.add_argument('--project', default='poe-repair-animals-compose')
    p.add_argument('--out', type=Path, default=Path('/datasets/mmolefe/poe_repair_min/outputs/fixed_x'))
    p.add_argument('--resume', type=Path)
    p.add_argument('--count-animals', action='store_true', help='CPU detector at each grid; species-blind')
    p.add_argument('--register-only', action='store_true', help='Create the online W&B run before Slurm starts')
    a = p.parse_args()
    if min(a.steps, a.eval_every) < 1 or not 1 <= a.tokens <= 6:
        p.error('steps/eval-every must be positive; tokens must be 1..6')
    if a.lr <= 0 or a.anchor_weight < 0 or not 0 <= a.on_policy_fraction <= 1:
        p.error('invalid learning rate, anchor weight, or on-policy fraction')
    if not a.out.resolve().is_relative_to(Path('/datasets')):
        p.error('large outputs must live on /datasets')
    if not set(a.train_seeds).issubset(a.seeds):
        p.error('--seeds must include all training seeds for trajectory refresh')
    a.train_seeds, a.seeds = sorted(set(a.train_seeds)), sorted(set(a.seeds))
    if a.resume and not a.run_id:
        p.error('--resume requires the original --run-id')
    a.run_id = a.run_id or uuid.uuid4().hex[:8]
    return a


def run_config(args):
    return {**{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
               if k not in ('resume', 'register_only')},
            'objective': 'joint_prediction_correction', 'joint_prompt': JOINT,
            'initial_phrase': PHRASE, 'guidance': GUIDANCE, 'sampling_steps': SAMPLING_STEPS,
            'resolution': [1024, 1024], 'trainable': 'private text input rows only',
            'dtype': 'bfloat16', 'grid_columns': [k for k, _ in COLUMNS],
            'held_out_seeds': sorted(set(args.seeds)-set(args.train_seeds)),
            'noise_from': 'a_cat__x__a_dog'}


def main():
    args = arguments()
    out = args.out / args.run_id
    out.mkdir(parents=True, exist_ok=True)
    for key in ('WANDB_DIR', 'WANDB_CACHE_DIR', 'WANDB_DATA_DIR', 'WANDB_ARTIFACT_DIR'):
        os.environ[key] = str(out / key.lower())
        Path(os.environ[key]).mkdir(parents=True, exist_ok=True)
    import wandb
    config = run_config(args)
    config_path = out / 'config.json'
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError('run ID already has a different configuration')
    config_path.write_text(json.dumps(config, indent=2))
    with wandb.init(entity=args.entity, project=args.project, id=args.run_id, resume='allow',
                    name=args.name, group='fixed-x-cat-dog', config=config, mode='online', dir=str(out)) as run:
        (out / 'wandb.json').write_text(json.dumps({'url': run.url, 'id': run.id}, indent=2))
        print(f'W&B: {run.url}', flush=True)
        run.define_metric('optimizer_step')
        run.define_metric('train/*', step_metric='optimizer_step')
        run.define_metric('eval/*', step_metric='optimizer_step')
        if args.register_only:
            run.summary['status'] = 'registered; awaiting Slurm execution'
            return 0
        train(args, out, run, wandb)
    return 0


def train(args, out, run, wandb):
    import torch
    import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "scripts", "shared"))  # shared helpers
    from concept_pair_sheet import NOISE_PAIR
    from poe_repair.run import make_ctx
    from poe_repair.runtime import encode_prompt_sdxl, decode_latents, tweedie_mean, ddim_prev_from_x0_eps
    from poe_repair.experiments.interaction_term.cell import cell_from_slug
    from poe_repair.methods._sampling import add_time_ids, write_decoded_image
    from main_study_grids import grid

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required')
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError('This run requires a GPU supporting bfloat16 gradients')
    torch.manual_seed(args.random_seed)
    rng = random.Random(args.random_seed)
    run.summary['status'] = 'loading SDXL'
    ctx = make_ctx(num_inference_steps=SAMPLING_STEPS, guidance_scale=GUIDANCE, dtype=torch.bfloat16)
    unet, sched = ctx.models['unet'], ctx.scheduler
    for model in ctx.models.values():
        if isinstance(model, torch.nn.Module):
            model.requires_grad_(False).eval()
    # Diffusers 0.31 checkpoints transformer blocks only in training mode. SDXL
    # uses no batch norm; explicitly disable dropout while enabling checkpoints.
    unet.enable_gradient_checkpointing()
    unet.train()
    for module in unet.modules():
        if isinstance(module, torch.nn.Dropout):
            module.eval()
    if hasattr(ctx.models['vae'], 'enable_tiling'):
        ctx.models['vae'].enable_tiling()
    with torch.no_grad():
        fixed = {key: encode_prompt_sdxl(prompt, models=ctx.models, device=ctx.device, dtype=ctx.dtype)
                 for key, prompt in [('cat', 'cat'), ('dog', 'dog'), ('joint', JOINT),
                                     ('empty', ''), ('phrase', PHRASE)]}
    private = PrivatePrompt(ctx.models, args.tokens)
    with torch.no_grad():
        assert all(torch.equal(a, b) for a, b in zip(private.encode(), fixed['phrase']))
    optimizer = torch.optim.Adam(private.rows, lr=args.lr)
    start = 0
    if args.resume:
        saved = torch.load(args.resume, map_location=ctx.device, weights_only=True)
        if saved['config'] != run_config(args):
            raise ValueError('checkpoint configuration differs')
        private.rows.load_state_dict(saved['rows'])
        optimizer.load_state_dict(saved['optimizer'])
        rng.setstate(saved['rng_state'])
        start = saved['step']
    elif (out / 'latest.pt').exists():
        raise FileExistsError('checkpoint exists; use --resume to avoid overwriting a run')
    time_ids = add_time_ids(height=1024, width=1024, batch_size=1, device=ctx.device, dtype=ctx.dtype)
    sched.set_timesteps(SAMPLING_STEPS)
    starts, noise_sources = {}, {}
    for seed in args.seeds:
        cell = cell_from_slug(NOISE_PAIR, seed)
        assert (cell.height, cell.width) == (1024, 1024)
        starts[seed], noise_sources[seed] = checked_start(cell, ctx, sched.timesteps[0])
    (out / 'noise_sources.json').write_text(json.dumps(noise_sources, indent=2))
    print('Cached starting noise verified for all seeds', flush=True)

    def predict(lat, i, emb):
        seq, pool = emb
        t = sched.timesteps[i]
        return unet(sched.scale_model_input(lat, t), t, encoder_hidden_states=seq,
                    added_cond_kwargs={'text_embeds': pool, 'time_ids': time_ids},
                    timestep_cond=None).sample

    def advance(lat, i, eps):
        t = sched.timesteps[i]
        eps = eps.to(ctx.dtype)
        abar = sched.alphas_cumprod[int(t)].to(device=ctx.device, dtype=ctx.dtype)
        return ddim_prev_from_x0_eps(scheduler=sched, timestep=t, step_index=i,
                                    x0=tweedie_mean(lat, abar, eps), eps=eps)

    def record(lat, i, predictions, seed):
        target = correction_target(predictions['cat'], predictions['dog'],
                                   predictions['joint'], predictions['empty'])
        gap_mse = (target-predictions['empty'].float()).square().mean()
        return dict(latent=lat.detach().cpu(), target=target.detach().cpu(),
                    gap_mse=float(gap_mse), step=i, seed=seed)

    def write_image(lat, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        write_decoded_image(decode_latents(ctx.models, lat).cpu(), path)

    # Check the real 1024px backward pass before spending time on reference grids.
    # No optimizer update: the experiment still starts from the exact phrase.
    run.summary['status'] = 'checking GPU memory and gradient path'
    torch.cuda.reset_peak_memory_stats()
    probe_lat = starts[args.train_seeds[0]]
    with torch.no_grad():
        probe_eps = {k: predict(probe_lat, 25, fixed[k])
                     for k in ('cat', 'dog', 'joint', 'empty')}
        probe_target = correction_target(probe_eps['cat'], probe_eps['dog'],
                                         probe_eps['joint'], probe_eps['empty'])
    probe_loss = (predict(probe_lat, 25, private.encode()).float()-probe_target).square().mean()
    probe_loss.backward()
    if not all(r.grad is not None and torch.isfinite(r.grad).all() and r.grad.norm() > 0
               for r in private.rows):
        raise RuntimeError('GPU preflight failed: missing/nonfinite token gradients')
    optimizer.zero_grad(set_to_none=True)
    peak_gib = torch.cuda.max_memory_allocated() / 2**30
    run.summary['preflight_peak_gpu_gib'] = peak_gib
    print(f'GPU forward/backward preflight passed; peak allocated {peak_gib:.2f} GiB', flush=True)
    del probe_lat, probe_eps, probe_target, probe_loss
    torch.cuda.empty_cache()

    teacher = []
    teacher_path = out / 'teacher_states.pt'
    run.summary['status'] = 'building teacher trajectories and fixed reference columns'
    with torch.no_grad():
        if teacher_path.exists():
            teacher = torch.load(teacher_path, map_location='cpu', weights_only=True)
        else:
            for seed in args.seeds:
                lat = starts[seed].clone()
                for i in range(SAMPLING_STEPS):
                    predictions = {key: predict(lat, i, fixed[key]) for key in ('cat', 'dog', 'joint', 'empty')}
                    teacher.append(record(lat, i, predictions, seed))
                    lat = advance(lat, i, guided_sum(predictions['empty'], [predictions['joint']]))
                write_image(lat, out / 'references' / f'seed{seed:02d}' / 'joint.png')
                print(f'cached joint trajectory seed={seed}', flush=True)
            torch.save(teacher, teacher_path.with_suffix('.tmp'))
            teacher_path.with_suffix('.tmp').replace(teacher_path)

        for seed in args.seeds:
            for key, conditions in [('cat', ['cat']), ('dog', ['dog']), ('poe', ['cat', 'dog'])]:
                path = out / 'references' / f'seed{seed:02d}' / f'{key}.png'
                if path.exists():
                    continue
                lat = starts[seed].clone()
                for i in range(SAMPLING_STEPS):
                    empty = predict(lat, i, fixed['empty'])
                    lat = advance(lat, i, guided_sum(empty, [predict(lat, i, fixed[c]) for c in conditions]))
                write_image(lat, path)
                print(f'reference seed={seed} condition={key}', flush=True)
    train_teacher = [s for s in teacher if s['seed'] in args.train_seeds]
    on_policy = []

    def save(step):
        state = dict(step=step, rows={k: v.detach().cpu() for k, v in private.rows.state_dict().items()},
                     optimizer=optimizer.state_dict(), rng_state=rng.getstate(), config=run_config(args))
        path = out / 'latest.pt'
        torch.save(state, path.with_suffix('.tmp'))
        path.with_suffix('.tmp').replace(path)
        if step == start or step % args.eval_every == 0 or step == args.steps:
            dest = out / 'checkpoints' / f'x-step{step:06d}.pt'
            dest.parent.mkdir(exist_ok=True)
            torch.save(state, dest)

    @torch.no_grad()
    def evaluate(step):
        nonlocal on_policy
        directory = out / 'evaluations' / f'step{step:06d}'
        directory.mkdir(parents=True, exist_ok=True)
        x = private.encode()
        refreshed, metrics, sidecars = [], {}, []
        for split, seeds in [('train', set(args.train_seeds)), ('held_out', set(args.seeds)-set(args.train_seeds))]:
            probes = [s for s in teacher if s['seed'] in seeds and s['step'] in (0, 10, 25, 40)]
            if probes:
                errors, gaps = [], []
                for s in probes:
                    pred = predict(s['latent'].to(ctx.device), s['step'], x).float()
                    errors.append(float((pred-s['target'].to(ctx.device)).square().mean()))
                    gaps.append(s['gap_mse'])
                metrics[f'eval/{split}_relative_correction_error'] = (sum(errors)/max(sum(gaps), 1e-12))**0.5
        for seed in args.seeds:
            for key in ('x', 'poe_x'):
                lat = starts[seed].clone()
                for i in range(SAMPLING_STEPS):
                    empty, ex = predict(lat, i, fixed['empty']), predict(lat, i, x)
                    conds = [ex]
                    if key == 'poe_x':
                        cat, dog = predict(lat, i, fixed['cat']), predict(lat, i, fixed['dog'])
                        conds = [cat, dog, ex]
                        if seed in args.train_seeds:
                            joint = predict(lat, i, fixed['joint'])
                            refreshed.append(record(lat, i, dict(cat=cat, dog=dog, joint=joint, empty=empty), seed))
                    lat = advance(lat, i, guided_sum(empty, conds))
                path = directory / f'seed{seed:02d}' / f'{key}.png'
                write_image(lat, path)
                rec = dict(seed=seed, condition=key, optimizer_step=step,
                           split='train' if seed in args.train_seeds else 'held_out',
                           guidance=GUIDANCE, sampling_steps=SAMPLING_STEPS, noise_from=NOISE_PAIR)
                if key == 'poe_x':
                    from PIL import Image
                    import numpy as np
                    joint_path = out / 'references' / f'seed{seed:02d}' / 'joint.png'
                    with Image.open(path) as a, Image.open(joint_path) as b:
                        rec['joint_pixel_mae_0_255'] = float(np.abs(np.asarray(a).astype(float)-np.asarray(b).astype(float)).mean())
                    if args.count_animals:
                        from poe_repair.experiments.compose_scorer_validation.detection_scorer import count_instances
                        rec['animal_count'], rec['boxes'] = count_instances(path, device=torch.device('cpu'))
                        rec['count_notice'] = 'Species-blind: two dogs count as success.'
                    sidecars.append(rec)
                path.with_suffix('.json').write_text(json.dumps(rec, indent=2))
                print(f'grid step={step} seed={seed} condition={key}', flush=True)
        on_policy = refreshed  # contains training seeds only, never held-out states
        for split in ('train', 'held_out'):
            values = [r['joint_pixel_mae_0_255'] for r in sidecars if r['split'] == split]
            if values:
                metrics[f'eval/{split}_joint_pixel_mae'] = sum(values)/len(values)
        rows = [(f'seed {seed} ({"train" if seed in args.train_seeds else "held out"})',
                 lambda key, seed=seed: (directory if key in ('x', 'poe_x') else out / 'references') /
                 f'seed{seed:02d}' / f'{key}.png') for seed in args.seeds]
        sheet = directory / 'comparison.png'
        grid(rows, COLUMNS, sheet, f'One fixed x for cat-dog | {step} optimization steps',
             'Frozen SDXL and text encoders; only private token rows learn. Same cached noise across each row. 50 DDIM steps, CFG 7.5.')
        (directory / 'metrics.json').write_text(json.dumps({'metrics': metrics, 'seeds': sidecars}, indent=2))
        run.log({'optimizer_step': step, **metrics,
                 'eval/comparison_grid': wandb.Image(str(sheet), caption=f'{step} updates; rows=seeds; columns=conditions')})
        # Store the exact embedding checkpoint alongside each grid in W&B.
        artifact = wandb.Artifact(f'fixed-x-{args.run_id}', type='model', metadata={'optimizer_step': step})
        artifact.add_file(str(out / 'checkpoints' / f'x-step{step:06d}.pt'))
        run.log_artifact(artifact, aliases=['latest', f'step-{step}'])
        run.summary['last_evaluated_step'] = step

    save(start)
    evaluate(start)
    run.summary['status'] = 'training'
    with (out / 'training.jsonl').open('a') as log:
        for step in range(start+1, args.steps+1):
            use_policy = bool(on_policy) and rng.random() < args.on_policy_fraction
            state = rng.choice(on_policy if use_policy else train_teacher)
            lat, target = state['latent'].to(ctx.device), state['target'].to(ctx.device)
            optimizer.zero_grad(set_to_none=True)
            x = private.encode()
            pred = predict(lat, state['step'], x).float()
            mse = (pred-target).square().mean()
            anchor = private.anchor_loss()
            loss = mse + args.anchor_weight * anchor
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(private.rows, 1.0, error_if_nonfinite=True)
            if not torch.isfinite(loss) or float(norm) == 0:
                raise RuntimeError('nonfinite loss or no gradient to x')
            assert all(p.grad is None for k in ('unet', 'text_encoder', 'text_encoder_2')
                       for p in ctx.models[k].parameters())
            with torch.no_grad():
                distance = sum((a.float()-b.float()).square().sum() for a, b in zip(x, fixed['empty'])).sqrt()
            optimizer.step()
            rec = {'optimizer_step': step, 'train/mse': float(mse.detach()),
                   'train/loss': float(loss.detach()), 'train/anchor': float(anchor.detach()),
                   'train/relative_correction_error': (float(mse.detach())/max(state['gap_mse'], 1e-12))**0.5,
                   'train/empty_distance': float(distance), 'train/gradient_norm': float(norm),
                   'train/timestep_index': state['step'], 'train/seed': state['seed'],
                   'train/on_policy': int(use_policy)}
            run.log(rec)
            log.write(json.dumps(rec)+'\n')
            log.flush()
            if step % 10 == 0 or step == 1:
                print(json.dumps(rec), flush=True)
            del x, pred, mse, anchor, loss, lat, target
            if step % 100 == 0 or step % args.eval_every == 0 or step == args.steps:
                save(step)
            if step % args.eval_every == 0 or step == args.steps:
                evaluate(step)
    run.summary['status'] = 'complete'
    run.summary['final_step'] = args.steps


if __name__ == '__main__':
    raise SystemExit(main())
