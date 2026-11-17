#!/usr/bin/env python
"""Where do "cat" and "dog" look while SDXL draws: apart in the joint prompt, on top of each other in the product?

For cat x dog on seeds 1 to 8, two drawings from the same starting noise:
    joint    "a cat and a dog": one branch, the "cat" and "dog" tokens read from it
    product  "a cat" x "a dog": the "cat" token from the cat branch, the "dog" token from the dog branch
At every step the cross-attention probabilities of every cross-attention layer working at 32 x 32
are averaged over heads and layers. As Attend-and-Excite does, the start-of-text token is dropped
and each spatial position is renormalised over the prompt's own word tokens, so a word's map shows
where it wins against the other words rather than the start token's background. Two overlaps are
kept per step: the histogram intersection of the two maps (each normalised to sum 1; 1 means the
same places) and the overlap of each map's top 10% of positions (intersection over union). Every
step's two maps are saved, so other measures can be read afterwards. Nothing is pushed.

Writes, under MOUNT_ROOT/outputs/classifier_guidance_ladder/rung2/attention/:
    overlap.json, overlap-through-time.png, maps-seedNN.png, tiles/seedNN__<joint|product>.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from poe_repair import paths  # noqa: E402
from poe_repair.composers._helpers import init_latents_for_cell  # noqa: E402
from poe_repair.methods._sampling import run_cfg, run_cfg_poe, write_decoded_image  # noqa: E402
from poe_repair.run import make_ctx  # noqa: E402
from poe_repair.runtime import encode_prompt_sdxl  # noqa: E402
import classifier_guidance_from_the_paper as single  # noqa: E402

OUT = paths.MOUNT_ROOT / "outputs/classifier_guidance_ladder/rung2/attention"
SEEDS = tuple(range(1, 9))
SHOWN_STEPS = (2, 5, 10, 20, 40)


class Store:
    def __init__(self):
        self.steps, self.current = [], []

    def new_step(self, *_):
        if self.current:
            self.steps.append(torch.stack(self.current).mean(0))
        self.current = []

    def finish(self):
        self.new_step()


class StoreCrossAttention:
    """The plain attention computation, keeping the probabilities of 32 x 32 layers."""

    def __init__(self, store: Store):
        self.store = store

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None, **kwargs):
        batch, seq, _ = hidden_states.shape
        query = attn.head_to_batch_dim(attn.to_q(hidden_states))
        key = attn.head_to_batch_dim(attn.to_k(encoder_hidden_states))
        value = attn.head_to_batch_dim(attn.to_v(encoder_hidden_states))
        probs = attn.get_attention_scores(query, key, attention_mask)
        if seq == 32 * 32:
            self.store.current.append(probs.view(batch, attn.heads, seq, -1).mean(1).float().cpu())
        # (the maps are renormalised over word tokens after averaging; see words_only)
        out = attn.batch_to_head_dim(torch.bmm(probs, value))
        return attn.to_out[1](attn.to_out[0](out))


def token_index(tokenizer, prompt: str, word: str) -> int:
    ids = tokenizer(prompt).input_ids
    target = tokenizer(word, add_special_tokens=False).input_ids
    assert len(target) == 1, (word, target)
    return ids.index(target[0])


def words_only(m: torch.Tensor, n_words: int) -> torch.Tensor:
    """(positions, tokens) -> the same over tokens 1..n_words, renormalised per position."""
    w = m[:, 1:1 + n_words]
    return w / w.sum(-1, keepdim=True)


def overlap(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a / a.sum(), b / b.sum()
    return float(torch.minimum(a, b).sum())


def top_iou(a: torch.Tensor, b: torch.Tensor, frac: float = 0.10) -> float:
    k = max(1, int(frac * a.numel()))
    ia, ib = set(torch.topk(a, k).indices.tolist()), set(torch.topk(b, k).indices.tolist())
    return len(ia & ib) / len(ia | ib)


def main() -> int:
    if not torch.cuda.is_available():
        raise SystemExit("guard: torch sees no CUDA card")
    (OUT / "tiles").mkdir(parents=True, exist_ok=True)
    ctx = make_ctx(output_root=OUT / "raw")
    unet = ctx.models["unet"]
    enc = lambda p: encode_prompt_sdxl(p, models=ctx.models, device=ctx.device, dtype=ctx.dtype)
    seq_e, pool_e = enc("")
    seq_a, pool_a = enc("a cat")
    seq_b, pool_b = enc("a dog")
    seq_j, pool_j = enc("a cat and a dog")
    tok = ctx.models["tokenizer"]
    i_cat_a, i_dog_b = token_index(tok, "a cat", "cat"), token_index(tok, "a dog", "dog")
    i_cat_j, i_dog_j = token_index(tok, "a cat and a dog", "cat"), token_index(tok, "a cat and a dog", "dog")
    n_a = len(tok("a cat").input_ids) - 2
    n_b = len(tok("a dog").input_ids) - 2
    n_joint = len(tok("a cat and a dog").input_ids) - 2
    common = dict(models=ctx.models, scheduler=ctx.scheduler, seq_e=seq_e, pool_e=pool_e,
                  guidance_scale=ctx.guidance_scale, num_inference_steps=ctx.num_inference_steps,
                  height=1024, width=1024, device=ctx.device, dtype=ctx.dtype)
    originals = dict(unet.attn_processors)
    record = {"seeds": SEEDS, "token_indices": {"cat_in_a_cat": i_cat_a, "dog_in_a_dog": i_dog_b,
                                                "cat_in_joint": i_cat_j, "dog_in_joint": i_dog_j}, "overlap": {}}
    maps_shown = {}
    for seed in SEEDS:
        for mode in ("joint", "product"):
            store = Store()
            procs = {n: (StoreCrossAttention(store) if ".attn2." in n else p) for n, p in originals.items()}
            unet.set_attn_processor(procs)
            hook = unet.register_forward_pre_hook(store.new_step)
            init, sigma = init_latents_for_cell(single.cell_for(seed), ctx)
            try:
                if mode == "joint":
                    o = run_cfg(init_latents=init, euler_init_noise_sigma=sigma, seq_cond=seq_j, pool_cond=pool_j, **common)
                else:
                    o = run_cfg_poe(init_latents=init, euler_init_noise_sigma=sigma, seq_a=seq_a, pool_a=pool_a,
                                    seq_b=seq_b, pool_b=pool_b, **common)
            finally:
                hook.remove()
                unet.set_attn_processor(dict(originals))
            store.finish()
            write_decoded_image(o.image, OUT / "tiles" / f"seed{seed:02d}__{mode}.png")
            curve, curve_top, shown, saved = [], [], {}, []
            for s, m in enumerate(store.steps):                # m: (batch, 1024, tokens)
                if mode == "joint":
                    w = words_only(m[0], n_joint)
                    cat, dog = w[:, i_cat_j - 1], w[:, i_dog_j - 1]
                else:
                    cat = words_only(m[0], n_a)[:, i_cat_a - 1]
                    dog = words_only(m[1], n_b)[:, i_dog_b - 1]
                curve.append(overlap(cat, dog))
                curve_top.append(top_iou(cat, dog))
                saved.append(torch.stack([cat, dog]).view(2, 32, 32))
                if s in SHOWN_STEPS:
                    shown[s] = (cat.view(32, 32).numpy(), dog.view(32, 32).numpy())
            record["overlap"][f"seed {seed}, {mode}"] = curve
            record.setdefault("top10_iou", {})[f"seed {seed}, {mode}"] = curve_top
            torch.save(torch.stack(saved), OUT / f"maps-seed{seed:02d}-{mode}.pt")
            maps_shown[(seed, mode)] = shown
            print(f"seed {seed} {mode}: overlap step 2/10/40 {curve[2]:.2f} {curve[10]:.2f} {curve[40]:.2f}; "
                  f"top-10% IoU {curve_top[2]:.2f} {curve_top[10]:.2f} {curve_top[40]:.2f}", flush=True)
        (OUT / "overlap.json").write_text(json.dumps(record, indent=1))

    fig, ax = plt.subplots(figsize=(8, 4))
    for seed in SEEDS:
        ax.plot(record["overlap"][f"seed {seed}, joint"], color="#2b6cb0", alpha=0.6, lw=1.2)
        ax.plot(record["overlap"][f"seed {seed}, product"], color="#c05621", alpha=0.6, lw=1.2)
    ax.plot([], [], color="#2b6cb0", label='joint prompt: "cat" and "dog" in one branch')
    ax.plot([], [], color="#c05621", label="product: cat branch's \"cat\" against dog branch's \"dog\"")
    ax.set_xlabel("sampler step (0 = pure noise, of 50)")
    ax.set_ylabel("overlap of the two words' attention (1 = same place)")
    ax.set_title("Where \"cat\" and \"dog\" look, seeds 1 to 8, one line per seed", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "overlap-through-time.png", dpi=120)
    plt.close(fig)
    for seed in SEEDS:
        fig, axes = plt.subplots(4, len(SHOWN_STEPS), figsize=(2.2 * len(SHOWN_STEPS), 9))
        for c, s in enumerate(SHOWN_STEPS):
            for r, (mode, k, label) in enumerate((("joint", 0, "joint: cat"), ("joint", 1, "joint: dog"),
                                                  ("product", 0, "product: cat"), ("product", 1, "product: dog"))):
                ax = axes[r][c]
                ax.imshow(maps_shown[(seed, mode)][s][k], cmap="magma")
                ax.set_xticks([]); ax.set_yticks([])
                if r == 0:
                    ax.set_title(f"step {s}", fontsize=9)
                if c == 0:
                    ax.set_ylabel(label, fontsize=9)
        fig.suptitle(f"Seed {seed}: where each word looks (32 x 32, averaged over heads and layers)", fontsize=10)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(OUT / f"maps-seed{seed:02d}.png", dpi=90)
        plt.close(fig)
    print("ATTENTION DONE", OUT, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
