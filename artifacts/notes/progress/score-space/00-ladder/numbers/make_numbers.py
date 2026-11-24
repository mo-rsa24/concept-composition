"""Every size and value the score-path ladder prints, read from the digit environment's own code.

It imports the repository's real modules (canvas_unet.py, canvas_sheet.py, w4_train.py's constants are read as text so
CLIP is never loaded), builds the real CanvasUNet on the CPU with random weights only to count its weights and trace its
shapes, and writes numbers.json beside this file. No trained weights are read, no GPU is used and nothing is sampled.

Run from the repository root:
    CUDA_VISIBLE_DEVICES= uv run --no-project --with torch --with numpy python \
        artifacts/notes/progress/score-space/00-ladder/numbers/make_numbers.py
"""
import itertools
import json
import re
import subprocess
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[5]
MNIST = REPO / "2-classifier-guidance" / "scripts" / "mnist"
sys.path.insert(0, str(MNIST))
import canvas_unet as cu  # noqa: E402
from canvas_sheet import HELD_OUT, PAIRS  # noqa: E402

commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()

model = cu.CanvasUNet().eval()  # the defaults the trainer uses: ch 64, mults (1, 2, 2), 11 labels
weights = sum(p.numel() for p in model.parameters())
by_part = {
    "time_vector_mlp": sum(p.numel() for p in model.temb.parameters()),
    "label_table": sum(p.numel() for p in model.cemb.parameters()),
    "input_conv": sum(p.numel() for p in model.inp.parameters()),
    "down_path": sum(p.numel() for p in model.down.parameters()),
    "middle": sum(p.numel() for p in model.mid.parameters()),
    "up_path": sum(p.numel() for p in model.up.parameters()),
    "output": sum(p.numel() for p in model.out.parameters()),
}
assert sum(by_part.values()) == weights

# Trace shapes through the real forward pass.
shapes = []
x = torch.zeros(1, 1, cu.H, cu.W)
t = torch.tensor([999])
y = torch.tensor([cu.NULL])
with torch.no_grad():
    emb = model.temb(cu.timestep_embedding(t, model.ch)) + model.cemb(y)
    h = model.inp(x)
    shapes.append(("input conv", list(h.shape[1:])))
    hs = [h]
    for m in model.down:
        h = m(h, emb) if isinstance(m, cu.ResBlock) else m(h)
        hs.append(h)
        shapes.append(("down " + type(m).__name__, list(h.shape[1:])))
    for m in model.mid:
        h = m(h, emb) if isinstance(m, cu.ResBlock) else m(h)
        shapes.append(("middle " + type(m).__name__, list(h.shape[1:])))
    for m in model.up:
        if isinstance(m, cu.ResBlock):
            h = m(torch.cat([h, hs.pop()], 1), emb)
        else:
            h = m(h)
        shapes.append(("up " + type(m).__name__, list(h.shape[1:])))
    eps = model.out(h)
    shapes.append(("output", list(eps.shape[1:])))
    assert torch.equal(eps, model(x, t, y))

levels = []
for name, shp in shapes:
    if name.startswith(("down ResBlock", "middle ResBlock")) and (not levels or levels[-1][1:] != shp[1:]):
        levels.append(shp)

steps = cu.ddim_timesteps(50)
abar = cu.ABAR
window = [int(v) for v in steps[:16]]
schedule = [{"step": i, "t": tt, "abar": round(float(abar[tt]), 6), "sqrt_abar": round(float(abar[tt].sqrt()), 4),
             "sqrt_one_minus_abar": round(float((1 - abar[tt]).sqrt()), 4)} for i, tt in enumerate(window)]

w4 = (REPO / "2-classifier-guidance" / "scripts" / "noise_text_critic" / "w4_train.py").read_text()
def const(name):
    return re.search(rf"^{name} = (.+)$", w4, re.M).group(1).strip().strip('"')

seen_digits = range(5)
caption_bank = len(seen_digits) + 3 * len(list(itertools.combinations(seen_digits, 2)))
pixels = cu.H * cu.W

out = {
    "source_commit": commit,
    "files": {
        "model": "2-classifier-guidance/scripts/mnist/canvas_unet.py",
        "canvases_for_the_models": "2-classifier-guidance/scripts/mnist/canvas_sheet.py",
        "test_canvases": "2-classifier-guidance/scripts/abstract_checkers/build_env.py",
        "existing_d_t_maps": "2-classifier-guidance/scripts/abstract_checkers/v2_learn.py, maps()",
        "text_with_latent": "2-classifier-guidance/scripts/noise_text_critic/w4_train.py",
    },
    "canvas": {"height": cu.H, "width": cu.W, "digit_side": 28, "pixels": pixels,
               "pixel_range": "-1 to +1 (training_batch divides by 127.5 and subtracts 1)"},
    "unet": {
        "class": "CanvasUNet(ch=64, mults=(1, 2, 2), n_classes=11)",
        "weights": weights,
        "weights_by_part": by_part,
        "base_channels": model.ch,
        "vector_width": model.ch * 4,
        "labels": 11,
        "null_label": cu.NULL,
        "label_meaning": "0 to 9 are digits, 10 is no digit; a pair has no label of its own",
        "levels": levels,
        "attention": "one 4-head self-attention at the 8 x 16 grid, in the middle",
        "res_blocks": {"down": 6, "middle": 2, "up": 9},
        "time_and_label_join": "emb = temb(timestep_embedding(t, 64)) + cemb(y), added inside every ResBlock",
        "shapes_traced": shapes,
    },
    "training_of_the_digit_models": {
        "T": cu.T, "betas": "linear 1e-4 to 0.02", "null_label_rate": 0.1,
        "steps": 40000, "batch": 128, "lr": 2e-4, "ema": 0.999,
        "shares": {"share-0.0": "never saw two digits on one canvas",
                   "share-0.01": "1% of canvases hold two digits, from the 40 pairs not held out; each enters once per digit it holds"},
        "held_out_pairs": [list(p) for p in HELD_OUT],
        "pairs_total": len(PAIRS),
        "pairs_seen_by_share_0_01": len(PAIRS) - len(HELD_OUT),
        "pairs_among_5_to_9_held_out": [list(p) for p in HELD_OUT if min(p) >= 5],
        "source": "canvas_unet.py argparse defaults and training_batch; canvas_sheet.py HELD_OUT",
    },
    "sampler": {"kind": "DDIM", "steps": 50, "eta": 0.0,
                "window_steps_0_to_15": schedule,
                "window_note": "ddim_timesteps(50)[:16], the window w4_train.py calls STEPS16. v2_learn.py also defines "
                               "SDXL_MATCHED, t = 712 down to 480, used by Loop 8's count checker."},
    "score_path": {
        "per_step": [1, cu.H, cu.W],
        "steps": 16,
        "shape": [16, cu.H, cu.W],
        "numbers": 16 * pixels,
        "forwards_per_canvas_per_step": "3 for one pair (no digit, a, b), batched by product_eps",
        "note": "d_t(c) = eps(x_t, t, c) - eps(x_t, t, 10). Stacking it is this ladder's definition from the session note; "
                "no code in the repository builds it yet.",
    },
    "w4_text_with_latent": {
        "text_model": const("TEXT_MODEL"), "text_width": 768, "text_layers": 12, "temperature": float(const("TAU")),
        "caption_bank": caption_bank, "updates": int(const("UPDATES")), "batch": int(const("BATCH")),
        "encoder_widths": [64, 128, 256, 256],
        "source": "w4_train.py constants; widths and 768 from noise-text-critic/04-train-our-own/design.md; 12 text layers is openai/clip-vit-large-patch14 text_config.num_hidden_layers, as the NoisyCLIP ladder overview ov3 prints it",
    },
    "score_path_encoder": "not designed yet (step X4); every size drawn for it is illustrative",
}
(HERE / "numbers.json").write_text(json.dumps(out, indent=1))
print(json.dumps({"weights": weights, "levels": levels, "window": window, "caption_bank": caption_bank}, indent=0))
