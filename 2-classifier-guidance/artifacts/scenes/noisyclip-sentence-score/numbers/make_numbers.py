"""Every full-scale number on the NoisyCLIP ladder, each with where it comes from.

Run with CPU torch and transformers (no weights are downloaded; the model is built on the meta device):

    CUDA_VISIBLE_DEVICES= uv run --no-project --with torch --with transformers --with numpy python make_numbers.py

kinds: "config" read from a public config file, "counted" computed from one, "computed" calculated here,
"paper" printed by arXiv 2512.08505v2 at the page given, "code" read from the released code at commit 016b913,
"compared" read from the checkpoint comparison output.
"""
import json
import math
import re
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SRC = HERE / "sources"
COMPARE = HERE.parents[4] / "artifacts/notes/progress/noise-text-critic/01-reading/tools/checkpoint-compare.txt"

cfg = json.loads((SRC / "clip-vit-large-patch14-config.json").read_text())
pre = json.loads((SRC / "clip-vit-large-patch14-preprocessor.json").read_text())
sched = json.loads((SRC / "sdxl-scheduler-config.json").read_text())
v, t = cfg["vision_config"], cfg["text_config"]


def phi(x):
    return 0.5 * math.erfc(-x / math.sqrt(2))


# ------------------------------------------------------------------ the two towers
patches_per_side = v["image_size"] // v["patch_size"]
clip = {
    "kind": "config, openai/clip-vit-large-patch14",
    "image_size": v["image_size"], "patch_size": v["patch_size"], "patches_per_side": patches_per_side,
    "patches": patches_per_side ** 2, "image_tokens": patches_per_side ** 2 + 1,
    "vision_width": v["hidden_size"], "vision_layers": v["num_hidden_layers"], "vision_heads": v["num_attention_heads"],
    "vision_mlp": v["intermediate_size"], "text_width": t["hidden_size"], "text_layers": t["num_hidden_layers"],
    "text_heads": t["num_attention_heads"], "text_mlp": t["intermediate_size"], "text_tokens": t["max_position_embeddings"],
    "vocab": t["vocab_size"], "shared_width": cfg["projection_dim"],
}

# ------------------------------------------------------------------ which parameters exist, counted on the meta device
import torch  # noqa: E402
from transformers import CLIPConfig, CLIPModel  # noqa: E402

with torch.device("meta"):
    model = CLIPModel(CLIPConfig.from_dict(cfg))
groups = {"vision_model": 0, "visual_projection": 0, "text_model": 0, "text_projection": 0, "logit_scale": 0}
for name, p in model.named_parameters():
    groups[name.split(".")[0]] += p.numel()
only_vision_flag_trains = sum(p.numel() for name, p in model.named_parameters() if "text" not in name and name != "logit_scale")
total = sum(groups.values())
parameters = {
    "kind": "counted from the config on the meta device",
    **groups, "total": total,
    "tuned_in_published_weights": groups["vision_model"],
    "frozen_in_published_weights": total - groups["vision_model"],
    "tuned_share_percent": round(100 * groups["vision_model"] / total, 1),
    "released_only_vision_flag_would_train": only_vision_flag_trains,
    "note": "vision_model as the tuned group is inferred from the tensors compared: visual_projection, text_projection and "
            "the first text layer's key projection are unchanged, the first vision layer changed. Only those were compared.",
}

# ------------------------------------------------------------------ the checkpoint comparison
text = COMPARE.read_text()
rel = float(re.search(r"vision_model\.encoder\.layers\.0\.self_attn\.k_proj\.weight:.*rel_norm ([0-9.e+-]+)", text).group(1))
tuned_scale, base_scale = map(float, re.search(r"values \(([0-9.]+), ([0-9.]+)\)", text).groups())
compared = {
    "kind": "compared, tools/checkpoint-compare.txt",
    "visual_projection_max_abs_diff": float(re.search(r"visual_projection\.weight:.*max\|diff\| ([0-9.e+-]+)", text).group(1)),
    "text_projection_max_abs_diff": float(re.search(r"text_projection\.weight:.*max\|diff\| ([0-9.e+-]+)", text).group(1)),
    "first_text_layer_max_abs_diff": float(re.search(r"text_model\.encoder\.layers\.0\.self_attn\.k_proj\.weight:.*max\|diff\| ([0-9.e+-]+)", text).group(1)),
    "first_vision_layer_relative_change_percent": round(100 * rel, 1),
    "logit_scale_noisyclip": tuned_scale, "logit_scale_base": round(base_scale, 5),
    "score_multiplier_noisyclip": round(math.exp(tuned_scale), 2), "score_multiplier_base": round(math.exp(base_scale), 2),
}

# ------------------------------------------------------------------ the linear map and its clamp
W = np.array([[60, -60, 25, -70], [60, -5, 15, -50], [60, 10, -5, -35]], float)
bias = np.array([150, 140, 130], float)
per = {}
for name, row, b in zip("RGB", W, bias):
    sd = float(np.sqrt((row ** 2).sum()))
    low, high = phi(-b / sd), phi(-(255 - b) / sd)
    per[name] = {"std": round(sd, 2), "below_0_percent": round(100 * low, 1), "above_255_percent": round(100 * high, 1),
                 "clamped_percent": round(100 * (low + high), 1)}
mean_clamped = round(float(np.mean([100 * (phi(-b / np.sqrt((r ** 2).sum())) + phi(-(255 - b) / np.sqrt((r ** 2).sum())))
                                    for r, b in zip(W, bias)])), 1)
rng = np.random.default_rng(0)
n, any_c, all_c = 0, 0, 0
for _ in range(20):
    z = rng.standard_normal((1_000_000, 4))
    rgb = z @ W.T + bias
    clamped = (rgb < 0) | (rgb > 255)
    any_c += int(clamped.any(1).sum()); all_c += int(clamped.all(1).sum()); n += len(z)
std = np.array(pre["image_std"])
jac = W / (255 * std[:, None])
linear_map = {
    "kind": "code, latent_to_image.py lines 10 to 18; shares computed on independent unit-variance channels",
    "weights": W.astype(int).tolist(), "biases": bias.astype(int).tolist(), "per_colour": per,
    "mean_clamped_percent": mean_clamped,
    "pixels_with_any_colour_clamped_percent": round(100 * any_c / n, 1),
    "pixels_with_all_three_clamped_percent": round(100 * all_c / n, 1),
    "monte_carlo_pixels": n,
    "clip_input_per_latent_unit": np.round(jac, 3).tolist(),
    "clip_input_per_latent_unit_note": "weight / (255 x CLIP std), the gradient factor of one unclamped pixel before the resize",
}

# ------------------------------------------------------------------ preprocessing that the released code leaves out
preprocess = {
    "kind": "config, the base model's preprocessor; the released code does not contain this step",
    "input_side": 128, "resize_to": pre["size"], "crop": pre["crop_size"], "resample_code": pre["resample"], "resample": "bicubic",
    "scale_factor": round(pre["size"] / 128, 2), "image_mean": pre["image_mean"], "image_std": pre["image_std"],
}

# ------------------------------------------------------------------ the noise schedule at 50 steps
betas = np.linspace(sched["beta_start"] ** 0.5, sched["beta_end"] ** 0.5, sched["num_train_timesteps"]) ** 2
alpha_bar = np.cumprod(1 - betas)
ratio = sched["num_train_timesteps"] // 50
timesteps = [(49 - k) * ratio + sched["steps_offset"] for k in range(50)]
schedule = {
    "kind": "computed from SDXL's scheduler_config.json: scaled_linear betas, leading spacing, offset 1",
    "sampler_in_released_code": sched["_class_name"],
    "timesteps": timesteps,
    "sigma": [round(float(np.sqrt(1 - alpha_bar[x])), 4) for x in timesteps],
    "signal": [round(float(np.sqrt(alpha_bar[x])), 4) for x in timesteps],
    "euler_sigma": [round(float(np.sqrt((1 - alpha_bar[x]) / alpha_bar[x])), 3) for x in timesteps],
    "train_window": [20, 29], "push_window": [0, 15],
}
for k in (0, 10, 15, 20, 29, 49):
    schedule[f"step_{k}"] = {"t": timesteps[k], "sigma": schedule["sigma"][k], "signal": schedule["signal"][k]}

# ------------------------------------------------------------------ what the paper prints
paper = {
    "kind": "paper, arXiv 2512.08505v2; the paper's own results on its own data, none of them on this project's pictures",
    "train_latents": {"value": 50000, "page": 8}, "train_window": {"value": "latents 20 to 29 of 50", "page": 6},
    "learning_rate": {"value": "5.7e-5", "page": 8}, "weight_decay": {"value": 0.1, "page": 8},
    "epochs": {"value": 10, "page": 8}, "batch": {"value": 128, "page": 8},
    "recall_at_1_latents_21_30": {"noisyclip": 0.506, "frozen_clip_l14": 0.145, "page": 14, "scale": "share of images whose right caption ranks first of 5, 0 to 1, chance 0.20"},
    "separation_mark": {"value": "latent 10, against about 20 iterations later for frozen CLIP", "page": 10},
}

out = {"clip": clip, "parameters": parameters, "checkpoint": compared, "linear_map": linear_map, "preprocess": preprocess,
       "schedule": schedule, "paper": paper}
(HERE / "numbers.json").write_text(json.dumps(out, indent=1))
for sect in ("parameters", "checkpoint"):
    print(sect, json.dumps(out[sect]))
print({k: linear_map[k] for k in ("per_colour", "mean_clamped_percent", "pixels_with_any_colour_clamped_percent", "pixels_with_all_three_clamped_percent")})
print({k: schedule[k] for k in schedule if k.startswith("step_")})
