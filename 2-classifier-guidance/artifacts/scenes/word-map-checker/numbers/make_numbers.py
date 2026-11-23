#!/usr/bin/env python
"""Every full-scale number the word-map checker ladder prints, written to numbers.json.

Three kinds of value, each carrying its source:
  counted   computed here from a config file, or by building the head's layers and counting weights
  printed   copied from the paper with its page, or from a checked note or result card with its path
  inferred  worked out from something printed, marked as inferred on every page that uses it

The head is rebuilt here from the structure read in the authors' train_relation_classifier/model.py at
commit a2ae427. The authors' files are read as data and never imported or run. No model weights are
loaded, and nothing touches SDXL beyond its public config file.

Run from anywhere:
  CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 uv run --no-project --with torch --with pyyaml python make_numbers.py
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
import yaml

HERE = Path(__file__).resolve().parent
JOB = Path(__file__).resolve().parent / "sources"
CODE = Path(__import__("os").environ.get("LEARN_TO_STEER_CODE", JOB / "learn-to-steer"))
COMMIT = "a2ae42746ec93c5936410964be01c85c0ca2266e"
CONFIGS = JOB / "configs"
NOTE = "artifacts/notes/progress/abstract-checkers/01-reading/learn-to-steer.md"
RESULT = "2-classifier-guidance/report/classifier-guided-poe/does-an-early-push-separate-the-cat-and-dog.md"


def item(value, source, kind="counted", meaning=""):
    return {"value": value, "source": source, "kind": kind, "meaning": meaning}


# ---------------------------------------------------------------- where the word maps come from
def cross_attention_layout(unet: dict) -> list[dict]:
    """One row per cross-attention depth: grid side, layers, heads per layer and maps per word."""
    heads = unet["attention_head_dim"]
    n = len(unet["block_out_channels"])
    heads = heads if isinstance(heads, list) else [heads] * n
    tf = unet.get("transformer_layers_per_block") or 1
    tf = tf if isinstance(tf, list) else [tf] * n
    side = unet["sample_size"]
    per = unet["layers_per_block"]
    rows: dict[int, dict] = {}

    def add(grid, layers, h, where):
        r = rows.setdefault(grid, {"grid": grid, "layers": 0, "heads": h, "where": []})
        assert r["heads"] == h, "one grid side, one head count"
        r["layers"] += layers
        r["where"].append(f"{where}: {layers}")

    for i, kind in enumerate(unet["down_block_types"]):
        grid = side // 2 ** i
        if kind.startswith("CrossAttn"):
            add(grid, per * tf[i], heads[i], f"down_blocks.{i}")
    bottom = side // 2 ** (n - 1)
    add(bottom, tf[-1], heads[-1], "mid_block")
    for i, kind in enumerate(unet["up_block_types"]):
        d = n - 1 - i
        grid = side // 2 ** d
        if kind.startswith("CrossAttn"):
            add(grid, (per + 1) * tf[d], heads[d], f"up_blocks.{i}")
    out = sorted(rows.values(), key=lambda r: -r["grid"])
    for r in out:
        r["maps_per_word"] = r["layers"] * r["heads"]
    return out


# ---------------------------------------------------------------- the head, rebuilt to count it
class Positional(nn.Module):
    def __init__(self, d, max_len):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d, 2).float() * (-torch.log(torch.tensor(10000.0)) / d))
        pe = torch.zeros(1, max_len, d)
        pe[0, :, 0::2] = torch.sin(position * div)
        pe[0, :, 1::2] = torch.cos(position * div)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, :x.shape[1], :]


def encoder_layer(d, nhead, ff, dropout):
    return nn.TransformerEncoderLayer(d, nhead, ff, dropout, "relu", batch_first=True, norm_first=True)


class Aggregator(nn.Module):
    """model.py CrossAttentionExtractor: L x H maps per word in, one 16 x 16 map out."""

    def __init__(self, n_maps, side, d, steps, c):
        super().__init__()
        self.side = side
        self.input_projection = nn.Linear(side * side, d)
        self.timestep_embedding = nn.Embedding(steps, d)
        self.positional = Positional(d, n_maps)
        self.layers = nn.ModuleList(encoder_layer(d, c["nhead"], c["dim_feedforward"], c["dropout"])
                                    for _ in range(c["num_layers"]))
        self.query_vector = nn.Parameter(torch.zeros(1, 1, d))
        self.layer_attention = nn.MultiheadAttention(d, c["layer_attn_nheads"], batch_first=True)
        self.output_projection = nn.Linear(d, side * side)

    def forward(self, maps, step):
        b, c, h, w = maps.shape
        x = self.positional(self.input_projection(maps.view(b, c, -1)))
        e = self.timestep_embedding(step).unsqueeze(1)
        for layer in self.layers:
            x = layer(x + e)
        pooled, weights = self.layer_attention(self.query_vector.repeat(b, 1, 1), x, x)
        return self.output_projection(pooled.squeeze(1)).view(b, h, w), weights


class Classifier(nn.Module):
    """model.py RelationClassifierTransformer: two 16 x 16 maps and the step in, five logits out."""

    def __init__(self, side, d, steps, classes, c):
        super().__init__()
        self.projection = nn.Linear(side * side * 2, d)
        self.positional = Positional(d, 256)
        self.transformer = nn.TransformerEncoder(
            encoder_layer(d, c["nhead"], c["dim_feedforward"], c["dropout"]), c["num_layers"])
        self.timestep_embedding = nn.Embedding(steps, c["timestep_embedding_dim"])
        self.fc = nn.Sequential(nn.Linear(d + c["timestep_embedding_dim"], 256), nn.LayerNorm(256), nn.LeakyReLU(0.1),
                                nn.Dropout(c["dropout"]), nn.Linear(256, 128), nn.LayerNorm(128), nn.LeakyReLU(0.1),
                                nn.Dropout(c["dropout"]), nn.Linear(128, classes))

    def forward(self, two_maps, step):
        x = self.projection(two_maps.view(two_maps.size(0), -1))
        tokens = self.positional(x.unsqueeze(1))
        x = self.transformer(tokens).squeeze(1)
        return self.fc(torch.cat([x, self.timestep_embedding(step)], 1)), tokens.shape[1]


def count(m):
    return sum(p.numel() for p in m.parameters())


def head_numbers(config_name: str, n_maps: int) -> dict:
    cfg = yaml.safe_load((CODE / "train_relation_classifier/configs" / f"{config_name}.yaml").read_text())["model"]
    side, d, steps = cfg["map_size"], cfg["d_model"], cfg["num_timesteps"]
    agg = Aggregator(n_maps, side, d, steps, cfg["extractor"]).eval()
    cls = Classifier(side, d, steps, 5, cfg["classifier"]).eval()
    with torch.no_grad():
        maps = torch.zeros(2, n_maps, side, side)
        step = torch.tensor([5, 5])
        one, weights = agg(maps, step)
        logits, tokens = cls(torch.stack([one, one], 1), step)
    assert one.shape == (2, side, side) and logits.shape == (2, 5) and weights.shape == (2, 1, n_maps)
    return {
        "config": f"train_relation_classifier/configs/{config_name}.yaml",
        "maps_per_word_in": n_maps, "map_side": side, "d": d, "step_table_rows": steps,
        "aggregator": {k: cfg["extractor"][k] for k in ("num_layers", "nhead", "dim_feedforward", "layer_attn_nheads")},
        "classifier": {k: cfg["classifier"][k] for k in ("num_layers", "nhead", "dim_feedforward", "timestep_embedding_dim")},
        "classifier_transformer_tokens": tokens,
        "trained_weights": {"aggregator": count(agg), "classifier": count(cls), "total": count(agg) + count(cls)},
        "shapes": {"one_word_in": [n_maps, side, side], "aggregated": [side, side], "stacked": [2, side, side],
                   "flattened": 2 * side * side, "projected": d, "logits": 5},
    }


# ---------------------------------------------------------------- the sampler's timesteps
def leading_timesteps(cfg: dict, steps: int) -> list[int]:
    assert cfg["timestep_spacing"] == "leading"
    ratio = cfg["num_train_timesteps"] // steps
    return [i * ratio + cfg["steps_offset"] for i in range(steps)][::-1]


def main():
    sdxl = json.loads((CONFIGS / "stable-diffusion-xl-base-1.0-unet.json").read_text())
    sd14 = json.loads((CONFIGS / "stable-diffusion-v1-4-unet.json").read_text())
    sched = json.loads((CONFIGS / "sdxl-scheduler.json").read_text())

    sdxl_rows, sd14_rows = cross_attention_layout(sdxl), cross_attention_layout(sd14)
    sdxl_maps = sum(r["maps_per_word"] for r in sdxl_rows)
    sdxl_layers = sum(r["layers"] for r in sdxl_rows)
    sd14_maps = sum(r["maps_per_word"] for r in sd14_rows)
    assert sdxl_layers == 70, "journey 13's network_view_sizes.json counts 70 transformer blocks"
    assert [(r["grid"], r["layers"], r["heads"]) for r in sdxl_rows] == [(64, 10, 10), (32, 60, 20)]

    # SD 2.1's config returned 401, so its count is worked from the usual head list and marked inferred.
    sd21_rows = [(64, 5, 5), (32, 5, 10), (16, 5, 20), (8, 1, 20)]
    sd21_maps = sum(l * h for _, l, h in sd21_rows)
    flux_maps = 57 * 24
    assert round(flux_maps / sd21_maps) == 7, "the paper's '7x fewer maps' (p. 5)"

    ts = leading_timesteps(sched, 50)
    window = ts[:16]

    out = {
        "_about": "Full-scale numbers for the word-map checker ladder. The miniature's numbers are in "
                  "2-miniature/miniature_answer_key.json. Written by make_numbers.py.",
        "code": {"repo": "https://github.com/sapir1779/learn-to-steer", "commit": COMMIT,
                 "checkout": "numbers/sources/learn-to-steer", "note": "cloned read-only, never imported or run"},
        "word_maps": {
            "sdxl": item({"per_grid": sdxl_rows, "layers": sdxl_layers, "maps_per_word": sdxl_maps},
                         "stabilityai/stable-diffusion-xl-base-1.0 unet/config.json",
                         meaning="SDXL's cross-attention layers and heads, so the maps one word would hand a "
                                 "Learn-to-Steer head. SDXL is our proposal: the paper never ran it."),
            "sd_1_4": item({"per_grid": sd14_rows, "maps_per_word": sd14_maps}, "CompVis/stable-diffusion-v1-4 unet/config.json"),
            "sd_2_1": item({"per_grid": [{"grid": g, "layers": l, "heads": h, "maps_per_word": l * h} for g, l, h in sd21_rows],
                            "maps_per_word": sd21_maps},
                           "SD 2.1-base's config was not reachable (HTTP 401). Head list 5, 10, 20, 20 assumed; "
                           "1368 / 195 = 7.0 agrees with the paper's '7x fewer maps' (p. 5).", "inferred"),
            "flux": item({"layers": 57, "heads": 24, "maps_per_word": flux_maps},
                         "train_relation_classifier/model_utils.py:37 default input_channels, 57 * 24"),
            "our_attention_push": item({"grid": 32, "layers_averaged": 60, "maps_per_word_after_averaging": 1},
                                       "2-classifier-guidance/scripts/rung2/attention_push.py:42-43,77",
                                       meaning="heads averaged per layer, then every 32 x 32 layer averaged into one map"),
        },
        "head": {
            "sd2_config_on_195_maps": head_numbers("sd2_config", sd21_maps),
            "flux_dev_config_on_1368_maps": head_numbers("flux_dev_config", flux_maps),
            "sd2_config_on_sdxl_1300_maps_proposal": head_numbers("sd2_config", sdxl_maps),
            "flux_dev_config_on_sdxl_1300_maps_proposal": head_numbers("flux_dev_config", sdxl_maps),
        },
        "schedule": {
            "first_16_timesteps": item(window, "SDXL scheduler_config.json: leading spacing, steps_offset 1, 50 steps"),
            "pushed_steps_above_t_800": item(sum(t > 800 for t in window),
                                             "counted on first_16_timesteps; the sketch's 10 of 16"),
            "bucket_rule": item({"steps 0 to 5": 5, "steps 6 to 10": 10, "steps 11 to 15": 15, "steps 16 to 20": 20,
                                 "steps 21 to 25": 25, "steps 26 to 49": 49},
                                "pipelines/test_time_optimization.py:59-74, training timesteps from configs",
                                meaning="the step number the head's time table is asked for"),
        },
        "paper": {
            "models": item(["FLUX.1-schnell", "FLUX.1-dev", "SD 2.1", "SD 1.4"], f"{NOTE}; paper p. 5", "printed"),
            "geneval_position": item({"SD 2.1": {"vanilla": 0.07, "learn_to_steer": 0.54, "best_baseline": "COMPASS 0.51", "storm_handcrafted": 0.19},
                                      "FLUX.1-dev": {"vanilla": 0.20, "learn_to_steer": 0.61, "best_baseline": "COMPASS 0.60"},
                                      "SD 1.4": {"vanilla": 0.05, "learn_to_steer": 0.36, "best_baseline": "COMPASS 0.46"}},
                                     "paper p. 5, Table 1", "printed",
                                     "fraction of 400 images with the right relation, 0 to 1"),
            "unseen_16_categories": item({"FLUX.1-dev": {"vanilla": 0.18, "compass": 0.58, "learn_to_steer": 0.60},
                                          "SD 2.1": {"vanilla": 0.15, "compass": 0.52, "learn_to_steer": 0.60},
                                          "SD 1.4": {"vanilla": 0.07, "compass": 0.40, "learn_to_steer": 0.47}},
                                         "paper p. 5, Table 2; 200 prompts, CLIPSeg scorer (p. 14)", "printed",
                                         "words reported absent from training, not a planned split"),
            "leakage_fix": item({"off": 0.425, "on": 0.57}, "paper p. 8, Table 4, FLUX.1-schnell validation", "printed"),
            "aggregation": item({"full_concatenation": 0.70, "mean_per_layer": 0.69, "mean_global": 0.78},
                                "paper p. 13, Table 9, FLUX.1-dev validation", "printed",
                                "the global mean scored highest and gave dull backgrounds"),
            "step_size": item({"SD 2.1 and Flux": 5, "SD 1.4": 7.5}, "paper p. 16", "printed"),
            "iterations_per_step": item(15, "paper p. 16", "printed"),
            "optimised_fraction": item(0.5, "paper p. 4, p. 16", "printed"),
            "cost_seconds": item({"SD 2.1": [4.5, 90], "FLUX.1-dev": [11, 360]}, "paper p. 7", "printed"),
        },
        "code_against_paper": [
            {"what": "pooling the maps into one", "paper": "a four-layer, single-head transformer with a learned query (p. 15)",
             "code": "one MultiheadAttention layer with a learned query: 4 heads on FLUX.1-dev, 2 on SD 2.1",
             "where": "train_relation_classifier/model.py:133-135,176-177; configs layer_attn_nheads"},
            {"what": "the classifier's transformer", "paper": "two-channel image flattened, projected, passed through a transformer (p. 15)",
             "code": "the whole flattened pair becomes one token, so its self-attention has only itself to attend to",
             "where": "train_relation_classifier/model.py:76-80"},
            {"what": "iterations per step", "paper": "15 (p. 16)",
             "code": "range(15 + 1) passes, each of which updates if the loss is above its target: up to 16 updates",
             "where": "pipelines/test_time_optimization.py:175-187; run.py:49 (FLUX.1-dev), 55 (SD 2.1), 61 (SD 1.4)"},
            {"what": "what 'timestep' means to the head", "paper": "the denoising timestep t (p. 4)",
             "code": "the step index 0 to 49, snapped up to one of 5, 10, 15, 20, 25, 49",
             "where": "pipelines/test_time_optimization.py:59-74,86-91,115"},
            {"what": "training epochs on SD 2.1", "paper": "150 for SD models (p. 16)", "code": "num_epochs 100 in sd2_config.yaml, 150 in sd14_config.yaml",
             "where": "train_relation_classifier/configs/sd2_config.yaml:30"},
            {"what": "inversion guidance on SD", "paper": "guidance scale 0 (p. 5)",
             "code": "guidance_scale 1.0, which this pipeline treats as no classifier-free guidance (guidance_scale > 1.0 turns it on)",
             "where": "generate_relations_data.py:163; pipeline_sd2_inversion_debug.py:113"},
        ],
        "what_the_paper_left_open_and_the_code_settles": [
            {"question": "how a multi-token name becomes one map", "answer": "the mean over the name's tokens", "where": "utils.py:127-129"},
            {"question": "which guidance branch supplies the maps", "answer": "the prompt's own branch: the optimiser runs the U-Net on the text embedding alone",
             "where": "pipelines/sd/pipeline_sd2_debug.py:347-350; sd2_test_time_optimization.py:33-41"},
            {"question": "the size of the maps", "answer": "resized to 16 x 16 and each renormalised to sum to 1 over the grid", "where": "utils.py:51-74,152-154; configs map_size"},
        ],
        "ours": {
            "attention_separation_push": item({"two_animals": 0, "seeds": 8, "seed_range": [9, 16], "steps": [0, 19],
                                               "note": "draft calls, strong pushes zoom into one face, no verdict recorded"},
                                              f"{RESULT}, table row result-sheet-layout", "printed"),
            "overlap_of_top_10_percent": item({"product_median_range": [0.506, 0.814], "joint_median_range": [0.225, 0.394],
                                               "steps": [2, 40], "seeds": 8},
                                              f"{RESULT}: intersection over union of the cat and dog maps' top 10%; the joint prompt is 'a cat and a dog' (scripts/rung2/attention_overlap.py:5)", "printed"),
            "default_push_cap": item(0.1, "poe_repair/methods/_classifier_guided.py:46 PUSH_CAP_FRACTION"),
            "proposed_window_and_cap": item({"steps": [0, 15], "of": 50, "cap": 0.4},
                                            "the abstract-checkers brief and the Learn-to-Steer note, line 112", "printed"),
        },
        "open_risk": {
            "detector_guidance_map": item({"mAP@0.5_at_t": {"0": 56.5, "200": 58.4, "400": 54.8, "600": 38.1, "800": 18.0}},
                                          "artifacts/notes/progress/abstract-checkers/01-reading/detector-guidance.md, line 66; "
                                          "that paper's p. 14, Table 1, on SD 2.1", "printed",
                                          "a word-map detector's score, 0 to 100, higher is better"),
        },
    }
    (HERE / "numbers.json").write_text(json.dumps(out, indent=1) + "\n")
    h = out["head"]
    print("SDXL:", [(r["grid"], r["layers"], r["heads"], r["maps_per_word"]) for r in sdxl_rows], "total maps per word", sdxl_maps)
    print("SD 1.4 maps", sd14_maps, "SD 2.1 maps (inferred)", sd21_maps, "Flux maps", flux_maps)
    for k, v in h.items():
        print(k, v["trained_weights"], "classifier tokens", v["classifier_transformer_tokens"])
    print("window", window[0], "to", window[-1], "above 800:", out["schedule"]["pushed_steps_above_t_800"]["value"])


if __name__ == "__main__":
    main()
