"""Every size and value the SDXL zoom ladder prints, computed and never typed. Writes sdxl_numbers.json.

Reads:
  source/sdxl-config/*.json      SDXL base 1.0's config files, copied read-only from the cluster's Hugging Face cache
                                 (snapshot 462165984030d82259a11f4367a4eed129e94a7b)
  source/diffusers-0.29.2/*.copy the diffusers files the co3 environment on mscluster runs, copied read-only
  /home/molef/PhD/concept-composition/2-classifier-guidance/scripts/leakage/z2_predictors.py and z2_bars.py
  the Z2 smoke run's outputs (maps, text cosines, picture) under /home/molef/.claude/jobs/0071c318/tmp/
  ../co3-sampler/co3_numbers.json, only to check that the schedule here matches the CO3 ladder's

Every code line a number leans on is asserted to still hold the expected text, so a moved line stops the build.
Run:  python3 sdxl_numbers.py
"""
import ast
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CFG = HERE / "source" / "sdxl-config"
DIF = HERE / "source" / "diffusers-0.29.2"
Z2 = Path("/home/molef/PhD/concept-composition/2-classifier-guidance/scripts/leakage")
SMOKE = Path("/home/molef/.claude/jobs/0071c318/tmp/z2-smoke/an_eagle__x__a_fox")
PICTURE = Path("/home/molef/.claude/jobs/0071c318/tmp/z2-smoke2/ours.png")
CO3_JSON = HERE.parent / "co3-sampler" / "co3_numbers.json"


def cfg(name):
    return json.loads((CFG / name).read_text())


def line(path, n, must):
    text = Path(path).read_text().splitlines()[n - 1]
    assert must in text, f"{Path(path).name}:{n} no longer holds {must!r}: {text!r}"
    return text


def softmax(v):
    e = np.exp(v - v.max())
    return e / e.sum()


def main():
    unet, te1, te2 = cfg("unet__config.json"), cfg("text_encoder__config.json"), cfg("text_encoder_2__config.json")
    sch, vae, idx = cfg("scheduler__scheduler_config.json"), cfg("vae__config.json"), cfg("model_index.json")
    U = DIF / "unet_2d_condition.py.copy"
    B = DIF / "unet_2d_blocks.py.copy"
    A = DIF / "attention.py.copy"
    AP = DIF / "attention_processor.py.copy"
    P = DIF / "pipeline_stable_diffusion_xl.py.copy"

    # ---- the U-Net's shape, from its config and the constructor lines that read it ------------------------------
    line(U, 241, "num_attention_heads = num_attention_heads or attention_head_dim")
    assert unet["num_attention_heads"] is None
    heads = list(unet["attention_head_dim"])  # diffusers reads this list as the number of heads (line 241)
    widths = unet["block_out_channels"]
    line(B, 806, "out_channels // num_attention_heads")
    head_dim = [w // h for w, h in zip(widths, heads)]
    grids = [unet["sample_size"] // 2 ** i for i in range(len(widths))]
    line(U, 362, "add_downsample=not is_final_block")
    line(U, 439, "num_layers=reversed_layers_per_block[i] + 1")
    line(U, 392, "transformer_layers_per_block=transformer_layers_per_block[-1]")
    lpb, tl = unet["layers_per_block"], unet["transformer_layers_per_block"]
    down, up = unet["down_block_types"], unet["up_block_types"]

    levels, skips = [], [(widths[0], grids[0], "conv_in")]
    for i, (kind, w, g) in enumerate(zip(down, widths, grids)):
        attn = kind.startswith("CrossAttn")
        levels.append(dict(part=f"down {i}", kind=kind, channels=w, grid=g, resblocks=lpb,
                           transformer_stacks=lpb if attn else 0, blocks_per_stack=tl[i] if attn else 0,
                           transformer_blocks=lpb * tl[i] if attn else 0, heads=heads[i] if attn else None,
                           head_dim=head_dim[i] if attn else None, patches=g * g))
        for _ in range(lpb):
            skips.append((w, g, f"down {i}"))
        if i < len(widths) - 1:
            skips.append((w, g // 2, f"down {i} downsampler"))
    assert unet["mid_block_type"] == "UNetMidBlock2DCrossAttn"
    # The middle block: one ResBlock, then num_layers (default 1, not passed by get_mid_block or the U-Net) pairs of
    # transformer stack and ResBlock, so 2 ResBlocks and 1 stack.
    line(B, 751, "num_layers: int = 1,")
    line(B, 784, "resnets = [")
    line(B, 801, "for i in range(num_layers):")
    line(B, 827, "resnets.append(")
    assert "num_layers" not in "".join(Path(U).read_text().splitlines()[383:405])
    levels.append(dict(part="middle", kind=unet["mid_block_type"], channels=widths[-1], grid=grids[-1], resblocks=2,
                       transformer_stacks=1, blocks_per_stack=tl[-1], transformer_blocks=tl[-1], heads=heads[-1],
                       head_dim=head_dim[-1], patches=grids[-1] ** 2))
    rw, rg, rtl, rh = widths[::-1], grids[::-1], tl[::-1], heads[::-1]
    prev = widths[-1]
    up_inputs = []
    for i, kind in enumerate(up):
        attn = kind.startswith("CrossAttn")
        ins = []
        for j in range(lpb + 1):
            sc, _, _ = skips.pop()
            ins.append((prev if j == 0 else rw[i]) + sc)
        prev = rw[i]
        up_inputs.append(ins)
        levels.append(dict(part=f"up {i}", kind=kind, channels=rw[i], grid=rg[i], resblocks=lpb + 1,
                           transformer_stacks=lpb + 1 if attn else 0, blocks_per_stack=rtl[i] if attn else 0,
                           transformer_blocks=(lpb + 1) * rtl[i] if attn else 0, heads=rh[i] if attn else None,
                           head_dim=rw[i] // rh[i] if attn else None, patches=rg[i] ** 2, resblock_inputs=ins))
    assert not skips, "every skip is consumed by an up block"
    n_res = sum(l["resblocks"] for l in levels)
    n_tb = sum(l["transformer_blocks"] for l in levels)
    by_grid = {}
    for l in levels:
        if l["transformer_blocks"]:
            by_grid[l["grid"]] = by_grid.get(l["grid"], 0) + l["transformer_blocks"]
    skip_list = [dict(channels=c, grid=g, made_by=m) for c, g, m in
                 [(widths[0], grids[0], "conv_in")] + [(l["channels"], l["grid"], l["part"]) for l in levels[:0]]]

    # skip connections listed again in order (the pop above emptied them)
    skips2 = [(widths[0], grids[0], "conv_in")]
    for i, w in enumerate(widths):
        skips2 += [(w, grids[i], f"down {i}")] * lpb
        if i < len(widths) - 1:
            skips2.append((w, grids[i] // 2, f"down {i} downsampler"))
    skip_list = [dict(channels=c, grid=g, made_by=m) for c, g, m in skips2]

    # ---- the conditioning: time vector, pooled text, size numbers ---------------------------------------------
    line(U, 555, "time_embed_dim = time_embedding_dim or block_out_channels[0] * 4")
    line(U, 558, "timestep_input_dim = block_out_channels[0]")
    line(U, 672, "self.add_time_proj = Timesteps(addition_time_embed_dim")
    line(U, 673, "self.add_embedding = TimestepEmbedding(projection_class_embeddings_input_dim, time_embed_dim)")
    line(P, 728, "add_time_ids = list(original_size + crops_coords_top_left + target_size)")
    time_vec = widths[0] * 4
    n_size_numbers = 6  # original_size (2) + crops_coords_top_left (2) + target_size (2), line 728
    pooled = te2["projection_dim"]
    added = pooled + n_size_numbers * unet["addition_time_embed_dim"]
    assert added == unet["projection_class_embeddings_input_dim"], (added, unet["projection_class_embeddings_input_dim"])

    # ---- the text side ---------------------------------------------------------------------------------------
    line(P, 402, "pooled_prompt_embeds = prompt_embeds[0]")
    line(P, 404, "prompt_embeds = prompt_embeds.hidden_states[-2]")
    line(P, 411, "prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)")
    word = te1["hidden_size"] + te2["hidden_size"]
    assert word == unet["cross_attention_dim"]
    assert te1["max_position_embeddings"] == te2["max_position_embeddings"]
    tokens = te1["max_position_embeddings"]
    text = dict(tokens=tokens, vocab=te1["vocab_size"],
                encoder_1=dict(name="CLIP ViT-L text encoder", cls=idx["text_encoder"][1], width=te1["hidden_size"],
                               layers=te1["num_hidden_layers"], heads=te1["num_attention_heads"],
                               layer_read=te1["num_hidden_layers"] - 1),
                encoder_2=dict(name="OpenCLIP bigG text encoder", cls=idx["text_encoder_2"][1], width=te2["hidden_size"],
                               layers=te2["num_hidden_layers"], heads=te2["num_attention_heads"],
                               layer_read=te2["num_hidden_layers"] - 1, pooled=pooled),
                word_vectors=[tokens, word])

    # ---- the VAE ---------------------------------------------------------------------------------------------
    factor = 2 ** (len(vae["block_out_channels"]) - 1)
    assert vae["sample_size"] // factor == unet["sample_size"]
    vae_n = dict(latent=[vae["latent_channels"], unet["sample_size"], unet["sample_size"]],
                 pixels=[vae["out_channels"], vae["sample_size"], vae["sample_size"]], factor=factor,
                 scaling_factor=vae["scaling_factor"], numbers_latent=vae["latent_channels"] * unet["sample_size"] ** 2,
                 numbers_pixels=vae["out_channels"] * vae["sample_size"] ** 2)

    # ---- one transformer block's shapes at each level that has one ------------------------------------------
    line(A, 759, "inner_dim = int(dim * mult)")
    line(A, 750, "mult: int = 4")
    line(AP, 141, "self.scale = dim_head**-0.5")
    block_shapes = {}
    for g, w, h in ((grids[1], widths[1], heads[1]), (grids[2], widths[2], heads[2])):
        block_shapes[str(g)] = dict(grid=g, patches=g * g, channels=w, heads=h, head_dim=w // h,
                                    self_q=[w, w], self_kv=[w, w], cross_q=[w, w], cross_kv=[word, w],
                                    ff=[w, 2 * 4 * w, w], ff_inner=4 * w, self_matrix=[g * g, g * g],
                                    cross_matrix=[g * g, tokens], scale=(w // h) ** -0.5)

    # ---- the schedule, as diffusers' DDIMScheduler builds it from the scheduler config -----------------------
    n_steps = 50
    betas = np.linspace(sch["beta_start"] ** 0.5, sch["beta_end"] ** 0.5, sch["num_train_timesteps"], dtype=np.float32) ** 2
    ac = np.cumprod(1.0 - betas, axis=0)
    assert sch["timestep_spacing"] == "leading"
    ratio = sch["num_train_timesteps"] // n_steps
    ts = (np.arange(0, n_steps) * ratio).round()[::-1].astype(np.int64) + sch["steps_offset"]
    steps = []
    for i, t in enumerate(ts):
        a = float(ac[t])
        a_co3 = float(np.concatenate([[1.0], ac])[t])  # CO3's alpha(t) reads alphas_cumprod[t - 1]
        steps.append(dict(step=i, t=int(t), alpha_bar=a, sqrt_alpha_bar=a ** 0.5, sqrt_one_minus=(1 - a) ** 0.5,
                          share_pct=100 * a, alpha_bar_co3=a_co3))
    co3 = json.loads(CO3_JSON.read_text())
    for s, c in zip(steps, co3["steps"]):
        assert s["t"] == c["t"] and abs(s["alpha_bar_co3"] - c["alpha_bar"]) < 1e-7, (s, c)
    shown = {0: 0.6, 5: 1.7, 20: 18, 45: 92}
    for k, v in shown.items():
        nd = 1 if v < 10 else 0
        assert round(steps[k]["share_pct"], nd) == v, (k, steps[k]["share_pct"], v)
        assert round(100 * steps[k]["alpha_bar_co3"], nd) == v, (k, steps[k]["alpha_bar_co3"], v)

    # ---- Z2's capture: which steps, which layers, which columns ----------------------------------------------
    bars = ast.parse((Z2 / "z2_bars.py").read_text())
    consts = {}
    for node in bars.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            try:
                consts[node.targets[0].id] = ast.literal_eval(node.value)
            except ValueError:
                pass
    import re
    n_corr = int(re.search(r"tuple\(range\((\d+)\)\)", line(Z2 / "z2_bars.py", 29, "CORRECTED_STEPS = tuple(range(")).group(1))
    corrected = tuple(range(n_corr))
    read_steps = list(corrected) + [10, 15, 20, 25, 30, 35, 40, 45]
    line(Z2 / "z2_bars.py", 30, "READ_STEPS = CORRECTED_STEPS + (10, 15, 20, 25, 30, 35, 40, 45)")
    zp = Z2 / "z2_predictors.py"
    line(zp, 53, "class KeepCrossAttention:")
    line(zp, 65, "p = probs.view(batch, attn.heads, seq, -1)[self.entry].mean(0).float()")
    line(zp, 102, 'name.endswith("attn2.processor")')
    line(zp, 113, '"dA": (eps[2] - eps[0]).cpu()')
    rec = json.loads((SMOKE / "seed-6001.json").read_text())
    txt = json.loads((SMOKE / "text.json").read_text())
    assert rec["steps"]["0"]["layers"] == n_tb, (rec["steps"]["0"]["layers"], n_tb)
    assert rec["captured_steps"] == read_steps
    d = np.load(SMOKE / "seed-6001-maps.npz")
    assert list(d["steps"]) == read_steps and d["maps"].shape == (len(read_steps), 4, unet["sample_size"], unet["sample_size"])
    maps = d["maps"].astype(np.float64)
    panel_steps = [3, 20, 40]
    real = {}
    for s in panel_steps:
        k = read_steps.index(s)
        real[str(s)] = {name: dict(min=float(maps[k, j].min()), max=float(maps[k, j].max()),
                                   mean_share_per_layer=float(maps[k, j].mean() / n_tb))
                        for j, name in enumerate(["eagle", "fox", "end-of-text", "padding"])}
    z2 = dict(read_steps=read_steps, corrected_steps=list(corrected), read_timesteps=[steps[s]["t"] for s in read_steps],
              layers_captured=rec["steps"]["0"]["layers"], token_columns=rec["token_columns"], prompt=rec["prompt"],
              seed=rec["seed"], wandb=json.loads((SMOKE.parent / "summary.json").read_text())["wandb_run"],
              text_cosine_joint=txt["text_joint"], text_cosine_alone=txt["text_alone"], text_cosine_pooled=txt["text_pooled"],
              dt_cos={str(s): rec["steps"][str(s)]["dt_cos"] for s in panel_steps},
              maps_panel=real, daam_cut=consts["DAAM_CUT"])

    # ---- the hand-worked head: 4 canvas patches, 3 words, 2 numbers per vector ------------------------------
    words = ["eagle", "and", "fox"]
    patches = ["p1 eagle's head", "p2 eagle's body", "p3 fox", "p4 snow"]
    Q = np.array([[2, 0], [2, 1], [1, 2], [-1, -1]], dtype=float)
    K = np.array([[2, 0], [0, 0], [0, 2]], dtype=float)
    V = np.array([[1, 0], [0, 0], [0, 1]], dtype=float)
    dk = Q.shape[1]
    S = Q @ K.T / np.sqrt(dk)
    W = np.stack([softmax(r) for r in S])
    O = W @ V
    toy = dict(words=words, patches=patches, d=dk, Q=Q.tolist(), K=K.tolist(), V=V.tolist(),
               scores=S.round(4).tolist(), weights=W.round(4).tolist(), outputs=O.round(4).tolist(),
               row_sums=W.sum(1).round(6).tolist(), eagle_column=W[:, 0].round(4).tolist(), fox_column=W[:, 2].round(4).tolist(),
               grid=[[0, 3], [1, 2]])

    out = dict(
        about="Every size and value the SDXL zoom ladder prints. Built by sdxl_numbers.py; never edit by hand.",
        sources=dict(snapshot="mscluster:~/.cache/huggingface/hub/models--stabilityai--stable-diffusion-xl-base-1.0/"
                              "snapshots/462165984030d82259a11f4367a4eed129e94a7b",
                     diffusers="0.29.2, mscluster /home-mscluster/mmolefe/miniforge3/envs/co3/lib/python3.10/site-packages/diffusers",
                     transformers="4.44.1 (same environment)", z2_code=str(zp), smoke=str(SMOKE), picture=str(PICTURE)),
        unet=dict(in_channels=unet["in_channels"], out_channels=unet["out_channels"], sample_size=unet["sample_size"],
                  widths=widths, grids=grids, heads=heads, head_dim=head_dim, layers_per_block=lpb,
                  transformer_layers_per_block=tl, levels=levels, resblocks=n_res, transformer_blocks=n_tb,
                  cross_attention_layers=n_tb, transformer_blocks_by_grid={str(k): v for k, v in by_grid.items()},
                  skips=skip_list, up_resblock_inputs=up_inputs, time_features=widths[0], time_vector=time_vec,
                  size_numbers=n_size_numbers, size_features=n_size_numbers * unet["addition_time_embed_dim"],
                  added_input=added, cross_attention_dim=unet["cross_attention_dim"],
                  linear_projection=unet["use_linear_projection"]),
        text=text, vae=vae_n, block=block_shapes,
        schedule=dict(source="DDIMScheduler rebuilt in numpy from scheduler_config.json: scaled_linear betas, "
                             "leading spacing, steps_offset 1", beta_start=sch["beta_start"], beta_end=sch["beta_end"],
                      train_steps=sch["num_train_timesteps"], sampling_steps=n_steps, steps=steps,
                      shown_steps=list(shown)),
        z2=z2, toy=toy)
    (HERE / "sdxl_numbers.json").write_text(json.dumps(out, indent=1) + "\n")
    print(f"resblocks {n_res}, transformer blocks {n_tb} {by_grid}, up inputs {up_inputs}")
    print("shares", {k: round(steps[k]['share_pct'], 2) for k in shown}, "toy weights", W.round(3).tolist())
    print("wrote sdxl_numbers.json")


if __name__ == "__main__":
    main()
