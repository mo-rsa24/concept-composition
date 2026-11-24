# SDXL down the layers: the code

The lines that compute each level of the ladder, in the order the levels are read. Each block is copied byte for byte
from the file it lives in, with its line numbers, under the equation or sentence it computes. `gen_codewalk.py` writes
this page and refuses to if any block's first or last line has moved.

**Which code.** diffusers 0.29.2, the version the `co3` environment on mscluster runs
(`/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python`, transformers 4.44.1). The
files were copied read-only into `source/diffusers-0.29.2/` on 2026-11-24. CO3's lines come from the authors' checkout at
`/home/molef/PhD/replications/co3`, commit `35ea972313276014716dca3d821023728e8bc917`. Z2's capture comes from
`2-classifier-guidance/scripts/leakage/z2_predictors.py`.

**Words this uses.** Z2 is Loop 12's step that reads three things inside SDXL (text similarity, attention overlap and
the $d_t$ maps) to find which one predicts that a pair of animals borrows parts. Its design is in
[the borrowing design note](../../../../artifacts/notes/progress/leakage/02-what-predicts-borrowing/design.md).

## Level 0: the prompt becomes word vectors and a pooled vector

Each encoder runs on the 77 token ids. SDXL keeps the layer before last of each (`hidden_states[-2]`), joins them along
the width (768 + 1280 = 2048 per slot), and keeps bigG's
projected end-of-text vector as the pooled vector (1280). CO3's `encode_prompt` in
`composers/utils/ptp_utils.py` (lines 42 to 67) does the same.

`/home-mscluster/mmolefe/miniforge3/envs/co3/lib/python3.10/site-packages/diffusers/pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl.py`, lines 399 to 411

```python
 399                  prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=True)
 400  
 401                  # We are only ALWAYS interested in the pooled output of the final text encoder
 402                  pooled_prompt_embeds = prompt_embeds[0]
 403                  if clip_skip is None:
 404                      prompt_embeds = prompt_embeds.hidden_states[-2]
 405                  else:
 406                      # "2" because SDXL always indexes from the penultimate layer.
 407                      prompt_embeds = prompt_embeds.hidden_states[-(clip_skip + 2)]
 408  
 409                  prompt_embeds_list.append(prompt_embeds)
 410  
 411              prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
```

## Level 0: the timestep and the pooled vector join into one time vector

$t$ becomes 320 sine and cosine features, then a small network makes the time vector
(1280). The pooled vector (1280) and 6 size numbers (each turned
into 256 features) are joined into 2816 numbers, sent through a second
small network to 1280, and added to the time vector.

`/home-mscluster/mmolefe/miniforge3/envs/co3/lib/python3.10/site-packages/diffusers/models/unets/unet_2d_condition.py`, lines 925 to 925

```python
 925          t_emb = self.time_proj(timesteps)
```

`/home-mscluster/mmolefe/miniforge3/envs/co3/lib/python3.10/site-packages/diffusers/models/unets/unet_2d_condition.py`, lines 970 to 980

```python
 970              text_embeds = added_cond_kwargs.get("text_embeds")
 971              if "time_ids" not in added_cond_kwargs:
 972                  raise ValueError(
 973                      f"{self.__class__} has the config param `addition_embed_type` set to 'text_time' which requires the keyword argument `time_ids` to be passed in `added_cond_kwargs`"
 974                  )
 975              time_ids = added_cond_kwargs.get("time_ids")
 976              time_embeds = self.add_time_proj(time_ids.flatten())
 977              time_embeds = time_embeds.reshape((text_embeds.shape[0], -1))
 978              add_embeds = torch.concat([text_embeds, time_embeds], dim=-1)
 979              add_embeds = add_embeds.to(emb.dtype)
 980              aug_emb = self.add_embedding(add_embeds)
```

`/home-mscluster/mmolefe/miniforge3/envs/co3/lib/python3.10/site-packages/diffusers/models/unets/unet_2d_condition.py`, lines 1135 to 1135

```python
1135          emb = self.time_embedding(t_emb, timestep_cond)
```

`/home-mscluster/mmolefe/miniforge3/envs/co3/lib/python3.10/site-packages/diffusers/models/unets/unet_2d_condition.py`, lines 1152 to 1152

```python
1152          emb = emb + aug_emb if aug_emb is not None else emb
```

## Figure 0.5a: the U-Net's shape is read from its config

`attention_head_dim` in SDXL's config is `[5, 10, 20]`, and `num_attention_heads` is empty, so line 241 takes the list
as the number of heads per level. Each head is then 64 numbers wide (line 806 of `unet_2d_blocks.py`
divides the channels by the heads). Up blocks get one ResBlock more than down blocks (line 439), which is how the up side
uses every skip.

`/home-mscluster/mmolefe/miniforge3/envs/co3/lib/python3.10/site-packages/diffusers/models/unets/unet_2d_condition.py`, lines 241 to 241

```python
 241          num_attention_heads = num_attention_heads or attention_head_dim
```

`/home-mscluster/mmolefe/miniforge3/envs/co3/lib/python3.10/site-packages/diffusers/models/unets/unet_2d_condition.py`, lines 353 to 367

```python
 353              is_final_block = i == len(block_out_channels) - 1
 354  
 355              down_block = get_down_block(
 356                  down_block_type,
 357                  num_layers=layers_per_block[i],
 358                  transformer_layers_per_block=transformer_layers_per_block[i],
 359                  in_channels=input_channel,
 360                  out_channels=output_channel,
 361                  temb_channels=blocks_time_embed_dim,
 362                  add_downsample=not is_final_block,
 363                  resnet_eps=norm_eps,
 364                  resnet_act_fn=act_fn,
 365                  resnet_groups=norm_num_groups,
 366                  cross_attention_dim=cross_attention_dim[i],
 367                  num_attention_heads=num_attention_heads[i],
```

`/home-mscluster/mmolefe/miniforge3/envs/co3/lib/python3.10/site-packages/diffusers/models/unets/unet_2d_condition.py`, lines 439 to 439

```python
 439                  num_layers=reversed_layers_per_block[i] + 1,
```

`/home-mscluster/mmolefe/miniforge3/envs/co3/lib/python3.10/site-packages/diffusers/models/unets/unet_2d_condition.py`, lines 555 to 558

```python
 555              time_embed_dim = time_embedding_dim or block_out_channels[0] * 4
 556  
 557              self.time_proj = Timesteps(block_out_channels[0], flip_sin_to_cos, freq_shift)
 558              timestep_input_dim = block_out_channels[0]
```

## Figure 0.5a: the path through the U-Net in the forward pass

`conv_in` takes the 4-channel latent to 320 channels; each down block hands its outputs to
`down_block_res_samples` (the skips); each up block takes its share back; `conv_out` returns 4 channels,
the noise guess.

`/home-mscluster/mmolefe/miniforge3/envs/co3/lib/python3.10/site-packages/diffusers/models/unets/unet_2d_condition.py`, lines 1162 to 1162

```python
1162          sample = self.conv_in(sample)
```

`/home-mscluster/mmolefe/miniforge3/envs/co3/lib/python3.10/site-packages/diffusers/models/unets/unet_2d_condition.py`, lines 1202 to 1223

```python
1202          for downsample_block in self.down_blocks:
1203              if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
1204                  # For t2i-adapter CrossAttnDownBlock2D
1205                  additional_residuals = {}
1206                  if is_adapter and len(down_intrablock_additional_residuals) > 0:
1207                      additional_residuals["additional_residuals"] = down_intrablock_additional_residuals.pop(0)
1208  
1209                  sample, res_samples = downsample_block(
1210                      hidden_states=sample,
1211                      temb=emb,
1212                      encoder_hidden_states=encoder_hidden_states,
1213                      attention_mask=attention_mask,
1214                      cross_attention_kwargs=cross_attention_kwargs,
1215                      encoder_attention_mask=encoder_attention_mask,
1216                      **additional_residuals,
1217                  )
1218              else:
1219                  sample, res_samples = downsample_block(hidden_states=sample, temb=emb)
1220                  if is_adapter and len(down_intrablock_additional_residuals) > 0:
1221                      sample += down_intrablock_additional_residuals.pop(0)
1222  
1223              down_block_res_samples += res_samples
```

`/home-mscluster/mmolefe/miniforge3/envs/co3/lib/python3.10/site-packages/diffusers/models/unets/unet_2d_condition.py`, lines 1262 to 1266

```python
1262          for i, upsample_block in enumerate(self.up_blocks):
1263              is_final_block = i == len(self.up_blocks) - 1
1264  
1265              res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
1266              down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]
```

`/home-mscluster/mmolefe/miniforge3/envs/co3/lib/python3.10/site-packages/diffusers/models/unets/unet_2d_condition.py`, lines 1296 to 1296

```python
1296          sample = self.conv_out(sample)
```

## Figure 0.5b: one transformer stack

GroupNorm, then the 1280 × 32 × 32 feature map becomes a row of 1024 patches and goes through a
linear layer (`use_linear_projection` is true in SDXL's config). The blocks run in order, then the row is folded back to
32 × 32 and the stack's input is added back.

`/home-mscluster/mmolefe/miniforge3/envs/co3/lib/python3.10/site-packages/diffusers/models/transformers/transformer_2d.py`, lines 404 to 407

```python
 404          if self.is_input_continuous:
 405              batch_size, _, height, width = hidden_states.shape
 406              residual = hidden_states
 407              hidden_states, inner_dim = self._operate_on_continuous_inputs(hidden_states)
```

`/home-mscluster/mmolefe/miniforge3/envs/co3/lib/python3.10/site-packages/diffusers/models/transformers/transformer_2d.py`, lines 479 to 492

```python
 479      def _operate_on_continuous_inputs(self, hidden_states):
 480          batch, _, height, width = hidden_states.shape
 481          hidden_states = self.norm(hidden_states)
 482  
 483          if not self.use_linear_projection:
 484              hidden_states = self.proj_in(hidden_states)
 485              inner_dim = hidden_states.shape[1]
 486              hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * width, inner_dim)
 487          else:
 488              inner_dim = hidden_states.shape[1]
 489              hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * width, inner_dim)
 490              hidden_states = self.proj_in(hidden_states)
 491  
 492          return hidden_states, inner_dim
```

`/home-mscluster/mmolefe/miniforge3/envs/co3/lib/python3.10/site-packages/diffusers/models/transformers/transformer_2d.py`, lines 417 to 417

```python
 417          for block in self.transformer_blocks:
```

`/home-mscluster/mmolefe/miniforge3/envs/co3/lib/python3.10/site-packages/diffusers/models/transformers/transformer_2d.py`, lines 514 to 527

```python
 514      def _get_output_for_continuous_inputs(self, hidden_states, residual, batch_size, height, width, inner_dim):
 515          if not self.use_linear_projection:
 516              hidden_states = (
 517                  hidden_states.reshape(batch_size, height, width, inner_dim).permute(0, 3, 1, 2).contiguous()
 518              )
 519              hidden_states = self.proj_out(hidden_states)
 520          else:
 521              hidden_states = self.proj_out(hidden_states)
 522              hidden_states = (
 523                  hidden_states.reshape(batch_size, height, width, inner_dim).permute(0, 3, 1, 2).contiguous()
 524              )
 525  
 526          output = hidden_states + residual
 527          return output
```

## Figure 0.5b: one transformer block

LayerNorm, self-attention (patch to patch, `encoder_hidden_states` is None), add back; LayerNorm, cross-attention (keys
and values from the word vectors), add back; LayerNorm, feed-forward, add back. In SDXL `norm_type` is `layer_norm`, so
only those branches run.

`/home-mscluster/mmolefe/miniforge3/envs/co3/lib/python3.10/site-packages/diffusers/models/attention.py`, lines 433 to 433

```python
 433              norm_hidden_states = self.norm1(hidden_states)
```

`/home-mscluster/mmolefe/miniforge3/envs/co3/lib/python3.10/site-packages/diffusers/models/attention.py`, lines 453 to 464

```python
 453          attn_output = self.attn1(
 454              norm_hidden_states,
 455              encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
 456              attention_mask=attention_mask,
 457              **cross_attention_kwargs,
 458          )
 459          if self.norm_type == "ada_norm_zero":
 460              attn_output = gate_msa.unsqueeze(1) * attn_output
 461          elif self.norm_type == "ada_norm_single":
 462              attn_output = gate_msa * attn_output
 463  
 464          hidden_states = attn_output + hidden_states
```

`/home-mscluster/mmolefe/miniforge3/envs/co3/lib/python3.10/site-packages/diffusers/models/attention.py`, lines 490 to 496

```python
 490              attn_output = self.attn2(
 491                  norm_hidden_states,
 492                  encoder_hidden_states=encoder_hidden_states,
 493                  attention_mask=encoder_attention_mask,
 494                  **cross_attention_kwargs,
 495              )
 496              hidden_states = attn_output + hidden_states
```

`/home-mscluster/mmolefe/miniforge3/envs/co3/lib/python3.10/site-packages/diffusers/models/attention.py`, lines 503 to 503

```python
 503              norm_hidden_states = self.norm3(hidden_states)
```

`/home-mscluster/mmolefe/miniforge3/envs/co3/lib/python3.10/site-packages/diffusers/models/attention.py`, lines 516 to 516

```python
 516              ff_output = self.ff(norm_hidden_states)
```

`/home-mscluster/mmolefe/miniforge3/envs/co3/lib/python3.10/site-packages/diffusers/models/attention.py`, lines 523 to 523

```python
 523          hidden_states = ff_output + hidden_states
```

The feed-forward widens each patch to 5120 (`mult` = 4), through GEGLU, which makes 2 × 5120
numbers and lets one half gate the other.

`/home-mscluster/mmolefe/miniforge3/envs/co3/lib/python3.10/site-packages/diffusers/models/attention.py`, lines 750 to 750

```python
 750          mult: int = 4,
```

`/home-mscluster/mmolefe/miniforge3/envs/co3/lib/python3.10/site-packages/diffusers/models/attention.py`, lines 758 to 767

```python
 758          if inner_dim is None:
 759              inner_dim = int(dim * mult)
 760          dim_out = dim_out if dim_out is not None else dim
 761  
 762          if activation_fn == "gelu":
 763              act_fn = GELU(dim, inner_dim, bias=bias)
 764          if activation_fn == "gelu-approximate":
 765              act_fn = GELU(dim, inner_dim, approximate="tanh", bias=bias)
 766          elif activation_fn == "geglu":
 767              act_fn = GEGLU(dim, inner_dim, bias=bias)
```

## Figure 0.5c: the noise guess becomes the clean guess

$\hat x_0 = (x_t - \sqrt{1-\bar\alpha_t}\,\hat\epsilon)/\sqrt{\bar\alpha_t}$, then
$x_{t-20} = \sqrt{\bar\alpha_{t-20}}\,\hat x_0 + \sqrt{1-\bar\alpha_{t-20}}\,\hat\epsilon^{\phi}$. CO3's
plain step writes both lines itself; with `use_cfgpp` on (the released setting) the second term uses the empty prompt's
guess $\hat\epsilon^{\phi}$. Its `at` is the scheduler's $\bar\alpha$ at $t - 1$ (the CO3 ladder's
`co3_numbers.py` asserts this), which moves the share of picture at step 0 from
0.578% to 0.584%.

`/home/molef/PhD/replications/co3/composers/Co3.py`, lines 354 to 357

```python
 354          noise_pred = noise_pred_uncond + self.config.guidance_scale * (noise_pred_cond - noise_pred_uncond)
 355          denoised_tweedie = (x - (1-at).sqrt() * noise_pred) / at.sqrt()
 356          if self.config.use_cfgpp:
 357              denoised_latent = at_next.sqrt() * denoised_tweedie + (1-at_next).sqrt() * noise_pred_uncond
```

## Level 2: one attention head

`head_to_batch_dim` splits the 1280 numbers of each query, key and value into 20 heads of
64. `get_attention_scores` computes $QK^\top$ times `scale` = $1/\sqrt{64}$ and takes the
softmax along the last axis, the 77 word slots, so each patch's row sums to 1. `torch.bmm(attention_probs,
value)` is the weighted sum of values.

`/home-mscluster/mmolefe/miniforge3/envs/co3/lib/python3.10/site-packages/diffusers/models/attention_processor.py`, lines 141 to 141

```python
 141          self.scale = dim_head**-0.5 if self.scale_qk else 1.0
```

`/home-mscluster/mmolefe/miniforge3/envs/co3/lib/python3.10/site-packages/diffusers/models/attention_processor.py`, lines 625 to 651

```python
 625          dtype = query.dtype
 626          if self.upcast_attention:
 627              query = query.float()
 628              key = key.float()
 629  
 630          if attention_mask is None:
 631              baddbmm_input = torch.empty(
 632                  query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype, device=query.device
 633              )
 634              beta = 0
 635          else:
 636              baddbmm_input = attention_mask
 637              beta = 1
 638  
 639          attention_scores = torch.baddbmm(
 640              baddbmm_input,
 641              query,
 642              key.transpose(-1, -2),
 643              beta=beta,
 644              alpha=self.scale,
 645          )
 646          del baddbmm_input
 647  
 648          if self.upcast_softmax:
 649              attention_scores = attention_scores.float()
 650  
 651          attention_probs = attention_scores.softmax(dim=-1)
```

`/home-mscluster/mmolefe/miniforge3/envs/co3/lib/python3.10/site-packages/diffusers/models/attention_processor.py`, lines 804 to 823

```python
 804          query = attn.to_q(hidden_states)
 805  
 806          if encoder_hidden_states is None:
 807              encoder_hidden_states = hidden_states
 808          elif attn.norm_cross:
 809              encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
 810  
 811          key = attn.to_k(encoder_hidden_states)
 812          value = attn.to_v(encoder_hidden_states)
 813  
 814          query = attn.head_to_batch_dim(query)
 815          key = attn.head_to_batch_dim(key)
 816          value = attn.head_to_batch_dim(value)
 817  
 818          attention_probs = attn.get_attention_scores(query, key, attention_mask)
 819          hidden_states = torch.bmm(attention_probs, value)
 820          hidden_states = attn.batch_to_head_dim(hidden_states)
 821  
 822          # linear proj
 823          hidden_states = attn.to_out[0](hidden_states)
```

The sampler itself runs `AttnProcessor2_0` (the default when PyTorch has `scaled_dot_product_attention`, lines 227 and
228), which computes the same softmax inside one fused call and never returns the weights:

`/home-mscluster/mmolefe/miniforge3/envs/co3/lib/python3.10/site-packages/diffusers/models/attention_processor.py`, lines 1584 to 1586

```python
1584          hidden_states = F.scaled_dot_product_attention(
1585              query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
1586          )
```

## Level 2: Z2 keeps the weights

`KeepCrossAttention` runs the explicit path above for the cross-attention layers only (names ending in
`attn2.processor`), averages the heads for the prompt's entry of the batch (`[self.entry].mean(0)`), and keeps four
columns: the two nouns, the end-of-text slot and the mean of the padding slots. `capture` makes one extra U-Net call on
[empty, prompt, A alone, B alone] at each read step, upscales each layer's 77-column map to
128 × 128, sums over the 70 layers, and takes $d_t(A) = \epsilon(A) -
\epsilon(\varnothing)$ from the same call.

`/home/molef/PhD/concept-composition/2-classifier-guidance/scripts/leakage/z2_predictors.py`, lines 53 to 70

```python
  53  class KeepCrossAttention:
  54      """Plain attention; for the batch entry `entry`, keeps the head-mean probabilities of the chosen key columns."""
  55  
  56      def __init__(self, store, name, entry, cols):
  57          self.store, self.name, self.entry, self.cols = store, name, entry, cols
  58  
  59      def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None, **kwargs):
  60          batch, seq, _ = hidden_states.shape
  61          query = attn.head_to_batch_dim(attn.to_q(hidden_states))
  62          key = attn.head_to_batch_dim(attn.to_k(encoder_hidden_states))
  63          value = attn.head_to_batch_dim(attn.to_v(encoder_hidden_states))
  64          probs = attn.get_attention_scores(query, key, attention_mask)
  65          p = probs.view(batch, attn.heads, seq, -1)[self.entry].mean(0).float()  # seq x 77
  66          a, b, eot, pad_from = self.cols
  67          kept = torch.stack([p[:, a], p[:, b], p[:, eot], p[:, pad_from:].mean(1)], 0)  # 4 x seq
  68          self.store.append(kept)
  69          out = attn.batch_to_head_dim(torch.bmm(probs, value))
  70          return attn.to_out[1](attn.to_out[0](out))
```

`/home/molef/PhD/concept-composition/2-classifier-guidance/scripts/leakage/z2_predictors.py`, lines 95 to 113

```python
  95      @torch.no_grad()
  96      def capture(self, x, timestep):
  97          emb, pool = self.model.text_embeds[0][:4], self.model.text_embeds[1][:4]
  98          added = {"time_ids": self.model.add_time_ids.repeat(4, 1), "text_embeds": pool}
  99          saved = dict(self.inner.attn_processors)
 100          store, procs = [], {}
 101          for name, proc in saved.items():
 102              procs[name] = KeepCrossAttention(store, name, 1, self.cols) if name.endswith("attn2.processor") else proc
 103          self.inner.set_attn_processor(procs)
 104          try:
 105              eps = self.inner(x.repeat(4, 1, 1, 1), timestep, encoder_hidden_states=emb, added_cond_kwargs=added)["sample"].float()
 106          finally:
 107              self.inner.set_attn_processor(saved)
 108          maps = torch.zeros(len(COLUMNS), GRID, GRID, device=x.device)
 109          for kept in store:
 110              side = int(round(kept.shape[1] ** 0.5))
 111              maps += F.interpolate(kept.view(1, len(COLUMNS), side, side), size=(GRID, GRID), mode="bilinear",
 112                                    align_corners=False)[0]
 113          return {"maps": maps.cpu(), "dA": (eps[2] - eps[0]).cpu(), "dB": (eps[3] - eps[0]).cpu(), "layers": len(store)}
```

