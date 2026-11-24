"""Level 3: writes code-walk.md, the lines that compute each level of the ladder, copied byte for byte with their line
numbers from the files they live in, each block under the equation or sentence it computes. Every block's first and
last lines are asserted to hold the expected text, so a moved line stops the build.

Sources: the diffusers 0.29.2 files the co3 environment on mscluster runs (copied read-only into
source/diffusers-0.29.2/), CO3's checkout at /home/molef/PhD/replications/co3, and Loop 12's z2_predictors.py.
Run:  python3 gen_codewalk.py
"""
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
N = json.loads((HERE / "sdxl_numbers.json").read_text())
DIF = HERE / "source" / "diffusers-0.29.2"
REMOTE = "/home-mscluster/mmolefe/miniforge3/envs/co3/lib/python3.10/site-packages/diffusers/"
FILES = {
    "pipe": (DIF / "pipeline_stable_diffusion_xl.py.copy", REMOTE + "pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl.py"),
    "unet": (DIF / "unet_2d_condition.py.copy", REMOTE + "models/unets/unet_2d_condition.py"),
    "t2d": (DIF / "transformer_2d.py.copy", REMOTE + "models/transformers/transformer_2d.py"),
    "attn": (DIF / "attention.py.copy", REMOTE + "models/attention.py"),
    "proc": (DIF / "attention_processor.py.copy", REMOTE + "models/attention_processor.py"),
    "co3": (Path("/home/molef/PhD/replications/co3/composers/Co3.py"), "/home/molef/PhD/replications/co3/composers/Co3.py"),
    "z2": (Path(N["sources"]["z2_code"]), N["sources"]["z2_code"]),
}


def block(key, a, b, first, last=None):
    path, shown = FILES[key]
    lines = path.read_text().splitlines()
    assert first in lines[a - 1], f"{path.name}:{a} no longer holds {first!r}: {lines[a - 1]!r}"
    if last:
        assert last in lines[b - 1], f"{path.name}:{b} no longer holds {last!r}: {lines[b - 1]!r}"
    body = "\n".join(f"{n:>4}  {lines[n - 1]}" for n in range(a, b + 1))
    return f"`{shown}`, lines {a} to {b}\n\n```python\n{body}\n```\n"


U, T, Z, B = N["unet"], N["text"], N["z2"], N["block"]["32"]
co3_commit = subprocess.run(["git", "-C", "/home/molef/PhD/replications/co3", "rev-parse", "HEAD"], capture_output=True,
                            text=True).stdout.strip()

out = [f"""# SDXL down the layers: the code

The lines that compute each level of the ladder, in the order the levels are read. Each block is copied byte for byte
from the file it lives in, with its line numbers, under the equation or sentence it computes. `gen_codewalk.py` writes
this page and refuses to if any block's first or last line has moved.

**Which code.** diffusers {N['sources']['diffusers'].split(',')[0]}, the version the `co3` environment on mscluster runs
(`/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python`, transformers {N['sources']['transformers'].split(' ')[0]}). The
files were copied read-only into `source/diffusers-0.29.2/` on 2026-11-24. CO3's lines come from the authors' checkout at
`/home/molef/PhD/replications/co3`, commit `{co3_commit}`. Z2's capture comes from
`2-classifier-guidance/scripts/leakage/z2_predictors.py`.

**Words this uses.** Z2 is Loop 12's step that reads three things inside SDXL (text similarity, attention overlap and
the $d_t$ maps) to find which one predicts that a pair of animals borrows parts. Its design is in
[the borrowing design note](../../../../artifacts/notes/progress/leakage/02-what-predicts-borrowing/design.md).

## Level 0: the prompt becomes word vectors and a pooled vector

Each encoder runs on the 77 token ids. SDXL keeps the layer before last of each (`hidden_states[-2]`), joins them along
the width ({T['encoder_1']['width']} + {T['encoder_2']['width']} = {T['word_vectors'][1]} per slot), and keeps bigG's
projected end-of-text vector as the pooled vector ({T['encoder_2']['pooled']}). CO3's `encode_prompt` in
`composers/utils/ptp_utils.py` (lines 42 to 67) does the same.

{block('pipe', 399, 411, 'prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=True)', 'prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)')}
## Level 0: the timestep and the pooled vector join into one time vector

$t$ becomes {U['time_features']} sine and cosine features, then a small network makes the time vector
({U['time_vector']}). The pooled vector ({T['encoder_2']['pooled']}) and {U['size_numbers']} size numbers (each turned
into {U['size_features'] // U['size_numbers']} features) are joined into {U['added_input']} numbers, sent through a second
small network to {U['time_vector']}, and added to the time vector.

{block('unet', 925, 925, 't_emb = self.time_proj(timesteps)')}
{block('unet', 970, 980, 'text_embeds = added_cond_kwargs.get("text_embeds")', 'aug_emb = self.add_embedding(add_embeds)')}
{block('unet', 1135, 1135, 'emb = self.time_embedding(t_emb, timestep_cond)')}
{block('unet', 1152, 1152, 'emb = emb + aug_emb if aug_emb is not None else emb')}
## Figure 0.5a: the U-Net's shape is read from its config

`attention_head_dim` in SDXL's config is `{U['heads']}`, and `num_attention_heads` is empty, so line 241 takes the list
as the number of heads per level. Each head is then {U['head_dim'][1]} numbers wide (line 806 of `unet_2d_blocks.py`
divides the channels by the heads). Up blocks get one ResBlock more than down blocks (line 439), which is how the up side
uses every skip.

{block('unet', 241, 241, 'num_attention_heads = num_attention_heads or attention_head_dim')}
{block('unet', 353, 367, 'is_final_block = i == len(block_out_channels) - 1', 'num_attention_heads=num_attention_heads[i],')}
{block('unet', 439, 439, 'num_layers=reversed_layers_per_block[i] + 1,')}
{block('unet', 555, 558, 'time_embed_dim = time_embedding_dim or block_out_channels[0] * 4', 'timestep_input_dim = block_out_channels[0]')}
## Figure 0.5a: the path through the U-Net in the forward pass

`conv_in` takes the {U['in_channels']}-channel latent to {U['widths'][0]} channels; each down block hands its outputs to
`down_block_res_samples` (the skips); each up block takes its share back; `conv_out` returns {U['out_channels']} channels,
the noise guess.

{block('unet', 1162, 1162, 'sample = self.conv_in(sample)')}
{block('unet', 1202, 1223, 'for downsample_block in self.down_blocks:', 'down_block_res_samples += res_samples')}
{block('unet', 1262, 1266, 'for i, upsample_block in enumerate(self.up_blocks):', 'down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]')}
{block('unet', 1296, 1296, 'sample = self.conv_out(sample)')}
## Figure 0.5b: one transformer stack

GroupNorm, then the {B['channels']} × 32 × 32 feature map becomes a row of {B['patches']} patches and goes through a
linear layer (`use_linear_projection` is true in SDXL's config). The blocks run in order, then the row is folded back to
32 × 32 and the stack's input is added back.

{block('t2d', 404, 407, 'if self.is_input_continuous:', 'hidden_states, inner_dim = self._operate_on_continuous_inputs(hidden_states)')}
{block('t2d', 479, 492, 'def _operate_on_continuous_inputs(self, hidden_states):', 'return hidden_states, inner_dim')}
{block('t2d', 417, 417, 'for block in self.transformer_blocks:')}
{block('t2d', 514, 527, 'def _get_output_for_continuous_inputs', 'return output')}
## Figure 0.5b: one transformer block

LayerNorm, self-attention (patch to patch, `encoder_hidden_states` is None), add back; LayerNorm, cross-attention (keys
and values from the word vectors), add back; LayerNorm, feed-forward, add back. In SDXL `norm_type` is `layer_norm`, so
only those branches run.

{block('attn', 433, 433, 'norm_hidden_states = self.norm1(hidden_states)')}
{block('attn', 453, 464, 'attn_output = self.attn1(', 'hidden_states = attn_output + hidden_states')}
{block('attn', 490, 496, 'attn_output = self.attn2(', 'hidden_states = attn_output + hidden_states')}
{block('attn', 503, 503, 'norm_hidden_states = self.norm3(hidden_states)')}
{block('attn', 516, 516, 'ff_output = self.ff(norm_hidden_states)')}
{block('attn', 523, 523, 'hidden_states = ff_output + hidden_states')}
The feed-forward widens each patch to {B['ff_inner']} (`mult` = 4), through GEGLU, which makes 2 × {B['ff_inner']}
numbers and lets one half gate the other.

{block('attn', 750, 750, 'mult: int = 4,')}
{block('attn', 758, 767, 'if inner_dim is None:', 'act_fn = GEGLU(dim, inner_dim, bias=bias)')}
## Figure 0.5c: the noise guess becomes the clean guess

$\\hat x_0 = (x_t - \\sqrt{{1-\\bar\\alpha_t}}\\,\\hat\\epsilon)/\\sqrt{{\\bar\\alpha_t}}$, then
$x_{{t-20}} = \\sqrt{{\\bar\\alpha_{{t-20}}}}\\,\\hat x_0 + \\sqrt{{1-\\bar\\alpha_{{t-20}}}}\\,\\hat\\epsilon^{{\\phi}}$. CO3's
plain step writes both lines itself; with `use_cfgpp` on (the released setting) the second term uses the empty prompt's
guess $\\hat\\epsilon^{{\\phi}}$. Its `at` is the scheduler's $\\bar\\alpha$ at $t - 1$ (the CO3 ladder's
`co3_numbers.py` asserts this), which moves the share of picture at step 0 from
{100 * N['schedule']['steps'][0]['alpha_bar']:.3f}% to {100 * N['schedule']['steps'][0]['alpha_bar_co3']:.3f}%.

{block('co3', 354, 357, 'noise_pred = noise_pred_uncond + self.config.guidance_scale * (noise_pred_cond - noise_pred_uncond)', 'denoised_latent = at_next.sqrt() * denoised_tweedie + (1-at_next).sqrt() * noise_pred_uncond')}
## Level 2: one attention head

`head_to_batch_dim` splits the {B['channels']} numbers of each query, key and value into {B['heads']} heads of
{B['head_dim']}. `get_attention_scores` computes $QK^\\top$ times `scale` = $1/\\sqrt{{{B['head_dim']}}}$ and takes the
softmax along the last axis, the {T['tokens']} word slots, so each patch's row sums to 1. `torch.bmm(attention_probs,
value)` is the weighted sum of values.

{block('proc', 141, 141, 'self.scale = dim_head**-0.5 if self.scale_qk else 1.0')}
{block('proc', 625, 651, 'dtype = query.dtype', 'attention_probs = attention_scores.softmax(dim=-1)')}
{block('proc', 804, 823, 'query = attn.to_q(hidden_states)', 'hidden_states = attn.to_out[0](hidden_states)')}
The sampler itself runs `AttnProcessor2_0` (the default when PyTorch has `scaled_dot_product_attention`, lines 227 and
228), which computes the same softmax inside one fused call and never returns the weights:

{block('proc', 1584, 1586, 'hidden_states = F.scaled_dot_product_attention(')}
## Level 2: Z2 keeps the weights

`KeepCrossAttention` runs the explicit path above for the cross-attention layers only (names ending in
`attn2.processor`), averages the heads for the prompt's entry of the batch (`[self.entry].mean(0)`), and keeps four
columns: the two nouns, the end-of-text slot and the mean of the padding slots. `capture` makes one extra U-Net call on
[empty, prompt, A alone, B alone] at each read step, upscales each layer's {T['tokens']}-column map to
{U['sample_size']} × {U['sample_size']}, sums over the {Z['layers_captured']} layers, and takes $d_t(A) = \\epsilon(A) -
\\epsilon(\\varnothing)$ from the same call.

{block('z2', 53, 70, 'class KeepCrossAttention:', 'return attn.to_out[1](attn.to_out[0](out))')}
{block('z2', 95, 113, '@torch.no_grad()', 'return {"maps": maps.cpu()')}
"""]
(HERE / "code-walk.md").write_text("".join(out))
print("wrote code-walk.md")
