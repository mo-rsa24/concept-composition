"""CO3 at the settings of its released scripts (commit 35ea972), which the owner chose on board question 16.

The authors' benchmark script `sample_co3_promptset.sh` on their `rare_bench` branch (commit aacbb8f) uses the same
settings. Prompts are split into concepts by the function their `sample_rarebench.py` calls.

The release imports `composers.utils.gaussian_smoothing`, which neither branch ships, and never uses it. A stub stands in
for that one import, so no file of the authors' code changes, and it raises if anything ever calls it.
"""
import sys
import types

import stanza
import torch


class _MissingGaussianSmoothing:
    def __init__(self, *args, **kwargs):
        raise RuntimeError("the released CO3 code does not ship GaussianSmoothing, and this run must not reach it")


_stub = types.ModuleType("composers.utils.gaussian_smoothing")
_stub.GaussianSmoothing = _MissingGaussianSmoothing
sys.modules["composers.utils.gaussian_smoothing"] = _stub

from composers.Co3 import Co3  # noqa: E402
from composers.config import Co3Config  # noqa: E402
from composers.utils.ptp_utils import get_prompts_and_concepts_coarse  # noqa: E402

RELEASED_SETTINGS = dict(guidance_scale=0.8, use_cfgpp=True, n_timesteps=50, sd_version="xl", resolution_h=1024,
                         resolution_w=1024, negative_prompt="", num_ts_to_correct=6, num_latent_corrector_steps=5,
                         num_resampling_steps=3, corrector_algo="co3-hybrid", modulate_comp_weights=True, beta=0.9,
                         lmda=0.8)


def load_parser():
    """The parser `sample_rarebench.py` builds."""
    return stanza.Pipeline("en", processors="tokenize,pos,lemma,depparse", download_method=None)


def co3_strings(prompt, parser):
    """The authors' `get_prompts_and_concepts_string`: the concept prompts, then the full prompt, joined by plus signs."""
    prompts, concepts = get_prompts_and_concepts_coarse(prompt, parser, remove_adj_from_contrastive_prompts=True,
                                                        remove_art_from_contrastive_prompts=True)
    return prompts, "+".join(prompts) + f"+{prompt}", "+".join([*concepts, prompt])


def load(first_prompt, parser, out):
    _, prompt_string, concept_string = co3_strings(first_prompt, parser)
    config = Co3Config(prompt=prompt_string, prompt_orig=first_prompt, concept=concept_string, output_path=str(out),
                       output_path_all=str(out), **RELEASED_SETTINGS)
    model = Co3(config)
    model.latentCorrector.verbose = False
    return model, config


def set_prompt(model, config, prompt, parser):
    """Point the model at one prompt, as the authors' loop does. Returns the number of concepts found."""
    prompts, prompt_string, concept_string = co3_strings(prompt, parser)
    config.prompt, config.prompt_orig, config.concept = prompt_string, prompt, concept_string
    model.config = config
    model.prepare_prompts(config)
    model.prepare_embeds()
    model.init_sampling()
    return len(prompts)


def start_noise(model, seed):
    """The same draw the RTD renderer makes for this seed, so the methods share a starting latent."""
    generator = torch.Generator(device=model.unet.device).manual_seed(seed)
    noise = torch.randn((1, 4, 128, 128), generator=generator, device=model.unet.device, dtype=model.unet.dtype)
    return noise * model.scheduler.init_noise_sigma
