"""Time plain SDXL on one GPU: 1024x1024, 50 steps, guidance 7.5, fp16.

Prints one JSON line: card name, seconds per picture at batch 1 and per picture at batch 4, peak memory.
"""
import json
import time

import torch
from diffusers import StableDiffusionXLPipeline

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
).to("cuda")
pipe.set_progress_bar_config(disable=True)
prompt = "a photo of a cheetah and a cougar"


def run(batch, n_batches):
    g = torch.Generator("cuda").manual_seed(6001)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_batches):
        pipe(prompt=[prompt] * batch, num_inference_steps=50, guidance_scale=7.5, generator=g)
    torch.cuda.synchronize()
    return (time.time() - t0) / (batch * n_batches)


run(1, 1)  # warm-up
torch.cuda.reset_peak_memory_stats()
b1 = run(1, 4)
b4 = run(4, 2)
print(json.dumps({
    "card": torch.cuda.get_device_name(0),
    "s_per_picture_batch1": round(b1, 2),
    "s_per_picture_batch4": round(b4, 2),
    "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 1),
    "torch": torch.__version__,
}))
