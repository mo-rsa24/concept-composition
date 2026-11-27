"""Fetch Paper 2's model weights into HF_HOME on /datasets, then check each loads its config offline.

Setup plan task 6 (plans/paper-2-dials/plans/01-setup.md), and with --set judges the species judge's step 0.14 (CLIP
ViT-L/14 and the ImageNet classifier convnext_large.fb_in22k_ft_in1k, which timm loads from timm/<name> on the hub). Run only through paper2_fetch_weights.sh,
which sets HF_HOME=/datasets/mmolefe/.cache/huggingface and the disk guard. Five repositories: the
plan's four, plus madebyollin/sdxl-vae-fp16-fix, which CO3 loads beside SDXL
(composition/debottam_co3/composers/Co3.py, line 69) and which an offline job could not fetch. CLIP
ViT-L/14 and the ImageNet classifier are the species judge's step 0.14.

For each: whether it was in the cache before, what was fetched, its size on disk, and whether its
config opens with HF_HUB_OFFLINE=1. Writes the table to --out (sizes.txt) and exits 1 if any config
file is missing offline. The configs are opened as files; loading them through transformers is
printed beside, and a failure there is reported, not fatal, since co3's transformers may predate
Qwen3-VL (the vision-language environment's load check covers that model).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# (repository, what to fetch, the config file that must open offline)
SETUP_MODELS = [
    ("stabilityai/stable-diffusion-xl-base-1.0",
     {"allow_patterns": ["model_index.json", "*/config.json", "scheduler/*", "tokenizer/*", "tokenizer_2/*",
                         "*fp16.safetensors"]},
     "model_index.json"),
    ("madebyollin/sdxl-vae-fp16-fix", {"allow_patterns": ["config.json", "diffusion_pytorch_model.safetensors"]},
     "config.json"),
    ("facebook/dinov2-large", {"ignore_patterns": ["*.msgpack", "*.h5", "*.onnx", "*.ot", "*.gguf"]}, "config.json"),
    ("Qwen/Qwen3-VL-8B-Instruct", {"ignore_patterns": ["*.msgpack", "*.h5", "*.onnx", "*.ot", "*.gguf"]},
     "config.json"),
    ("IDEA-Research/grounding-dino-tiny", {"ignore_patterns": ["*.msgpack", "*.h5", "*.onnx", "*.ot", "*.gguf"]},
     "config.json"),
]
# Species judge step 0.14: the two judges' models the setup fetch leaves out. timm loads its ImageNet classifier
# from the hub repository timm/<name>, into the same HF_HOME. Each repository holds its weights twice, as
# model.safetensors and pytorch_model.bin (1.71 GB each for CLIP, 0.79 GB each for the classifier, read from the hub on
# 2026-11-26); transformers and timm both load the safetensors file when it is there, so the .bin copy is skipped.
JUDGE_SKIP = ["*.bin", "*.msgpack", "*.h5", "*.onnx", "*.ot", "*.gguf"]
JUDGE_MODELS = [
    ("openai/clip-vit-large-patch14", {"ignore_patterns": JUDGE_SKIP}, "config.json"),
    ("timm/convnext_large.fb_in22k_ft_in1k", {"ignore_patterns": JUDGE_SKIP}, "config.json"),
]
# The follow-on judges after step 0 failed: BioCLIP, which open_clip loads from the hub repository (its config is
# open_clip_config.json, not config.json). DreamSim's weights do not come from the hub; paper2_fetch_followon.sh
# fetches them through the dreamsim package.
FOLLOWON_MODELS = [
    ("imageomics/bioclip", {"ignore_patterns": ["*.msgpack", "*.h5", "*.onnx", "*.ot", "*.gguf"]}, "open_clip_config.json"),
]
SETS = {"setup": SETUP_MODELS, "judges": JUDGE_MODELS, "followon": FOLLOWON_MODELS}


def repo_dir(hf_home: Path, repo: str) -> Path:
    return hf_home / "hub" / ("models--" + repo.replace("/", "--"))


def du_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    out = subprocess.run(["du", "-sb", str(path)], capture_output=True, text=True).stdout.split()
    return int(out[0]) if out else 0


def gb(n: int) -> str:
    return f"{n / 1e9:.2f} GB"


def offline_config(repo: str, filename: str) -> tuple[bool, str]:
    """Open the config file and, where transformers knows the model, load it, both offline."""
    code = f"""
import os, sys, json
os.environ['HF_HUB_OFFLINE'] = '1'
from huggingface_hub import hf_hub_download
p = hf_hub_download({repo!r}, {filename!r}, local_files_only=True)
json.load(open(p))
msg = 'config file opens offline'
if {filename!r} == 'config.json' and {repo!r} != 'madebyollin/sdxl-vae-fp16-fix':
    try:
        from transformers import AutoConfig
        AutoConfig.from_pretrained({repo!r}, local_files_only=True)
        msg += '; transformers loads it'
    except Exception as e:
        msg += '; transformers here cannot load it (' + type(e).__name__ + ')'
if {repo!r} == 'imageomics/bioclip':
    try:
        import open_clip
        open_clip.create_model_and_transforms('hf-hub:imageomics/bioclip')
        msg += '; open_clip builds it offline'
    except Exception as e:
        msg += '; open_clip cannot build it offline (' + type(e).__name__ + ': ' + str(e)[:120] + ')'
if {repo!r}.startswith('timm/'):
    try:
        import timm
        timm.create_model({repo!r}.split('/', 1)[1], pretrained=True)
        msg += '; timm builds it offline'
    except Exception as e:
        msg += '; timm cannot build it offline (' + type(e).__name__ + ')'
print(msg)
"""
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                       env={**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
    err = r.stderr.strip().splitlines()
    return r.returncode == 0, r.stdout.strip() or (err[-1] if err else "")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True, help="sizes.txt")
    ap.add_argument("--check-only", action="store_true", help="fetch nothing; report the cache and the offline checks")
    ap.add_argument("--set", choices=sorted(SETS), default="setup",
                    help="setup: task 6's five repositories; judges: CLIP ViT-L/14 and the ImageNet classifier (step 0.14)")
    a = ap.parse_args(argv)
    hf_home = Path(os.environ["HF_HOME"])
    if not str(hf_home).startswith("/datasets/"):
        raise SystemExit(f"HF_HOME is {hf_home}, not on /datasets: refusing to fetch")
    os.environ.pop("HF_HUB_OFFLINE", None)
    os.environ.pop("TRANSFORMERS_OFFLINE", None)
    from huggingface_hub import snapshot_download

    lines = [f"HF_HOME {hf_home}", f"started {time.strftime('%Y-%m-%dT%H:%M:%S')}", ""]
    header = f"{'repository':42s} {'cached before':>13s} {'size':>10s} {'seconds':>8s}  offline config"
    lines.append(header)
    ok_all = True
    for repo, patterns, cfg in SETS[a.set]:
        d = repo_dir(hf_home, repo)
        before = (d / "snapshots").is_dir() and any((d / "snapshots").iterdir())
        t0 = time.time()
        if not a.check_only:
            snapshot_download(repo, **patterns)
        secs = time.time() - t0
        ok, msg = offline_config(repo, cfg)
        ok_all &= ok
        line = f"{repo:42s} {'yes' if before else 'no':>13s} {gb(du_bytes(d)):>10s} {secs:8.0f}  {'ok' if ok else 'MISSING'}: {msg}"
        print(line, flush=True)
        lines.append(line)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("\n".join(lines) + "\n")
    print(f"wrote {a.out}")
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
