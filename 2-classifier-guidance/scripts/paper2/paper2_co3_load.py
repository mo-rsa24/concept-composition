"""Build CO3 from the fetched weights, offline, and point it at one pair, without drawing a picture.

Setup plan task 8 (plans/paper-2-dials/plans/01-setup.md). Run only through paper2_co3_load.sh. Loads
CO3 exactly as its row does (paper2_rows.py: load_parser, load, set_prompt, start_noise, which come from
method_search/co3_common.py), with HF_HUB_OFFLINE=1, so any file not in HF_HOME raises instead of
downloading. CO3 turns on xformers attention as it loads, which needs a visible card, so the build runs
on the held node's device 0. Writes the package versions, the device, the seconds to load and to set a
prompt, the concepts CO3's parser found, and the peak memory, to --out (load.txt).
"""
from __future__ import annotations

import argparse
import importlib.metadata as md
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
for p in (HERE.parent / "method_search", HERE.parent / "rung2"):
    sys.path.insert(0, str(p))


def version(pkg: str) -> str:
    try:
        return md.version(pkg)
    except md.PackageNotFoundError:
        return "not installed"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--pair", default="a cheetah and a cougar")
    a = ap.parse_args(argv)
    assert os.environ.get("HF_HUB_OFFLINE") == "1", "run through paper2_co3_load.sh: HF_HUB_OFFLINE must be 1"
    import torch
    import paper2_rows as rows
    import co3_common

    lines = [f"python {sys.executable}", f"co3_common {Path(co3_common.__file__).resolve()}",
             f"HF_HOME {os.environ.get('HF_HOME')}"]
    lines += [f"{p} {version(p)}" for p in ("torch", "diffusers", "transformers", "xformers", "stanza",
                                            "huggingface-hub", "accelerate")]
    t0 = time.time()
    parser = rows.load_parser()
    t1 = time.time()
    model, config = rows.load(a.pair, parser, a.out.parent / "co3-scratch")
    t2 = time.time()
    n = rows.set_prompt(model, config, a.pair, parser)
    noise = rows.start_noise(model, 13001)
    t3 = time.time()
    lines += [f"device {model.unet.device}, unet dtype {model.unet.dtype}",
              f"seconds: parser {t1 - t0:.1f}, CO3 build {t2 - t1:.1f}, set prompt and start noise {t3 - t2:.1f}",
              f"prompt {a.pair!r}: {n} concept prompts, CO3 strings {config.prompt!r}",
              f"start noise {tuple(noise.shape)} {noise.dtype}",
              f"timesteps {len(model.scheduler.timesteps)}, corrected steps {config.num_ts_to_correct}",
              f"peak card memory {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB" if torch.cuda.is_available()
              else "peak card memory: no card"]
    for line in lines:
        print(line, flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("\n".join(lines) + "\n")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
