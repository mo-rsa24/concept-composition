"""Load Qwen3-VL-8B-Instruct in the vision-language environment and ask one crop one question.

Run only through paper2_vlm_env_check.sh, with the environment's own python. Offline (HF_HUB_OFFLINE=1),
so the model must already be in HF_HOME (task 6). Loads the processor and the model the way the
vision-language judge does (paper2_judge_vlm.QwenAsker, bf16 on the card), then asks one question of one
crop and prints the yes probability. The crop: the first body of one of the joint prompt's pictures on
/datasets (task 7), cut with the judges' own crop and the dry run's stand-in box, or a crop given with
--crop. Writes the versions, seconds to load, seconds per answer, peak card memory and the answer to
--out (load.txt).
"""
from __future__ import annotations

import argparse
import importlib.metadata as md
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

JOINT = Path("/datasets/mmolefe/poe_repair_min/paper2/paper2-upload-joint")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--crop", type=Path, default=None, help="a crop to ask about; default: one cut from a joint picture")
    ap.add_argument("--species", default="cheetah", help="asked as 'Is this a <species>?'")
    a = ap.parse_args(argv)
    assert os.environ.get("HF_HUB_OFFLINE") == "1", "run through paper2_vlm_env_check.sh: HF_HUB_OFFLINE must be 1"
    import torch
    from PIL import Image
    from paper2_common import STAND_IN_BOXES, crop_body, with_article
    from paper2_judge_vlm import VLM_NAME, QwenAsker

    if a.crop:
        crop, source = Image.open(a.crop).convert("RGB"), str(a.crop)
    else:
        pics = sorted(JOINT.glob("*.png"))
        if not pics:
            raise SystemExit(f"no crop given and no pictures in {JOINT} (task 7); pass --crop")
        crop, _ = crop_body(Image.open(pics[0]).convert("RGB"), STAND_IN_BOXES, 0)
        source = f"{pics[0]} body 0 (stand-in box)"
    lines = [f"python {sys.executable}"] + [f"{p} {md.version(p)}" for p in ("torch", "transformers", "accelerate")]
    lines.append(f"cuda {torch.cuda.is_available()} {torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''}")
    t0 = time.time()
    asker = QwenAsker("cuda")
    t1 = time.time()
    question = f"Is this {with_article(a.species)}?"
    p = asker.p_yes(crop, question)
    t2 = time.time()
    lines += [f"model {VLM_NAME}, loaded in {t1 - t0:.1f} seconds",
              f"crop {source}, {crop.size[0]} by {crop.size[1]}",
              f"question {question!r}: yes probability {p:.4f}, answered in {t2 - t1:.2f} seconds",
              f"peak card memory {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB"]
    for line in lines:
        print(line, flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("\n".join(lines) + "\n")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
