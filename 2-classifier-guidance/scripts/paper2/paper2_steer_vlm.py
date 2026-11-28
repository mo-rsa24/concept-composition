"""The vision-language candidate for the steering judge: Qwen3-VL-8B reads rules version 4 and answers "two clear?".

Run inside the paper2-vlm environment (transformers new enough for Qwen3-VL), on one bigbatch RTX 3090, through
paper2_steer.sbatch. For every unsealed row of <run>/table.csv it shows the whole picture with the reviewers' own rubric
(paper2_sheets.rules_text(4), minus the line about the calls file), names the two asked animals, and asks whether the
picture is two clear. The answer is read as the probability of yes from the first answer token (the yes tokens' mass
over the yes and no tokens' together), as paper2_judge_vlm.py reads its answers. No threshold is set here: the fit step
of paper2_steer_judge.py learns it on the training folds only, so this candidate is scored by the same split rule.

Writes <run>/vlm/answers.json {item: p_yes} and <run>/vlm/timing.json.

    sbatch ... paper2_steer.sbatch vlm paper2_steer_vlm.py --run /datasets/mmolefe/poe_repair_min/paper2/paper2-judge-steer
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from paper2_sheets import rules_text  # noqa: E402

VLM_NAME = "Qwen/Qwen3-VL-8B-Instruct"
SIDE = 768                       # the picture is shown at 768 by 768, about 2,300 image tokens fewer than 1024


def question(a: str, b: str) -> str:
    rules = "\n".join(l for l in rules_text(4).splitlines() if "calls-template.csv" not in l)
    return (f"You are a careful reviewer applying these rules to one picture.\n\n{rules}\n\n"
            f"In this picture, A is a {a} and B is a {b}. Following the rules exactly, is this picture 'two clear': "
            f"exactly two bodies, one plain {a} and one plain {b}, neither flagged, and no other body? "
            f"Answer yes or no.")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args(argv)
    import csv

    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor
    with open(a.run / "table.csv", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["sealed"] != "True"][: a.limit]
    out = a.run / "vlm"
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda")
    model = AutoModelForImageTextToText.from_pretrained(VLM_NAME, torch_dtype=torch.bfloat16).to(dev).eval()
    proc = AutoProcessor.from_pretrained(VLM_NAME)
    tok = proc.tokenizer

    def first_ids(words):
        return sorted({tok.encode(w, add_special_tokens=False)[0] for w in words})
    yes_ids = first_ids(["Yes", "yes", " Yes", " yes", "YES"])
    no_ids = first_ids(["No", "no", " No", " no", "NO"])
    answers, t0, warm = {}, time.time(), min(8, len(rows) // 2)
    for i, r in enumerate(rows):
        if i == warm:
            torch.cuda.synchronize()
            t0 = time.time()
        im = Image.open(r["picture"]).convert("RGB").resize((SIDE, SIDE), Image.BICUBIC)
        messages = [{"role": "user", "content": [{"type": "image", "image": im},
                                                 {"type": "text", "text": question(r["a"], r["b"])}]}]
        inp = proc.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True,
                                       return_tensors="pt").to(dev)
        with torch.no_grad():
            p = model(**inp).logits[0, -1].float().softmax(-1)
        y, n = float(p[yes_ids].sum()), float(p[no_ids].sum())
        answers[r["item"]] = round(y / (y + n), 5) if y + n > 0 else 0.5
        if (i + 1) % 100 == 0:
            print(f"{i + 1}/{len(rows)}", flush=True)
    torch.cuda.synchronize()
    per = (time.time() - t0) / max(1, len(rows) - warm)
    (out / "answers.json").write_text(json.dumps(answers, indent=0))
    (out / "timing.json").write_text(json.dumps({"seconds_per_picture": round(per, 4), "side": SIDE,
                                                 "device": torch.cuda.get_device_name(0), "model": VLM_NAME,
                                                 "question_example": question("cat", "dog")}, indent=1))
    print(f"wrote {out / 'answers.json'}: {len(answers)} answers, {per:.3f} s a picture", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
