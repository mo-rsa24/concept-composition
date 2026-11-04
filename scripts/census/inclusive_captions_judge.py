#!/usr/bin/env python
"""Ask a vision-language model whether a training photo really holds what its set claims.

The detector-and-classifier screen counts animals and names each one. It cannot see a hybrid that
happens to come in a pair: two instances, one called cat and one called dog, passes even when the
"cat" has a dog's muzzle. SD 3.5 produces exactly that, so the pool needs a judge that looks at
the picture and answers in words.

LLaVA 1.6 Mistral 7B, already on this cluster, is asked one short question at a time rather than
for a verdict, because a model asked for a verdict gives the agreeable answer and a model asked to
count gives a number. The questions for a cat-and-dog photo:

  1. how many animals are in the picture
  2. is there exactly one cat and one dog
  3. is the cat a normal cat and the dog a normal dog
  4. is each one clearly visible rather than hidden behind the other

The third question's wording was chosen by measurement, not by how careful it sounds. Four
wordings were asked of the same 48 hand-labelled images. "with no features of the other species
mixed into it" agreed on 20 of 48, because naming the failure sends the model looking for it, and
it answered no on a clean cat beside a clean dog. "Is the cat a normal cat, and the dog a normal
dog" agreed on 45. Its three disagreements all run the same way: it keeps a marginal picture a
person rejects, and it rejects none that a person keeps, so it is slightly permissive and never
over-strict.

A photo is kept only when the count is the number its set wants and every question answers yes.
Writes judged.json with every answer, so a keep or a rejection can be read back and argued with.

Validate the judge before trusting it: run it over a set you have labelled by eye and compare.

    /home-mscluster/mmolefe/miniforge3/envs/co3/bin/python \
      scripts/census/inclusive_captions_judge.py --kind both --dir <a folder of png> --limit 48
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from PIL import Image

MODEL_ID = "llava-hf/llava-v1.6-mistral-7b-hf"

QUESTIONS = {
    "both": [
        ("count", "How many animals are in this picture? Answer with a single number."),
        ("pair", "Is there exactly one cat and exactly one dog in this picture? "
                 "Answer yes or no."),
        ("clean", "Is the cat in this picture a normal cat, and the dog a normal dog? "
                  "Answer yes or no."),
        ("distinct", "Are both animals clearly visible as separate animals, rather than one "
                     "hiding the other? Answer yes or no."),
    ],
    "cat": [
        ("count", "How many animals are in this picture? Answer with a single number."),
        ("pair", "Is the only animal in this picture a cat? Answer yes or no."),
        ("clean", "Is it an ordinary cat, a real live animal rather than a drawing, a statue, a "
                  "toy or an animal on a screen? Answer yes or no."),
        ("distinct", "Is the cat clearly visible rather than small or hidden? Answer yes or no."),
    ],
    "dog": [
        ("count", "How many animals are in this picture? Answer with a single number."),
        ("pair", "Is the only animal in this picture a dog? Answer yes or no."),
        ("clean", "Is it an ordinary dog, a real live animal rather than a drawing, a statue, a "
                  "toy or an animal on a screen? Answer yes or no."),
        ("distinct", "Is the dog clearly visible rather than small or hidden? Answer yes or no."),
    ],
}
WANTED_COUNT = {"both": 2, "cat": 1, "dog": 1}


class Judge:
    """One loaded LLaVA, asked one question at a time about one picture."""

    def __init__(self, device: str = "cuda") -> None:
        from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor

        self.processor = LlavaNextProcessor.from_pretrained(MODEL_ID)
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            MODEL_ID, torch_dtype=torch.float16, low_cpu_mem_usage=True).to(device).eval()
        self.device = device

    def ask(self, image: Image.Image, question: str) -> str:
        prompt = f"[INST] <image>\n{question} [/INST]"
        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=20, do_sample=False)
        text = self.processor.decode(out[0], skip_special_tokens=True)
        return text.split("[/INST]")[-1].strip()


def yes(answer: str) -> bool:
    return bool(re.match(r"\s*yes\b", answer, re.IGNORECASE))


def number(answer: str) -> int | None:
    words = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "zero": 0}
    m = re.search(r"\d+", answer)
    if m:
        return int(m.group())
    for word, value in words.items():
        if re.search(rf"\b{word}\b", answer, re.IGNORECASE):
            return value
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--kind", choices=sorted(QUESTIONS), required=True)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--limit", type=int, default=0, help="judge only the first N, for validation")
    ap.add_argument("--out", default=None, help="where judged.json goes; defaults beside the images")
    args = ap.parse_args()

    directory = Path(args.dir)
    images = sorted(directory.glob("*.png"))
    if args.limit:
        images = images[:args.limit]
    print(f"{len(images)} images to judge in {directory}", flush=True)

    # One row written per image, as it is judged, because a node going down mid-pass should
    # cost the images it had left rather than every image it had already looked at.
    target = Path(args.out) if args.out else directory / "judged.json"
    partial = target.with_suffix(".jsonl")
    done = {}
    if partial.exists():
        for line in partial.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                done[row["file"]] = row
        print(f"{len(done)} already judged, resuming", flush=True)
    images = [p for p in images if str(p) not in done]

    judge = Judge() if images else None
    wanted = WANTED_COUNT[args.kind]
    rows, kept = list(done.values()), sum(r["verdict"] == "keep" for r in done.values())
    log = partial.open("a")
    for i, path in enumerate(images):
        with Image.open(path) as im:
            rgb = im.convert("RGB")
            answers = {name: judge.ask(rgb, q) for name, q in QUESTIONS[args.kind]}
        n = number(answers["count"])
        verdict = ("keep" if n == wanted and all(yes(answers[k])
                                                 for k in ("pair", "clean", "distinct"))
                   else "reject")
        kept += verdict == "keep"
        row = {"file": str(path), "n_animals": n, "answers": answers, "verdict": verdict}
        rows.append(row)
        log.write(json.dumps(row) + "\n")
        log.flush()
        if (i + 1) % 10 == 0:
            print(f"{i + 1} judged, {kept} kept", flush=True)

    log.close()
    target.write_text(json.dumps(
        {"kind": args.kind, "model": MODEL_ID, "n": len(rows), "kept": kept, "rows": rows},
        indent=2))
    print(f"done: {kept} of {len(rows)} kept, written to {target}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
