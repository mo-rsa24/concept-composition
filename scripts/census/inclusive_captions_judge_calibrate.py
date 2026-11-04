#!/usr/bin/env python
"""Try several wordings of the hybrid question and score each against labels made by eye.

The first judge kept 9 of 48 images a person kept 37 of. Three of its four questions were right:
the count said two animals on 44 of 48 and caught both the three-animal frame and the single
animal, and the one-cat-one-dog question caught the two-cat frames. The fourth question, which
asked whether each animal had "no features of the other species mixed into it", answered no on 39
of 48 including pictures holding an ordinary cat beside an ordinary dog. A question that names the
failure invites the model to find it.

So the wording is measured rather than argued about. Each candidate is asked of the same 48
images, its answers are combined with the three questions that already work, and the result is
scored against the calls in checks/eye-labels-first-48.json.

    /home-mscluster/mmolefe/miniforge3/envs/co3/bin/python \
      scripts/census/inclusive_captions_judge_calibrate.py \
        --judged <judged_first48.json> --labels <eye-labels-first-48.json>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.census.inclusive_captions_judge import Judge, number, yes  # noqa: E402

CANDIDATES = {
    "A_mixed_in": ("Look closely at each animal. Is each one an ordinary cat or an ordinary dog, "
                   "with no features of the other species mixed into it? Answer yes or no.", True),
    "B_any_hybrid": ("Does this picture contain an animal that is a mix of a cat and a dog, with "
                     "the body of one and the face of the other? Answer yes or no.", False),
    "C_normal_each": ("Is the cat in this picture a normal cat, and the dog a normal dog? "
                      "Answer yes or no.", True),
    "D_list": ("List every animal in this picture, one per line, naming its species.", None),
}


def list_is_pair(answer: str) -> bool:
    low = answer.lower()
    return low.count("cat") == 1 and low.count("dog") == 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--judged", required=True, help="the first run's judged_first48.json")
    ap.add_argument("--labels", required=True, help="the eye's calls, as a json of rejects")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    judged = json.loads(Path(args.judged).read_text())
    labels = json.loads(Path(args.labels).read_text())
    rejected_by_eye = {int(k) for k in labels["rejects"]}

    judge = Judge()
    rows = judged["rows"]
    extra: dict[str, list[str]] = {name: [] for name in CANDIDATES}
    for i, row in enumerate(rows):
        with Image.open(row["file"]) as im:
            rgb = im.convert("RGB")
            for name, (question, _) in CANDIDATES.items():
                extra[name].append(judge.ask(rgb, question))
        if (i + 1) % 10 == 0:
            print(f"{i + 1} re-asked", flush=True)

    report = {}
    for name, (_, keep_on_yes) in CANDIDATES.items():
        agree = keeps = 0
        disagreements = []
        for i, row in enumerate(rows):
            answer = extra[name][i]
            if keep_on_yes is None:
                clean = list_is_pair(answer)
            else:
                clean = yes(answer) if keep_on_yes else not yes(answer)
            keep = (number(row["answers"]["count"]) == 2
                    and yes(row["answers"]["pair"]) and clean)
            keeps += keep
            eye_keep = i not in rejected_by_eye
            if keep == eye_keep:
                agree += 1
            else:
                disagreements.append({"tile": i, "judge": "keep" if keep else "reject",
                                      "eye": "keep" if eye_keep else "reject",
                                      "answer": answer[:120]})
        report[name] = {"agree": agree, "of": len(rows), "judge_keeps": keeps,
                        "eye_keeps": len(rows) - len(rejected_by_eye),
                        "disagreements": disagreements}
        print(f"{name}: agrees on {agree} of {len(rows)}, keeps {keeps}", flush=True)

    Path(args.out).write_text(json.dumps(
        {"candidates": {k: v[0] for k, v in CANDIDATES.items()},
         "answers": extra, "report": report}, indent=2))
    print(f"written to {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
