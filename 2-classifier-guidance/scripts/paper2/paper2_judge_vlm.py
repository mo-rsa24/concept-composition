"""Vision-language model on crops: Qwen3-VL-8B-Instruct asked three yes-or-no questions about each body.

Species judge plan, "The judges", vision-language row; run as step 0.20 (`paper2-judge-vlm`).

Each crop is asked "Is this a {A}?", then "Is this a {B}?", then "Does this animal have body
parts of a {the other}?", where the other is whichever asked species got the lower answer to its
own question. Each answer's probability of yes is read from the first answer token, as Soft-TIFA
reads its answers: the yes tokens' probability mass over the yes and no tokens' together. A body
is called other when both species answers fall below t, mixed when both reach t, else the species
that reached it, and flagged when the body-parts answer reaches t.

The two-clear score is the geometric mean of the yes probabilities rule 4 needs: one body a yes
to A, the other a yes to B, and both a no to the body-parts question (one minus its yes), taking
the better of the two ways of assigning the two bodies. Rule 4 needs exactly two bodies, so a
picture with any other number of bodies scores 0.

    python3 paper2_judge_vlm.py --pictures list.csv --out <dir> --dry-run
    python paper2_judge_vlm.py --pictures list.csv --bodies bodies.json --out \
        /datasets/mmolefe/poe_repair_min/paper2/paper2-judge-vlm --yes-threshold 0.5
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from paper2_common import Judge, base_parser, run, with_article  # noqa: E402

VLM_NAME = "Qwen/Qwen3-VL-8B-Instruct"
YES_THRESHOLD_GRID = (0.3, 0.5, 0.7)
ANSWER_HINT = " Answer yes or no."


class StandIn:
    """Fixed answers: the first body a clear A, the second a clear B, every later one neither."""

    FIXED = ({"yes_a": 0.92, "yes_b": 0.08, "yes_parts": 0.10},
             {"yes_a": 0.12, "yes_b": 0.88, "yes_parts": 0.20},
             {"yes_a": 0.05, "yes_b": 0.06, "yes_parts": 0.05})

    def __init__(self):
        self.k = 0

    def reset(self):
        self.k = 0

    def __call__(self, crop, a, b):
        raw = dict(self.FIXED[min(self.k, 2)])
        raw["parts_of"] = "B" if raw["yes_a"] >= raw["yes_b"] else "A"
        self.k += 1
        return raw


class QwenAsker:
    def __init__(self, device):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self.torch = torch
        self.device = device
        self.model = AutoModelForImageTextToText.from_pretrained(
            VLM_NAME, torch_dtype=torch.bfloat16).to(device).eval()
        self.proc = AutoProcessor.from_pretrained(VLM_NAME)
        tok = self.proc.tokenizer

        def first_ids(words):
            ids = set()
            for w in words:
                t = tok.encode(w, add_special_tokens=False)
                if t:
                    ids.add(t[0])
            return sorted(ids)
        self.yes_ids = first_ids(["Yes", "yes", " Yes", " yes", "YES"])
        self.no_ids = first_ids(["No", "no", " No", " no", "NO"])

    def p_yes(self, crop, question: str) -> float:
        torch = self.torch
        messages = [{"role": "user", "content": [{"type": "image", "image": crop},
                                                 {"type": "text", "text": question + ANSWER_HINT}]}]
        inp = self.proc.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                            return_dict=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            logits = self.model(**inp).logits[0, -1].float()
        probs = logits.softmax(-1)
        y, n = float(probs[self.yes_ids].sum()), float(probs[self.no_ids].sum())
        return y / (y + n) if y + n > 0 else 0.5

    def __call__(self, crop, a, b):
        ya = self.p_yes(crop, f"Is this {with_article(a)}?")
        yb = self.p_yes(crop, f"Is this {with_article(b)}?")
        other = b if ya >= yb else a
        yp = self.p_yes(crop, f"Does this animal have body parts of {with_article(other)}?")
        return {"yes_a": ya, "yes_b": yb, "yes_parts": yp, "parts_of": "B" if other == b else "A"}


def geometric_mean(xs: list[float]) -> float:
    prod = 1.0
    for x in xs:
        prod *= max(x, 0.0)
    return prod ** (1.0 / len(xs))


class VlmJudge(Judge):
    name = "vlm"

    def __init__(self, args):
        super().__init__(args)
        self.asker = StandIn() if args.dry_run else QwenAsker(args.device)

    def settings(self):
        return {"yes_threshold": self.args.yes_threshold,
                "model": "stand-in" if self.args.dry_run else VLM_NAME,
                "questions": ["Is this {A}?", "Is this {B}?",
                              "Does this animal have body parts of {the other}?"],
                "answer_hint": ANSWER_HINT.strip()}

    def raw(self, crop, a, b):
        return self.asker(crop, a, b)

    def decide(self, raw, settings):
        t = settings["yes_threshold"]
        ya, yb = raw["yes_a"] >= t, raw["yes_b"] >= t
        if not ya and not yb:
            return "other", False
        if ya and yb:
            return "mixed", False
        return ("A" if ya else "B"), raw["yes_parts"] >= t

    def two_clear_score(self, bodies, settings):
        if len(bodies) != 2:
            return 0.0
        r0, r1 = bodies[0]["raw"], bodies[1]["raw"]
        no0, no1 = 1.0 - r0["yes_parts"], 1.0 - r1["yes_parts"]
        return max(geometric_mean([r0["yes_a"], r1["yes_b"], no0, no1]),
                   geometric_mean([r1["yes_a"], r0["yes_b"], no0, no1]))


def main(argv=None):
    ap = base_parser(__doc__.splitlines()[0])
    ap.add_argument("--yes-threshold", type=float, choices=YES_THRESHOLD_GRID, default=0.5,
                    help="t: an answer counts as yes at or above this probability")
    run(VlmJudge(ap.parse_args(argv)))


if __name__ == "__main__":
    main()
