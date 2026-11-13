"""SynGen's parse of the 276 benchmark prompts, aligned to SDXL's tokenizer, for the SDXL port (Plan 2).

Plan: 1-lora-correction/artifacts/drips/answer-iclr-reviews/routes/plan-sd15-baselines-and-syngen-sdxl.md, "spaCy and the token indices".
co3 has no spaCy, and en_core_web_trf 3.6.1 needs transformers below 4.31, so the parse runs here, in the syngen
environment, through SynGen's own code: `SynGenDiffusionPipeline._extract_attribution_indices` (its three extractors,
`include_entities` False, and `unify_lists`) with `_align_indices` against SDXL's first tokenizer. Both methods read only
`doc`, `include_entities`, `tokenizer` and `_align_indices` from the pipeline, so they are bound to a small object that
carries those four, and the U-Net is never loaded.

Writes data/prompt-sets/attend-and-excite/syngen_indices_sdxl.json keyed by pair slug, for the 276 prompts and
Figure 5's two rows outside them (tent and snowy mountain, lighthouse and stormy sea). SD 1.5 and SDXL's first tokenizer
share CLIP's vocabulary, so these indices must equal the ones SynGen found itself on SD 1.5; that is asserted for every
prompt Plan 1's smoke run rendered (their sidecars record `syngen_indices`).

    /datasets/mmolefe/envs/syngen/bin/python 1-lora-correction/scripts/review_syngen_indices.py
"""

from __future__ import annotations

import glob
import json
import os
import sys
import types
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

REPO = Path(os.environ.get("POE_REPO", Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(REPO / "scripts"))
from review_baseline_cell import slug  # noqa: E402

SYNGEN_DIR = Path("/datasets/mmolefe/repos/Linguistic-Binding-in-Diffusion-Models")
SDXL = sorted(glob.glob("/home-mscluster/mmolefe/.cache/huggingface/hub/models--stabilityai--stable-diffusion-xl-base-1.0"
                        "/snapshots/*/tokenizer"))
PAIRS = REPO / "scripts" / "review_benchmark_pairs.txt"
# Figure 5's two rows outside the benchmark (group r6, as CO3's tiles for the same rows).
EXTRA = ("a tent|a snowy mountain|and|r6", "a lighthouse|a stormy sea|and|r6")
OUT = REPO / "data" / "prompt-sets" / "attend-and-excite" / "syngen_indices_sdxl.json"
SD15_SIDECARS = Path("/datasets/mmolefe/poe_repair_min/outputs/paper_review/sd15/syngen_default/r1")


def main() -> None:
    sys.path.insert(0, str(SYNGEN_DIR))
    import spacy
    from syngen_diffusion_pipeline import SynGenDiffusionPipeline as P
    from transformers import CLIPTokenizer

    assert SDXL, "no SDXL tokenizer in the Hugging Face cache"
    tok = CLIPTokenizer.from_pretrained(str(Path(SDXL[-1]).parent), subfolder="tokenizer")
    nlp = spacy.load("en_core_web_trf")
    parser = types.SimpleNamespace(tokenizer=tok, include_entities=False, doc=None)
    parser._align_indices = types.MethodType(P._align_indices, parser)

    table = {}
    for ln in PAIRS.read_text().splitlines() + list(EXTRA):
        if not ln.strip() or ln.startswith("#"):
            continue
        a, b, joiner, group = ln.strip().split("|")
        joint = f"{a} {joiner} {b}"
        parser.doc = nlp(joint)
        idx = P._extract_attribution_indices(parser, joint)
        table[slug(a, b)] = {"joint_prompt": joint, "set": group, "syngen_indices": idx}
    OUT.write_text(json.dumps(table, indent=1) + "\n")
    empty = sum(1 for v in table.values() if not v["syngen_indices"])
    by_set = {}
    for v in table.values():
        if not v["syngen_indices"]:
            by_set[v["set"]] = by_set.get(v["set"], 0) + 1
    print(f"{len(table)} prompts, {empty} with no modifier (loss zero): {by_set} -> {OUT}", flush=True)

    checked, bad = 0, []
    for k, v in table.items():
        side = SD15_SIDECARS / k / "seed09" / "syngen.json"
        if side.exists():
            sd15 = json.loads(side.read_text()).get("syngen_indices")
            checked += 1
            if sd15 != v["syngen_indices"]:
                bad.append((k, sd15, v["syngen_indices"]))
            print(f"  {k}: SD 1.5 {sd15}  SDXL {v['syngen_indices']}", flush=True)
    print(f"against SynGen's own SD 1.5 parse: {checked} prompts checked, {len(bad)} differ", flush=True)
    if bad or not checked:
        sys.exit(f"MISMATCH or nothing to check: {bad}")


if __name__ == "__main__":
    main()
