"""Let CO3 build without xformers: diffusers' own attention in its place, and the choice on record.

CO3's Co3.__init__ (upstream composers/Co3.py, line 67) calls pipe.enable_xformers_memory_efficient_attention().
`co3_bw`, the only environment that drives the Blackwell cards, has no xformers, so diffusers raises
ModuleNotFoundError there and CO3 never builds (paper2-co3-load.log on mscluster112, 2026-11-26). `co3` has xformers but
cannot drive a Blackwell card. The owner's decision: when xformers cannot be imported, that one call becomes a no-op
that prints one line, and CO3 keeps diffusers' default attention processor, which is PyTorch's scaled-dot-product
attention. When xformers imports, nothing changes.

Call `use_pytorch_attention_if_no_xformers()` before CO3 is built (paper2_co3_load.py, and paper2_identity_dial.py's
co3_sampler, which train, draw and closeness-cache all use). It returns "xformers" or "pytorch-sdpa", which every run
writes beside its settings so each row's attention kernel is on record. co3_common.py, paper2_rows.py and CO3's own code
are not edited.
"""
from __future__ import annotations

MESSAGE = "xformers missing: CO3 runs on PyTorch scaled-dot-product attention"
METHOD = "enable_xformers_memory_efficient_attention"


def xformers_importable() -> bool:
    try:
        import xformers  # noqa: F401
        import xformers.ops  # noqa: F401
    except Exception:
        return False
    return True


def _skip_xformers(self, *args, **kwargs):
    print(MESSAGE, flush=True)


def use_pytorch_attention_if_no_xformers(pipeline_classes=None, available: bool | None = None) -> str:
    """Make the pipelines' xformers switch a no-op when xformers cannot be imported; return the attention in use.

    `pipeline_classes` defaults to diffusers' DiffusionPipeline and every pipeline class CO3 builds that defines the
    method itself; `available` defaults to whether xformers imports. Both are arguments so a test can pass stand-ins.
    """
    if available is None:
        available = xformers_importable()
    if available:
        return "xformers"
    if pipeline_classes is None:
        import diffusers
        pipeline_classes = [diffusers.DiffusionPipeline]
        for name in ("StableDiffusionXLPipeline",):
            cls = getattr(diffusers, name, None)
            if cls is not None and METHOD in vars(cls):
                pipeline_classes.append(cls)
    for cls in pipeline_classes:
        setattr(cls, METHOD, _skip_xformers)
    return "pytorch-sdpa"
