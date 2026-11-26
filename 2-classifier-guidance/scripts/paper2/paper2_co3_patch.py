"""Let CO3 build without xformers: diffusers' own attention in its place, and the choice on record.

CO3 asks for xformers twice as it builds (upstream composers/Co3.py): line 67, pipe.enable_xformers_memory_efficient_attention(),
diffusers' DiffusionPipeline method; and line 79, self.unet.enable_xformers_memory_efficient_attention(), diffusers'
ModelMixin method (models/modeling_utils.py). Those are the only two. `co3_bw`, the only environment that drives the
Blackwell cards, has no xformers, so diffusers raises ModuleNotFoundError at either call and CO3 never builds
(paper2-co3-load.log on mscluster112, 2026-11-26). `co3` has xformers but cannot drive a Blackwell card. The owner's
decision: when xformers cannot be imported, both switches become a no-op that prints one line, and CO3 keeps diffusers'
default attention processor, which is PyTorch's scaled-dot-product attention. When xformers imports, nothing changes.

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


def default_classes() -> list:
    """The two classes CO3's calls reach, DiffusionPipeline and ModelMixin, plus any class CO3 builds that defines the
    switch itself (none does in diffusers 0.35.2, but a later version could)."""
    import diffusers
    from diffusers.models.modeling_utils import ModelMixin
    classes = [diffusers.DiffusionPipeline, ModelMixin]
    for name in ("StableDiffusionXLPipeline", "UNet2DConditionModel", "AutoencoderKL"):
        cls = getattr(diffusers, name, None)
        if cls is not None and METHOD in vars(cls):
            classes.append(cls)
    return classes


def use_pytorch_attention_if_no_xformers(classes=None, available: bool | None = None) -> str:
    """Make every listed class's xformers switch a no-op when xformers cannot be imported; return the attention in use.

    `classes` defaults to default_classes(); `available` defaults to whether xformers imports. Both are arguments so a
    test can pass stand-ins. "pytorch-sdpa" is returned only after every class is patched, and checked.
    """
    if available is None:
        available = xformers_importable()
    if available:
        return "xformers"
    if classes is None:
        classes = default_classes()
    for cls in classes:
        setattr(cls, METHOD, _skip_xformers)
    if any(getattr(cls, METHOD) is not _skip_xformers for cls in classes):
        raise SystemExit("the xformers switch is still live on a class CO3 calls")
    return "pytorch-sdpa"
