from functools import partial

from transformers import modeling_utils, masking_utils

from nanorlhf.kernels.api import rms_norm
from nanorlhf.kernels.utils.huggingface import flash_attention_forward
from nanorlhf.kernels.utils.vllm import paged_flash_attention_forward


def patch_kernel(model, use_paged_attention=False):
    """
    Patch the model to use optimized kernels for attention and RMS normalization.

    Args:
        model: The model to be patched.
        use_paged_attention (bool): Whether to use paged attention kernel. Defaults to False.

    Returns:
        The patched model.
    """
    # patch flash attention kernel
    if use_paged_attention:
        attn_impl = "nanoRLHF_paged"
        attn_func = paged_flash_attention_forward
        mask_func = lambda *args, **kwargs: None
    else:
        attn_impl = "nanoRLHF"
        attn_func = flash_attention_forward
        mask_func = masking_utils.eager_mask

    if attn_impl not in modeling_utils.ALL_ATTENTION_FUNCTIONS:
        modeling_utils.ALL_ATTENTION_FUNCTIONS._global_mapping[attn_impl] = attn_func
    if attn_impl not in masking_utils.ALL_MASK_ATTENTION_FUNCTIONS:
        masking_utils.ALL_MASK_ATTENTION_FUNCTIONS._global_mapping[attn_impl] = mask_func
    if hasattr(model.config, "_attn_implementation"):
        model.config._attn_implementation = attn_impl

    # patch rms norm kernel
    for module in model.modules():
        if "RMSNorm" in module.__class__.__qualname__:
            rms_eps = getattr(module, "eps", None)
            if rms_eps is None:
                rms_eps = getattr(module, "variance_epsilon", 1e-6)
            if hasattr(module, "weight"):
                module.forward = partial(rms_norm, weight=module.weight, eps=rms_eps)

    return model
