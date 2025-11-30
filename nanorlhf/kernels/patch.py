import importlib
from functools import partial

from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from nanorlhf.kernels.api import rms_norm, apply_rotary_pos_emb
from nanorlhf.kernels.utils.huggingface import flash_attention_forward


def patch_kernel(model):
    # patch flash attention kernel
    if "nanoRLHF" not in ALL_ATTENTION_FUNCTIONS:
        ALL_ATTENTION_FUNCTIONS["nanoRLHF"] = flash_attention_forward
    if not hasattr(model.config, "_attention_implementation"):
        model.config._attention_implementation = "nanoRLHF"

    # patch rms norm kernel
    for module in model.modules():
        if "RMSNorm" in module.__class__.__qualname__:
            rms_eps = getattr(module, "eps", None)
            if rms_eps is None:
                rms_eps = getattr(module, "variance_epsilon", 1e-6)
            if hasattr(module, "weight"):
                module.forward = partial(rms_norm, weight=module.weight, eps=rms_eps)

    # patch rotary position embedding kernel
    modeling_module = importlib.import_module(model.__class__.__module__)
    if hasattr(modeling_module, "apply_rotary_pos_emb"):
        modeling_module.apply_rotary_pos_emb = apply_rotary_pos_emb

    return model
