from nanorlhf.kernels.flash_attn.ops import FlashAttentionFunc
from nanorlhf.kernels.flash_attn_varlen.ops import FlashAttnVarlenFunc


def flash_attn_func(q, k, v, causal=True, softmax_scale=None):
    return FlashAttentionFunc.apply(q, k, v, causal, softmax_scale)


def flash_attn_varlen_func(q, k, v, cu_q, cu_k, causal=True, softmax_scale=None):
    return FlashAttnVarlenFunc.apply(q, k, v, cu_q, cu_k, causal, softmax_scale)
