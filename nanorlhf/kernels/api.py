from nanorlhf.kernels.flash_attn.ops import FlashAttentionFunc
from nanorlhf.kernels.flash_attn_varlen.ops import FlashAttnVarlenFunc
from nanorlhf.kernels.rmsnorm.ops import FusedRMSNormFunc
from nanorlhf.kernels.rotary.ops import apply_rotary_func
from nanorlhf.kernels.utils.padding import pad_input as _pad_input, unpad_input as _unpad_input


def flash_attn_func(q, k, v, causal=True, softmax_scale=None, **kwargs):
    return FlashAttentionFunc.apply(q, k, v, causal, softmax_scale)


def flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, causal=True, softmax_scale=None, **kwargs):
    return FlashAttnVarlenFunc.apply(q, k, v, cu_seqlens_q, cu_seqlens_k, causal, softmax_scale)


def rms_norm(x, weight, eps=1e-6):
    return FusedRMSNormFunc.apply(x, weight, eps)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    return apply_rotary_func(q, k, cos, sin, position_ids, unsqueeze_dim)


def pad_input(hidden_states, indices, batch, seqlen):
    return _pad_input(hidden_states, indices, batch, seqlen)


def unpad_input(hidden_states, attention_mask, unused_mask=None):
    return _unpad_input(hidden_states, attention_mask, unused_mask)

