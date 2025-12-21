from nanorlhf.kernels.flash_attn.ops import FlashAttentionFunc
from nanorlhf.kernels.flash_attn_varlen.ops import FlashAttnVarlenFunc
from nanorlhf.kernels.rmsnorm.ops import FusedRMSNormFunc
from nanorlhf.kernels.utils.padding import pad_input as _pad_input, unpad_input as _unpad_input
from nanorlhf.kernels.flash_attn_decode.ops import flash_attn_decode


def flash_attn_func(q, k, v, attention_mask=None, causal=True, softmax_scale=None, **kwargs):
    return FlashAttentionFunc.apply(q, k, v, attention_mask, causal, softmax_scale)


def flash_attn_varlen_func(q, k, v, cu_seq_lens_q, cu_seq_lens_k, max_seq_len_q, max_seq_len_k, causal=True, softmax_scale=None, **kwargs):
    return FlashAttnVarlenFunc.apply(q, k, v, cu_seq_lens_q, cu_seq_lens_k, max_seq_len_q, max_seq_len_k, causal, softmax_scale)


def flash_attn_decode_func(q, k, v, split_k=None, causal=True, softmax_scale=None, block_size_q=16, block_size_k=16):
    return flash_attn_decode(q, k, v, split_k, causal, softmax_scale, block_size_q, block_size_k)


def rms_norm(x, weight, eps=1e-6):
    return FusedRMSNormFunc.apply(x, weight, eps)


def pad_input(hidden_states, indices, batch, seq_len):
    return _pad_input(hidden_states, indices, batch, seq_len)


def unpad_input(hidden_states, attention_mask, unused_mask=None):
    return _unpad_input(hidden_states, attention_mask, unused_mask)

