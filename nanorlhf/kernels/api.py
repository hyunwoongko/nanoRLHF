from nanorlhf.kernels.flash_attn.ops import FlashAttentionFunc
from nanorlhf.kernels.flash_attn_varlen.ops import FlashAttnVarlenFunc
from nanorlhf.kernels.rmsnorm.ops import FusedRMSNormFunc
from nanorlhf.kernels.utils.padding import pad_input as _pad_input, unpad_input as _unpad_input
from nanorlhf.kernels.flash_attn_decode.ops import flash_attn_decode


def flash_attn_func(q, k, v, attention_mask=None, causal=True, softmax_scale=None, **kwargs):
    """
    Flash Attention function.

    Args:
        q (torch.Tensor): Query tensor of shape (..., seq_len_q, head_dim).
        k (torch.Tensor): Key tensor of shape (..., seq_len_k, head_dim).
        v (torch.Tensor): Value tensor of shape (..., seq_len_k, head_dim).
        attention_mask (torch.Tensor, optional): Attention mask tensor.
        causal (bool, optional): Whether to apply causal masking. Default is True.
        softmax_scale (float, optional): Scaling factor for softmax. Default is None.

    Returns:
        torch.Tensor: Output tensor after applying Flash Attention.
    """
    return FlashAttentionFunc.apply(q, k, v, attention_mask, causal, softmax_scale)


def flash_attn_varlen_func(q, k, v, cu_seq_lens_q, cu_seq_lens_k, max_seq_len_q, max_seq_len_k, causal=True, softmax_scale=None, **kwargs):
    """
    Flash Attention function for variable-length sequences.

    Args:
        q (torch.Tensor): Query tensor of shape (total_q_len, head_dim).
        k (torch.Tensor): Key tensor of shape (total_k_len, head_dim).
        v (torch.Tensor): Value tensor of shape (total_k_len, head_dim).
        cu_seq_lens_q (torch.Tensor): Cumulative sequence lengths for queries.
        cu_seq_lens_k (torch.Tensor): Cumulative sequence lengths for keys/values.
        max_seq_len_q (int): Maximum sequence length for queries.
        max_seq_len_k (int): Maximum sequence length for keys/values.
        causal (bool, optional): Whether to apply causal masking. Default is True.
        softmax_scale (float, optional): Scaling factor for softmax. Default is None.

    Returns:
        torch.Tensor: Output tensor after applying Flash Attention on variable-length sequences.
    """
    return FlashAttnVarlenFunc.apply(q, k, v, cu_seq_lens_q, cu_seq_lens_k, max_seq_len_q, max_seq_len_k, causal, softmax_scale)


def flash_attn_decode_func(q, k, v, split_k=None, causal=True, softmax_scale=None, block_size_q=16, block_size_k=16):
    """
    Flash Attention decode function.

    Args:
        q (torch.Tensor): Query tensor.
        k (torch.Tensor): Key tensor.
        v (torch.Tensor): Value tensor.
        split_k (int, optional): Split factor for keys. Default is None.
        causal (bool, optional): Whether to apply causal masking. Default is True.
        softmax_scale (float, optional): Scaling factor for softmax. Default is None.
        block_size_q (int, optional): Block size for queries. Default is 16.
        block_size_k (int, optional): Block size for keys. Default is 16.

    Returns:
        torch.Tensor: Output tensor after applying Flash Attention decode.
    """
    return flash_attn_decode(q, k, v, split_k, causal, softmax_scale, block_size_q, block_size_k)


def rms_norm(x, weight, eps=1e-6):
    """
    Fused RMS Normalization function.

    Args:
        x (torch.Tensor): Input tensor.
        weight (torch.Tensor): Weight tensor.
        eps (float, optional): Epsilon for numerical stability. Default is 1e-6.

    Returns:
        torch.Tensor: Normalized tensor.
    """
    return FusedRMSNormFunc.apply(x, weight, eps)


def pad_input(hidden_states, indices, batch, seq_len):
    """
    Pad input sequences based on provided indices.

    Args:
        hidden_states (torch.Tensor): Input tensor of shape (total_nnz, ...).
        indices (torch.Tensor): Indices of non-masked tokens.
        batch (int): Batch size.
        seq_len (int): Sequence length.

    Returns:
        torch.Tensor: Padded tensor of shape (batch, seq_len, ...).
    """
    return _pad_input(hidden_states, indices, batch, seq_len)


def unpad_input(hidden_states, attention_mask, unused_mask=None):
    """
    Unpad input sequences based on attention mask.

    Args:
        hidden_states (torch.Tensor): Input tensor of shape (batch, seq_len, ...).
        attention_mask (torch.Tensor): Attention mask tensor.
        unused_mask (torch.Tensor, optional): Unused mask tensor. Default is None.

    Returns:
        torch.Tensor: Unpadded tensor of shape (total_nnz, ...).
    """
    return _unpad_input(hidden_states, attention_mask, unused_mask)

