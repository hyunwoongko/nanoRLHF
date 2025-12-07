import torch
import triton

from nanorlhf.kernels.flash_attn_decode.reduce_k import flash_attn_decode_kernel_reduce_k
from nanorlhf.kernels.flash_attn_decode.split_k import flash_attn_decode_kernel_split_k


def _get_split_k(
    seq_len_q,
    seq_len_k,
    block_size_k,
    max_split_k=32,
    min_interactions_per_split=4096,
):
    min_chunk = 4 * block_size_k
    if seq_len_k <= min_chunk:
        return 1

    max_splits_from_length = seq_len_k // min_chunk
    if max_splits_from_length <= 1:
        return 1

    split_k = 1
    while True:
        next_split = split_k * 2
        if next_split > max_splits_from_length or next_split > max_split_k:
            break
        k_per_split = (seq_len_k + next_split - 1) // next_split
        interactions = seq_len_q * k_per_split
        if interactions < min_interactions_per_split:
            break
        split_k = next_split

    return split_k


def flash_attn_decode(q, k, v, split_k=None, causal=True, softmax_scale=None, block_size_q=16, block_size_k=16):
    assert q.ndim == 4 and k.ndim == 4 and v.ndim == 4
    bsz, num_heads, seq_len_q, dim = q.shape
    _, _, seq_len_k, dim_k = k.shape
    assert dim == dim_k and k.shape == v.shape

    device = q.device
    bh = bsz * num_heads

    def merge_heads(x):
        return x.contiguous().view(bh, x.shape[2], dim)

    q_merged = merge_heads(q)
    k_merged = merge_heads(k)
    v_merged = merge_heads(v)

    if softmax_scale is None:
        softmax_scale = 1.0 / (dim ** 0.5)

    if seq_len_q == seq_len_k:
        split_k = 1
    elif split_k is None:
        # automatically determine split_k if not provided.
        split_k = _get_split_k(seq_len_q, seq_len_k, block_size_k)

    block_n_per_split = (seq_len_k + split_k - 1) // split_k
    seq_len_q_ceil = triton.cdiv(seq_len_q, block_size_q) * block_size_q

    ez_dot_v = torch.empty(
        (bh, split_k, seq_len_q_ceil, dim),
        device=device,
        dtype=torch.float32,
    )
    max_q = torch.empty(
        (bh, split_k, seq_len_q_ceil),
        device=device,
        dtype=torch.float32,
    )
    ez_sum = torch.empty_like(max_q)
    o = torch.empty_like(q_merged)

    stride_q_bh, stride_q_seq, stride_q_dim = q_merged.stride()
    stride_k_bh, stride_k_seq, stride_k_dim = k_merged.stride()
    stride_v_bh, stride_v_seq, stride_v_dim = v_merged.stride()
    stride_ez_dot_v_bh, stride_ez_dot_v_split, stride_ez_dot_v_seq, stride_ez_dot_v_dim = ez_dot_v.stride()
    stride_max_q_bh, stride_max_q_split, stride_max_q_seq = max_q.stride()
    stride_ez_sum_bh, stride_ez_sum_split, stride_ez_sum_seq = ez_sum.stride()
    stride_o_out_bh, stride_o_out_seq, stride_o_out_dim = o.stride()

    grid_split_k = triton.cdiv(seq_len_q, block_size_q), bh, split_k
    flash_attn_decode_kernel_split_k[grid_split_k](
        q_merged, k_merged, v_merged, ez_dot_v,
        max_q, ez_sum,
        seq_len_q, seq_len_k,
        stride_q_bh, stride_q_seq, stride_q_dim,
        stride_k_bh, stride_k_seq, stride_k_dim,
        stride_v_bh, stride_v_seq, stride_v_dim,
        stride_ez_dot_v_bh, stride_ez_dot_v_split, stride_ez_dot_v_seq, stride_ez_dot_v_dim,
        stride_max_q_bh, stride_max_q_split, stride_max_q_seq,
        stride_ez_sum_bh, stride_ez_sum_split, stride_ez_sum_seq,
        softmax_scale,
        block_n_per_split,
        causal=causal,
        dim=dim,
        block_size_q=block_size_q,
        block_size_k=block_size_k,
    )

    grid_reduce_k = (bh, triton.cdiv(seq_len_q, block_size_q))
    flash_attn_decode_kernel_reduce_k[grid_reduce_k](
        ez_dot_v, max_q, ez_sum, o,
        seq_len_q,
        stride_ez_dot_v_bh, stride_ez_dot_v_split, stride_ez_dot_v_seq, stride_ez_dot_v_dim,
        stride_max_q_bh, stride_max_q_split, stride_max_q_seq,
        stride_ez_sum_bh, stride_ez_sum_split, stride_ez_sum_seq,
        stride_o_out_bh, stride_o_out_seq, stride_o_out_dim,
        dim=dim,
        block_size_q=block_size_q,
        split_k=split_k,
        causal=causal,
    )
    o = o.view(bsz, num_heads, seq_len_q, dim)
    return o
