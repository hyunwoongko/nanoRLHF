from typing import Optional

import torch
import triton

from nanorlhf.kernels.flash_attn_decode.reduce_k import flash_attn_decode_kernel_reduce_k
from nanorlhf.kernels.flash_attn_decode.split_k import (
    flash_attn_decode_kernel_split_k_paged,
    flash_attn_decode_kernel_split_k,
)

KVCACHE_BLOCK_SIZE = 256


def get_split_k(batch_size: int, n_group_q: int, heads_per_group_q: int, seq_len_k: int) -> int:
    """
    Heuristic to determine split_k for flash attention decoding.

    Args:
        batch_size (int): Batch size.
        n_group_q (int): Number of query head groups.
        heads_per_group_q (int): Number of heads per query group.
        seq_len_k (int): Sequence length of keys.

    Returns:
        int: Calculated split_k value.

    References:
        https://github.com/Dao-AILab/flash-attention/blob/672381f72c927a4b4a92f30755dc5829c3d0eaa3/flash_attn/flash_attn_triton_amd/fwd_decode.py
    """
    bh = max(batch_size * heads_per_group_q, 1)
    split_k = max(seq_len_k, 1024) // bh
    max_chunk_size = 64
    while split_k > 0 and seq_len_k / split_k < max_chunk_size:
        split_k = split_k // 2
    while batch_size * heads_per_group_q * n_group_q * split_k >= 1024:
        split_k = split_k // 2
    split_k = min(max(split_k, 1), 512)
    return split_k


def flash_attn_decode_paged(
    q_bh: torch.Tensor,
    k_slots: torch.Tensor,
    v_slots: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    num_heads: int,
    kv_heads: int,
    causal: bool = True,
    softmax_scale: Optional[float] = None,
    block_size_k: int = 64,
    kv_block_size: int = KVCACHE_BLOCK_SIZE,
) -> torch.Tensor:
    """
    Flash Attention decoding with paged key-value cache.

    Args:
        q_bh (torch.Tensor): Query tensor of shape (batch_size * num_heads, 1, dim).
        k_slots (torch.Tensor): Key cache tensor of shape (num_kv_slots, kv_heads, dim).
        v_slots (torch.Tensor): Value cache tensor of shape (num_kv_slots, kv_heads, dim).
        block_tables (torch.Tensor): Block table tensor of shape (batch_size, max_num_blocks).
        context_lens (torch.Tensor): Context lengths tensor of shape (batch_size,).
        num_heads (int): Total number of attention heads.
        kv_heads (int): Number of key-value heads.
        causal (bool): Whether to apply causal masking.
        softmax_scale (Optional[float]): Scaling factor for softmax.
        block_size_k (int): Block size for keys.
        kv_block_size (int): Block size for key-value cache.

    Returns:
        torch.Tensor: Output tensor of shape (batch_size * num_heads, 1, dim).
    """
    assert q_bh.ndim == 3
    bh, seq_len_q, dim = q_bh.shape
    assert seq_len_q == 1
    block_size_q = 16
    split_k = 1  # fix split_k to 1 for cuda graph

    if softmax_scale is None:
        softmax_scale = 1.0 / (dim**0.5)

    assert block_tables.ndim == 2
    max_num_blocks = int(block_tables.shape[1])

    seq_len_k_cap = max_num_blocks * kv_block_size
    block_n_per_split = (seq_len_k_cap + split_k - 1) // split_k
    seq_len_q_ceil = triton.cdiv(seq_len_q, block_size_q) * block_size_q

    ez_dot_v = torch.empty((bh, split_k, seq_len_q_ceil, dim), device=q_bh.device, dtype=torch.float32)
    max_q = torch.empty((bh, split_k, seq_len_q_ceil), device=q_bh.device, dtype=torch.float32)
    ez_sum = torch.empty_like(max_q)
    o = torch.empty_like(q_bh)

    stride_q_bh, stride_q_seq, stride_q_dim = q_bh.stride()
    stride_cache_slot, stride_cache_head, stride_cache_dim = k_slots.stride()
    stride_bt_b, stride_bt_blk = block_tables.stride()
    stride_cl_b = context_lens.stride()[0]

    stride_ez_dot_v_bh, stride_ez_dot_v_split, stride_ez_dot_v_seq, stride_ez_dot_v_dim = ez_dot_v.stride()
    stride_max_q_bh, stride_max_q_split, stride_max_q_seq = max_q.stride()
    stride_ez_sum_bh, stride_ez_sum_split, stride_ez_sum_seq = ez_sum.stride()
    stride_o_out_bh, stride_o_out_seq, stride_o_out_dim = o.stride()

    grid_split = (triton.cdiv(seq_len_q, block_size_q), bh, split_k)
    flash_attn_decode_kernel_split_k_paged[grid_split](
        q_bh,
        k_slots,
        v_slots,
        block_tables,
        context_lens,
        ez_dot_v,
        max_q,
        ez_sum,
        seq_len_q,
        stride_q_bh,
        stride_q_seq,
        stride_q_dim,
        stride_cache_slot,
        stride_cache_head,
        stride_cache_dim,
        stride_bt_b,
        stride_bt_blk,
        stride_cl_b,
        stride_ez_dot_v_bh,
        stride_ez_dot_v_split,
        stride_ez_dot_v_seq,
        stride_ez_dot_v_dim,
        stride_max_q_bh,
        stride_max_q_split,
        stride_max_q_seq,
        stride_ez_sum_bh,
        stride_ez_sum_split,
        stride_ez_sum_seq,
        softmax_scale,
        block_n_per_split,
        kv_block_size=kv_block_size,
        kv_heads=kv_heads,
        num_heads=num_heads,
        causal=causal,
        dim=dim,
        block_size_q=block_size_q,
        block_size_k=block_size_k,
        max_num_blocks=max_num_blocks,
    )

    grid_reduce = (bh, triton.cdiv(seq_len_q, block_size_q))
    flash_attn_decode_kernel_reduce_k[grid_reduce](
        ez_dot_v,
        max_q,
        ez_sum,
        o,
        seq_len_q,
        stride_ez_dot_v_bh,
        stride_ez_dot_v_split,
        stride_ez_dot_v_seq,
        stride_ez_dot_v_dim,
        stride_max_q_bh,
        stride_max_q_split,
        stride_max_q_seq,
        stride_ez_sum_bh,
        stride_ez_sum_split,
        stride_ez_sum_seq,
        stride_o_out_bh,
        stride_o_out_seq,
        stride_o_out_dim,
        dim=dim,
        block_size_q=block_size_q,
        split_k=split_k,
        causal=causal,
    )

    return o


# deprecated, kept for reference
def flash_attn_decode(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = True,
    softmax_scale: Optional[float] = None,
    split_k: Optional[int] = None,
    block_size_q: int = 16,
    block_size_k: int = 64,
) -> torch.Tensor:
    """
    Flash Attention decoding from loaded key-value tensors by python.

    Args:
        q (torch.Tensor): Query tensor of shape (batch_size, num_heads, seq_len_q, dim) or (batch_size * num_heads, seq_len_q, dim).
        k (torch.Tensor): Key tensor of shape (batch_size, num_heads, seq_len_k, dim) or (batch_size * num_heads, seq_len_k, dim).
        v (torch.Tensor): Value tensor of shape (batch_size, num_heads, seq_len_k, dim) or (batch_size * num_heads, seq_len_k, dim).
        causal (bool): Whether to apply causal masking.
        softmax_scale (Optional[float]): Scaling factor for softmax.
        split_k (Optional[int]): Number of splits along the key dimension.
        block_size_q (int): Block size for queries.
        block_size_k (int): Block size for keys.

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, num_heads, seq_len_q, dim).

    Notes:
        This function is deprecated because it can not be used with CUDA graph.
        Use `flash_attn_decode_paged` instead.
    """
    if q.ndim == 4:
        bsz, num_heads, seq_len_q, dim = q.shape
        seq_len_k = k.shape[2]
        bh = bsz * num_heads
        assert k.shape == v.shape == (bsz, num_heads, seq_len_k, dim)

        def merge_heads(x):
            return x.contiguous().view(bh, x.shape[2], dim)

        q = merge_heads(q)
        k = merge_heads(k)
        v = merge_heads(v)
    elif q.ndim == 3:
        bh, seq_len_q, dim = q.shape
        seq_len_k = k.shape[1]
        bsz = 1
        num_heads = bh
    else:
        raise ValueError("q, k, v must be 3D or 4D tensors")

    if softmax_scale is None:
        softmax_scale = 1.0 / (dim**0.5)

    if seq_len_q == seq_len_k:
        split_k = 1
    elif split_k is None:
        # automatically determine split_k if not provided.
        split_k = get_split_k(bsz, num_heads, 1, seq_len_k)

    block_n_per_split = (seq_len_k + split_k - 1) // split_k
    seq_len_q_ceil = triton.cdiv(seq_len_q, block_size_q) * block_size_q

    ez_dot_v = torch.empty(
        (bh, split_k, seq_len_q_ceil, dim),
        device=q.device,
        dtype=torch.float32,
    )
    max_q = torch.empty(
        (bh, split_k, seq_len_q_ceil),
        device=q.device,
        dtype=torch.float32,
    )
    ez_sum = torch.empty_like(max_q)
    o = torch.empty_like(q)

    stride_q_bh, stride_q_seq, stride_q_dim = q.stride()
    stride_k_bh, stride_k_seq, stride_k_dim = k.stride()
    stride_v_bh, stride_v_seq, stride_v_dim = v.stride()
    stride_ez_dot_v_bh, stride_ez_dot_v_split, stride_ez_dot_v_seq, stride_ez_dot_v_dim = ez_dot_v.stride()
    stride_max_q_bh, stride_max_q_split, stride_max_q_seq = max_q.stride()
    stride_ez_sum_bh, stride_ez_sum_split, stride_ez_sum_seq = ez_sum.stride()
    stride_o_out_bh, stride_o_out_seq, stride_o_out_dim = o.stride()

    grid_split_k = triton.cdiv(seq_len_q, block_size_q), bh, split_k
    flash_attn_decode_kernel_split_k[grid_split_k](
        q,
        k,
        v,
        ez_dot_v,
        max_q,
        ez_sum,
        seq_len_q,
        seq_len_k,
        stride_q_bh,
        stride_q_seq,
        stride_q_dim,
        stride_k_bh,
        stride_k_seq,
        stride_k_dim,
        stride_v_bh,
        stride_v_seq,
        stride_v_dim,
        stride_ez_dot_v_bh,
        stride_ez_dot_v_split,
        stride_ez_dot_v_seq,
        stride_ez_dot_v_dim,
        stride_max_q_bh,
        stride_max_q_split,
        stride_max_q_seq,
        stride_ez_sum_bh,
        stride_ez_sum_split,
        stride_ez_sum_seq,
        softmax_scale,
        block_n_per_split,
        causal=causal,
        dim=dim,
        block_size_q=block_size_q,
        block_size_k=block_size_k,
    )

    grid_reduce_k = (bh, triton.cdiv(seq_len_q, block_size_q))
    flash_attn_decode_kernel_reduce_k[grid_reduce_k](
        ez_dot_v,
        max_q,
        ez_sum,
        o,
        seq_len_q,
        stride_ez_dot_v_bh,
        stride_ez_dot_v_split,
        stride_ez_dot_v_seq,
        stride_ez_dot_v_dim,
        stride_max_q_bh,
        stride_max_q_split,
        stride_max_q_seq,
        stride_ez_sum_bh,
        stride_ez_sum_split,
        stride_ez_sum_seq,
        stride_o_out_bh,
        stride_o_out_seq,
        stride_o_out_dim,
        dim=dim,
        block_size_q=block_size_q,
        split_k=split_k,
        causal=causal,
    )
    o = o.view(bsz, num_heads, seq_len_q, dim)
    return o
