from typing import Optional, Tuple

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"block_size_q": 64, "tile_size_kv": 64, "num_warps": 4, "num_stages": 2}),
        triton.Config({"block_size_q": 128, "tile_size_kv": 64, "num_warps": 8, "num_stages": 2}),
        triton.Config({"block_size_q": 64, "tile_size_kv": 128, "num_warps": 8, "num_stages": 2}),
        triton.Config({"block_size_q": 32, "tile_size_kv": 64, "num_warps": 4, "num_stages": 2}),
        triton.Config({"block_size_q": 64, "tile_size_kv": 32, "num_warps": 4, "num_stages": 2}),
    ],
    key=["dim"],
)
@triton.jit
def flash_attn_varlen_bwd_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    do_ptr,
    cu_seq_lens_q_ptr,
    cu_seq_lens_k_ptr,
    max_q_ptr,
    ez_sum_ptr,
    dq_ptr,
    dk_ptr,
    dv_ptr,
    batch_size,
    num_heads,
    stride_q_tok,
    stride_q_head,
    stride_q_dim,
    stride_k_tok,
    stride_k_head,
    stride_k_dim,
    stride_v_tok,
    stride_v_head,
    stride_v_dim,
    stride_o_tok,
    stride_o_head,
    stride_o_dim,
    stride_max_q_head,
    stride_max_q_tok,
    stride_ez_sum_head,
    stride_ez_sum_tok,
    stride_dq_tok,
    stride_dq_head,
    stride_dq_dim,
    stride_dk_tok,
    stride_dk_head,
    stride_dk_dim,
    stride_dv_tok,
    stride_dv_head,
    stride_dv_dim,
    softmax_scale,
    causal: tl.constexpr,
    block_size_q: tl.constexpr,
    tile_size_kv: tl.constexpr,
    dim: tl.constexpr,
):
    """
    Flash Attention backward kernel for variable-length sequences.

    Notes:
        You can find detailed explanations about how this kernel works with variable-length sequences
        in the docstring of `flash_attn_varlen_fwd_kernel`.
    """
    pid_bh = tl.program_id(0)
    pid_m = tl.program_id(1)

    seq_id = pid_bh // num_heads
    head_id = pid_bh % num_heads

    q_start = tl.load(cu_seq_lens_q_ptr + seq_id)
    q_end = tl.load(cu_seq_lens_q_ptr + seq_id + 1)
    seq_len_q = q_end - q_start

    k_start = tl.load(cu_seq_lens_k_ptr + seq_id)
    k_end = tl.load(cu_seq_lens_k_ptr + seq_id + 1)
    seq_len_k = k_end - k_start

    block_q_start = pid_m * block_size_q
    offs_q = block_q_start + tl.arange(0, block_size_q)
    q_mask = offs_q < seq_len_q

    if block_q_start >= seq_len_q:
        return

    q_indices = q_start + offs_q

    offs_d = tl.arange(0, dim)
    offs_kv = tl.arange(0, tile_size_kv)

    # per-head, per-sequence bases
    q_head_seq_base = q_ptr + head_id * stride_q_head + q_start * stride_q_tok
    k_head_seq_base = k_ptr + head_id * stride_k_head + k_start * stride_k_tok
    v_head_seq_base = v_ptr + head_id * stride_v_head + k_start * stride_v_tok
    o_head_seq_base = o_ptr + head_id * stride_o_head + q_start * stride_o_tok
    do_head_seq_base = do_ptr + head_id * stride_o_head + q_start * stride_o_tok

    dq_head_seq_base = dq_ptr + head_id * stride_dq_head + q_start * stride_dq_tok
    dk_head_seq_base = dk_ptr + head_id * stride_dk_head + k_start * stride_dk_tok
    dv_head_seq_base = dv_ptr + head_id * stride_dv_head + k_start * stride_dv_tok

    max_q_head_base = max_q_ptr + head_id * stride_max_q_head
    ez_sum_head_base = ez_sum_ptr + head_id * stride_ez_sum_head

    # q / o / do blocks
    q_block_ptr = tl.make_block_ptr(
        base=q_head_seq_base,
        shape=(seq_len_q, dim),
        strides=(stride_q_tok, stride_q_dim),
        offsets=(block_q_start, 0),
        block_shape=(block_size_q, dim),
        order=(1, 0),
    )
    o_block_ptr = tl.make_block_ptr(
        base=o_head_seq_base,
        shape=(seq_len_q, dim),
        strides=(stride_o_tok, stride_o_dim),
        offsets=(block_q_start, 0),
        block_shape=(block_size_q, dim),
        order=(1, 0),
    )
    do_block_ptr = tl.make_block_ptr(
        base=do_head_seq_base,
        shape=(seq_len_q, dim),
        strides=(stride_o_tok, stride_o_dim),
        offsets=(block_q_start, 0),
        block_shape=(block_size_q, dim),
        order=(1, 0),
    )

    q = tl.load(
        q_block_ptr,
        boundary_check=(0, 1),
        padding_option="zero",
    ).to(tl.float32)

    o = tl.load(
        o_block_ptr,
        boundary_check=(0, 1),
        padding_option="zero",
    ).to(tl.float32)

    do = tl.load(do_block_ptr, boundary_check=(0, 1), padding_option="zero").to(tl.float32)

    max_q = tl.load(max_q_head_base + q_indices * stride_max_q_tok, mask=q_mask, other=0.0)
    ez_sum = tl.load(ez_sum_head_base + q_indices * stride_ez_sum_tok, mask=q_mask, other=1.0)

    u = tl.sum(do * o, axis=1)
    dq = tl.zeros((block_size_q, dim), dtype=tl.float32)
    for kv_start in range(0, seq_len_k, tile_size_kv):
        k_block_ptr = tl.make_block_ptr(
            base=k_head_seq_base,
            shape=(seq_len_k, dim),
            strides=(stride_k_tok, stride_k_dim),
            offsets=(kv_start, 0),
            block_shape=(tile_size_kv, dim),
            order=(1, 0),
        )
        v_block_ptr = tl.make_block_ptr(
            base=v_head_seq_base,
            shape=(seq_len_k, dim),
            strides=(stride_v_tok, stride_v_dim),
            offsets=(kv_start, 0),
            block_shape=(tile_size_kv, dim),
            order=(1, 0),
        )

        k = tl.load(
            k_block_ptr,
            boundary_check=(0, 1),
            padding_option="zero",
        ).to(tl.float32)
        v = tl.load(
            v_block_ptr,
            boundary_check=(0, 1),
            padding_option="zero",
        ).to(tl.float32)

        scores = tl.dot(q.to(k.dtype), tl.trans(k), out_dtype=tl.float32) * softmax_scale
        kv_idx = kv_start + offs_kv
        kv_mask = kv_idx < seq_len_k
        base_mask = (~q_mask[:, None]) | (~kv_mask[None, :])

        if causal:
            offset = seq_len_k - seq_len_q
            q_pos = (offset + offs_q)[:, None]
            kv_pos = kv_idx[None, :]
            causal_mask = kv_pos > q_pos
            mask = base_mask | causal_mask
        else:
            mask = base_mask

        scores = tl.where(mask, -float("inf"), scores)
        p = tl.exp(scores - max_q[:, None]) / ez_sum[:, None]
        p = tl.where(mask, 0.0, p)

        dv_tile = tl.dot(tl.trans(p), do, out_dtype=tl.float32)
        dot_do_v = tl.dot(do, tl.trans(v), out_dtype=tl.float32)
        ds = (dot_do_v - u[:, None]) * p

        dq += tl.dot(ds, k, out_dtype=tl.float32) * softmax_scale
        dk_tile = tl.dot(tl.trans(ds), q, out_dtype=tl.float32) * softmax_scale

        # We can't use block ptrs for atomic add
        # because atomic add doesn't support block ptrs
        dv_ptrs = dv_head_seq_base + kv_idx[:, None] * stride_dv_tok + offs_d[None, :] * stride_dv_dim
        dk_ptrs = dk_head_seq_base + kv_idx[:, None] * stride_dk_tok + offs_d[None, :] * stride_dk_dim

        mask_2d = kv_mask[:, None] & (offs_d[None, :] < dim)

        tl.atomic_add(dv_ptrs, dv_tile, mask=mask_2d)
        tl.atomic_add(dk_ptrs, dk_tile, mask=mask_2d)

    dq_block_ptr = tl.make_block_ptr(
        base=dq_head_seq_base,
        shape=(seq_len_q, dim),
        strides=(stride_dq_tok, stride_dq_dim),
        offsets=(block_q_start, 0),
        block_shape=(block_size_q, dim),
        order=(1, 0),
    )
    tl.store(
        dq_block_ptr,
        dq.to(tl.float32),
        boundary_check=(0, 1),
    )


def flash_attn_varlen_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    do: torch.Tensor,
    cu_seq_lens_q: torch.Tensor,
    cu_seq_lens_k: torch.Tensor,
    max_q: torch.Tensor,
    ez_sum: torch.Tensor,
    batch_size: int,
    num_heads: int,
    max_seq_len_q: int,
    max_seq_len_k: int,
    causal: bool = True,
    softmax_scale: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Flash Attention backward pass for variable-length sequences.
    
    Args:
        q (torch.Tensor): Query tensor of shape (total_q, num_heads, dim).
        k (torch.Tensor): Key tensor of shape (total_k, num_heads, dim).
        v (torch.Tensor): Value tensor of shape (total_k, num_heads, dim).
        o (torch.Tensor): Output tensor of shape (total_q, num_heads, dim).
        do (torch.Tensor): Gradient of output tensor of shape (total_q, num_heads, dim).
        cu_seq_lens_q (torch.Tensor): Cumulative sequence lengths for queries.
        cu_seq_lens_k (torch.Tensor): Cumulative sequence lengths for keys/values.
        max_q (torch.Tensor): Max logits per query for numerical stability.
        ez_sum (torch.Tensor): Exponential sums per query for normalization.
        batch_size (int): Number of sequences in the batch.
        num_heads (int): Number of attention heads.
        max_seq_len_q (int): Maximum sequence length for queries.
        max_seq_len_k (int): Maximum sequence length for keys/values.
        causal (bool): Whether to apply causal masking.
        softmax_scale (Optional[float]): Scaling factor for softmax.
    """
    assert q.is_cuda and k.is_cuda and v.is_cuda and o.is_cuda and do.is_cuda
    assert q.dim() == 3 and k.dim() == 3 and v.dim() == 3
    total_q, num_heads_q, dim = q.shape
    total_k, num_heads_k, dim_k = k.shape
    assert num_heads_q == num_heads and num_heads_k == num_heads
    assert dim == dim_k
    assert max_q.shape == (num_heads, total_q)
    assert ez_sum.shape == (num_heads, total_q)

    dq = torch.zeros_like(q, dtype=torch.float32)
    dk = torch.zeros_like(k, dtype=torch.float32)
    dv = torch.zeros_like(v, dtype=torch.float32)

    stride_q_tok, stride_q_head, stride_q_dim = q.stride()
    stride_k_tok, stride_k_head, stride_k_dim = k.stride()
    stride_v_tok, stride_v_head, stride_v_dim = v.stride()
    stride_o_tok, stride_o_head, stride_o_dim = o.stride()
    stride_dq_tok, stride_dq_head, stride_dq_dim = dq.stride()
    stride_dk_tok, stride_dk_head, stride_dk_dim = dk.stride()
    stride_dv_tok, stride_dv_head, stride_dv_dim = dv.stride()
    stride_max_q_head, stride_max_q_tok = max_q.stride()
    stride_ez_sum_head, stride_ez_sum_tok = ez_sum.stride()

    if softmax_scale is None:
        softmax_scale = 1.0 / (dim**0.5)

    def grid(meta):
        return batch_size * num_heads, triton.cdiv(max_seq_len_q, meta["block_size_q"])

    flash_attn_varlen_bwd_kernel[grid](
        q,
        k,
        v,
        o,
        do,
        cu_seq_lens_q,
        cu_seq_lens_k,
        max_q,
        ez_sum,
        dq,
        dk,
        dv,
        batch_size,
        num_heads,
        stride_q_tok,
        stride_q_head,
        stride_q_dim,
        stride_k_tok,
        stride_k_head,
        stride_k_dim,
        stride_v_tok,
        stride_v_head,
        stride_v_dim,
        stride_o_tok,
        stride_o_head,
        stride_o_dim,
        stride_max_q_head,
        stride_max_q_tok,
        stride_ez_sum_head,
        stride_ez_sum_tok,
        stride_dq_tok,
        stride_dq_head,
        stride_dq_dim,
        stride_dk_tok,
        stride_dk_head,
        stride_dk_dim,
        stride_dv_tok,
        stride_dv_head,
        stride_dv_dim,
        softmax_scale,
        causal=causal,
        dim=dim,
    )
    return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype)
