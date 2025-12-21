from typing import Optional, Tuple

import torch
import triton
import triton.language as tl


@triton.jit
def flash_attn_kernel_bwd(
    q_ptr,
    k_ptr,
    v_ptr,
    do_ptr,
    dq_ptr,
    dk_ptr,
    dv_ptr,
    max_q_ptr,
    ez_sum_ptr,
    mask_ptr,
    seq_len_q,
    seq_len_kv,
    stride_q_bh,
    stride_q_seq,
    stride_q_dim,
    stride_k_bh,
    stride_k_seq,
    stride_k_dim,
    stride_v_bh,
    stride_v_seq,
    stride_v_dim,
    stride_do_bh,
    stride_do_seq,
    stride_do_dim,
    stride_dq_bh,
    stride_dq_seq,
    stride_dq_dim,
    stride_dk_bh,
    stride_dk_seq,
    stride_dk_dim,
    stride_dv_bh,
    stride_dv_seq,
    stride_dv_dim,
    stride_max_q_bh,
    stride_max_q_seq,
    stride_ez_sum_bh,
    stride_ez_sum_seq,
    stride_mask_b,
    stride_mask_h,
    stride_mask_q,
    stride_mask_k,
    num_heads,
    mask_num_heads,
    softmax_scale,
    causal: tl.constexpr,
    has_mask: tl.constexpr,
    dim: tl.constexpr,
    block_size_q: tl.constexpr,
    tile_size_kv: tl.constexpr,
):
    """
    Forward:
        s_ij = (q_i · k_j) * scale
        p_ij = softmax_j(s_ij)
        o_i  = Σ_j p_ij v_j

    Backward:
        dV_j = Σ_i p_ij^T · dO_i
        dP_ij = dO_i · v_j^T
        dS_ij = (dP_ij - Σ_k dP_ik p_ik) * p_ij
        dQ_i = Σ_j dS_ij · k_j * scale
        dK_j = Σ_i dS_ij · q_i * scale
    """
    pid_q = tl.program_id(0)
    pid_bh = tl.program_id(1)

    q_bh = q_ptr + pid_bh * stride_q_bh
    k_bh = k_ptr + pid_bh * stride_k_bh
    v_bh = v_ptr + pid_bh * stride_v_bh
    do_bh = do_ptr + pid_bh * stride_do_bh

    dq_bh = dq_ptr + pid_bh * stride_dq_bh
    dk_bh = dk_ptr + pid_bh * stride_dk_bh
    dv_bh = dv_ptr + pid_bh * stride_dv_bh

    q_start = pid_q * block_size_q
    if q_start >= seq_len_q:
        return

    offs_q = q_start + tl.arange(0, block_size_q)
    offs_kv = tl.arange(0, tile_size_kv)
    q_mask = offs_q < seq_len_q

    q_block_ptr = tl.make_block_ptr(
        base=q_bh,
        shape=(seq_len_q, dim),
        offsets=(q_start, 0),
        block_shape=(block_size_q, dim),
        strides=(stride_q_seq, stride_q_dim),
        order=(1, 0),
    )
    do_block_ptr = tl.make_block_ptr(
        base=do_bh,
        shape=(seq_len_q, dim),
        offsets=(q_start, 0),
        block_shape=(block_size_q, dim),
        strides=(stride_do_seq, stride_do_dim),
        order=(1, 0),
    )
    dq_block_ptr = tl.make_block_ptr(
        base=dq_bh,
        shape=(seq_len_q, dim),
        offsets=(q_start, 0),
        block_shape=(block_size_q, dim),
        strides=(stride_dq_seq, stride_dq_dim),
        order=(1, 0),
    )

    q = tl.load(q_block_ptr, boundary_check=(0, 1), padding_option="zero")
    do = tl.load(do_block_ptr, boundary_check=(0, 1), padding_option="zero")

    # Load max_q and ez_sum
    max_q = tl.load(
        max_q_ptr + pid_bh * stride_max_q_bh + offs_q * stride_max_q_seq,
        mask=q_mask,
        other=0.0,
    )
    ez_sum = tl.load(
        ez_sum_ptr + pid_bh * stride_ez_sum_bh + offs_q * stride_ez_sum_seq,
        mask=q_mask,
        other=1.0,
    )

    ez_sum = tl.maximum(ez_sum, 1e-6)
    dq = tl.zeros((block_size_q, dim), dtype=tl.float32)

    if has_mask:
        batch_idx = pid_bh // num_heads
        head_idx = pid_bh % num_heads
        mask_head_idx = tl.minimum(head_idx, mask_num_heads - 1)
        mask_bh = mask_ptr + batch_idx * stride_mask_b + mask_head_idx * stride_mask_h

    for kv_start in range(0, seq_len_kv, tile_size_kv):
        k_block_ptr = tl.make_block_ptr(
            base=k_bh,
            shape=(seq_len_kv, dim),
            offsets=(kv_start, 0),
            block_shape=(tile_size_kv, dim),
            strides=(stride_k_seq, stride_k_dim),
            order=(1, 0),
        )
        v_block_ptr = tl.make_block_ptr(
            base=v_bh,
            shape=(seq_len_kv, dim),
            offsets=(kv_start, 0),
            block_shape=(tile_size_kv, dim),
            strides=(stride_v_seq, stride_v_dim),
            order=(1, 0),
        )
        k = tl.load(
            k_block_ptr,
            boundary_check=(0, 1),
            padding_option="zero",
        )
        v = tl.load(
            v_block_ptr,
            boundary_check=(0, 1),
            padding_option="zero",
        )

        scores = tl.dot(q.to(k.dtype), tl.trans(k), out_dtype=tl.float32) * softmax_scale

        if has_mask:
            mask_block_ptr = tl.make_block_ptr(
                base=mask_bh,  # noqa
                shape=(seq_len_q, seq_len_kv),
                offsets=(q_start, kv_start),
                block_shape=(block_size_q, tile_size_kv),
                strides=(stride_mask_q, stride_mask_k),
                order=(1, 0),
            )
            mask_tile = tl.load(
                mask_block_ptr,
                boundary_check=(0, 1),
                padding_option="zero",
            )
            scores = scores + mask_tile.to(tl.float32)

        kv_idx = kv_start + offs_kv
        kv_mask = kv_idx < seq_len_kv
        base_mask = (~q_mask[:, None]) | (~kv_mask[None, :])

        if causal:
            offset = seq_len_kv - seq_len_q
            q_pos = (offset + offs_q)[:, None]
            kv_pos = kv_idx[None, :]
            causal_mask = kv_pos > q_pos
            mask = base_mask | causal_mask
        else:
            mask = base_mask

        scores = tl.where(mask, -float("inf"), scores)
        p = tl.exp(scores - max_q[:, None]) / ez_sum[:, None]
        p_half = p.to(q.dtype)

        # dv_tile = p^T @ do
        dv_tile = tl.dot(tl.trans(p_half), do, out_dtype=tl.float32)

        # dp = do @ V^T
        dp = tl.dot(do, tl.trans(v), out_dtype=tl.float32)

        # ds = (dp - Σ_k dp_ik p_ik) * p_ij
        row_dp_p = tl.sum(dp * p, axis=1)
        ds = (dp - row_dp_p[:, None]) * p
        ds_half = ds.to(q.dtype)

        dq += tl.dot(ds_half, k, out_dtype=tl.float32) * softmax_scale
        dk_tile = tl.dot(tl.trans(ds_half), q, out_dtype=tl.float32) * softmax_scale

        dk_ptrs = dk_bh + kv_idx[:, None] * stride_dk_seq + tl.arange(0, dim)[None, :] * stride_dk_dim
        dv_ptrs = dv_bh + kv_idx[:, None] * stride_dv_seq + tl.arange(0, dim)[None, :] * stride_dv_dim

        dk_tile_out = dk_tile.to(k.dtype)
        dv_tile_out = dv_tile.to(v.dtype)

        tl.atomic_add(dk_ptrs, dk_tile_out, mask=kv_mask[:, None])
        tl.atomic_add(dv_ptrs, dv_tile_out, mask=kv_mask[:, None])

    dq_out = dq.to(q.dtype)
    tl.store(dq_block_ptr, dq_out, boundary_check=(0, 1))


def flash_attn_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    do: torch.Tensor,
    max_q: torch.Tensor,
    ez_sum: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Backward pass for Flash Attention.

    Args:
        q: Query tensor of shape (bsz, num_heads, seq_len_q, dim_head).
        k: Key tensor of shape (bsz, num_heads, seq_len_kv, dim_head).
        v: Value tensor of shape (bsz, num_heads, seq_len_kv, dim_head).
        do: Gradient of output tensor of shape (bsz, num_heads, seq_len_q, dim_head).
        max_q: Max logits per query block of shape (bsz * num_heads, seq_len_q).
        ez_sum: Exponential sum per query block of shape (bsz * num_heads, seq_len_q).
        attention_mask: Optional attention mask tensor.
        softmax_scale: Optional scaling factor for softmax.
        causal: Whether to apply causal masking.

    Returns:
        Tuple of gradients (dq, dk, dv) with the same shapes as q, k, v respectively.
    """

    bsz, num_heads_q, seq_len_q, dim_head = q.shape
    seq_len_kv = k.shape[2]
    assert max_q.shape == ez_sum.shape == (bsz * num_heads_q, seq_len_q)

    bh = bsz * num_heads_q

    def merge_heads(x):
        return x.contiguous().view(bh, x.shape[2], dim_head)

    def unmerge_heads(x, b, h):
        bh, seq_len, dim = x.shape
        assert bh == b * h
        return x.view(b, h, seq_len, dim)

    def grid(meta):
        return triton.cdiv(seq_len_q, meta["block_size_q"]), bh

    q = merge_heads(q)
    k = merge_heads(k)
    v = merge_heads(v)
    do = merge_heads(do)

    dq = torch.zeros_like(q)
    dk = torch.zeros_like(k)
    dv = torch.zeros_like(v)

    stride_q_bh, stride_q_seq, stride_q_dim = q.stride()
    stride_k_bh, stride_k_seq, stride_k_dim = k.stride()
    stride_v_bh, stride_v_seq, stride_v_dim = v.stride()
    stride_do_bh, stride_do_seq, stride_do_dim = do.stride()
    stride_dq_bh, stride_dq_seq, stride_dq_dim = dq.stride()
    stride_dk_bh, stride_dk_seq, stride_dk_dim = dk.stride()
    stride_dv_bh, stride_dv_seq, stride_dv_dim = dv.stride()
    stride_max_q_bh, stride_max_q_seq = max_q.stride()
    stride_ez_sum_bh, stride_ez_sum_seq = ez_sum.stride()

    has_mask = attention_mask is not None
    if has_mask:
        assert attention_mask.ndim == 4
        assert attention_mask.shape[0] == bsz
        assert attention_mask.shape[-2] == seq_len_q or attention_mask.shape[-1] == seq_len_kv

        mask = attention_mask.contiguous()
        mask_num_heads = mask.shape[1]
        stride_mask_b, stride_mask_h, stride_mask_q, stride_mask_k = mask.stride()
    else:
        mask = q  # dummy
        mask_num_heads = 1
        stride_mask_b = stride_mask_h = stride_mask_q = stride_mask_k = 0

    if softmax_scale is None:
        softmax_scale = 1.0 / (dim_head**0.5)

    flash_attn_kernel_bwd[grid](
        q,
        k,
        v,
        do,
        dq,
        dk,
        dv,
        max_q,
        ez_sum,
        mask,
        seq_len_q,
        seq_len_kv,
        stride_q_bh,
        stride_q_seq,
        stride_q_dim,
        stride_k_bh,
        stride_k_seq,
        stride_k_dim,
        stride_v_bh,
        stride_v_seq,
        stride_v_dim,
        stride_do_bh,
        stride_do_seq,
        stride_do_dim,
        stride_dq_bh,
        stride_dq_seq,
        stride_dq_dim,
        stride_dk_bh,
        stride_dk_seq,
        stride_dk_dim,
        stride_dv_bh,
        stride_dv_seq,
        stride_dv_dim,
        stride_max_q_bh,
        stride_max_q_seq,
        stride_ez_sum_bh,
        stride_ez_sum_seq,
        stride_mask_b,
        stride_mask_h,
        stride_mask_q,
        stride_mask_k,
        num_heads=num_heads_q,
        mask_num_heads=mask_num_heads,
        softmax_scale=softmax_scale,
        causal=causal,
        has_mask=has_mask,
        dim=dim_head,
        block_size_q=64,
        tile_size_kv=64,
    )

    dq = unmerge_heads(dq, bsz, num_heads_q)
    dk = unmerge_heads(dk, bsz, num_heads_q)
    dv = unmerge_heads(dv, bsz, num_heads_q)
    return dq, dk, dv
