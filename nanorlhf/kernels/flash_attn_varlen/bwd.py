import math

import torch
import triton
import triton.language as tl


@triton.jit
def flash_attn_varlen_bwd_kernel(
    q_ptr, k_ptr, v_ptr,
    o_ptr, do_ptr,
    cu_q_ptr, cu_k_ptr,
    max_q_ptr, ez_sum_ptr,
    dq_ptr, dk_ptr, dv_ptr,
    B, H,
    stride_q_tok, stride_q_head, stride_q_dim,
    stride_k_tok, stride_k_head, stride_k_dim,
    stride_v_tok, stride_v_head, stride_v_dim,
    stride_o_tok, stride_o_head, stride_o_dim,
    stride_max_q_head, stride_max_q_tok,
    stride_ez_sum_head, stride_ez_sum_tok,
    stride_dq_tok, stride_dq_head, stride_dq_dim,
    stride_dk_tok, stride_dk_head, stride_dk_dim,
    stride_dv_tok, stride_dv_head, stride_dv_dim,
    softmax_scale,
    causal: tl.constexpr,
    block_size_q: tl.constexpr,
    tile_size_kv: tl.constexpr,
    dim: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_m = tl.program_id(1)

    seq_id = pid_bh // H
    head_id = pid_bh % H

    q_start = tl.load(cu_q_ptr + seq_id)
    q_end = tl.load(cu_q_ptr + seq_id + 1)
    seqlen_q = q_end - q_start

    k_start = tl.load(cu_k_ptr + seq_id)
    k_end = tl.load(cu_k_ptr + seq_id + 1)
    seqlen_k = k_end - k_start

    block_q_start = pid_m * block_size_q
    offs_q = block_q_start + tl.arange(0, block_size_q)
    q_mask = offs_q < seqlen_q

    if block_q_start >= seqlen_q:
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

    # Q / O / dO blocks
    q_block_ptr = tl.make_block_ptr(
        base=q_head_seq_base,
        shape=(seqlen_q, dim),
        strides=(stride_q_tok, stride_q_dim),
        offsets=(block_q_start, 0),
        block_shape=(block_size_q, dim),
        order=(1, 0),
    )
    o_block_ptr = tl.make_block_ptr(
        base=o_head_seq_base,
        shape=(seqlen_q, dim),
        strides=(stride_o_tok, stride_o_dim),
        offsets=(block_q_start, 0),
        block_shape=(block_size_q, dim),
        order=(1, 0),
    )
    do_block_ptr = tl.make_block_ptr(
        base=do_head_seq_base,
        shape=(seqlen_q, dim),
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
    dO = tl.load(
        do_block_ptr,
        boundary_check=(0, 1),
        padding_option="zero",
    ).to(tl.float32)

    max_q = tl.load(max_q_head_base + q_indices * stride_max_q_tok, mask=q_mask, other=-float("inf"))
    ez_sum = tl.load(ez_sum_head_base + q_indices * stride_ez_sum_tok, mask=q_mask, other=1.0)

    u = tl.sum(dO * o, axis=1)

    dq = tl.zeros((block_size_q, dim), dtype=tl.float32)

    for kv_start in range(0, seqlen_k, tile_size_kv):
        kv_rel = kv_start + offs_kv
        kv_mask = kv_rel < seqlen_k

        k_block_ptr = tl.make_block_ptr(
            base=k_head_seq_base,
            shape=(seqlen_k, dim),
            strides=(stride_k_tok, stride_k_dim),
            offsets=(kv_start, 0),
            block_shape=(tile_size_kv, dim),
            order=(1, 0),
        )
        v_block_ptr = tl.make_block_ptr(
            base=v_head_seq_base,
            shape=(seqlen_k, dim),
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

        scores = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * softmax_scale

        q_pos = offs_q[:, None]
        kv_pos = kv_rel[None, :]

        base_mask = (~q_mask[:, None]) | (~kv_mask[None, :])

        if causal:
            causal_mask = kv_pos > q_pos
            mask = base_mask | causal_mask
        else:
            mask = base_mask

        scores = tl.where(mask, -float("inf"), scores)

        scores_shifted = scores - max_q[:, None]
        P = tl.exp(scores_shifted) / ez_sum[:, None]
        P = tl.where(mask, 0.0, P)

        dV_tile = tl.dot(tl.trans(P), dO, out_dtype=tl.float32)
        dot_dOV = tl.dot(dO, tl.trans(v), out_dtype=tl.float32)
        dS = (dot_dOV - u[:, None]) * P

        dq += tl.dot(dS, k, out_dtype=tl.float32) * softmax_scale
        dK_tile = tl.dot(tl.trans(dS), q, out_dtype=tl.float32) * softmax_scale

        # We can't use block ptrs for atomic add
        # because atomic add doesn't support block ptrs
        dv_ptrs = dv_head_seq_base \
                  + kv_rel[:, None] * stride_dv_tok \
                  + offs_d[None, :] * stride_dv_dim
        dk_ptrs = dk_head_seq_base \
                  + kv_rel[:, None] * stride_dk_tok \
                  + offs_d[None, :] * stride_dk_dim

        mask_2d = kv_mask[:, None] & (offs_d[None, :] < dim)

        tl.atomic_add(dv_ptrs, dV_tile, mask=mask_2d)
        tl.atomic_add(dk_ptrs, dK_tile, mask=mask_2d)

    dq_block_ptr = tl.make_block_ptr(
        base=dq_head_seq_base,
        shape=(seqlen_q, dim),
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


def flash_attn_varlen_bwd(q, k, v, o, dO, cu_q, cu_k, max_q, ez_sum, B, H, max_seqlen_q, max_seqlen_k, causal=True, softmax_scale=None):
    assert q.is_cuda and k.is_cuda and v.is_cuda and o.is_cuda and dO.is_cuda
    assert q.dim() == 3 and k.dim() == 3 and v.dim() == 3
    total_q, H_q, dim = q.shape
    total_k, H_k, dim_k = k.shape
    assert H_q == H and H_k == H
    assert dim == dim_k
    assert max_q.shape == (H, total_q)
    assert ez_sum.shape == (H, total_q)

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
        softmax_scale = 1.0 / (dim ** 0.5)

    block_size_q = 64
    tile_size_kv = 64

    grid = (B * H, triton.cdiv(max_seqlen_q, block_size_q))

    flash_attn_varlen_bwd_kernel[grid](
        q, k, v,
        o, dO,
        cu_q, cu_k,
        max_q, ez_sum,
        dq, dk, dv,
        B, H,
        stride_q_tok, stride_q_head, stride_q_dim,
        stride_k_tok, stride_k_head, stride_k_dim,
        stride_v_tok, stride_v_head, stride_v_dim,
        stride_o_tok, stride_o_head, stride_o_dim,
        stride_max_q_head, stride_max_q_tok,
        stride_ez_sum_head, stride_ez_sum_tok,
        stride_dq_tok, stride_dq_head, stride_dq_dim,
        stride_dk_tok, stride_dk_head, stride_dk_dim,
        stride_dv_tok, stride_dv_head, stride_dv_dim,
        softmax_scale,
        causal=causal,
        block_size_q=block_size_q,
        tile_size_kv=tile_size_kv,
        dim=dim,
    )
    return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype)
