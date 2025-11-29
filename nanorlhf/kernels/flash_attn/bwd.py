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
    key=["seq_len_kv", "dim"],
)
@triton.jit
def flash_attn_kernel_bwd(
    q_ptr, k_ptr, v_ptr, do_ptr,
    dq_ptr, dk_ptr, dv_ptr,
    seq_len_q, seq_len_kv,
    stride_q_bh, stride_q_seq, stride_q_dim,
    stride_k_bh, stride_k_seq, stride_k_dim,
    stride_v_bh, stride_v_seq, stride_v_dim,
    stride_do_bh, stride_do_seq, stride_do_dim,
    stride_dq_bh, stride_dq_seq, stride_dq_dim,
    stride_dk_bh, stride_dk_seq, stride_dk_dim,
    stride_dv_bh, stride_dv_seq, stride_dv_dim,
    softmax_scale,
    causal: tl.constexpr,
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

    Notes:
        We haven't save softmax intermediates during forward pass,
        so they must be recomputed in backward. (2-pass softmax)
    """
    # Program ids
    pid_q = tl.program_id(0)
    pid_bh = tl.program_id(1)

    # Pointers to batch head
    q_bh = q_ptr + pid_bh * stride_q_bh
    k_bh = k_ptr + pid_bh * stride_k_bh
    v_bh = v_ptr + pid_bh * stride_v_bh
    do_bh = do_ptr + pid_bh * stride_do_bh

    # Pointers to batch head gradients
    dq_bh = dq_ptr + pid_bh * stride_dq_bh
    dk_bh = dk_ptr + pid_bh * stride_dk_bh
    dv_bh = dv_ptr + pid_bh * stride_dv_bh

    q_start = pid_q * block_size_q
    if q_start >= seq_len_q:
        return

    offs_q = q_start + tl.arange(0, block_size_q)
    offs_kv = tl.arange(0, tile_size_kv)

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

    # Load q, dO
    q = tl.load(q_block_ptr, boundary_check=(0, 1), padding_option="zero")
    dO = tl.load(do_block_ptr, boundary_check=(0, 1), padding_option="zero")

    # Recompute softmax forward to compute max_q and ez_sum
    max_q = tl.full((block_size_q,), -float("inf"), dtype=tl.float32)
    ez_sum = tl.zeros((block_size_q,), dtype=tl.float32)
    for kv_start in range(0, seq_len_kv, tile_size_kv):
        k_block_ptr = tl.make_block_ptr(
            base=k_bh,
            shape=(seq_len_kv, dim),
            offsets=(kv_start, 0),
            block_shape=(tile_size_kv, dim),
            strides=(stride_k_seq, stride_k_dim),
            order=(1, 0),
        )
        k = tl.load(
            k_block_ptr,
            boundary_check=(0, 1),
            padding_option="zero",
        )

        # Compute scores
        scores = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * softmax_scale

        # Apply causal mask if needed
        if causal:
            q_pos = offs_q[:, None]
            kv_pos = (kv_start + offs_kv)[None, :]
            mask = kv_pos > q_pos
            scores = tl.where(mask, -float("inf"), scores)

        # Streaming softmax
        current_max_q = tl.max(scores, axis=1)
        new_max_q = tl.maximum(max_q, current_max_q)
        rescale = tl.exp(max_q - new_max_q)
        current_ez = tl.exp(scores - new_max_q[:, None])
        ez_sum = ez_sum * rescale + tl.sum(current_ez, axis=1)
        max_q = new_max_q

    # Prevent division by zero
    ez_sum = tl.maximum(ez_sum, 1e-6)

    # Start computing backward
    dq = tl.zeros((block_size_q, dim), dtype=tl.float32)
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

        # Compute scores
        scores = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * softmax_scale

        # Apply causal mask if needed
        if causal:
            q_pos = offs_q[:, None]
            kv_pos = (kv_start + offs_kv)[None, :]
            mask = kv_pos > q_pos
            scores = tl.where(mask, -float("inf"), scores)

        # Compute softmax probabilities
        p = tl.exp(scores - max_q[:, None]) / ez_sum[:, None]
        p_half = p.to(q.dtype)

        # dv_tile = p^T @ dO
        dv_tile = tl.dot(tl.trans(p_half), dO, out_dtype=tl.float32)

        # dP = dO @ V^T
        dP = tl.dot(dO, tl.trans(v), out_dtype=tl.float32)

        # dS = (dP - Σ_k dP_ik · p_ik) * p_ij
        dS = (dP - tl.sum(dP * p, axis=1)[:, None]) * p
        dS_half = dS.to(q.dtype)

        # dQ = Σ_j dS_ij · k_j * scale
        dq += tl.dot(dS_half, k, out_dtype=tl.float32) * softmax_scale
        # dK = Σ_i dS_ij · q_i * scale
        dk_tile = tl.dot(tl.trans(dS_half), q, out_dtype=tl.float32) * softmax_scale

        # store dK, dV with atomic adds
        kv_idx = kv_start + offs_kv
        mask_kv = kv_idx < seq_len_kv

        # Compute pointers
        dv_ptrs = dv_bh + kv_idx[:, None] * stride_dv_seq + tl.arange(0, dim)[None, :] * stride_dv_dim
        dk_ptrs = dk_bh + kv_idx[:, None] * stride_dk_seq + tl.arange(0, dim)[None, :] * stride_dk_dim

        # Cast to storage dtype before atomic_add
        dv_tile_out = dv_tile.to(v.dtype)
        dk_tile_out = dk_tile.to(k.dtype)

        # Do atomic adds
        tl.atomic_add(dv_ptrs, dv_tile_out, mask=mask_kv[:, None])
        tl.atomic_add(dk_ptrs, dk_tile_out, mask=mask_kv[:, None])

    # Store dQ
    dq_out = dq.to(q.dtype)
    tl.store(dq_block_ptr, dq_out, boundary_check=(0, 1))


def flash_attn_bwd(q, k, v, do, causal=True, softmax_scale=None):
    bsz, num_heads, seq_len_q, dim_head = q.shape
    seq_len_kv = k.shape[2]
    assert k.shape == v.shape == (bsz, num_heads, seq_len_kv, dim_head)

    bh = bsz * num_heads

    def merge_heads(x):
        return x.contiguous().view(bh, x.shape[2], dim_head)

    def unmerge_heads(x, b, h):
        bh, seq_len, dim = x.shape
        assert bh == b * h
        return x.view(b, h, seq_len, dim)

    def grid(meta):
        return triton.cdiv(seq_len_q, meta["block_size_q"]), bh

    q_m = merge_heads(q)
    k_m = merge_heads(k)
    v_m = merge_heads(v)
    do_m = merge_heads(do)

    dq_m = torch.zeros_like(q_m)
    dk_m = torch.zeros_like(k_m)
    dv_m = torch.zeros_like(v_m)

    stride_q_bh, stride_q_seq, stride_q_dim = q_m.stride()
    stride_k_bh, stride_k_seq, stride_k_dim = k_m.stride()
    stride_v_bh, stride_v_seq, stride_v_dim = v_m.stride()
    stride_do_bh, stride_do_seq, stride_do_dim = do_m.stride()
    stride_dq_bh, stride_dq_seq, stride_dq_dim = dq_m.stride()
    stride_dk_bh, stride_dk_seq, stride_dk_dim = dk_m.stride()
    stride_dv_bh, stride_dv_seq, stride_dv_dim = dv_m.stride()

    if softmax_scale is None:
        softmax_scale = 1.0 / (dim_head ** 0.5)

    flash_attn_kernel_bwd[grid](
        q_m, k_m, v_m, do_m,
        dq_m, dk_m, dv_m,
        seq_len_q, seq_len_kv,
        stride_q_bh, stride_q_seq, stride_q_dim,
        stride_k_bh, stride_k_seq, stride_k_dim,
        stride_v_bh, stride_v_seq, stride_v_dim,
        stride_do_bh, stride_do_seq, stride_do_dim,
        stride_dq_bh, stride_dq_seq, stride_dq_dim,
        stride_dk_bh, stride_dk_seq, stride_dk_dim,
        stride_dv_bh, stride_dv_seq, stride_dv_dim,
        softmax_scale=softmax_scale,
        causal=causal,
        dim=dim_head,
    )

    dq = unmerge_heads(dq_m, bsz, num_heads)
    dk = unmerge_heads(dk_m, bsz, num_heads)
    dv = unmerge_heads(dv_m, bsz, num_heads)
    return dq, dk, dv
