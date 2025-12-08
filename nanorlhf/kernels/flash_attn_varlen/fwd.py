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
    key=["dim"]
)
@triton.jit
def flash_attn_varlen_fwd_kernel(
    q_ptr, k_ptr, v_ptr,
    cu_seqlens_q_ptr, cu_seqlens_k_ptr,
    o_ptr, max_q_ptr, ez_sum_ptr,
    bsz, num_heads,
    stride_q_tok, stride_q_head, stride_q_dim,
    stride_k_tok, stride_k_head, stride_k_dim,
    stride_v_tok, stride_v_head, stride_v_dim,
    stride_o_tok, stride_o_head, stride_o_dim,
    stride_max_q_head, stride_max_q_tok,
    stride_ez_sum_head, stride_ez_sum_tok,
    softmax_scale,
    max_seqlen_k,
    block_table_ptr,
    block_table_batch_stride,
    page_block_size,
    use_block_table: tl.constexpr,
    causal: tl.constexpr,
    block_size_q: tl.constexpr,
    tile_size_kv: tl.constexpr,
    dim: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_m = tl.program_id(1)

    seq_id = pid_bh // num_heads
    head_id = pid_bh % num_heads

    q_start = tl.load(cu_seqlens_q_ptr + seq_id)
    q_end = tl.load(cu_seqlens_q_ptr + seq_id + 1)
    seqlen_q = q_end - q_start

    k_start = tl.load(cu_seqlens_k_ptr + seq_id)
    k_end = tl.load(cu_seqlens_k_ptr + seq_id + 1)
    seqlen_k = k_end - k_start

    block_q_start = pid_m * block_size_q
    offs_q = block_q_start + tl.arange(0, block_size_q)
    offs_kv = tl.arange(0, tile_size_kv)
    offs_d = tl.arange(0, dim)
    q_mask = offs_q < seqlen_q

    if block_q_start >= seqlen_q:
        return

    q_indices = q_start + offs_q

    # Per-head, per-sequence bases (token axis is the first dim of shape)
    q_head_seq_base = q_ptr + head_id * stride_q_head + q_start * stride_q_tok
    k_head_seq_base = k_ptr + head_id * stride_k_head + k_start * stride_k_tok
    v_head_seq_base = v_ptr + head_id * stride_v_head + k_start * stride_v_tok
    o_head_seq_base = o_ptr + head_id * stride_o_head + q_start * stride_o_tok

    # This is for vllm-style paged attention
    k_head_cache_base = k_ptr + head_id * stride_k_head
    v_head_cache_base = v_ptr + head_id * stride_v_head

    max_q_head_base = max_q_ptr + head_id * stride_max_q_head
    ez_sum_head_base = ez_sum_ptr + head_id * stride_ez_sum_head

    q_block_ptr = tl.make_block_ptr(
        base=q_head_seq_base,
        shape=(seqlen_q, dim),
        strides=(stride_q_tok, stride_q_dim),
        offsets=(block_q_start, 0),
        block_shape=(block_size_q, dim),
        order=(1, 0),
    )
    q = tl.load(
        q_block_ptr,
        boundary_check=(0, 1),
        padding_option="zero",
    )
    max_q = tl.full((block_size_q,), -float("inf"), dtype=tl.float32)
    ez_sum = tl.zeros((block_size_q,), dtype=tl.float32)
    ez_dot_v = tl.zeros((block_size_q, dim), dtype=tl.float32)

    for kv_start in range(0, seqlen_k, tile_size_kv):
        kv_idx = kv_start + offs_kv
        kv_mask = kv_idx < seqlen_k

        if not use_block_table:
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
            )
            v = tl.load(
                v_block_ptr,
                boundary_check=(0, 1),
                padding_option="zero",
            )
        else:
            page_idx = kv_idx // page_block_size
            page_offs = kv_idx % page_block_size

            table_row = seq_id * block_table_batch_stride + page_idx
            page_id = tl.load(block_table_ptr + table_row, mask=kv_mask, other=0)
            cache_row = page_id * page_block_size + page_offs
            cache_row_mask = kv_mask & (cache_row < max_seqlen_k)

            k_row_ptrs = (
                k_head_cache_base
                + cache_row[:, None] * stride_k_tok
                + offs_d[None, :] * stride_k_dim
            )
            v_row_ptrs = (
                v_head_cache_base
                + cache_row[:, None] * stride_v_tok
                + offs_d[None, :] * stride_v_dim
            )

            elem_mask = cache_row_mask[:, None] & (offs_d[None, :] < dim)
            k = tl.load(k_row_ptrs, mask=elem_mask, other=0.0)
            v = tl.load(v_row_ptrs, mask=elem_mask, other=0.0)

        scores = tl.dot(q, tl.trans(k)) * softmax_scale
        base_mask = (~q_mask[:, None]) | (~kv_mask[None, :])

        if causal:
            offset = seqlen_k - seqlen_q
            q_pos = (offset + offs_q)[:, None]
            kv_pos = kv_idx[None, :]
            causal_mask = kv_pos > q_pos
            mask = base_mask | causal_mask
        else:
            mask = base_mask

        scores = tl.where(mask, -float("inf"), scores)
        current_max_q = tl.max(scores, axis=1)
        new_max_q = tl.maximum(max_q, current_max_q)
        rescale = tl.exp(max_q - new_max_q)
        current_ez = tl.exp(scores - new_max_q[:, None])
        ez_sum = ez_sum * rescale + tl.sum(current_ez, axis=1)
        ez_dot_v = ez_dot_v * rescale[:, None] + tl.dot(current_ez.to(v.dtype), v, out_dtype=tl.float32)
        max_q = new_max_q

    # Prevent division by zero
    ez_sum = tl.maximum(ez_sum, 1e-6)
    o = ez_dot_v / ez_sum[:, None]

    # Output block store
    o_block_ptr = tl.make_block_ptr(
        base=o_head_seq_base,
        shape=(seqlen_q, dim),
        strides=(stride_o_tok, stride_o_dim),
        offsets=(block_q_start, 0),
        block_shape=(block_size_q, dim),
        order=(1, 0),
    )
    tl.store(
        o_block_ptr,
        o.to(q.dtype),
        boundary_check=(0, 1),
    )

    tl.store(max_q_head_base + q_indices * stride_max_q_tok, max_q, mask=q_mask)
    tl.store(ez_sum_head_base + q_indices * stride_ez_sum_tok, ez_sum, mask=q_mask)


def flash_attn_varlen_fwd(
    q, k, v,
    cu_seqlens_q, cu_seqlens_k,
    bsz, num_heads,
    max_seqlen_q, max_seqlen_k,
    causal=True, softmax_scale=None,
    block_table=None, page_block_size=None,
):
    assert q.is_cuda and k.is_cuda and v.is_cuda
    assert q.dim() == 3 and k.dim() == 3 and v.dim() == 3
    assert q.shape[1] == k.shape[1] == v.shape[1]
    assert q.shape[2] == k.shape[2] == v.shape[2]
    assert cu_seqlens_q.shape[0] == bsz + 1
    assert cu_seqlens_k.shape[0] == bsz + 1

    total_q, num_heads_q, dim = q.shape
    total_k, num_heads_k, dim_k = k.shape
    assert num_heads_q == num_heads_k == num_heads
    assert dim == dim_k

    o = torch.empty_like(q)
    max_q = torch.empty(num_heads, total_q, device=q.device, dtype=torch.float32)
    ez_sum = torch.empty(num_heads, total_q, device=q.device, dtype=torch.float32)

    stride_q_tok, stride_q_head, stride_q_dim = q.stride()
    stride_k_tok, stride_k_head, stride_k_dim = k.stride()
    stride_v_tok, stride_v_head, stride_v_dim = v.stride()
    stride_o_tok, stride_o_head, stride_o_dim = o.stride()
    stride_max_q_head, stride_max_q_tok = max_q.stride()
    stride_ez_sum_head, stride_ez_sum_tok = ez_sum.stride()

    if softmax_scale is None:
        softmax_scale = 1.0 / (dim ** 0.5)

    use_block_table = block_table is not None
    if use_block_table:
        assert page_block_size is not None
        assert block_table.dim() == 2
        block_table = block_table.contiguous()
        block_table_batch_stride = block_table.shape[1]
        block_table_ptr = block_table.view(-1)
        page_block_size = int(page_block_size)
    else:
        # dummy values
        block_table_batch_stride = 0
        page_block_size = 1
        block_table_ptr = q

    def grid(meta):
        return bsz * num_heads, triton.cdiv(max_seqlen_q, meta["block_size_q"])

    flash_attn_varlen_fwd_kernel[grid](
        q, k, v,
        cu_seqlens_q, cu_seqlens_k,
        o, max_q, ez_sum,
        bsz, num_heads,
        stride_q_tok, stride_q_head, stride_q_dim,
        stride_k_tok, stride_k_head, stride_k_dim,
        stride_v_tok, stride_v_head, stride_v_dim,
        stride_o_tok, stride_o_head, stride_o_dim,
        stride_max_q_head, stride_max_q_tok,
        stride_ez_sum_head, stride_ez_sum_tok,
        softmax_scale,
        max_seqlen_k,
        block_table_ptr,
        block_table_batch_stride,
        page_block_size,
        use_block_table,
        causal=causal,
        dim=dim,
    )
    return o, max_q, ez_sum
