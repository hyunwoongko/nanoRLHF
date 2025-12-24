import triton
import triton.language as tl


@triton.jit
def flash_attn_decode_kernel_split_k_paged(
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    block_tables_ptr,
    context_lens_ptr,
    ez_dot_v_ptr,
    max_q_ptr,
    ez_sum_ptr,
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
    kv_block_size: tl.constexpr,
    kv_heads: tl.constexpr,
    num_heads: tl.constexpr,
    causal: tl.constexpr,
    dim: tl.constexpr,
    block_size_q: tl.constexpr,
    block_size_k: tl.constexpr,
    max_num_blocks: tl.constexpr,
):
    """
    Decoding kernel for Flash Attention with Split-K and paged KV cache.

    Discussion:
        Q. Why Split-K is needed for decoding?
            In decoding, the query length (seq_len_q) is just 1 (for one token at a time),
            So it does not make sense to split along the query length dimension.
            However, the key/value cache can grow very long (seq_len_k) as more tokens are generated.
            To efficiently handle long key/value caches, we split along the key/value length dimension (Split-K).

        Q. Can we compute the final output directly in this kernel?
            No. We can't directly compute the final softmax output from partial results of each split,
            because softmax is a non-linear operation. To address this, we call Reduce-K kernel after this kernel.
            The Reduce-K kernel will aggregate the partial results from all splits to produce the final output.

        Q. What is difference between this paged kernel and the deprecated kernel?
            This kernel gets the huge KV cache directly and loads only the needed KV tensors based on the block table.
            But the deprecated kernel requires preloaded KV tensors by Python code.
            The problem is the KV cache loading function written in Python can not be used with CUDA graphs,
            because it has dynamic control flow.
    """
    pid_q_block = tl.program_id(0)
    pid_bh = tl.program_id(1)
    pid_split = tl.program_id(2)

    q_start = pid_q_block * block_size_q
    if q_start >= seq_len_q:
        return

    b = pid_bh // num_heads
    h = pid_bh - b * num_heads

    group_size = num_heads // kv_heads
    kv_h = h // group_size

    ctx_len = tl.load(context_lens_ptr + b * stride_cl_b).to(tl.int32)

    offs_q = q_start + tl.arange(0, block_size_q)
    offs_k = tl.arange(0, block_size_k)
    q_mask = offs_q < seq_len_q

    q_bh = q_ptr + pid_bh * stride_q_bh
    q_block_ptr = tl.make_block_ptr(
        base=q_bh,
        shape=(seq_len_q, dim),
        offsets=(q_start, 0),
        block_shape=(block_size_q, dim),
        strides=(stride_q_seq, stride_q_dim),
        order=(1, 0),
    )
    q = tl.load(q_block_ptr, boundary_check=(0, 1), padding_option='zero')

    seq_len_k_cap = max_num_blocks * kv_block_size
    seq_len_k = tl.minimum(ctx_len, seq_len_k_cap)

    k_low = pid_split * block_n_per_split
    k_high = tl.minimum((pid_split + 1) * block_n_per_split, seq_len_k)

    max_q = tl.full((block_size_q,), -float('inf'), dtype=tl.float32)
    ez_sum = tl.zeros((block_size_q,), dtype=tl.float32)
    ez_dot_v = tl.zeros((block_size_q, dim), dtype=tl.float32)

    if k_low >= k_high:
        ez_dot_v_offset = ez_dot_v_ptr + pid_bh * stride_ez_dot_v_bh + pid_split * stride_ez_dot_v_split
        ez_dot_v_block_ptr = tl.make_block_ptr(
            base=ez_dot_v_offset,
            shape=(seq_len_q, dim),
            offsets=(q_start, 0),
            block_shape=(block_size_q, dim),
            strides=(stride_ez_dot_v_seq, stride_ez_dot_v_dim),
            order=(1, 0),
        )
        tl.store(ez_dot_v_block_ptr, ez_dot_v)

        max_q_block_ptr = (
            max_q_ptr + pid_bh * stride_max_q_bh + pid_split * stride_max_q_split + offs_q * stride_max_q_seq
        )
        ez_sum_block_ptr = (
            ez_sum_ptr + pid_bh * stride_ez_sum_bh + pid_split * stride_ez_sum_split + offs_q * stride_ez_sum_seq
        )
        tl.store(max_q_block_ptr, max_q, mask=q_mask)
        tl.store(ez_sum_block_ptr, ez_sum, mask=q_mask)
        return

    for kv_start in range(k_low, k_high, block_size_k):
        kv_idx = kv_start + offs_k
        kv_mask = kv_idx < k_high

        block_idx = kv_idx // kv_block_size
        block_off = kv_idx - block_idx * kv_block_size

        bt_base = block_tables_ptr + b * stride_bt_b
        page_id = tl.load(bt_base + block_idx * stride_bt_blk, mask=kv_mask, other=-1).to(tl.int32)

        kv_mask = kv_mask & (page_id >= 0)
        slot = page_id * kv_block_size + block_off
        d = tl.arange(0, dim)

        k_ptrs = (
            k_cache_ptr + slot[:, None] * stride_cache_slot + kv_h * stride_cache_head + d[None, :] * stride_cache_dim
        )
        v_ptrs = (
            v_cache_ptr + slot[:, None] * stride_cache_slot + kv_h * stride_cache_head + d[None, :] * stride_cache_dim
        )

        k_tile = tl.load(k_ptrs, mask=kv_mask[:, None], other=0.0)
        v_tile = tl.load(v_ptrs, mask=kv_mask[:, None], other=0.0)

        scores = tl.dot(q.to(k_tile.dtype), tl.trans(k_tile)) * softmax_scale
        base_mask = (~q_mask[:, None]) | (~kv_mask[None, :])

        if causal:
            offset = seq_len_k - seq_len_q
            q_pos = offset + offs_q[:, None]
            kv_pos = kv_idx[None, :]
            scores = tl.where(base_mask | (kv_pos > q_pos), -float('inf'), scores)
        else:
            scores = tl.where(base_mask, -float('inf'), scores)

        current_max_q = tl.max(scores, axis=1)
        new_max_q = tl.maximum(max_q, current_max_q)
        rescale = tl.exp(max_q - new_max_q)

        current_ez = tl.exp(scores - new_max_q[:, None])
        ez_sum = ez_sum * rescale + tl.sum(current_ez, axis=1)
        ez_dot_v = ez_dot_v * rescale[:, None] + tl.dot(current_ez.to(v_tile.dtype), v_tile, out_dtype=tl.float32)
        max_q = new_max_q

    ez_dot_v_offset = ez_dot_v_ptr + pid_bh * stride_ez_dot_v_bh + pid_split * stride_ez_dot_v_split
    ez_dot_v_block_ptr = tl.make_block_ptr(
        base=ez_dot_v_offset,
        shape=(seq_len_q, dim),
        offsets=(q_start, 0),
        block_shape=(block_size_q, dim),
        strides=(stride_ez_dot_v_seq, stride_ez_dot_v_dim),
        order=(1, 0),
    )
    tl.store(ez_dot_v_block_ptr, ez_dot_v)

    max_q_block_ptr = max_q_ptr + pid_bh * stride_max_q_bh + pid_split * stride_max_q_split + offs_q * stride_max_q_seq
    ez_sum_block_ptr = (
        ez_sum_ptr + pid_bh * stride_ez_sum_bh + pid_split * stride_ez_sum_split + offs_q * stride_ez_sum_seq
    )
    tl.store(max_q_block_ptr, max_q, mask=q_mask)
    tl.store(ez_sum_block_ptr, ez_sum, mask=q_mask)


# deprecated, kept for reference
@triton.jit
def flash_attn_decode_kernel_split_k(
    q_ptr,
    k_ptr,
    v_ptr,
    ez_dot_v_ptr,
    max_q_ptr,
    ez_sum_ptr,
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
    causal: tl.constexpr,
    dim: tl.constexpr,
    block_size_q: tl.constexpr,
    block_size_k: tl.constexpr,
):
    pid_q_block = tl.program_id(0)
    pid_bh = tl.program_id(1)
    pid_split_k = tl.program_id(2)

    q_start = pid_q_block * block_size_q
    if q_start >= seq_len_q:
        return

    offs_q = q_start + tl.arange(0, block_size_q)
    offs_k = tl.arange(0, block_size_k)
    q_mask = offs_q < seq_len_q

    q_bh = q_ptr + pid_bh * stride_q_bh
    k_bh = k_ptr + pid_bh * stride_k_bh
    v_bh = v_ptr + pid_bh * stride_v_bh

    q_block_ptr = tl.make_block_ptr(
        base=q_bh,
        shape=(seq_len_q, dim),
        offsets=(q_start, 0),
        block_shape=(block_size_q, dim),
        strides=(stride_q_seq, stride_q_dim),
        order=(1, 0),
    )

    q = tl.load(
        q_block_ptr,
        boundary_check=(0, 1),
        padding_option="zero",
    )

    k_low = pid_split_k * block_n_per_split
    k_high = tl.minimum((pid_split_k + 1) * block_n_per_split, seq_len_k)

    max_q = tl.full((block_size_q,), -float("inf"), dtype=tl.float32)
    ez_sum = tl.zeros((block_size_q,), dtype=tl.float32)
    ez_dot_v = tl.zeros((block_size_q, dim), dtype=tl.float32)

    for kv_start in range(k_low, k_high, block_size_k):
        k_block_ptr = tl.make_block_ptr(
            base=k_bh,
            shape=(seq_len_k, dim),
            offsets=(kv_start, 0),
            block_shape=(block_size_k, dim),
            strides=(stride_k_seq, stride_k_dim),
            order=(1, 0),
        )
        v_block_ptr = tl.make_block_ptr(
            base=v_bh,
            shape=(seq_len_k, dim),
            offsets=(kv_start, 0),
            block_shape=(block_size_k, dim),
            strides=(stride_v_seq, stride_v_dim),
            order=(1, 0),
        )

        k_tile = tl.load(
            k_block_ptr,
            boundary_check=(0, 1),
            padding_option="zero",
        )
        v_tile = tl.load(
            v_block_ptr,
            boundary_check=(0, 1),
            padding_option="zero",
        )

        scores = tl.dot(q.to(k_tile.dtype), tl.trans(k_tile)) * softmax_scale

        kv_idx = kv_start + offs_k
        kv_mask = kv_idx < k_high
        # kv_mask = kv_idx < seq_len_k
        base_mask = (~q_mask[:, None]) | (~kv_mask[None, :])

        if causal:
            offset = seq_len_k - seq_len_q
            q_pos = offset + offs_q[:, None]
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
        ez_dot_v = ez_dot_v * rescale[:, None] + tl.dot(current_ez.to(v_tile.dtype), v_tile, out_dtype=tl.float32)
        max_q = new_max_q

    # we save ez_dot_v and ez_sum separately and do the division in the reduce-k kernel.
    ez_dot_v_offset = ez_dot_v_ptr + pid_bh * stride_ez_dot_v_bh + pid_split_k * stride_ez_dot_v_split
    ez_dot_v_block_ptr = tl.make_block_ptr(
        base=ez_dot_v_offset,
        shape=(seq_len_q, dim),
        offsets=(q_start, 0),
        block_shape=(block_size_q, dim),
        strides=(stride_ez_dot_v_seq, stride_ez_dot_v_dim),
        order=(1, 0),
    )
    tl.store(ez_dot_v_block_ptr, ez_dot_v)

    max_q_block_ptr = (
        max_q_ptr + pid_bh * stride_max_q_bh + pid_split_k * stride_max_q_split + offs_q * stride_max_q_seq
    )
    ez_sum_block_ptr = (
        ez_sum_ptr + pid_bh * stride_ez_sum_bh + pid_split_k * stride_ez_sum_split + offs_q * stride_ez_sum_seq
    )
    tl.store(max_q_block_ptr, max_q, mask=q_mask)
    tl.store(ez_sum_block_ptr, ez_sum, mask=q_mask)
