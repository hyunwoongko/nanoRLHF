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
    key=["seq_len_kv", "dim"]
)
@triton.jit
def flash_attn_kernel_fwd(
    q_ptr, k_ptr, v_ptr, o_ptr,
    seq_len_q, seq_len_kv,
    stride_q_bh, stride_q_seq, stride_q_dim,
    stride_k_bh, stride_k_seq, stride_k_dim,
    stride_v_bh, stride_v_seq, stride_v_dim,
    stride_o_bh, stride_o_seq, stride_o_dim,
    softmax_scale,
    causal: tl.constexpr,
    dim: tl.constexpr,
    block_size_q: tl.constexpr,
    tile_size_kv: tl.constexpr,
):
    """
    How does the flash-attention kernel tile each tensor?
        The attention kernel partitions the query sequence length into blocks.
        One program block processes `block_size_q` query tokens.
        Key/Value are not block-partitioned; a single program block loops over
        the entire key/value sequence.

        In short:
            - Query: block tiling along the sequence-length dimension
            - Key/Value: loop tiling along the sequence-length dimension

    Why tile only along the sequence-length dimension?
        Sequence length can range from hundreds to tens of thousands, which is
        too large for a single block. The feature dimension `dim` is split across
        heads and is comparatively small. For example, in Qwen3-32B, dim=5120 and
        with 64 heads the per-head dim is only 128.

    Query original:
        ------------------- dim -------------------
        |------|------|------|------|------|------|  |
        | Q_00 | Q_01 | Q_02 | Q_03 | Q_04 | Q_05 |  | → token0
        |------|------|------|------|------|------|  |
        | Q_10 | Q_11 | Q_12 | Q_13 | Q_14 | Q_15 |  s → token1
        |------|------|------|------|------|------|  e
        | Q_20 | Q_21 | Q_22 | Q_23 | Q_24 | Q_25 |  q → token2
        |------|------|------|------|------|------|  |
        | Q_30 | Q_31 | Q_32 | Q_33 | Q_34 | Q_35 |  l → token3
        |------|------|------|------|------|------|  e
        | Q_40 | Q_41 | Q_42 | Q_43 | Q_44 | Q_45 |  n → token4
        |------|------|------|------|------|------|  |
        | Q_50 | Q_51 | Q_52 | Q_53 | Q_54 | Q_55 |  | → token5
        |------|------|------|------|------|------|  |

    Query blocked (block_size_q=2):
        ------------------- dim -------------------  b
        |------|------|------|------|------|------|  l
        | Q_00 | Q_01 | Q_02 | Q_03 | Q_04 | Q_05 |  o → token0
        |------|------|------|------|------|------|  c
        | Q_10 | Q_11 | Q_12 | Q_13 | Q_14 | Q_15 |  k → token1
        |------|------|------|------|------|------|  1

        ------------------- dim -------------------  b
        |------|------|------|------|------|------|  l
        | Q_20 | Q_21 | Q_22 | Q_23 | Q_24 | Q_25 |  o → token2
        |------|------|------|------|------|------|  c
        | Q_30 | Q_31 | Q_32 | Q_33 | Q_34 | Q_35 |  k → token3
        |------|------|------|------|------|------|  2
                            ...

    Query blocked * Key^T:
        ------------------- dim -------------------  b         |  ---- loop1 ----     ---- loop2 ----
        |------|------|------|------|------|------|  l         |  |------|------|     |------|------|
        | Q_00 | Q_01 | Q_02 | Q_03 | Q_04 | Q_05 |  o         |  | K_00 | K_10 |     | K_20 | K_30 |
        |------|------|------|------|------|------|  c     *   |  |------|------|     |------|------|
        | Q_10 | Q_11 | Q_12 | Q_13 | Q_14 | Q_15 |  k         |  | K_01 | K_11 |     | K_21 | K_31 |
        |------|------|------|------|------|------|  1         |  |------|------|     |------|------|
                                                               d  | K_02 | K_12 |     | K_22 | K_32 |
                                                               i  |------|------|  →  |------|------|  ...
                                                               m  | K_03 | K_13 |     | K_23 | K_33 |
                                                               |  |------|------|     |------|------|
                                                               |  | K_04 | K_14 |     | K_24 | K_34 |
                                                               |  |------|------|     |------|------|
                                                               |  | K_05 | K_15 |     | K_25 | K_35 |
                                                               |  |------|------|     |------|------|
                                                                     ↓      ↓            ↓      ↓
                                                                  token0  token1      token2  token3
    Streaming softmax (Online softmax):
        Vanilla softmax for one query token requires the entire Key/Value range.
        Because we loop over Key/Value tiles, we cannot use the standard formula
        directly. We therefore use an streaming (online) softmax that updates
        statistics per tile.

        Standard softmax:
            >>> import numpy as np
            >>>
            >>> def standard_softmax(x):
            ...     x_max = np.max(x)
            ...     ez = np.exp(x - x_max)
            ...     return ez / ez.sum()

        Streaming softmax:
            >>> def streaming_softmax(x, tile_size):
            ...    x_max = -np.inf
            ...    ez_sum = 0.0
            ...    for idx in range(0, x.size, tile_size):
            ...        current_x = x[idx:idx + tile_size]
            ...        current_x_max = np.max(current_x)
            ...        new_x_max = np.maximum(current_x_max, x_max)
            ...        rescale = np.exp(x_max - new_x_max)
            ...        ez_sum *= rescale
            ...        ez_sum += np.exp(current_x - new_x_max).sum()
            ...        x_max = new_x_max
            ...    return np.exp(x - x_max) / ez_sum

        The two functions produce identical results:
            >>> x = np.random.randn(1024)
            >>> y_standard = standard_softmax(x)
            >>> y_streaming = streaming_softmax(x, tile_size=64)
            >>> print(np.allclose(y_standard, y_streaming))  # True

        Key idea:
            - Quantities like `x_max` and `ez_sum` cannot be computed in one pass
              per tile subset, so we update them incrementally per tile.
            - After the final updates, we use the final `x_max` and `ez_sum`
              to recover the normalized softmax values.
    """
    pid_q = tl.program_id(0)
    pid_bh = tl.program_id(1)

    # We need to account for the (batch * head) dimension by adding a stride
    # for the base pointer on the bh dimension. In practice batch and head are
    # reshaped into a single dimension when entering the kernel.
    q_bh = q_ptr + pid_bh * stride_q_bh
    k_bh = k_ptr + pid_bh * stride_k_bh
    v_bh = v_ptr + pid_bh * stride_v_bh
    o_bh = o_ptr + pid_bh * stride_o_bh

    # Query is processed in blocks; its pointer depends on the block id.
    # Key/Value are processed via a loop; create local pointers and adjust in-loop.
    q_start = pid_q * block_size_q
    offs_q = q_start + tl.arange(0, block_size_q)
    offs_kv = tl.arange(0, tile_size_kv)

    q_block_ptr = tl.make_block_ptr(
        base=q_bh,
        shape=(seq_len_q, dim),
        offsets=(q_start, 0),  # ← The dim axis is not block-partitioned, so all blocks start at 0 on dim.
        block_shape=(block_size_q, dim),
        strides=(stride_q_seq, stride_q_dim),
        order=(1, 0),
    )
    o_block_ptr = tl.make_block_ptr(
        base=o_bh,
        shape=(seq_len_q, dim),
        offsets=(q_start, 0),  # ← The dim axis is not block-partitioned, so all blocks start at 0 on dim.
        block_shape=(block_size_q, dim),
        strides=(stride_o_seq, stride_o_dim),
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

    for kv_start in range(0, seq_len_kv, tile_size_kv):
        # Create K and V block pointers
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

        # Load K and V tiles
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

        # (Q * K^T) / sqrt(d)
        scores = tl.dot(q, tl.trans(k)) * softmax_scale

        # Apply causal mask if needed
        if causal:
            q_pos = offs_q[:, None]
            kv_pos = (kv_start + offs_kv)[None, :]
            mask = kv_pos > q_pos
            scores = tl.where(mask, -float("inf"), scores)

        # Streaming softmax update
        current_max_q = tl.max(scores, axis=1)
        new_max_q = tl.maximum(max_q, current_max_q)
        rescale = tl.exp(max_q - new_max_q)
        current_ez = tl.exp(scores - new_max_q[:, None])
        ez_sum = ez_sum * rescale + tl.sum(current_ez, axis=1)
        ez_dot_v = ez_dot_v * rescale[:, None] + tl.dot(current_ez, v)
        max_q = new_max_q

    o = ez_dot_v / ez_sum[:, None]

    tl.store(
        o_block_ptr, o,
        boundary_check=(0, 1),
    )


def flash_attn_fwd(q, k, v, causal=True, softmax_scale=None):
    bsz, num_heads, seq_len_q, dim_head = q.shape
    seq_len_kv = k.shape[2]
    assert k.shape == v.shape == (bsz, num_heads, seq_len_kv, dim_head)

    bh = bsz * num_heads

    def merge_heads(x):
        return x.contiguous().view(bh, x.shape[2], dim_head)

    def grid(meta):
        return triton.cdiv(seq_len_q, meta["block_size_q"]), bh

    q_merged = merge_heads(q)
    k_merged = merge_heads(k)
    v_merged = merge_heads(v)
    o = torch.empty_like(q_merged)

    stride_q_bh, stride_q_seq, stride_q_dim = q_merged.stride()
    stride_k_bh, stride_k_seq, stride_k_dim = k_merged.stride()
    stride_v_bh, stride_v_seq, stride_v_dim = v_merged.stride()
    stride_o_bh, stride_o_seq, stride_o_dim = o.stride()

    if softmax_scale is None:
        softmax_scale = 1.0 / (dim_head ** 0.5)

    flash_attn_kernel_fwd[grid](
        q_merged, k_merged, v_merged, o,
        seq_len_q, seq_len_kv,
        stride_q_bh, stride_q_seq, stride_q_dim,
        stride_k_bh, stride_k_seq, stride_k_dim,
        stride_v_bh, stride_v_seq, stride_v_dim,
        stride_o_bh, stride_o_seq, stride_o_dim,
        softmax_scale=softmax_scale,
        causal=causal,
        dim=dim_head,
    )

    return o.view(bsz, num_heads, seq_len_q, dim_head)
