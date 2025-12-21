from typing import Optional, Tuple

import torch
import triton
import triton.language as tl


@triton.jit
def flash_attn_kernel_fwd(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    max_q_ptr,
    ez_sum_ptr,
    mask_ptr,
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
    stride_o_bh,
    stride_o_seq,
    stride_o_dim,
    stride_max_bh,
    stride_max_seq,
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
        directly. We therefore use streaming (online) softmax that updates
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

    # batch * head base
    q_bh = q_ptr + pid_bh * stride_q_bh
    k_bh = k_ptr + pid_bh * stride_k_bh
    v_bh = v_ptr + pid_bh * stride_v_bh
    o_bh = o_ptr + pid_bh * stride_o_bh

    # Query block index
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
    o_block_ptr = tl.make_block_ptr(
        base=o_bh,
        shape=(seq_len_q, dim),
        offsets=(q_start, 0),
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

    if has_mask:
        batch_idx = pid_bh // num_heads
        head_idx = pid_bh % num_heads
        mask_head_idx = tl.minimum(head_idx, mask_num_heads - 1)
        mask_bh = mask_ptr + batch_idx * stride_mask_b + mask_head_idx * stride_mask_h

    for kv_start in range(0, seq_len_k, tile_size_kv):
        k_block_ptr = tl.make_block_ptr(
            base=k_bh,
            shape=(seq_len_k, dim),
            offsets=(kv_start, 0),
            block_shape=(tile_size_kv, dim),
            strides=(stride_k_seq, stride_k_dim),
            order=(1, 0),
        )
        v_block_ptr = tl.make_block_ptr(
            base=v_bh,
            shape=(seq_len_k, dim),
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

        scores = tl.dot(q.to(k.dtype), tl.trans(k)) * softmax_scale

        if has_mask:
            mask_block_ptr = tl.make_block_ptr(
                base=mask_bh,  # noqa
                shape=(seq_len_q, seq_len_k),
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
            # mask tile is already has -inf where masked.
            scores = scores + mask_tile.to(tl.float32)

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

    tl.store(
        o_block_ptr,
        o.to(q.dtype),
        boundary_check=(0, 1),
    )

    # Save max_q and ez_sum for backward pass
    max_q_ptrs = max_q_ptr + pid_bh * stride_max_bh + offs_q * stride_max_seq
    ez_sum_ptrs = ez_sum_ptr + pid_bh * stride_ez_sum_bh + offs_q * stride_ez_sum_seq

    tl.store(max_q_ptrs, max_q, mask=q_mask)
    tl.store(ez_sum_ptrs, ez_sum, mask=q_mask)


def flash_attn_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Flash attention forward pass.

    Args:
        q: Query tensor of shape (bsz, num_heads, seq_len_q, dim) or (bh, seq_len_q, dim)
        k: Key tensor of shape (bsz, num_heads, seq_len_k, dim) or (bh, seq_len_k, dim)
        v: Value tensor of shape (bsz, num_heads, seq_len_k, dim) or (bh, seq_len_k, dim)
        attention_mask: Optional attention mask of shape (bsz, num_heads_mask, seq_len_q, seq_len_k)
        softmax_scale: Optional scaling factor for softmax
        causal: Whether to apply causal masking

    Returns:
        Tuple of output tensor, max_q, ez_sum for backward pass
    """
    input_ndim = q.ndim

    if input_ndim == 4:
        bsz, num_heads_q, seq_len_q, dim = q.shape
        seq_len_k = k.shape[2]
        bh = bsz * num_heads_q
        assert k.shape[-1] == dim and v.shape[-1] == dim
        assert k.shape[0] == bsz and v.shape[0] == bsz

        def merge_heads(x):
            return x.contiguous().view(bh, x.shape[2], dim)

        q = merge_heads(q)
        k = merge_heads(k)
        v = merge_heads(v)
    elif input_ndim == 3:
        bh, seq_len_q, dim = q.shape
        seq_len_k = k.shape[1]
        bsz = 1
        num_heads_q = bh
        assert k.shape == v.shape == (bh, seq_len_k, dim)
    else:
        raise ValueError("q, k, v must be 3D or 4D tensors")

    def grid(meta):
        return triton.cdiv(seq_len_q, meta["block_size_q"]), bh

    o = torch.empty_like(q)
    max_q = torch.full((bh, seq_len_q), -float("inf"), device=q.device, dtype=torch.float32)
    ez_sum = torch.zeros(bh, seq_len_q, device=q.device, dtype=torch.float32)

    stride_q_bh, stride_q_seq, stride_q_dim = q.stride()
    stride_k_bh, stride_k_seq, stride_k_dim = k.stride()
    stride_v_bh, stride_v_seq, stride_v_dim = v.stride()
    stride_o_bh, stride_o_seq, stride_o_dim = o.stride()
    stride_max_bh, stride_max_seq = max_q.stride()
    stride_ez_sum_bh, stride_ez_sum_seq = ez_sum.stride()

    has_mask = attention_mask is not None
    if has_mask:
        assert attention_mask.ndim == 4
        assert attention_mask.shape[0] == bsz
        assert attention_mask.shape[-2] == seq_len_q or attention_mask.shape[-1] == seq_len_k

        mask = attention_mask.contiguous()
        mask_num_heads = mask.shape[1]
        stride_mask_b, stride_mask_h, stride_mask_q, stride_mask_k = mask.stride()
    else:
        mask = q  # dummy
        mask_num_heads = 1
        stride_mask_b = stride_mask_h = stride_mask_q = stride_mask_k = 0

    if softmax_scale is None:
        softmax_scale = 1.0 / (dim**0.5)

    flash_attn_kernel_fwd[grid](
        q,
        k,
        v,
        o,
        max_q,
        ez_sum,
        mask,
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
        stride_o_bh,
        stride_o_seq,
        stride_o_dim,
        stride_max_bh,
        stride_max_seq,
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
        dim=dim,
        block_size_q=64,
        tile_size_kv=64,
    )

    if input_ndim == 4:
        o = o.view(bsz, num_heads_q, seq_len_q, dim)
    return o, max_q, ez_sum
