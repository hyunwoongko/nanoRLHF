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
def flash_attn_varlen_fwd_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    cu_seq_lens_q_ptr,
    cu_seq_lens_k_ptr,
    o_ptr,
    max_q_ptr,
    ez_sum_ptr,
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
    softmax_scale,
    causal: tl.constexpr,
    block_size_q: tl.constexpr,
    tile_size_kv: tl.constexpr,
    dim: tl.constexpr,
):
    """
    Flash Attention forward kernel for variable-length sequences.

    Discussion:
        Q. How does this kernel handle variable-length sequences?
            This kernel uses cumulative sequence length pointers (cu_seq_lens_q and cu_seq_lens_k) to determine the
            start and end positions of each sequence within the batch. By calculating the sequence lengths dynamically
            for each sequence, it can process sequences of variable lengths without computation on padding tokens.

        Q. How this kernel uses cumulative sequence lengths to manage variable-length sequences?
            We can see the following code snippet in the kernel implementation:

                ```
                seq_id = pid_bh // num_heads

                q_start = tl.load(cu_seq_lens_q_ptr + seq_id)
                q_end = tl.load(cu_seq_lens_q_ptr + seq_id + 1)
                seq_len_q = q_end - q_start

                k_start = tl.load(cu_seq_lens_k_ptr + seq_id)
                k_end = tl.load(cu_seq_lens_k_ptr + seq_id + 1)
                seq_len_k = k_end - k_start
                ```

            The kernel takes in two pointers, `cu_seq_lens_q_ptr` and `cu_seq_lens_k_ptr`, which point to arrays
            containing the cumulative sequence lengths for the query and key/value sequences, respectively. For each
            sequence in the batch, the kernel calculates the start and end indices by loading values from these arrays.
            The sequence length is determined by subtracting the start index from the end index. This allows the kernel
            to identify the valid tokens for each sequence and avoid processing padding tokens, ensuring efficient
            computation for variable-length sequences.
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
    offs_kv = tl.arange(0, tile_size_kv)
    q_mask = offs_q < seq_len_q

    if block_q_start >= seq_len_q:
        return

    q_indices = q_start + offs_q

    # Per-head, per-sequence bases (token axis is the first dim of shape)
    q_head_seq_base = q_ptr + head_id * stride_q_head + q_start * stride_q_tok
    k_head_seq_base = k_ptr + head_id * stride_k_head + k_start * stride_k_tok
    v_head_seq_base = v_ptr + head_id * stride_v_head + k_start * stride_v_tok
    o_head_seq_base = o_ptr + head_id * stride_o_head + q_start * stride_o_tok

    max_q_head_base = max_q_ptr + head_id * stride_max_q_head
    ez_sum_head_base = ez_sum_ptr + head_id * stride_ez_sum_head

    q_block_ptr = tl.make_block_ptr(
        base=q_head_seq_base,
        shape=(seq_len_q, dim),
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
        )
        v = tl.load(
            v_block_ptr,
            boundary_check=(0, 1),
            padding_option="zero",
        )
        scores = tl.dot(q.to(k.dtype), tl.trans(k)) * softmax_scale
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

    # Output block store
    o_block_ptr = tl.make_block_ptr(
        base=o_head_seq_base,
        shape=(seq_len_q, dim),
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
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seq_lens_q: torch.Tensor,
    cu_seq_lens_k: torch.Tensor,
    batch_size: int,
    num_heads: int,
    max_seq_len_q: int,
    max_seq_len_k: int,
    causal: bool = True,
    softmax_scale: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Flash Attention forward function for variable-length sequences.

    Args:
        q (torch.Tensor): Query tensor of shape (total_q, num_heads, dim).
        k (torch.Tensor): Key tensor of shape (total_k, num_heads, dim).
        v (torch.Tensor): Value tensor of shape (total_k, num_heads, dim).
        cu_seq_lens_q (torch.Tensor): Cumulative sequence lengths for queries of shape (batch_size + 1,).
        cu_seq_lens_k (torch.Tensor): Cumulative sequence lengths for keys/values of shape (batch_size + 1,).
        batch_size (int): Number of sequences in the batch.
        num_heads (int): Number of attention heads.
        max_seq_len_q (int): Maximum sequence length for queries.
        max_seq_len_k (int): Maximum sequence length for keys/values.
        causal (bool): Whether to apply causal masking. Default is True.
        softmax_scale (Optional[float]): Scaling factor for softmax. If None, defaults to 1/sqrt(dim).

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            - Output tensor of shape (total_q, num_heads, dim).
            - Max logits tensor of shape (num_heads, total_q).
            - Exponential sum tensor of shape (num_heads, total_q).
    """
    assert q.is_cuda and k.is_cuda and v.is_cuda
    assert q.dim() == 3 and k.dim() == 3 and v.dim() == 3
    assert q.shape[1] == k.shape[1] == v.shape[1]
    assert q.shape[2] == k.shape[2] == v.shape[2]
    assert cu_seq_lens_q.shape[0] == batch_size + 1
    assert cu_seq_lens_k.shape[0] == batch_size + 1

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
        softmax_scale = 1.0 / (dim**0.5)

    def grid(meta):
        return batch_size * num_heads, triton.cdiv(max_seq_len_q, meta["block_size_q"])

    flash_attn_varlen_fwd_kernel[grid](
        q,
        k,
        v,
        cu_seq_lens_q,
        cu_seq_lens_k,
        o,
        max_q,
        ez_sum,
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
        softmax_scale,
        causal=causal,
        dim=dim,
    )
    return o, max_q, ez_sum
