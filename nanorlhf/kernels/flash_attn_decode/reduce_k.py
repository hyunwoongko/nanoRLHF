import triton
import triton.language as tl


@triton.jit
def flash_attn_decode_kernel_reduce_k(
    ez_dot_v_ptr,
    max_q_ptr,
    ez_sum_ptr,
    o_out_ptr,
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
    dim: tl.constexpr,
    block_size_q: tl.constexpr,
    split_k: tl.constexpr,
    causal: tl.constexpr,
):
    """
    A kernel to reduce the split_k dimension in flash attention decoding.

    Discussion:
        Q. Why do we need to reduce over split_k?
            In flash attention with split_k, the attention computation is divided into multiple splits along the key
            dimension. Each split computes partial results for the output. To obtain the final output, we need to
            aggregate these partial results by reducing over the split_k dimension. This kernel performs that reduction
            by combining the intermediate results (max_q, ez_sum, ez_dot_v) from each split to produce the final output.
    """
    pid_bh = tl.program_id(0)
    pid_q_block = tl.program_id(1)

    q_start = pid_q_block * block_size_q
    if q_start >= seq_len_q:
        return

    offs_q = q_start + tl.arange(0, block_size_q)
    q_mask = offs_q < seq_len_q

    max_q_global = tl.full((block_size_q,), -float('inf'), dtype=tl.float32)

    for split in range(split_k):
        max_q_block_ptr = max_q_ptr + pid_bh * stride_max_q_bh + split * stride_max_q_split + offs_q * stride_max_q_seq
        max_q_split = tl.load(max_q_block_ptr, mask=q_mask, other=-float('inf'))
        max_q_global = tl.maximum(max_q_global, max_q_split)

    ez_sum_global = tl.zeros((block_size_q,), dtype=tl.float32)
    ez_dot_v_global = tl.zeros((block_size_q, dim), dtype=tl.float32)

    for split in range(split_k):
        max_q_block_ptr = max_q_ptr + pid_bh * stride_max_q_bh + split * stride_max_q_split + offs_q * stride_max_q_seq
        ez_sum_block_ptr = (
            ez_sum_ptr + pid_bh * stride_ez_sum_bh + split * stride_ez_sum_split + offs_q * stride_ez_sum_seq
        )
        max_q_split = tl.load(max_q_block_ptr, mask=q_mask, other=-float('inf'))
        ez_sum_split = tl.load(ez_sum_block_ptr, mask=q_mask, other=0.0)

        if causal:
            max_q_offset = max_q_split - max_q_global
            alpha = tl.where(max_q_offset > -float('inf'), tl.exp(max_q_offset), 0.0)
        else:
            alpha = tl.exp(max_q_split - max_q_global)

        ez_sum_global += ez_sum_split * alpha

        ez_dot_v_split_offset = ez_dot_v_ptr + pid_bh * stride_ez_dot_v_bh + split * stride_ez_dot_v_split
        ez_dot_v_block_ptr = tl.make_block_ptr(
            base=ez_dot_v_split_offset,
            shape=(seq_len_q, dim),
            offsets=(q_start, 0),
            block_shape=(block_size_q, dim),
            strides=(stride_ez_dot_v_seq, stride_ez_dot_v_dim),
            order=(1, 0),
        )
        ez_dot_v_split = tl.load(ez_dot_v_block_ptr)
        ez_dot_v_global += ez_dot_v_split * alpha[:, None]

    ez_sum_global = tl.maximum(ez_sum_global, 1e-6)
    o_block = ez_dot_v_global / ez_sum_global[:, None]

    o_out_bh = o_out_ptr + pid_bh * stride_o_out_bh
    o_out_block_ptr = tl.make_block_ptr(
        base=o_out_bh,
        shape=(seq_len_q, dim),
        offsets=(q_start, 0),
        block_shape=(block_size_q, dim),
        strides=(stride_o_out_seq, stride_o_out_dim),
        order=(1, 0),
    )
    tl.store(
        o_out_block_ptr,
        o_block.to(o_out_ptr.dtype.element_ty),
        boundary_check=(0, 1),
    )
