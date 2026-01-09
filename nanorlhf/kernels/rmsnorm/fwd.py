import triton
import triton.language as tl


@triton.jit
def rmsnorm_kernel_fwd(
    x_ptr,
    w_ptr,
    y_ptr,
    M,
    N,
    eps,
    stride_xm,
    stride_xn,
    stride_ym,
    stride_yn,
    block_size: tl.constexpr,
):
    """
    A kernel to compute the forward pass of RMSNorm.

    Args:
        x_ptr: Pointer to the input tensor of shape (M, N).
        w_ptr: Pointer to the weight tensor of shape (N,).
        y_ptr: Pointer to the output tensor of shape (M, N).
        M: Number of rows in the input/output tensors.
        N: Number of columns in the input/output tensors.
        eps: Small epsilon value for numerical stability.
        stride_xm: Stride for rows in the input tensor.
        stride_xn: Stride for columns in the input tensor.
        stride_ym: Stride for rows in the output tensor.
        stride_yn: Stride for columns in the output tensor.
        block_size: Block size for processing columns.
    """
    row = tl.program_id(0)

    offs = tl.arange(0, block_size)
    mask = offs < N

    x_row_ptrs = x_ptr + row * stride_xm + offs * stride_xn
    x = tl.load(x_row_ptrs, mask=mask, other=0.0)

    # variance & rms
    x2 = x * x
    mean = tl.sum(x2, axis=0) / N
    inv_rms = tl.rsqrt(mean + eps)

    # scale by weight
    w = tl.load(w_ptr + offs, mask=mask, other=0.0)
    y = x * inv_rms * w

    y_row_ptrs = y_ptr + row * stride_ym + offs * stride_yn
    tl.store(y_row_ptrs, y, mask=mask)
