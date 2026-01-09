import triton
import triton.language as tl


@triton.jit
def rmsnorm_kernel_bwd(
    x_ptr,
    w_ptr,
    dy_ptr,
    dx_ptr,
    dw_ptr,
    M,
    N,
    eps,
    stride_xm,
    stride_xn,
    stride_dym,
    stride_dyn,
    stride_dxm,
    stride_dxn,
    block_size: tl.constexpr,
):
    """
    A kernel to compute the backward pass of RMSNorm.

    Args:
        x_ptr: Pointer to the input tensor of shape (M, N).
        w_ptr: Pointer to the weight tensor of shape (N,).
        dy_ptr: Pointer to the gradient of the output tensor of shape (M, N).
        dx_ptr: Pointer to the gradient of the input tensor of shape (M, N).
        dw_ptr: Pointer to the gradient of the weight tensor of shape (N,).
        M: Number of rows in the input/output tensors.
        N: Number of columns in the input/output tensors.
        eps: Small epsilon value for numerical stability.
        stride_xm: Stride for rows in the input tensor.
        stride_xn: Stride for columns in the input tensor.
        stride_dym: Stride for rows in the output gradient tensor.
        stride_dyn: Stride for columns in the output gradient tensor.
        stride_dxm: Stride for rows in the input gradient tensor.
        stride_dxn: Stride for columns in the input gradient tensor.
        block_size: Block size for processing columns.
    """
    row = tl.program_id(0)
    offs = tl.arange(0, block_size)
    mask = offs < N

    # load x, dy, w
    x_row_ptrs = x_ptr + row * stride_xm + offs * stride_xn
    dy_row_ptrs = dy_ptr + row * stride_dym + offs * stride_dyn

    x = tl.load(x_row_ptrs, mask=mask, other=0.0)
    dy = tl.load(dy_row_ptrs, mask=mask, other=0.0)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0)

    # var, rms, inv_rms
    x2 = x * x
    mean = tl.sum(x2, axis=0) / N
    inv_rms = tl.rsqrt(mean + eps)  # 1 / rms

    # h = x / rms = x * inv_rms
    h = x * inv_rms

    # dL/dw_j_row = dy_j * h_j
    dw_contrib = dy * h
    tl.atomic_add(dw_ptr + offs, dw_contrib, mask=mask)

    # grad_h = dy * w
    gh = dy * w

    # dot = sum_j gh_j * x_j
    dot = tl.sum(gh * x, axis=0)

    # grad_x = gh / rms - x * (dot / (N * rms^3))
    # using inv_rms: 1/rms = inv_rms, 1/rms^3 = inv_rms^3
    inv_rms3 = inv_rms * inv_rms * inv_rms
    coeff = dot * inv_rms3 / N  # scalar
    dx = gh * inv_rms - x * coeff

    dx_row_ptrs = dx_ptr + row * stride_dxm + offs * stride_dxn
    tl.store(dx_row_ptrs, dx, mask=mask)
