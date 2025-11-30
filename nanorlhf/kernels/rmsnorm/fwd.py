import triton
import triton.language as tl


@triton.jit
def rmsnorm_kernel_fwd(
    x_ptr,  # (M, N)
    w_ptr,  # (N,)
    y_ptr,  # (M, N)
    M,
    N,
    eps,
    stride_xm,
    stride_xn,
    stride_ym,
    stride_yn,
    block_size: tl.constexpr,
):
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
