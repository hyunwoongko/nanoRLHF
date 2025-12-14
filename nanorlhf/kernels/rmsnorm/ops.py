import torch

from nanorlhf.kernels.rmsnorm.bwd import rmsnorm_kernel_bwd
from nanorlhf.kernels.rmsnorm.fwd import rmsnorm_kernel_fwd


class FusedRMSNormFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden_states, weight, eps):
        assert hidden_states.is_cuda, "FusedRMSNorm only supports CUDA tensors."

        input_dtype = hidden_states.dtype
        device = hidden_states.device

        x = hidden_states.to(torch.float32)

        hidden_size = weight.shape[0]
        assert x.shape[-1] == hidden_size, "hidden_size mismatch"

        orig_shape = x.shape
        M = x.numel() // hidden_size
        x_2d = x.view(M, hidden_size).contiguous()

        y_2d = torch.empty_like(x_2d)

        stride_xm, stride_xn = x_2d.stride()
        stride_ym, stride_yn = y_2d.stride()

        block_size = 1 << (hidden_size - 1).bit_length()
        grid = (M,)

        rmsnorm_kernel_fwd[grid](
            x_2d, weight.to(torch.float32), y_2d,
            M, hidden_size,
            eps,
            stride_xm, stride_xn,
            stride_ym, stride_yn,
            block_size=block_size,
        )

        y = y_2d.view(orig_shape).to(input_dtype)

        ctx.save_for_backward(x_2d, weight.to(torch.float32))
        ctx.eps = eps
        ctx.hidden_size = hidden_size
        ctx.orig_shape = orig_shape
        ctx.input_dtype = input_dtype
        ctx.device = device
        return y

    @staticmethod
    def backward(ctx, grad_output):
        x_2d, w_f32 = ctx.saved_tensors
        eps = ctx.eps
        hidden_size = ctx.hidden_size
        orig_shape = ctx.orig_shape
        input_dtype = ctx.input_dtype
        device = ctx.device

        g = grad_output.to(torch.float32)
        M = x_2d.shape[0]

        g_2d = g.contiguous().view(M, hidden_size)
        dx_2d = torch.empty_like(x_2d, device=device, dtype=torch.float32)
        dw = torch.zeros_like(w_f32, device=device, dtype=torch.float32)

        stride_xm, stride_xn = x_2d.stride()
        stride_dym, stride_dyn = g_2d.stride()
        stride_dxm, stride_dxn = dx_2d.stride()

        block_size = 1 << (hidden_size - 1).bit_length()
        grid = (M,)

        rmsnorm_kernel_bwd[grid](
            x_2d,
            w_f32,
            g_2d,
            dx_2d,
            dw,
            M, hidden_size,
            eps,
            stride_xm, stride_xn,
            stride_dym, stride_dyn,
            stride_dxm, stride_dxn,
            block_size=block_size,
        )

        dx = dx_2d.view(orig_shape).to(input_dtype)
        grad_weight = dw.to(w_f32.dtype)
        return dx, grad_weight, None
