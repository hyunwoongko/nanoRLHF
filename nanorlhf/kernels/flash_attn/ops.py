import torch

from nanorlhf.kernels.flash_attn.bwd import flash_attn_bwd
from nanorlhf.kernels.flash_attn.fwd import flash_attn_fwd


class FlashAttentionFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, causal=True, softmax_scale=None):
        o, max_q, ez_sum = flash_attn_fwd(q, k, v, causal=causal, softmax_scale=softmax_scale)
        ctx.save_for_backward(q, k, v)
        ctx.causal = causal
        ctx.softmax_scale = softmax_scale
        ctx.max_q = max_q
        ctx.ez_sum = ez_sum
        return o

    @staticmethod
    def backward(ctx, grad_o):
        q, k, v = ctx.saved_tensors
        causal = ctx.causal
        softmax_scale = ctx.softmax_scale
        max_q = ctx.max_q
        ez_sum = ctx.ez_sum
        dq, dk, dv = flash_attn_bwd(q, k, v, grad_o, max_q, ez_sum, causal=causal, softmax_scale=softmax_scale)
        return dq, dk, dv, None, None


