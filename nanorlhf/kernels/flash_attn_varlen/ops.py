import torch

from .fwd import flash_attn_varlen_fwd
from .bwd import flash_attn_varlen_bwd


class FlashAttnVarlenFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, cu_q, cu_k, causal=True, softmax_scale=None):
        assert q.dim() == 3 and k.dim() == 3 and v.dim() == 3
        assert q.shape[0] == cu_q[-1].item()
        assert k.shape[0] == cu_k[-1].item()

        B = cu_q.shape[0] - 1
        _, H, D = q.shape

        seqlens_q = (cu_q[1:] - cu_q[:-1]).to(torch.int32)
        seqlens_k = (cu_k[1:] - cu_k[:-1]).to(torch.int32)

        max_seqlen_q = int(seqlens_q.max().item())
        max_seqlen_k = int(seqlens_k.max().item())

        o, max_q, ez_sum = flash_attn_varlen_fwd(
            q, k, v,
            cu_q, cu_k,
            B, H,
            max_seqlen_q, max_seqlen_k,
            causal=causal,
            softmax_scale=softmax_scale,
        )

        ctx.save_for_backward(q, k, v, o, cu_q, cu_k, max_q, ez_sum)
        ctx.B = B
        ctx.H = H
        ctx.max_seqlen_q = max_seqlen_q
        ctx.max_seqlen_k = max_seqlen_k
        ctx.causal = causal
        ctx.softmax_scale = softmax_scale

        return o

    @staticmethod
    def backward(ctx, dO):
        q, k, v, o, cu_q, cu_k, max_q, ez_sum = ctx.saved_tensors
        B = ctx.B
        H = ctx.H
        max_seqlen_q = ctx.max_seqlen_q
        max_seqlen_k = ctx.max_seqlen_k
        causal = ctx.causal
        softmax_scale = ctx.softmax_scale

        dq, dk, dv = flash_attn_varlen_bwd(
            q, k, v,
            o, dO,
            cu_q, cu_k,
            max_q, ez_sum,
            B, H,
            max_seqlen_q, max_seqlen_k,
            causal=causal,
            softmax_scale=softmax_scale,
        )

        return dq, dk, dv, None, None, None, None