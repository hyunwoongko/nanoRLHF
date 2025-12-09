import torch

from nanorlhf.kernels.flash_attn_varlen.bwd import flash_attn_varlen_bwd
from nanorlhf.kernels.flash_attn_varlen.fwd import flash_attn_varlen_fwd


class FlashAttnVarlenFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, cu_seqlens_q, cu_seqlens_k, causal=True, softmax_scale=None):
        assert q.dim() == 3 and k.dim() == 3 and v.dim() == 3
        assert q.shape[0] == cu_seqlens_q[-1].item()
        assert k.shape[0] == cu_seqlens_k[-1].item()

        bsz = cu_seqlens_q.shape[0] - 1
        _, num_heads, dim = q.shape

        seqlens_q = (cu_seqlens_q[1:] - cu_seqlens_q[:-1]).to(torch.int32)
        seqlens_k = (cu_seqlens_k[1:] - cu_seqlens_k[:-1]).to(torch.int32)

        max_seqlen_q = int(seqlens_q.max().item())
        max_seqlen_k = int(seqlens_k.max().item())

        o, max_q, ez_sum = flash_attn_varlen_fwd(
            q, k, v,
            cu_seqlens_q, cu_seqlens_k,
            bsz, num_heads,
            max_seqlen_q, max_seqlen_k,
            causal=causal,
            softmax_scale=softmax_scale,
        )

        ctx.save_for_backward(q, k, v, o, cu_seqlens_q, cu_seqlens_k, max_q, ez_sum)
        ctx.bsz = bsz
        ctx.num_heads = num_heads
        ctx.max_seqlen_q = max_seqlen_q
        ctx.max_seqlen_k = max_seqlen_k
        ctx.causal = causal
        ctx.softmax_scale = softmax_scale

        return o

    @staticmethod
    def backward(ctx, dO):
        q, k, v, o, cu_seqlens_q, cu_seqlens_k, max_q, ez_sum = ctx.saved_tensors
        bsz = ctx.bsz
        num_heads = ctx.num_heads
        max_seqlen_q = ctx.max_seqlen_q
        max_seqlen_k = ctx.max_seqlen_k
        causal = ctx.causal
        softmax_scale = ctx.softmax_scale

        dq, dk, dv = flash_attn_varlen_bwd(
            q, k, v,
            o, dO,
            cu_seqlens_q, cu_seqlens_k,
            max_q, ez_sum,
            bsz, num_heads,
            max_seqlen_q, max_seqlen_k,
            causal=causal,
            softmax_scale=softmax_scale,
        )

        return dq, dk, dv, None, None, None, None
