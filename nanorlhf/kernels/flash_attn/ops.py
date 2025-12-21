from typing import Optional, Tuple

import torch

from nanorlhf.kernels.flash_attn.bwd import flash_attn_bwd
from nanorlhf.kernels.flash_attn.fwd import flash_attn_fwd


class FlashAttentionFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal: bool = False,
        softmax_scale: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Flash Attention forward pass.

        Args:
            q (torch.Tensor): Query tensor of shape (batch_size, seq_len_q, dim).
            k (torch.Tensor): Key tensor of shape (batch_size, seq_len_k, dim).
            v (torch.Tensor): Value tensor of shape (batch_size, seq_len_k, dim).
            attention_mask (Optional[torch.Tensor]): Attention mask tensor.
            causal (bool): Whether to apply causal masking.
            softmax_scale (Optional[float]): Scaling factor for softmax.

        Returns:
            torch.Tensor: Output tensor after applying flash attention.
        """
        o, max_q, ez_sum = flash_attn_fwd(
            q, k, v, attention_mask=attention_mask, causal=causal, softmax_scale=softmax_scale
        )
        ctx.save_for_backward(q, k, v)
        ctx.causal = causal
        ctx.softmax_scale = softmax_scale
        ctx.max_q = max_q
        ctx.ez_sum = ez_sum
        ctx.attention_mask = attention_mask
        return o

    @staticmethod
    def backward(ctx, grad_o: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, None, None, None]:
        """
        Flash Attention backward pass.

        Args:
            grad_o (torch.Tensor): Gradient of the output tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, None, None, None]:
                Gradients with respect to q, k, v and None for other inputs.
        """
        q, k, v = ctx.saved_tensors
        causal = ctx.causal
        softmax_scale = ctx.softmax_scale
        max_q = ctx.max_q
        ez_sum = ctx.ez_sum
        attention_mask = ctx.attention_mask
        dq, dk, dv = flash_attn_bwd(
            q, k, v, grad_o, max_q, ez_sum, attention_mask=attention_mask, causal=causal, softmax_scale=softmax_scale
        )
        return dq, dk, dv, None, None, None
