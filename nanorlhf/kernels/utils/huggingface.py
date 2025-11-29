from typing import Optional

import torch
from transformers.modeling_flash_attention_utils import (
    _upad_input,
    _is_packed_sequence,
    _prepare_from_posids,
    fa_peft_integration_check,
    logger,
)

from nanorlhf.kernels.api import (
    flash_attn_func,
    flash_attn_varlen_func,
    pad_input,
    unpad_input,
)


def _maybe_repeat_kv(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    if k.shape[-2] == q.shape[-2]:
        return q, k, v

    num_heads = q.shape[-2]
    num_kv_heads = k.shape[-2]

    if num_heads % num_kv_heads != 0:
        raise ValueError(
            f"Unsupported head layout: query heads={num_heads}, kv heads={num_kv_heads} "
            " (cannot broadcast k/v to match q)."
        )

    num_groups = num_heads // num_kv_heads
    k = k.repeat_interleave(num_groups, dim=-2)
    v = v.repeat_interleave(num_groups, dim=-2)
    return q, k, v


def _get_target_dtype(query: torch.Tensor, module: torch.nn.Module) -> Optional[torch.dtype]:
    if query.dtype == torch.float32:
        if torch.is_autocast_enabled():
            return (
                torch.get_autocast_dtype("cuda")
                if hasattr(torch, "get_autocast_dtype")
                else torch.get_autocast_gpu_dtype()
            )
        elif hasattr(module.config, "_pre_quantization_dtype"):
            return module.config._pre_quantization_dtype
        else:
            return next(layer for layer in module.modules() if isinstance(layer, torch.nn.Linear)).weight.dtype
    return None


def _flash_attention_forward(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    query_length: int,
    is_causal: bool,
    softmax_scale: Optional[float] = None,
    position_ids: Optional[torch.Tensor] = None,
    cu_seq_lens_q: Optional[torch.LongTensor] = None,
    cu_seq_lens_k: Optional[torch.LongTensor] = None,
    target_dtype: Optional[torch.dtype] = None,
    **kwargs,
):
    query_states, key_states, value_states = fa_peft_integration_check(
        query_states, key_states, value_states, target_dtype
    )
    query_states, key_states, value_states = _maybe_repeat_kv(query_states, key_states, value_states)

    flash_kwargs = {"causal": is_causal, "softmax_scale": softmax_scale}
    is_fa_with_position_ids = _is_packed_sequence(position_ids, batch_size=query_states.size(0))
    is_fa_with_varlen_kwargs = all(kwarg is not None for kwarg in (cu_seq_lens_q, cu_seq_lens_k))

    if attention_mask is not None:
        q, k, v, indices_q, (cu_seq_lens_q, cu_seq_lens_k), _ = _upad_input(
            query_states, key_states, value_states, attention_mask, query_length, unpad_input
        )

        if "mps" in str(q.device):
            cu_seq_lens_k = cu_seq_lens_k.clone()

        out_unpad = flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q=cu_seq_lens_q,
            cu_seqlens_k=cu_seq_lens_k,
            **flash_kwargs,
        )
        if isinstance(out_unpad, tuple):
            out_unpad = out_unpad[0]

        out = pad_input(out_unpad, indices_q, query_states.size(0), query_length)

    elif is_fa_with_varlen_kwargs or is_fa_with_position_ids:
        if cu_seq_lens_q is None or cu_seq_lens_k is None:
            try:
                q, k, v, (cu_seq_lens_q, cu_seq_lens_k), _ = _prepare_from_posids(
                    query_states, key_states, value_states, position_ids
                )
            except TypeError:
                q, k, v, (cu_seq_lens_q, cu_seq_lens_k), _ = _prepare_from_posids(
                    query_states, key_states, value_states, position_ids, query_length
                )
        else:
            bsz = query_states.size(0)
            q_len = query_states.size(1)
            k_len = key_states.size(1)
            num_heads = query_states.size(2)

            q = query_states.reshape(-1, num_heads, query_states.size(-1))
            k = key_states.reshape(-1, key_states.size(2), key_states.size(-1))
            v = value_states.reshape(-1, value_states.size(2), value_states.size(-1))

            cu_seq_lens_q = torch.arange(
                0,
                (bsz * q_len) + 1,
                step=q_len,
                dtype=torch.int32,
                device=query_states.device,
            )
            cu_seq_lens_k = torch.arange(
                0,
                (bsz * k_len) + 1,
                step=k_len,
                dtype=torch.int32,
                device=query_states.device,
            )

        if "mps" in str(query_states.device):
            cu_seq_lens_k = cu_seq_lens_k.clone()

        out = flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q=cu_seq_lens_q,
            cu_seqlens_k=cu_seq_lens_k,
            **flash_kwargs,
        )
        if isinstance(out, tuple):
            out = out[0]

        out = out.view(query_states.size(0), -1, out.size(-2), out.size(-1))

    else:
        q_fixed = query_states.transpose(1, 2).contiguous()
        k_fixed = key_states.transpose(1, 2).contiguous()
        v_fixed = value_states.transpose(1, 2).contiguous()

        out = flash_attn_func(q_fixed, k_fixed, v_fixed, **flash_kwargs)
        if isinstance(out, tuple):
            out = out[0]

        out = out.transpose(1, 2).contiguous()

    return out


def flash_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float | None = None,
    is_causal: bool | None = None,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    if kwargs.get("output_attentions", False):
        logger.warning_once(
            "nanoRLHF `flash_attention` does not support `output_attentions=True`."
            " Please set your attention to `eager` if you want any of these features."
        )

    seq_len = query.shape[2]

    if any(dim == 0 for dim in query.shape):
        raise ValueError(
            "Tensor query has shape with a zero dimension.\n"
            "FlashAttention does not support inputs with dim=0.\n"
            "Please check your input shapes or use SDPA instead."
        )

    query_states = query.transpose(1, 2)
    key_states = key.transpose(1, 2)
    value_states = value.transpose(1, 2)

    target_dtype = _get_target_dtype(query_states, module)
    is_causal = is_causal if is_causal is not None else module.is_causal

    attn_output = _flash_attention_forward(
        query_states,
        key_states,
        value_states,
        attention_mask,
        query_length=seq_len,
        is_causal=is_causal,
        softmax_scale=scaling,
        target_dtype=target_dtype,
        **kwargs,
    )

    return attn_output, None
