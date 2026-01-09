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


def maybe_repeat_kv(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    """
    Repeat key and value tensors to match the number of query heads if necessary.
    This is helpful when the model uses group query attention, where the number of
    key/value heads is less than the number of query heads.

    Args:
        q (torch.Tensor): Query tensor of shape (..., num_heads_q, head_dim).
        k (torch.Tensor): Key tensor of shape (..., num_heads_kv, head_dim).
        v (torch.Tensor): Value tensor of shape (..., num_heads_kv, head_dim).
    """
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


def get_target_dtype(query: torch.Tensor, module: torch.nn.Module) -> Optional[torch.dtype]:
    """
    Determine the target dtype for Flash Attention based on the query tensor and module configuration.

    Args:
        query (torch.Tensor): The query tensor.
        module (torch.nn.Module): The attention module.

    Returns:
        Optional[torch.dtype]: The target dtype for Flash Attention, or None if not applicable.
    """
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


def convert_4d_mask_to_2d_mask(attention_mask):
    """
    Convert a 4D attention mask to a 2D padding mask.

    Args:
        attention_mask (torch.Tensor): The attention mask, either 4D or 2D.

    Returns:
        Optional[torch.Tensor]: The converted 2D padding mask, or None if no mask is provided.
    """
    padding_mask = None
    if attention_mask is not None:
        if attention_mask.dim() == 4:
            assert attention_mask.dtype != torch.bool
            attention_mask_row0 = attention_mask[:, 0, -1, :]
            padding_mask = attention_mask_row0 == 0

        elif attention_mask.dim() == 2:
            padding_mask = attention_mask
        else:
            raise ValueError(f"Unsupported attention_mask dim: {attention_mask.dim()}")

    return padding_mask


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
    max_length_q: Optional[int] = None,
    max_length_k: Optional[int] = None,
    target_dtype: Optional[torch.dtype] = None,
    **kwargs,
):
    """
    Core Flash Attention forward function handling various input formats.

    Args:
        query_states (torch.Tensor): Query tensor of shape (batch_size, seq_len, num_heads, head_dim).
        key_states (torch.Tensor): Key tensor of shape (batch_size, seq_len, num_heads, head_dim).
        value_states (torch.Tensor): Value tensor of shape (batch_size, seq_len, num_heads, head_dim).
        attention_mask (Optional[torch.Tensor]): Attention mask tensor.
        query_length (int): Length of the query sequence.
        is_causal (bool): Whether to apply causal masking.
        softmax_scale (Optional[float]): Scaling factor for softmax.
        position_ids (Optional[torch.Tensor]): Position IDs for packed sequences.
        cu_seq_lens_q (Optional[torch.LongTensor]): Cumulative sequence lengths for queries.
        cu_seq_lens_k (Optional[torch.LongTensor]): Cumulative sequence lengths for keys.
        max_length_q (Optional[int]): Maximum length for queries.
        max_length_k (Optional[int]): Maximum length for keys.
        target_dtype (Optional[torch.dtype]): Target dtype for Flash Attention.

    Returns:
        torch.Tensor: Output tensor after applying Flash Attention.

    Discussion:
        Q. Why are there multiple input formats supported?
            Flash Attention can handle different input formats to accommodate various model architectures
            and data representations. This includes support for packed sequences with position IDs,
            variable-length sequences with cumulative lengths, and standard fixed-length sequences
            with attention masks. By supporting these formats, Flash Attention can be seamlessly integrated
            into a wide range of transformer models.

        Q. How does the function decide which Flash Attention implementation to use?
            The function checks the presence of position IDs and cumulative sequence lengths to determine
            if the input represents packed sequences or variable-length sequences. If either of these
            conditions is met, it uses the variable-length Flash Attention implementation. Otherwise,
            it defaults to the fixed-length implementation.
    """
    query_states, key_states, value_states = fa_peft_integration_check(
        query_states, key_states, value_states, target_dtype
    )
    query_states, key_states, value_states = maybe_repeat_kv(query_states, key_states, value_states)

    flash_kwargs = {"causal": is_causal, "softmax_scale": softmax_scale}
    is_fa_with_position_ids = _is_packed_sequence(position_ids, batch_size=query_states.size(0))
    is_fa_with_varlen_kwargs = all(
        kwarg is not None for kwarg in (cu_seq_lens_q, cu_seq_lens_k, max_length_q, max_length_k)
    )

    if is_fa_with_varlen_kwargs or is_fa_with_position_ids:
        q, k, v, (cu_seq_lens_q, cu_seq_lens_k), (max_length_q, max_length_k) = _prepare_from_posids(
            query_states, key_states, value_states, position_ids
        )

        if "mps" in str(query_states.device):
            cu_seq_lens_k = cu_seq_lens_k.clone()

        output = flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seq_lens_q=cu_seq_lens_q,
            cu_seq_lens_k=cu_seq_lens_k,
            max_seq_len_q=max_length_q,
            max_seq_len_k=max_length_k,
            **flash_kwargs,
        )
        if isinstance(output, tuple):
            output = output[0]

        output = output.view(query_states.size(0), -1, output.size(-2), output.size(-1))

    elif attention_mask is not None:
        attention_mask = convert_4d_mask_to_2d_mask(attention_mask)

        q, k, v, indices_q, (cu_seq_lens_q, cu_seq_lens_k), (max_length_q, max_length_k) = _upad_input(
            query_states, key_states, value_states, attention_mask, query_length, unpad_input
        )

        if "mps" in str(q.device):
            cu_seq_lens_k = cu_seq_lens_k.clone()

        output_unpad = flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seq_lens_q=cu_seq_lens_q,
            cu_seq_lens_k=cu_seq_lens_k,
            max_seq_len_q=max_length_q,
            max_seq_len_k=max_length_k,
            **flash_kwargs,
        )
        if isinstance(output_unpad, tuple):
            output_unpad = output_unpad[0]

        output = pad_input(output_unpad, indices_q, query_states.size(0), query_length)
    else:
        q_fixed = query_states.transpose(1, 2).contiguous()
        k_fixed = key_states.transpose(1, 2).contiguous()
        v_fixed = value_states.transpose(1, 2).contiguous()

        # attention mask actually is not used here, but we provide flash_attn_func
        # with attention mask to provide this even without huggingface dependencies.
        output = flash_attn_func(q_fixed, k_fixed, v_fixed, attention_mask, **flash_kwargs)
        if isinstance(output, tuple):
            output = output[0]

        output = output.transpose(1, 2).contiguous()

    return output


def flash_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: Optional[float] = None,
    is_causal: Optional[bool] = None,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    """
    Forward function for Flash Attention compatible with Hugging Face transformers.

    Args:
        module (torch.nn.Module): The attention module.
        query (torch.Tensor): Query tensor of shape (batch_size, seq_len, num_heads, head_dim).
        key (torch.Tensor): Key tensor of shape (batch_size, seq_len, num_heads, head_dim).
        value (torch.Tensor): Value tensor of shape (batch_size, seq_len, num_heads, head_dim).
        attention_mask (Optional[torch.Tensor]): Attention mask tensor.
        scaling (Optional[float]): Scaling factor for softmax.
        is_causal (Optional[bool]): Whether to apply causal masking.

    Returns:
        tuple[torch.Tensor, None]: Output tensor after applying Flash Attention and None for compatibility.
    """
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

    target_dtype = get_target_dtype(query_states, module)
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
