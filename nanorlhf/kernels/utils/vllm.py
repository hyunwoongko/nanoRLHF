from dataclasses import dataclass
from typing import Optional

import torch
from transformers.modeling_flash_attention_utils import fa_peft_integration_check, logger

from nanorlhf.kernels import flash_attn_varlen_func
from nanorlhf.kernels.flash_attn_decode.ops import flash_attn_decode_paged
from nanorlhf.kernels.kvcache.load import load_kv_from_cache_prefill
from nanorlhf.kernels.kvcache.store import store_kv_to_cache
from nanorlhf.kernels.utils.huggingface import maybe_repeat_kv, get_target_dtype


@dataclass
class Context:
    is_prefill: bool = False
    slot_mapping: Optional[torch.Tensor] = None
    context_lens: Optional[torch.Tensor] = None
    block_tables: Optional[torch.Tensor] = None
    cu_seq_lens_q: Optional[torch.Tensor] = None
    cu_seq_lens_k: Optional[torch.Tensor] = None
    max_seq_len_q: Optional[int] = None
    max_seq_len_k: Optional[int] = None


GLOBAL_CONTEXT = Context()


def get_context() -> Context:
    return GLOBAL_CONTEXT


def set_context(
    is_prefill,
    slot_mapping=None,
    context_lens=None,
    block_tables=None,
    cu_seq_lens_q=None,
    cu_seq_lens_k=None,
    max_seq_len_q=None,
    max_seq_len_k=None,
):
    """
    Set the global context for flash attention operations.
    """
    global GLOBAL_CONTEXT
    GLOBAL_CONTEXT = Context(
        is_prefill=is_prefill,
        slot_mapping=slot_mapping,
        context_lens=context_lens,
        block_tables=block_tables,
        cu_seq_lens_q=cu_seq_lens_q,
        cu_seq_lens_k=cu_seq_lens_k,
        max_seq_len_q=max_seq_len_q,
        max_seq_len_k=max_seq_len_k,
    )


def reset_context():
    global GLOBAL_CONTEXT
    GLOBAL_CONTEXT = Context()


def store_cache(context, key_states_not_repeated, value_states_not_repeated, key_cache, value_cache):
    """
    Store key and value states into the KV cache based on the current context.

    Args:
        context (Context): The current context containing slot mapping and prefill status.
        key_states_not_repeated (torch.Tensor): New key states of shape (bsz, length, kv_heads, dim).
        value_states_not_repeated (torch.Tensor): New value states of shape (bsz, length, kv_heads, dim).
        key_cache (torch.Tensor): Key cache tensor of shape (num_blocks, block_size, kv_heads, head_dim).
        value_cache (torch.Tensor): Value cache tensor of shape (num_blocks, block_size, kv_heads, head_dim).
    """
    if context.is_prefill:
        assert context.cu_seq_lens_q is not None
        store_kv_to_cache(
            key_states_not_repeated=key_states_not_repeated,
            value_states_not_repeated=value_states_not_repeated,
            key_cache=key_cache,
            value_cache=value_cache,
            slot_mapping=context.slot_mapping,
        )
    else:
        k_for_cache = key_states_not_repeated[:, -1:, :, :]
        v_for_cache = value_states_not_repeated[:, -1:, :, :]
        store_kv_to_cache(
            key_states_not_repeated=k_for_cache,
            value_states_not_repeated=v_for_cache,
            key_cache=key_cache,
            value_cache=value_cache,
            slot_mapping=context.slot_mapping,
        )


def compute_prefill(
    context,
    query_states,
    key_states,
    value_states,
    key_cache,
    value_cache,
    dim,
    scaling,
    is_causal,
):
    """
    Compute the flash attention output during the prefill phase.

    Args:
        context (Context): The current context containing sequence lengths and block tables.
        query_states (torch.Tensor): Query states of shape (1, q_len, num_heads, dim).
        key_states (torch.Tensor): Key states of shape (1, k_len, num_heads, dim).
        value_states (torch.Tensor): Value states of shape (1, k_len, num_heads, dim).
        key_cache (torch.Tensor): Key cache tensor of shape (num_blocks, block_size, kv_heads, head_dim).
        value_cache (torch.Tensor): Value cache tensor of shape (num_blocks, block_size, kv_heads, head_dim).
        dim (int): Dimension of each head.
        scaling (float): Scaling factor for attention scores.
        is_causal (bool): Whether to apply causal masking.

    Returns:
        Tuple[torch.Tensor, None]: Output tensor of shape (1, q_len, num_heads, dim) and None.
    """
    assert context.cu_seq_lens_q is not None and context.cu_seq_lens_k is not None
    assert context.max_seq_len_q is not None and context.max_seq_len_k is not None

    device = query_states.device
    cu_seq_lens_q = context.cu_seq_lens_q.to(device=device, dtype=torch.int32)
    cu_seq_lens_k = context.cu_seq_lens_k.to(device=device, dtype=torch.int32)
    max_seq_len_q = int(context.max_seq_len_q)
    max_seq_len_k = int(context.max_seq_len_k)

    total_q = int(cu_seq_lens_q[-1].item())
    total_k = int(cu_seq_lens_k[-1].item())
    if context.block_tables is None:
        assert total_k == total_q, f"no-cache prefill expects total_k==total_q, got {total_k} vs {total_q}"

    batch_size, q_len, num_heads, _ = query_states.shape
    assert batch_size == 1, f"packed prefill expects batch=1, got {batch_size}"

    q = query_states.reshape(-1, num_heads, dim).contiguous()

    if context.block_tables is not None:
        k_bh, v_bh = load_kv_from_cache_prefill(
            context=context,
            cu_seq_lens_k=cu_seq_lens_k,
            key_cache=key_cache,
            value_cache=value_cache,
            num_heads=num_heads,
            dim=dim,
        )
    else:
        k_bh = key_states.reshape(-1, num_heads, dim).contiguous()
        v_bh = value_states.reshape(-1, num_heads, dim).contiguous()

    output = flash_attn_varlen_func(
        q,
        k_bh,
        v_bh,
        cu_seq_lens_q=cu_seq_lens_q,
        cu_seq_lens_k=cu_seq_lens_k,
        max_seq_len_q=max_seq_len_q,
        max_seq_len_k=max_seq_len_k,
        softmax_scale=scaling,
        causal=is_causal,
    )
    if isinstance(output, tuple):
        output = output[0]

    output = output.view(1, -1, output.size(-2), output.size(-1))
    return output, None


def compute_decode(
    context,
    query_states,
    key_cache,
    value_cache,
    batch_size,
    seq_len_q,
    num_heads,
    dim,
    scaling,
    is_causal,
):
    """
    Compute the flash attention output during the decode phase.

    Args:
        context (Context): The current context containing block tables and context lengths.
        query_states (torch.Tensor): Query states of shape (batch_size, seq_len_q, num_heads, dim).
        key_cache (torch.Tensor): Key cache tensor of shape (num_blocks, block_size, kv_heads, head_dim).
        value_cache (torch.Tensor): Value cache tensor of shape (num_blocks, block_size, kv_heads, head_dim).
        batch_size (int): Batch size.
        seq_len_q (int): Sequence length of queries.
        num_heads (int): Number of attention heads.
        dim (int): Dimension of each head.
        scaling (float): Scaling factor for attention scores.
        is_causal (bool): Whether to apply causal masking.

    Returns:
        Tuple[torch.Tensor, None]: Output tensor of shape (batch_size, seq_len_q, num_heads, dim) and None.
    """
    assert seq_len_q == 1
    assert context.block_tables is not None
    assert context.context_lens is not None

    q_bh = query_states.permute(0, 2, 1, 3).reshape(batch_size * num_heads, seq_len_q, dim).contiguous()

    num_blocks, block_size, kv_heads, dim_cache = key_cache.shape
    assert dim_cache == dim

    num_slots = num_blocks * block_size
    k_slots = key_cache.view(num_slots, kv_heads, dim)
    v_slots = value_cache.view(num_slots, kv_heads, dim)

    block_tables = context.block_tables.to(device=key_cache.device, dtype=torch.int32)
    context_lens = context.context_lens.to(device=key_cache.device, dtype=torch.int32)

    output = flash_attn_decode_paged(
        q_bh=q_bh,
        k_slots=k_slots,
        v_slots=v_slots,
        block_tables=block_tables,
        context_lens=context_lens,
        num_heads=num_heads,
        kv_heads=kv_heads,
        causal=is_causal,
        softmax_scale=scaling,
        kv_block_size=block_size,
    )
    output = output.view(batch_size, num_heads, seq_len_q, dim)
    return output, None


@torch.no_grad()
def paged_flash_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: Optional[float] = None,
    position_ids: Optional[torch.Tensor] = None,
    is_causal: Optional[bool] = None,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    """
    Flash attention forward function with KV cache support for NanoVLLM.

    Args:
        module (torch.nn.Module): The attention module.
        query (torch.Tensor): Query tensor of shape (batch_size, num_heads, seq_len_q, dim).
        key (torch.Tensor): Key tensor of shape (batch_size, num_heads, seq_len_k, dim).
        value (torch.Tensor): Value tensor of shape (batch_size, num_heads, seq_len_k, dim).
        attention_mask (Optional[torch.Tensor]): Attention mask tensor.
        scaling (Optional[float]): Scaling factor for attention scores.
        position_ids (Optional[torch.Tensor]): Position IDs tensor.
        is_causal (Optional[bool]): Whether to apply causal masking.
        **kwargs: Additional keyword arguments.

    Returns:
        Tuple[torch.Tensor, None]: Output tensor and None.
    """
    context = get_context()
    if kwargs.get("output_attentions", False):
        logger.warning_once(
            "nanoRLHF `flash_attention` does not support `output_attentions=True`."
            " Please set your attention to `eager` if you want any of these features."
        )

    if any(dim == 0 for dim in query.shape):
        raise ValueError(
            "Tensor query has shape with a zero dimension.\n"
            "FlashAttention does not support inputs with dim=0.\n"
            "Please check your input shapes or use SDPA instead."
        )

    batch_size, num_heads, seq_len_q, dim = query.shape
    query_states = query.transpose(1, 2)
    key_states = key.transpose(1, 2)
    value_states = value.transpose(1, 2)

    target_dtype = get_target_dtype(query_states, module)
    is_causal = is_causal if is_causal is not None else module.is_causal

    query_states, key_states, value_states = fa_peft_integration_check(
        query_states, key_states, value_states, target_dtype
    )

    key_states_not_repeated = key.transpose(1, 2)
    value_states_not_repeated = value.transpose(1, 2)
    query_states, key_states, value_states = maybe_repeat_kv(query_states, key_states, value_states)

    if scaling is None:
        scaling = 1.0 / (query_states.size(-1) ** 0.5)

    assert hasattr(module, "key_cache") and hasattr(module, "value_cache"), (
        "NanoRLHF paged_flash_attention_forward requires the attention module to have "
        "`key_cache` and `value_cache` attributes for KV cache."
    )

    key_cache, value_cache = module.key_cache, module.value_cache
    if context.slot_mapping is not None and context.slot_mapping.numel() > 0:
        store_cache(
            context=context,
            key_states_not_repeated=key_states_not_repeated,
            value_states_not_repeated=value_states_not_repeated,
            key_cache=key_cache,
            value_cache=value_cache,
        )

    if context.is_prefill:
        return compute_prefill(
            context=context,
            query_states=query_states,
            key_states=key_states,
            value_states=value_states,
            key_cache=key_cache,
            value_cache=value_cache,
            dim=dim,
            scaling=scaling,
            is_causal=is_causal,
        )
    else:
        return compute_decode(
            context=context,
            query_states=query_states,
            key_cache=key_cache,
            value_cache=value_cache,
            batch_size=batch_size,
            seq_len_q=seq_len_q,
            num_heads=num_heads,
            dim=dim,
            scaling=scaling,
            is_causal=is_causal,
        )
