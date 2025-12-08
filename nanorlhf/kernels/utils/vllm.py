from dataclasses import dataclass
from typing import Optional

import torch
from transformers.modeling_flash_attention_utils import (
    fa_peft_integration_check,
    logger,
    _is_packed_sequence,
)

from nanorlhf.kernels.flash_attn_decode.ops import flash_attn_decode
from nanorlhf.kernels.flash_attn_varlen.fwd import flash_attn_varlen_fwd
from nanorlhf.kernels.kvcache.store_kvcache import store_kvcache
from nanorlhf.kernels.utils.huggingface import _maybe_repeat_kv, _get_target_dtype


@dataclass
class Context:
    is_prefill: bool = False
    cu_seqlens_q: Optional[torch.Tensor] = None
    cu_seqlens_k: Optional[torch.Tensor] = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: Optional[torch.Tensor] = None
    context_lens: Optional[torch.Tensor] = None
    block_tables: Optional[torch.Tensor] = None


KVCACHE_BLOCK_SIZE = 256
GLOBAL_CONTEXT = Context()


def get_context():
    return GLOBAL_CONTEXT


def set_context(
    is_prefill,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    max_seqlen_q=0,
    max_seqlen_k=0,
    slot_mapping=None,
    context_lens=None,
    block_tables=None,
):
    global GLOBAL_CONTEXT
    GLOBAL_CONTEXT = Context(
        is_prefill,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        slot_mapping,
        context_lens,
        block_tables,
    )


def reset_context():
    global GLOBAL_CONTEXT
    GLOBAL_CONTEXT = Context()


@torch.no_grad()
def paged_flash_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: Optional[float] = None,
    is_causal: Optional[bool] = None,
    position_ids: Optional[torch.Tensor] = None,
    cu_seq_lens_q: Optional[torch.LongTensor] = None,
    cu_seq_lens_k: Optional[torch.LongTensor] = None,
    target_dtype: Optional[torch.dtype] = None,
    **kwargs,
) -> tuple[torch.Tensor, None]:
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

    query_states = query.transpose(1, 2)
    key_states = key.transpose(1, 2)
    value_states = value.transpose(1, 2)

    target_dtype = _get_target_dtype(query_states, module)
    is_causal = is_causal if is_causal is not None else module.is_causal

    query_states, key_states, value_states = fa_peft_integration_check(
        query_states, key_states, value_states, target_dtype
    )
    query_states, key_states, value_states = _maybe_repeat_kv(query_states, key_states, value_states)

    is_fa_with_position_ids = _is_packed_sequence(position_ids, batch_size=query_states.size(0))
    is_fa_with_varlen_kwargs = all(kwarg is not None for kwarg in (cu_seq_lens_q, cu_seq_lens_k))
    assert is_fa_with_position_ids or is_fa_with_varlen_kwargs, (
        "nanoRLHF paged_flash_attention_forward requires either "
        "`position_ids` for packed sequences or `cu_seq_lens_q` and `cu_seq_lens_k` for variable-length sequences."
    )

    q = query_states.reshape(-1, query_states.size(2), query_states.size(-1))
    k = key_states.reshape(-1, key_states.size(2), key_states.size(-1))
    v = value_states.reshape(-1, value_states.size(2), value_states.size(-1))

    if not (hasattr(module, "k_cache") and hasattr(module, "v_cache")):
        k_cache = v_cache = torch.tensor([])
        setattr(module, "k_cache", k_cache)
        setattr(module, "v_cache", v_cache)

    k_cache, v_cache = module.k_cache, module.v_cache
    if k_cache.numel() and v_cache.numel():
        store_kvcache(key_states, value_states, k_cache, v_cache, context.slot_mapping)

    if context.is_prefill:
        if context.block_tables is not None:
            k, v = k_cache, v_cache

        bsz = context.cu_seqlens_q.shape[0] - 1
        _, num_heads, dim = q.shape
        o, _, _ = flash_attn_varlen_fwd(
            q,
            k,
            v,
            bsz=bsz,
            num_heads=num_heads,
            max_seqlen_q=context.max_seqlen_q,
            cu_seqlens_q=context.cu_seqlens_q,
            max_seqlen_k=context.max_seqlen_k,
            cu_seqlens_k=context.cu_seqlens_k,
            softmax_scale=scaling,
            causal=is_causal,
            block_table=context.block_tables,
            page_block_size=KVCACHE_BLOCK_SIZE,
        )
    else:
        o = flash_attn_decode(
            q,
            k_cache,
            v_cache,
            softmax_scale=scaling,
            causal=is_causal,
            block_table=context.block_tables,
            page_block_size=KVCACHE_BLOCK_SIZE,
        )

    return o, None
