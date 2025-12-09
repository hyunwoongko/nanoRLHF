from dataclasses import dataclass
from typing import Optional

import torch
from transformers.modeling_flash_attention_utils import fa_peft_integration_check, logger

from nanorlhf.kernels import flash_attn_varlen_func
from nanorlhf.kernels.flash_attn_decode.ops import flash_attn_decode
from nanorlhf.kernels.kvcache.load import load_kv_from_cache_prefill, load_kv_from_cache_decode
from nanorlhf.kernels.kvcache.store import store_kv_to_cache_kernel
from nanorlhf.kernels.utils.huggingface import _maybe_repeat_kv, _get_target_dtype


@dataclass
class Context:
    is_prefill: bool = False
    slot_mapping: Optional[torch.Tensor] = None
    context_lens: Optional[torch.Tensor] = None
    block_tables: Optional[torch.Tensor] = None


KVCACHE_BLOCK_SIZE = 256
GLOBAL_CONTEXT = Context()


def get_context() -> Context:
    return GLOBAL_CONTEXT


def set_context(is_prefill, slot_mapping=None, context_lens=None, block_tables=None):
    global GLOBAL_CONTEXT
    GLOBAL_CONTEXT = Context(
        is_prefill=is_prefill,
        slot_mapping=slot_mapping,
        context_lens=context_lens,
        block_tables=block_tables,
    )


def reset_context():
    global GLOBAL_CONTEXT
    GLOBAL_CONTEXT = Context()


def store_kv_to_cache(
    context,
    key_states_not_repeated,
    value_states_not_repeated,
    key_cache,
    value_cache,
    bsz,
    device,
):
    if context.is_prefill:
        assert context.context_lens is not None, "prefill requires context_lens for KV cache write"
        lengths = context.context_lens.to(device)
        assert lengths.numel() == bsz

        k_pieces = []
        v_pieces = []
        for b in range(bsz):
            length = int(lengths[b].item())
            if length == 0:
                continue
            k_pieces.append(key_states_not_repeated[b, :length])
            v_pieces.append(value_states_not_repeated[b, :length])

        if k_pieces:
            k_for_cache = torch.cat(k_pieces, dim=0).unsqueeze(0)
            v_for_cache = torch.cat(v_pieces, dim=0).unsqueeze(0)
            store_kv_to_cache_kernel(
                key_states_not_repeated=k_for_cache,
                value_states_not_repeated=v_for_cache,
                key_cache=key_cache,
                value_cache=value_cache,
                slot_mapping=context.slot_mapping,
            )
    else:
        k_for_cache = key_states_not_repeated[:, -1:, :, :]
        v_for_cache = value_states_not_repeated[:, -1:, :, :]
        store_kv_to_cache_kernel(
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
    bsz, q_len, num_heads, _ = query_states.shape
    k_len = key_states.shape[1]

    cu_seqlens_q = torch.arange(
        0,
        (bsz * q_len) + 1,
        step=q_len,
        dtype=torch.int32,
        device=query_states.device,
    )
    cu_seqlens_k = torch.arange(
        0,
        (bsz * k_len) + 1,
        step=k_len,
        dtype=torch.int32,
        device=query_states.device,
    )

    if "mps" in str(query_states.device):
        cu_seqlens_k = cu_seqlens_k.clone()

    if context.block_tables is not None:
        k_bh, v_bh = load_kv_from_cache_prefill(
            context=context,
            cu_seqlens_k=cu_seqlens_k,
            key_cache=key_cache,
            value_cache=value_cache,
            num_heads=num_heads,
            dim=dim,
        )
    else:
        k_bh = key_states.reshape(-1, num_heads, key_states.size(-1))
        v_bh = value_states.reshape(-1, num_heads, value_states.size(-1))

    out = flash_attn_varlen_func(
        query_states.reshape(-1, num_heads, query_states.size(-1)),
        k_bh,
        v_bh,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        softmax_scale=scaling,
        causal=is_causal,
    )
    if isinstance(out, tuple):
        out = out[0]

    out = out.view(query_states.size(0), -1, out.size(-2), out.size(-1))
    return out, None


def compute_decode(
    context,
    query_states,
    key_cache,
    value_cache,
    bsz,
    seqlen_q,
    num_heads,
    dim,
    scaling,
    is_causal,
):
    assert seqlen_q == 1, f"decode expects seqlen_q=1, got {seqlen_q}"
    assert context.block_tables is not None, "Decode requires block_tables in context."
    assert context.context_lens is not None, "Decode requires context_lens in context."
    q_bh = query_states.permute(0, 2, 1, 3).reshape(bsz * num_heads, seqlen_q, dim).contiguous()

    k_bh, v_bh = load_kv_from_cache_decode(
        context=context,
        key_cache=key_cache,
        value_cache=value_cache,
        num_heads=num_heads,
        dim=dim,
    )
    out = flash_attn_decode(
        q_bh,
        k_bh,
        v_bh,
        softmax_scale=scaling,
        causal=is_causal,
    )
    out = out.view(bsz, num_heads, seqlen_q, dim)
    return out, None


@torch.no_grad()
def paged_flash_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: Optional[float] = None,
    is_causal: Optional[bool] = None,
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

    bsz, num_heads, seqlen_q, dim = query.shape
    device = query.device

    query_states = query.transpose(1, 2)
    key_states = key.transpose(1, 2)
    value_states = value.transpose(1, 2)

    target_dtype = _get_target_dtype(query_states, module)
    is_causal = is_causal if is_causal is not None else module.is_causal

    query_states, key_states, value_states = fa_peft_integration_check(
        query_states, key_states, value_states, target_dtype
    )

    key_states_not_repeated = key.transpose(1, 2)
    value_states_not_repeated = value.transpose(1, 2)
    query_states, key_states, value_states = _maybe_repeat_kv(query_states, key_states, value_states)

    if scaling is None:
        scaling = 1.0 / (query_states.size(-1) ** 0.5)

    assert hasattr(module, "key_cache") and hasattr(module, "value_cache"), (
        "NanoRLHF paged_flash_attention_forward requires the attention module to have "
        "`key_cache` and `value_cache` attributes for KV cache."
    )

    key_cache, value_cache = module.key_cache, module.value_cache
    if context.slot_mapping is not None and context.slot_mapping.numel() > 0:
        store_kv_to_cache(
            context=context,
            key_states_not_repeated=key_states_not_repeated,
            value_states_not_repeated=value_states_not_repeated,
            key_cache=key_cache,
            value_cache=value_cache,
            bsz=bsz,
            device=device,
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
            bsz=bsz,
            seqlen_q=seqlen_q,
            num_heads=num_heads,
            dim=dim,
            scaling=scaling,
            is_causal=is_causal,
        )
