import torch
import triton
import triton.language as tl


@triton.jit
def _store_kv_to_cache_kernel(
    new_k_ptr,
    new_v_ptr,
    cache_k_ptr,
    cache_v_ptr,
    slot_mapping_ptr,
    num_tokens: tl.constexpr,
    stride_new_tok,
    stride_new_head,
    stride_new_d,
    stride_cache_slot,
    stride_cache_head,
    stride_cache_d,
    kv_heads: tl.constexpr,
    dim: tl.constexpr,
    block_size_d: tl.constexpr,
):
    tok = tl.program_id(0)
    head = tl.program_id(1)
    if tok >= num_tokens or head >= kv_heads:
        return

    slot_idx = tl.load(slot_mapping_ptr + tok)
    offs_d = tl.arange(0, block_size_d)
    mask_d = offs_d < dim

    new_k_row = new_k_ptr + tok * stride_new_tok + head * stride_new_head
    new_v_row = new_v_ptr + tok * stride_new_tok + head * stride_new_head
    k_vals = tl.load(new_k_row + offs_d * stride_new_d, mask=mask_d, other=0.0)
    v_vals = tl.load(new_v_row + offs_d * stride_new_d, mask=mask_d, other=0.0)

    cache_k_row = cache_k_ptr + slot_idx * stride_cache_slot + head * stride_cache_head
    cache_v_row = cache_v_ptr + slot_idx * stride_cache_slot + head * stride_cache_head
    tl.store(cache_k_row + offs_d * stride_cache_d, k_vals, mask=mask_d)
    tl.store(cache_v_row + offs_d * stride_cache_d, v_vals, mask=mask_d)


def store_kv_to_cache_kernel(key_states_not_repeated, value_states_not_repeated, key_cache, value_cache, slot_mapping):
    if not (key_cache.numel() and value_cache.numel()) or slot_mapping is None or slot_mapping.numel() == 0:
        return

    device = key_cache.device
    assert value_states_not_repeated.shape == key_states_not_repeated.shape
    bsz, length, kv_heads, dim = key_states_not_repeated.shape

    key_flat = key_states_not_repeated.contiguous().view(-1, kv_heads, dim).to(device)
    value_flat = value_states_not_repeated.contiguous().view(-1, kv_heads, dim).to(device)
    num_tokens = key_flat.size(0)

    assert slot_mapping.numel() == num_tokens, f"slot_mapping len {slot_mapping.numel()} vs new tokens {num_tokens}"
    slot_mapping = slot_mapping.to(device).to(torch.int32)

    num_blocks, block_size, kv_heads_c, dim_cache = key_cache.shape
    assert kv_heads_c == kv_heads and dim_cache == dim, (
        f"KV cache shape mismatch: key_states {kv_heads}x{dim}, " f"cache {kv_heads_c}x{dim_cache}"
    )

    num_slots = num_blocks * block_size
    k_slots = key_cache.view(num_slots, kv_heads, dim)
    v_slots = value_cache.view(num_slots, kv_heads, dim)

    stride_new_tok, stride_new_head, stride_new_d = key_flat.stride()
    stride_cache_slot, stride_cache_head, stride_cache_d = k_slots.stride()

    block_size_d = 32
    while block_size_d < dim and block_size_d < 128:
        block_size_d *= 2

    grid = (num_tokens, kv_heads)
    _store_kv_to_cache_kernel[grid](
        key_flat,
        value_flat,
        k_slots,
        v_slots,
        slot_mapping,
        num_tokens,
        stride_new_tok,
        stride_new_head,
        stride_new_d,
        stride_cache_slot,
        stride_cache_head,
        stride_cache_d,
        kv_heads=kv_heads,
        dim=dim,
        block_size_d=block_size_d,
    )
