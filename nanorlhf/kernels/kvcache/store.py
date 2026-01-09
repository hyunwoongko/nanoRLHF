import torch
import triton
import triton.language as tl


@triton.jit
def store_kv_to_cache_kernel(
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
    """
    A kernel to store key and value states into the KV cache based on slot mapping.

    Args:
        new_k_ptr: Pointer to the new key states tensor of shape (num_tokens, kv_heads, dim).
        new_v_ptr: Pointer to the new value states tensor of shape (num_tokens, kv_heads, dim).
        cache_k_ptr: Pointer to the key cache tensor of shape (num_slots, kv_heads, dim).
        cache_v_ptr: Pointer to the value cache tensor of shape (num_slots, kv_heads, dim).
        slot_mapping_ptr: Pointer to the slot mapping tensor of shape (num_tokens,), mapping each token to a cache slot.
        num_tokens: Total number of new tokens.
        stride_new_tok: Stride for tokens in the new key/value tensors.
        stride_new_head: Stride for heads in the new key/value tensors.
        stride_new_d: Stride for dimensions in the new key/value tensors.
        stride_cache_slot: Stride for slots in the cache tensors.
        stride_cache_head: Stride for heads in the cache tensors.
        stride_cache_d: Stride for dimensions in the cache tensors.
        kv_heads: Number of key/value heads.
        dim: Dimension of each head.
        block_size_d: Block size for dimension processing.

    Discussion:
        Q. What is the purpose of this kernel?
            This kernel efficiently stores new key and value states into a pre-allocated KV cache based on a provided
            slot mapping. Each token's new key and value vectors are written to specific slots in the cache, allowing
            for dynamic updates of the KV cache during sequence generation or processing.

        Q. How does the slot mapping work?
            The slot mapping tensor indicates which cache slot each new token's key and value should be stored in.
            A value of -1 in the slot mapping indicates that the corresponding token should not be stored in the cache.
            This allows for flexible management of the KV cache, enabling reuse of slots and efficient memory usage.
    """
    token = tl.program_id(0)
    head = tl.program_id(1)
    if token >= num_tokens or head >= kv_heads:
        return

    slot_idx = tl.load(slot_mapping_ptr + token)
    offs_d = tl.arange(0, block_size_d)
    mask_d = offs_d < dim
    valid_slot = slot_idx >= 0

    new_k_row = new_k_ptr + token * stride_new_tok + head * stride_new_head
    new_v_row = new_v_ptr + token * stride_new_tok + head * stride_new_head
    k_values = tl.load(new_k_row + offs_d * stride_new_d, mask=mask_d, other=0.0)
    v_values = tl.load(new_v_row + offs_d * stride_new_d, mask=mask_d, other=0.0)

    cache_k_row = cache_k_ptr + slot_idx * stride_cache_slot + head * stride_cache_head
    cache_v_row = cache_v_ptr + slot_idx * stride_cache_slot + head * stride_cache_head

    store_mask = mask_d & valid_slot
    tl.store(cache_k_row + offs_d * stride_cache_d, k_values, mask=store_mask)
    tl.store(cache_v_row + offs_d * stride_cache_d, v_values, mask=store_mask)


def store_kv_to_cache(key_states_not_repeated, value_states_not_repeated, key_cache, value_cache, slot_mapping):
    """
    Store key and value states into the KV cache based on slot mapping.

    Args:
        key_states_not_repeated (torch.Tensor): New key states of shape (bsz, length, kv_heads, dim).
        value_states_not_repeated (torch.Tensor): New value states of shape (bsz, length, kv_heads, dim).
        key_cache (torch.Tensor): Key cache tensor of shape (num_blocks, block_size, kv_heads, head_dim).
        value_cache (torch.Tensor): Value cache tensor of shape (num_blocks, block_size, kv_heads, head_dim).
        slot_mapping (torch.Tensor): Slot mapping tensor of shape (num_tokens,), mapping each token to a cache slot.
    """
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
    store_kv_to_cache_kernel[grid](
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
