import torch


def load_kv_from_cache_prefill(context, cu_seq_lens_k, key_cache, value_cache, num_heads, dim):
    """
    Load keys and values from cache during prefill phase.

    Args:
        context: An object containing block_tables.
        cu_seq_lens_k (torch.Tensor): Cumulative sequence lengths for keys.
        key_cache (torch.Tensor): Key cache tensor of shape (num_blocks, block_size, kv_heads, head_dim).
        value_cache (torch.Tensor): Value cache tensor of shape (num_blocks, block_size, kv_heads, head_dim).
        num_heads (int): Number of attention heads.
        dim (int): Dimension of each head.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Loaded keys and values tensors.
    """
    assert context.block_tables is not None

    device = key_cache.device
    cu_seq_lens_k = cu_seq_lens_k.to(device)
    block_tables = context.block_tables.to(device)

    b = cu_seq_lens_k.numel() - 1
    num_blocks, block_size, kv_heads, hd = key_cache.shape
    assert hd == dim, f"head_dim mismatch: cache {hd}, q {dim}"
    assert block_tables.size(0) == b

    num_slots = num_blocks * block_size
    k_slots = key_cache.view(num_slots, kv_heads, dim)
    v_slots = value_cache.view(num_slots, kv_heads, dim)

    k_chunks = []
    v_chunks = []
    total_k = 0
    for _b in range(b):
        length = int((cu_seq_lens_k[_b + 1] - cu_seq_lens_k[_b]).item())
        if length == 0:
            continue

        blocks_b = block_tables[_b]
        pos = torch.arange(length, device=device, dtype=torch.int32)

        block_idx = pos // block_size
        block_off = pos % block_size

        page_ids = blocks_b[block_idx]
        slot_ids = page_ids * block_size + block_off

        k_seq = k_slots.index_select(0, slot_ids)
        v_seq = v_slots.index_select(0, slot_ids)

        k_chunks.append(k_seq)
        v_chunks.append(v_seq)
        total_k += length

    if total_k == 0:
        return (
            torch.empty(0, num_heads, dim, device=device, dtype=key_cache.dtype),
            torch.empty(0, num_heads, dim, device=device, dtype=value_cache.dtype),
        )

    k_all = torch.cat(k_chunks, dim=0)
    v_all = torch.cat(v_chunks, dim=0)

    if num_heads == kv_heads:
        k_used = k_all
        v_used = v_all
    else:
        assert num_heads % kv_heads == 0, f"num_heads ({num_heads}) must be multiple of kv_heads ({kv_heads})"
        group_size = num_heads // kv_heads
        k_used = k_all.repeat_interleave(group_size, dim=1)
        v_used = v_all.repeat_interleave(group_size, dim=1)

    return k_used, v_used


# deprecated, kept for reference
def load_kv_from_cache_decode(context, key_cache, value_cache, num_heads, dim):
    """
    Load keys and values from cache during decode phase.

    Args:
        context: An object containing block_tables and context_lens.
        key_cache (torch.Tensor): Key cache tensor of shape (num_blocks, block_size, kv_heads, head_dim).
        value_cache (torch.Tensor): Value cache tensor of shape (num_blocks, block_size, kv_heads, head_dim).
        num_heads (int): Number of attention heads.
        dim (int): Dimension of each head.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Loaded keys and values tensors.
    """
    assert context.block_tables is not None
    assert context.context_lens is not None

    device = key_cache.device
    block_tables = context.block_tables.to(device)
    context_lens = context.context_lens.to(device)

    b = block_tables.size(0)
    num_blocks, block_size, kv_heads, hd = key_cache.shape
    assert hd == dim, f"head_dim mismatch: cache {hd}, q {dim}"
    assert context_lens.size(0) == b

    max_len = int(context_lens.max().item())
    num_slots = num_blocks * block_size

    k_slots = key_cache.view(num_slots, kv_heads, dim)
    v_slots = value_cache.view(num_slots, kv_heads, dim)

    k_dense = torch.zeros(
        b * kv_heads,
        max_len,
        dim,
        device=device,
        dtype=key_cache.dtype,
    )
    v_dense = torch.zeros_like(k_dense)

    positions = torch.arange(max_len, device=device, dtype=torch.int32)
    for _b in range(b):
        length = int(context_lens[_b].item())
        if length == 0:
            continue

        blocks_b = block_tables[_b]
        pos_b = positions[:length]

        block_idx = pos_b // block_size
        block_off = pos_b % block_size
        page_ids = blocks_b[block_idx]
        slot_ids = page_ids * block_size + block_off

        k_seq = k_slots.index_select(0, slot_ids)
        v_seq = v_slots.index_select(0, slot_ids)

        k_seq_hld = k_seq.permute(1, 0, 2).contiguous()
        v_seq_hld = v_seq.permute(1, 0, 2).contiguous()

        start = _b * kv_heads
        end = (_b + 1) * kv_heads
        k_dense[start:end, :length] = k_seq_hld
        v_dense[start:end, :length] = v_seq_hld

    if num_heads == kv_heads:
        k_bh = k_dense
        v_bh = v_dense
    else:
        assert num_heads % kv_heads == 0, f"num_heads ({num_heads}) must be multiple of kv_heads ({kv_heads})"
        group_size = num_heads // kv_heads
        k_bh = k_dense.repeat_interleave(group_size, dim=0)
        v_bh = v_dense.repeat_interleave(group_size, dim=0)

    return k_bh, v_bh

