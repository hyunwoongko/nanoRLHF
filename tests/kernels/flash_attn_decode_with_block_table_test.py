import math

import torch

from nanorlhf.kernels.flash_attn_decode.ops import flash_attn_decode


torch.manual_seed(0)


def flash_attn_decode_ref(q, k, v, causal: bool = True, softmax_scale: float | None = None):
    """
    q, k, v: [B, H, T_q, D], [B, H, T_k, D]
    """
    device = q.device
    dtype = torch.float32

    bsz, num_heads, seq_len_q, dim = q.shape
    _, _, seq_len_k, dim_k = k.shape
    assert dim == dim_k

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(dim)

    q_f = q.to(dtype)
    k_f = k.to(dtype)
    v_f = v.to(dtype)

    scores = torch.matmul(q_f, k_f.transpose(-2, -1)) * softmax_scale

    if causal:
        offset = seq_len_k - seq_len_q
        q_pos = offset + torch.arange(seq_len_q, device=device)
        kv_pos = torch.arange(seq_len_k, device=device)
        causal_mask = kv_pos[None, :] > q_pos[:, None]
        scores = scores.masked_fill(causal_mask[None, None, :, :], float("-inf"))

    p = torch.softmax(scores, dim=-1)
    out = torch.matmul(p, v_f)
    return out.to(q.dtype)


def build_paged_kv_cache_decode(k, v, page_block_size: int):
    """
    디코딩용 paged KV 캐시 + block_table 생성

    입력:
        k, v: [B, H, T_k, D]
    출력:
        k_cache, v_cache: [B, H, cache_len, D]
        block_table: [B*H, max_pages_per_seq] (int32)
        max_seqlen_k: int (cache_len)
    """
    assert k.shape == v.shape
    device = k.device
    dtype = k.dtype

    bsz, num_heads, seq_len_k, dim = k.shape
    bh = bsz * num_heads

    num_pages = (seq_len_k + page_block_size - 1) // page_block_size
    cache_len = num_pages * page_block_size

    k_cache = torch.zeros(bsz, num_heads, cache_len, dim, device=device, dtype=dtype)
    v_cache = torch.zeros_like(k_cache)

    block_table = torch.zeros(bh, num_pages, device=device, dtype=torch.int32)

    for b in range(bsz):
        for h in range(num_heads):
            bh_idx = b * num_heads + h
            k_bh = k[b, h]
            v_bh = v[b, h]

            for p in range(num_pages):
                page_id = p
                block_table[bh_idx, p] = page_id

                page_start_tok = p * page_block_size
                page_end_tok = min((p + 1) * page_block_size, seq_len_k)
                this_page_len = page_end_tok - page_start_tok
                if this_page_len <= 0:
                    continue

                src_start = page_start_tok
                src_end = page_end_tok

                dst_row_start = page_id * page_block_size
                dst_row_end = dst_row_start + this_page_len

                k_cache[b, h, dst_row_start:dst_row_end] = k_bh[src_start:src_end]
                v_cache[b, h, dst_row_start:dst_row_end] = v_bh[src_start:src_end]

    max_seqlen_k = cache_len
    return k_cache, v_cache, block_table, max_seqlen_k


def run_single_test(
    bsz=2,
    num_heads=4,
    dim=64,
    seq_len_q=16,
    seq_len_k=64,
    causal=True,
    page_block_size=64,
    device="cuda",
    force_split_k: int | None = None,
):
    print(
        f"=== Test: B={bsz}, H={num_heads}, D={dim}, T_q={seq_len_q}, T_k={seq_len_k}, "
        f"causal={causal}, page_block_size={page_block_size}, split_k={force_split_k} ==="
    )

    q = torch.randn(bsz, num_heads, seq_len_q, dim, device=device, dtype=torch.float16)
    k = torch.randn(bsz, num_heads, seq_len_k, dim, device=device, dtype=torch.float16)
    v = torch.randn_like(k)

    out_ref = flash_attn_decode_ref(q, k, v, causal=causal, softmax_scale=None)
    o_triton_contig = flash_attn_decode(
        q,
        k,
        v,
        split_k=force_split_k,
        causal=causal,
        softmax_scale=None,
        block_size_q=16,
        block_size_k=32,
    )

    diff_contig_ref = (o_triton_contig - out_ref).abs().max().item()
    print(f"[contiguous K/V] max |diff| vs ref = {diff_contig_ref:.3e}")

    k_cache, v_cache, block_table, max_seqlen_k = build_paged_kv_cache_decode(k, v, page_block_size=page_block_size)
    o_triton_bt = flash_attn_decode(
        q,
        k_cache,
        v_cache,
        split_k=force_split_k,
        causal=causal,
        softmax_scale=None,
        block_size_q=16,
        block_size_k=32,
        block_table=block_table,
        page_block_size=page_block_size,
        max_seqlen_k=max_seqlen_k,
    )

    diff_bt_ref = (o_triton_bt - out_ref).abs().max().item()
    diff_bt_contig = (o_triton_bt - o_triton_contig).abs().max().item()
    print(f"[block_table K/V] max |diff| vs ref       = {diff_bt_ref:.3e}")
    print(f"[block_table K/V] max |diff| vs contiguous = {diff_bt_contig:.3e}")
    print()


if __name__ == "__main__":
    assert torch.cuda.is_available()
    device = "cuda"
    tests = [
        dict(bsz=2, num_heads=4, dim=32, seq_len_q=8, seq_len_k=64, causal=True),
        dict(bsz=2, num_heads=4, dim=64, seq_len_q=16, seq_len_k=128, causal=True),
        dict(bsz=2, num_heads=4, dim=64, seq_len_q=16, seq_len_k=128, causal=False),
        dict(bsz=1, num_heads=8, dim=64, seq_len_q=32, seq_len_k=512, causal=True),
    ]

    for i, cfg in enumerate(tests):
        if i == len(tests) - 1:
            run_single_test(**cfg, page_block_size=64, device=device, force_split_k=4)
        else:
            run_single_test(**cfg, page_block_size=64, device=device, force_split_k=None)
