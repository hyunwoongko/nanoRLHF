import math

import torch

from nanorlhf.kernels.flash_attn_varlen.fwd import flash_attn_varlen_fwd

torch.manual_seed(0)


def make_random_varlen_batch(
    bsz: int,
    num_heads: int,
    dim: int,
    max_seqlen_q: int,
    max_seqlen_k: int,
    device: str = "cuda",
    same_qk_len: bool = True,
):
    assert device.startswith("cuda")
    seqlens_q = torch.randint(1, max_seqlen_q + 1, (bsz,), device=device)
    if same_qk_len:
        seqlens_k = seqlens_q.clone()
    else:
        seqlens_k = torch.randint(1, max_seqlen_k + 1, (bsz,), device=device)

    cu_seqlens_q = torch.zeros(bsz + 1, device=device, dtype=torch.int32)
    cu_seqlens_k = torch.zeros(bsz + 1, device=device, dtype=torch.int32)
    cu_seqlens_q[1:] = torch.cumsum(seqlens_q, dim=0)
    cu_seqlens_k[1:] = torch.cumsum(seqlens_k, dim=0)

    total_q = int(cu_seqlens_q[-1].item())
    total_k = int(cu_seqlens_k[-1].item())

    q = torch.randn(total_q, num_heads, dim, device=device, dtype=torch.float16)
    k = torch.randn(total_k, num_heads, dim, device=device, dtype=torch.float16)
    v = torch.randn(total_k, num_heads, dim, device=device, dtype=torch.float16)

    return q, k, v, cu_seqlens_q, cu_seqlens_k, seqlens_q, seqlens_k


def flash_attn_ref(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    seqlens_q,
    seqlens_k,
    bsz,
    num_heads,
    causal: bool = True,
    softmax_scale: float | None = None,
):
    device = q.device
    dtype = torch.float32

    total_q, num_heads_q, dim = q.shape
    total_k, num_heads_k, dim_k = k.shape
    assert num_heads_q == num_heads_k == num_heads
    assert dim == dim_k

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(dim)

    out = torch.empty(total_q, num_heads, dim, device=device, dtype=dtype)

    for b in range(bsz):
        qs = int(cu_seqlens_q[b].item())
        qe = int(cu_seqlens_q[b + 1].item())
        ks = int(cu_seqlens_k[b].item())
        ke = int(cu_seqlens_k[b + 1].item())

        Lq = qe - qs
        Lk = ke - ks

        q_b = q[qs:qe].to(dtype)
        k_b = k[ks:ke].to(dtype)
        v_b = v[ks:ke].to(dtype)

        q_b_h = q_b.permute(1, 0, 2)
        k_b_h = k_b.permute(1, 0, 2)
        v_b_h = v_b.permute(1, 0, 2)

        scores = torch.matmul(q_b_h, k_b_h.transpose(-2, -1)) * softmax_scale

        if causal:
            offset = Lk - Lq
            q_pos = (offset + torch.arange(Lq, device=device))[:, None]
            kv_pos = torch.arange(Lk, device=device)[None, :]
            causal_mask = kv_pos > q_pos
            scores = scores.masked_fill(causal_mask.unsqueeze(0), float("-inf"))

        p = torch.softmax(scores, dim=-1)
        out_b_h = torch.matmul(p, v_b_h)
        out_b = out_b_h.permute(1, 0, 2)

        out[qs:qe] = out_b

    return out.to(q.dtype)


def build_paged_kv_cache(
    k,
    v,
    cu_seqlens_k,
    bsz,
    num_heads,
    page_block_size: int,
):
    device = k.device
    total_k, num_heads, dim = k.shape

    seqlens_k = cu_seqlens_k[1:] - cu_seqlens_k[:-1]
    max_len_k = int(seqlens_k.max().item())
    max_pages_per_seq = (max_len_k + page_block_size - 1) // page_block_size

    num_pages = bsz * max_pages_per_seq
    cache_len = num_pages * page_block_size

    k_cache = torch.zeros(cache_len, num_heads, dim, device=device, dtype=k.dtype)
    v_cache = torch.zeros_like(k_cache)

    block_table = torch.zeros(bsz, max_pages_per_seq, device=device, dtype=torch.int32)

    for b in range(bsz):
        ks = int(cu_seqlens_k[b].item())
        ke = int(cu_seqlens_k[b + 1].item())
        len_b = ke - ks

        num_pages_b = (len_b + page_block_size - 1) // page_block_size

        for p in range(num_pages_b):
            page_id = b * max_pages_per_seq + p
            block_table[b, p] = page_id

            page_start_tok = p * page_block_size
            page_end_tok = min((p + 1) * page_block_size, len_b)
            this_page_len = page_end_tok - page_start_tok

            if this_page_len <= 0:
                continue

            src_start = ks + page_start_tok
            src_end = ks + page_end_tok

            dst_row_start = page_id * page_block_size
            dst_row_end = dst_row_start + this_page_len

            k_cache[dst_row_start:dst_row_end] = k[src_start:src_end]
            v_cache[dst_row_start:dst_row_end] = v[src_start:src_end]

    max_seqlen_k_cache = cache_len
    return k_cache, v_cache, block_table, max_seqlen_k_cache


def run_single_test(
    bsz=4,
    num_heads=8,
    dim=64,
    max_seqlen_q=128,
    max_seqlen_k=128,
    causal=True,
    page_block_size=64,
    device="cuda",
):
    q, k, v, cu_seqlens_q, cu_seqlens_k, seqlens_q, seqlens_k = make_random_varlen_batch(
        bsz, num_heads, dim, max_seqlen_q, max_seqlen_k, device=device, same_qk_len=True
    )

    out_ref = flash_attn_ref(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        seqlens_q,
        seqlens_k,
        bsz,
        num_heads,
        causal=causal,
    )

    o_triton, max_q, ez_sum = flash_attn_varlen_fwd(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        bsz,
        num_heads,
        max_seqlen_q=int(seqlens_q.max().item()),
        max_seqlen_k=int(seqlens_k.max().item()),
        causal=causal,
        softmax_scale=None,
        block_table=None,
        page_block_size=None,
    )

    diff_contig = (o_triton - out_ref).abs().max().item()
    print(f"[contiguous K/V] max |diff| vs ref = {diff_contig:.3e}")

    k_cache, v_cache, block_table, max_seqlen_k_cache = build_paged_kv_cache(
        k, v, cu_seqlens_k, bsz, num_heads, page_block_size=page_block_size
    )

    o_triton_bt, max_q_bt, ez_sum_bt = flash_attn_varlen_fwd(
        q,
        k_cache,
        v_cache,
        cu_seqlens_q,
        cu_seqlens_k,
        bsz,
        num_heads,
        max_seqlen_q=int(seqlens_q.max().item()),
        max_seqlen_k=max_seqlen_k_cache,
        causal=causal,
        softmax_scale=None,
        block_table=block_table,
        page_block_size=page_block_size,
    )

    diff_bt_ref = (o_triton_bt - out_ref).abs().max().item()
    diff_bt_contig = (o_triton_bt - o_triton).abs().max().item()
    print(f"[block_table K/V] max |diff| vs ref       = {diff_bt_ref:.3e}")
    print(f"[block_table K/V] max |diff| vs contiguous = {diff_bt_contig:.3e}")
    print()


if __name__ == "__main__":
    assert torch.cuda.is_available()
    tests = [
        dict(bsz=2, num_heads=4, dim=32, max_seqlen_q=64, max_seqlen_k=64, causal=True),
        dict(bsz=4, num_heads=8, dim=64, max_seqlen_q=128, max_seqlen_k=128, causal=True),
        dict(bsz=2, num_heads=8, dim=64, max_seqlen_q=96, max_seqlen_k=96, causal=False),
    ]

    for cfg in tests:
        run_single_test(**cfg, page_block_size=64, device="cuda")
