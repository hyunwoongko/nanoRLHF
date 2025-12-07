import math
import time

import torch

# TODO: 경로는 네 프로젝트 구조에 맞게 수정해줘
from nanorlhf.kernels.flash_attn.fwd import flash_attn_fwd
from nanorlhf.kernels.flash_attn_decode.ops import flash_attn_decode


def attention_ref(q, k, v, causal=True, softmax_scale=None):
    """
    PyTorch reference attention (prefill / decode 둘 다 커버)
    q: [B, H, S_q, D]
    k: [B, H, S_k, D]
    v: [B, H, S_k, D]
    """
    bsz, num_heads, seq_len_q, dim = q.shape
    _, _, seq_len_k, _ = k.shape

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(dim)

    # [B, H, S_q, S_k]
    scores = torch.matmul(q, k.transpose(-1, -2)) * softmax_scale

    if causal:
        # 네 커널에서 쓰는 masking 방식과 동일하게 맞춤
        # offset = S_k - S_q
        device = q.device
        q_pos = torch.arange(seq_len_q, device=device) + (seq_len_k - seq_len_q)  # [S_q]
        kv_pos = torch.arange(seq_len_k, device=device)                            # [S_k]
        causal_mask = kv_pos[None, None, None, :] > q_pos[None, None, :, None]    # [1,1,S_q,S_k]
        scores = scores.masked_fill(causal_mask, float("-inf"))

    attn = torch.softmax(scores, dim=-1)      # [B, H, S_q, S_k]
    out = torch.matmul(attn, v)               # [B, H, S_q, D]
    return out


# ---------------------------------------------------------------------------
# 1) Prefill 시나리오 테스트: S_q = S_k
# ---------------------------------------------------------------------------

def run_prefill_accuracy_test(
    device="cuda",
    dtypes=(torch.float16, torch.bfloat16),
    batch_sizes=(1, 2),
    num_heads_list=(4, 8),
    seq_list=(64, 256, 1024),
    dims=(64, 128),
    causal=True,
):
    torch.manual_seed(0)
    print("\n================ Prefill accuracy test ================\n")

    for dtype in dtypes:
        print(f"\n=== dtype: {dtype} ===")
        for bsz in batch_sizes:
            for num_heads in num_heads_list:
                for dim in dims:
                    for seq_len in seq_list:
                        shape_str = f"[PREFILL] B={bsz}, H={num_heads}, D={dim}, S={seq_len}"
                        print(shape_str, flush=True)

                        q = torch.randn(bsz, num_heads, seq_len, dim, device=device, dtype=dtype)
                        k = torch.randn(bsz, num_heads, seq_len, dim, device=device, dtype=dtype)
                        v = torch.randn_like(k)

                        # reference는 float32로 계산
                        q_ref = q.float()
                        k_ref = k.float()
                        v_ref = v.float()
                        scale = 1.0 / math.sqrt(dim)

                        with torch.no_grad():
                            out_ref = attention_ref(q_ref, k_ref, v_ref, causal=causal, softmax_scale=scale).to(dtype)

                            # Triton prefill 커널
                            out_fwd, _, _ = flash_attn_fwd(
                                q, k, v,
                                causal=causal,
                                softmax_scale=scale,
                            )

                            # decode 커널을 prefill 모드처럼 (S_q = S_k) 사용
                            out_decode = flash_attn_decode(
                                q, k, v,
                                split_k=None,
                                causal=causal,
                                softmax_scale=scale,
                                block_size_q=16,
                                block_size_k=16,
                            )

                        # 에러 계산
                        def error_stats(x, y):
                            diff = (x - y).abs()
                            max_abs = diff.max().item()
                            denom = y.abs().max().item() + 1e-6
                            max_rel = max_abs / denom
                            return max_abs, max_rel

                        abs_fwd, rel_fwd = error_stats(out_fwd, out_ref)
                        abs_dec, rel_dec = error_stats(out_decode, out_ref)
                        abs_fwd_vs_dec, rel_fwd_vs_dec = error_stats(out_fwd, out_decode)

                        print(
                            f"  flash_attn_fwd vs ref    : max_abs={abs_fwd:.3e}, max_rel={rel_fwd:.3e}"
                        )
                        print(
                            f"  flash_attn_decode vs ref : max_abs={abs_dec:.3e}, max_rel={rel_dec:.3e}"
                        )
                        print(
                            f"  fwd vs decode            : max_abs={abs_fwd_vs_dec:.3e}, max_rel={rel_fwd_vs_dec:.3e}"
                        )


def benchmark_prefill(
    bsz=1,
    num_heads=8,
    dim=128,
    seq_len=1024,
    dtype=torch.float16,
    causal=True,
    warmup=10,
    iters=50,
    device="cuda",
):
    torch.manual_seed(0)
    print("\n================ Prefill benchmark ================\n")

    q = torch.randn(bsz, num_heads, seq_len, dim, device=device, dtype=dtype)
    k = torch.randn(bsz, num_heads, seq_len, dim, device=device, dtype=dtype)
    v = torch.randn_like(k)

    q_ref = q.float()
    k_ref = k.float()
    v_ref = v.float()

    scale = 1.0 / math.sqrt(dim)

    # warmup
    for _ in range(warmup):
        _ = attention_ref(q_ref, k_ref, v_ref, causal=causal, softmax_scale=scale)
        _ = flash_attn_fwd(q, k, v, causal=causal, softmax_scale=scale)
        _ = flash_attn_decode(q, k, v, split_k=None, causal=causal, softmax_scale=scale,
                              block_size_q=16, block_size_k=16)
    torch.cuda.synchronize()

    # PyTorch baseline
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = attention_ref(q_ref, k_ref, v_ref, causal=causal, softmax_scale=scale)
    torch.cuda.synchronize()
    t_ref = (time.perf_counter() - t0) * 1000 / iters

    # flash_attn_fwd
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = flash_attn_fwd(q, k, v, causal=causal, softmax_scale=scale)
    torch.cuda.synchronize()
    t_fwd = (time.perf_counter() - t0) * 1000 / iters

    # flash_attn_decode (S_q = S_k)
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = flash_attn_decode(q, k, v, split_k=None, causal=causal, softmax_scale=scale,
                              block_size_q=16, block_size_k=16)
    torch.cuda.synchronize()
    t_dec = (time.perf_counter() - t0) * 1000 / iters

    print(f"[PREFILL BENCH] B={bsz}, H={num_heads}, D={dim}, S={seq_len}, dtype={dtype}")
    print(f"  PyTorch          : {t_ref:.3f} ms/iter")
    print(f"  flash_attn_fwd   : {t_fwd:.3f} ms/iter")
    print(f"  flash_attn_decode: {t_dec:.3f} ms/iter")


# ---------------------------------------------------------------------------
# 2) Decode 시나리오 테스트: S_q << S_k
# ---------------------------------------------------------------------------

def run_decode_accuracy_test(
    device="cuda",
    dtypes=(torch.float16, torch.bfloat16),
    batch_sizes=(1, 2),
    num_heads_list=(4, 8),
    seq_q_list=(1, 16, 32),
    seq_k_list=(64, 256, 1024, 2048),
    dims=(64, 128),
    causal=True,
):
    torch.manual_seed(1)
    print("\n================ Decode accuracy test ================\n")

    for dtype in dtypes:
        print(f"\n=== dtype: {dtype} ===")
        for bsz in batch_sizes:
            for num_heads in num_heads_list:
                for dim in dims:
                    for seq_len_k in seq_k_list:
                        for seq_len_q in seq_q_list:
                            if seq_len_q > seq_len_k:
                                continue  # decode 시나리오에서는 보통 S_q <= S_k

                            shape_str = (
                                f"[DECODE] B={bsz}, H={num_heads}, D={dim}, "
                                f"S_q={seq_len_q}, S_k={seq_len_k}"
                            )
                            print(shape_str, flush=True)

                            q = torch.randn(bsz, num_heads, seq_len_q, dim, device=device, dtype=dtype)
                            k = torch.randn(bsz, num_heads, seq_len_k, dim, device=device, dtype=dtype)
                            v = torch.randn_like(k)

                            q_ref = q.float()
                            k_ref = k.float()
                            v_ref = v.float()
                            scale = 1.0 / math.sqrt(dim)

                            with torch.no_grad():
                                out_ref = attention_ref(q_ref, k_ref, v_ref, causal=causal, softmax_scale=scale).to(dtype)

                                out_triton = flash_attn_decode(
                                    q, k, v,
                                    split_k=None,
                                    causal=causal,
                                    softmax_scale=scale,
                                    block_size_q=16,
                                    block_size_k=16,
                                )

                            diff = (out_triton - out_ref).abs()
                            max_abs_error = diff.max().item()
                            denom = out_ref.abs().max().item() + 1e-6
                            max_rel_error = max_abs_error / denom

                            print(
                                f"  max_abs_error={max_abs_error:.3e}, "
                                f"max_rel_error={max_rel_error:.3e}"
                            )


def benchmark_decode(
    bsz=1,
    num_heads=8,
    dim=128,
    seq_len_q=1,
    seq_len_k=2048,
    dtype=torch.float16,
    causal=True,
    warmup=10,
    iters=50,
    device="cuda",
):
    torch.manual_seed(1)
    print("\n================ Decode benchmark ================\n")

    q = torch.randn(bsz, num_heads, seq_len_q, dim, device=device, dtype=dtype)
    k = torch.randn(bsz, num_heads, seq_len_k, dim, device=device, dtype=dtype)
    v = torch.randn_like(k)

    q_ref = q.float()
    k_ref = k.float()
    v_ref = v.float()

    scale = 1.0 / math.sqrt(dim)

    # warmup
    for _ in range(warmup):
        _ = attention_ref(q_ref, k_ref, v_ref, causal=causal, softmax_scale=scale)
        _ = flash_attn_decode(q, k, v, split_k=None, causal=causal, softmax_scale=scale,
                              block_size_q=16, block_size_k=16)
    torch.cuda.synchronize()

    # PyTorch baseline
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = attention_ref(q_ref, k_ref, v_ref, causal=causal, softmax_scale=scale)
    torch.cuda.synchronize()
    t_ref = (time.perf_counter() - t0) * 1000 / iters

    # Triton decode
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = flash_attn_decode(
            q, k, v,
            split_k=None,
            causal=causal,
            softmax_scale=scale,
            block_size_q=16,
            block_size_k=16,
        )
    torch.cuda.synchronize()
    t_triton = (time.perf_counter() - t0) * 1000 / iters

    print(
        f"[DECODE BENCH] B={bsz}, H={num_heads}, D={dim}, "
        f"S_q={seq_len_q}, S_k={seq_len_k}, dtype={dtype}"
    )
    print(f"  PyTorch: {t_ref:.3f} ms/iter")
    print(f"  Triton : {t_triton:.3f} ms/iter")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA device is required for Triton tests"

    device = "cuda"

    # 1) Prefill 정확도
    run_prefill_accuracy_test(device=device)

    # 2) Decode 정확도
    run_decode_accuracy_test(device=device)

    # 3) Prefill 벤치마크
    benchmark_prefill(
        bsz=4,
        num_heads=32,
        dim=128,
        seq_len=1024,
        dtype=torch.float16,
        device=device,
    )

    # 4) Decode 벤치마크 (진짜 decode 스타일: S_q=1)
    benchmark_decode(
        bsz=4,
        num_heads=32,
        dim=512,
        seq_len_q=1,
        seq_len_k=8192,
        dtype=torch.float16,
        device=device,
    )