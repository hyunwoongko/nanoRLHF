import time
import torch

from nanorlhf.kernels.api import flash_attn_varlen_func


def max_diff(a: torch.Tensor, b: torch.Tensor):
    diff = (a.float() - b.float()).abs()
    return diff.max().item(), diff.mean().item()


def torch_attn_varlen(q, k, v, cu_q, cu_k, b, h, causal=True, softmax_scale=None):
    """
    q: [total_q, H, D]
    k,v: [total_k, H, D]
    cu_q, cu_k: [B+1] int32
    output: [total_q, H, D]
    """
    device = q.device
    total_q, h_q, d = q.shape
    total_k, h_k, d_k = k.shape
    assert h_q == h_k == h
    assert d == d_k

    if softmax_scale is None:
        softmax_scale = 1.0 / (d**0.5)

    out = torch.zeros_like(q)

    for bi in range(b):
        q_start = cu_q[bi].item()
        q_end = cu_q[bi + 1].item()
        k_start = cu_k[bi].item()
        k_end = cu_k[bi + 1].item()

        seqlen_q = q_end - q_start
        seqlen_k = k_end - k_start
        if seqlen_q == 0 or seqlen_k == 0:
            continue

        q_b = q[q_start:q_end]    # [Lq, H, D]
        k_b = k[k_start:k_end]    # [Lk, H, D]
        v_b = v[k_start:k_end]

        # [H, Lq, D], [H, Lk, D]
        q_b = q_b.permute(1, 0, 2)
        k_b = k_b.permute(1, 0, 2)
        v_b = v_b.permute(1, 0, 2)

        scores = torch.matmul(q_b, k_b.transpose(-2, -1)) * softmax_scale  # [H, Lq, Lk]

        if causal:
            tq, tk = seqlen_q, seqlen_k
            mask = torch.triu(
                torch.ones(tq, tk, device=device, dtype=torch.bool), diagonal=1
            )  # [tq, tk]
            scores = scores.masked_fill(mask.unsqueeze(0), float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        o_b = torch.matmul(attn, v_b)  # [H, Lq, D]
        o_b = o_b.permute(1, 0, 2).contiguous()  # [Lq, H, D]

        out[q_start:q_end] = o_b

    return out


def make_random_varlen_qkv(b, h, max_tq, max_tk, d, device):
    seqlens = torch.randint(1, max_tq + 1, (b,), device=device, dtype=torch.int32)
    seqlens_q = seqlens
    seqlens_k = seqlens.clone()

    cu_q = torch.zeros(b + 1, device=device, dtype=torch.int32)
    cu_k = torch.zeros(b + 1, device=device, dtype=torch.int32)
    cu_q[1:] = seqlens_q.cumsum(0)
    cu_k[1:] = seqlens_k.cumsum(0)

    total_q = cu_q[-1].item()
    total_k = cu_k[-1].item()

    q0 = torch.randn(total_q, h, d, device=device, dtype=torch.float32)
    k0 = torch.randn(total_k, h, d, device=device, dtype=torch.float32)
    v0 = torch.randn(total_k, h, d, device=device, dtype=torch.float32)

    return q0, k0, v0, seqlens_q, seqlens_k, cu_q, cu_k


def run_varlen_accuracy_test(dtype: torch.dtype, device, b=32, h=32, max_tq=1024, max_tk=1024, d=64):
    print(f"\n[Varlen Packed Accuracy] dtype={dtype}, B={b}, H={h}, max_Tq={max_tq}, max_Tk={max_tk}, D={d}")

    # base fp32 tensors + lengths
    q0, k0, v0, seqlens_q, seqlens_k, cu_q, cu_k = make_random_varlen_qkv(
        b, h, max_tq, max_tk, d, device
    )

    # Torch reference path
    q_ref = q0.to(dtype).detach().clone().requires_grad_(True)
    k_ref = k0.to(dtype).detach().clone().requires_grad_(True)
    v_ref = v0.to(dtype).detach().clone().requires_grad_(True)

    o_ref = torch_attn_varlen(q_ref, k_ref, v_ref, cu_q, cu_k, b, h, causal=True)
    loss_ref = o_ref.float().square().mean()
    loss_ref.backward()
    dq_ref, dk_ref, dv_ref = q_ref.grad.detach(), k_ref.grad.detach(), v_ref.grad.detach()

    # Triton varlen path
    q_tr = q0.to(dtype).detach().clone().requires_grad_(True)
    k_tr = k0.to(dtype).detach().clone().requires_grad_(True)
    v_tr = v0.to(dtype).detach().clone().requires_grad_(True)

    o_tr = flash_attn_varlen_func(q_tr, k_tr, v_tr, cu_q, cu_k, causal=True)
    loss_tr = o_tr.float().square().mean()
    loss_tr.backward()
    dq_tr, dk_tr, dv_tr = q_tr.grad.detach(), k_tr.grad.detach(), v_tr.grad.detach()

    o_max, o_mean = max_diff(o_ref, o_tr)
    dq_max, dq_mean = max_diff(dq_ref, dq_tr)
    dk_max, dk_mean = max_diff(dk_ref, dk_tr)
    dv_max, dv_mean = max_diff(dv_ref, dv_tr)

    print(f"  output  max diff = {o_max:.3e}, mean diff = {o_mean:.3e}")
    print(f"  dQ      max diff = {dq_max:.3e}, mean diff = {dq_mean:.3e}")
    print(f"  dK      max diff = {dk_max:.3e}, mean diff = {dk_mean:.3e}")
    print(f"  dV      max diff = {dv_max:.3e}, mean diff = {dv_mean:.3e}")


def run_varlen_speed_test(dtype: torch.dtype, device, b=32, h=32, max_tq=1024, max_tk=1024, d=64,
                          warmup=5, steps=20):
    print(f"\n[Varlen Packed Speed] dtype={dtype}, B={b}, H={h}, max_Tq={max_tq}, max_Tk={max_tk}, D={d}, steps={steps}")

    # lengths + base tensors (fp32), then cast
    q0, k0, v0, seqlens_q, seqlens_k, cu_q, cu_k = make_random_varlen_qkv(
        b, h, max_tq, max_tk, d, device
    )

    q = q0.to(dtype)
    k = k0.to(dtype)
    v = v0.to(dtype)

    # Triton warmup
    for _ in range(warmup):
        q_t = q.detach().clone().requires_grad_(True)
        k_t = k.detach().clone().requires_grad_(True)
        v_t = v.detach().clone().requires_grad_(True)
        o = flash_attn_varlen_func(q_t, k_t, v_t, cu_q, cu_k, causal=True)
        loss = o.float().square().mean()
        loss.backward()
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(steps):
        q_t = q.detach().clone().requires_grad_(True)
        k_t = k.detach().clone().requires_grad_(True)
        v_t = v.detach().clone().requires_grad_(True)
        o = flash_attn_varlen_func(q_t, k_t, v_t, cu_q, cu_k, causal=True)
        loss = o.float().square().mean()
        loss.backward()
    torch.cuda.synchronize()
    triton_time = (time.perf_counter() - start) / steps

    # Torch varlen warmup
    for _ in range(warmup):
        q_t = q.detach().clone().requires_grad_(True)
        k_t = k.detach().clone().requires_grad_(True)
        v_t = v.detach().clone().requires_grad_(True)
        o = torch_attn_varlen(q_t, k_t, v_t, cu_q, cu_k, b, h, causal=True)
        loss = o.float().square().mean()
        loss.backward()
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(steps):
        q_t = q.detach().clone().requires_grad_(True)
        k_t = k.detach().clone().requires_grad_(True)
        v_t = v.detach().clone().requires_grad_(True)
        o = torch_attn_varlen(q_t, k_t, v_t, cu_q, cu_k, b, h, causal=True)
        loss = o.float().square().mean()
        loss.backward()
    torch.cuda.synchronize()
    torch_time = (time.perf_counter() - start) / steps

    print(f"  Triton: {triton_time * 1000:.3f} ms / step")
    print(f"  Torch : {torch_time * 1000:.3f} ms / step")
    if torch_time > 0:
        print(f"  Speedup (Torch / Triton) = {torch_time / triton_time:.2f}x")


if __name__ == "__main__":
    assert torch.cuda.is_available()
    device = torch.device("cuda")

    dtypes = [torch.float32, torch.float16]
    if torch.cuda.is_bf16_supported():
        dtypes.append(torch.bfloat16)

    for dt in dtypes:
        run_varlen_accuracy_test(dt, device)
        run_varlen_speed_test(dt, device)
