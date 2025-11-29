import time

import torch

from nanorlhf.kernels.api import flash_attn_func


def torch_attn(q, k, v, causal=True, softmax_scale=None):
    b, h, tq, d = q.shape
    tk = k.shape[2]
    if softmax_scale is None:
        softmax_scale = 1.0 / (d**0.5)

    scores = torch.matmul(q, k.transpose(-2, -1)) * softmax_scale

    if causal:
        mask = torch.triu(torch.ones(tq, tk, device=q.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))

    attn = torch.softmax(scores, dim=-1)
    out = torch.matmul(attn, v)
    return out


def max_diff(a: torch.Tensor, b: torch.Tensor):
    diff = (a.float() - b.float()).abs()
    return diff.max().item(), diff.mean().item()


def run_accuracy_test(dtype: torch.dtype, device, b=32, h=32, tq=1024, tk=1024, d=64):
    print(f"\n[Accuracy] dtype={dtype}, B={b}, H={h}, Tq={tq}, Tk={tk}, D={d}")

    q0 = torch.randn(b, h, tq, d, device=device, dtype=torch.float32)
    k0 = torch.randn(b, h, tk, d, device=device, dtype=torch.float32)
    v0 = torch.randn(b, h, tk, d, device=device, dtype=torch.float32)

    # Torch reference
    q_ref = q0.to(dtype).detach().clone().requires_grad_(True)
    k_ref = k0.to(dtype).detach().clone().requires_grad_(True)
    v_ref = v0.to(dtype).detach().clone().requires_grad_(True)

    o_ref = torch_attn(q_ref, k_ref, v_ref, causal=True)
    loss_ref = o_ref.float().square().mean()
    loss_ref.backward()
    dq_ref, dk_ref, dv_ref = q_ref.grad, k_ref.grad, v_ref.grad

    # Triton
    q_tr = q0.to(dtype).detach().clone().requires_grad_(True)
    k_tr = k0.to(dtype).detach().clone().requires_grad_(True)
    v_tr = v0.to(dtype).detach().clone().requires_grad_(True)

    o_tr = flash_attn_func(q_tr, k_tr, v_tr, causal=True)
    loss_tr = o_tr.float().square().mean()
    loss_tr.backward()
    dq_tr, dk_tr, dv_tr = q_tr.grad, k_tr.grad, v_tr.grad

    o_max, o_mean = max_diff(o_ref, o_tr)
    dq_max, dq_mean = max_diff(dq_ref, dq_tr)
    dk_max, dk_mean = max_diff(dk_ref, dk_tr)
    dv_max, dv_mean = max_diff(dv_ref, dv_tr)

    print(f"  output  max diff = {o_max:.3e}, mean diff = {o_mean:.3e}")
    print(f"  dQ      max diff = {dq_max:.3e}, mean diff = {dq_mean:.3e}")
    print(f"  dK      max diff = {dk_max:.3e}, mean diff = {dk_mean:.3e}")
    print(f"  dV      max diff = {dv_max:.3e}, mean diff = {dv_mean:.3e}")


def run_speed_test(dtype: torch.dtype, device, b=32, h=32, tq=1024, tk=1024, d=64, warmup=5, steps=20):
    print(f"\n[Speed] dtype={dtype}, B={b}, H={h}, Tq={tq}, Tk={tk}, D={d}, steps={steps}")

    q = torch.randn(b, h, tq, d, device=device, dtype=dtype)
    k = torch.randn(b, h, tk, d, device=device, dtype=dtype)
    v = torch.randn(b, h, tk, d, device=device, dtype=dtype)

    # Triton warmup
    for _ in range(warmup):
        q_t = q.detach().clone().requires_grad_(True)
        k_t = k.detach().clone().requires_grad_(True)
        v_t = v.detach().clone().requires_grad_(True)
        o = flash_attn_func(q_t, k_t, v_t, causal=True)
        loss = o.float().square().mean()
        loss.backward()
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(steps):
        q_t = q.detach().clone().requires_grad_(True)
        k_t = k.detach().clone().requires_grad_(True)
        v_t = v.detach().clone().requires_grad_(True)
        o = flash_attn_func(q_t, k_t, v_t, causal=True)
        loss = o.float().square().mean()
        loss.backward()
    torch.cuda.synchronize()
    triton_time = (time.perf_counter() - start) / steps

    # Torch warmup
    for _ in range(warmup):
        q_t = q.detach().clone().requires_grad_(True)
        k_t = k.detach().clone().requires_grad_(True)
        v_t = v.detach().clone().requires_grad_(True)
        o = torch_attn(q_t, k_t, v_t, causal=True)
        loss = o.float().square().mean()
        loss.backward()
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(steps):
        q_t = q.detach().clone().requires_grad_(True)
        k_t = k.detach().clone().requires_grad_(True)
        v_t = v.detach().clone().requires_grad_(True)
        o = torch_attn(q_t, k_t, v_t, causal=True)
        loss = o.float().square().mean()
        loss.backward()
    torch.cuda.synchronize()
    torch_time = (time.perf_counter() - start) / steps

    print(f"  Triton: {triton_time*1000:.3f} ms / step")
    print(f"  Torch : {torch_time*1000:.3f} ms / step")
    if torch_time > 0:
        print(f"  Speedup (Torch / Triton) = {torch_time / triton_time:.2f}x")


if __name__ == "__main__":
    assert torch.cuda.is_available()
    device = torch.device("cuda")

    dtypes = [torch.float32, torch.float16]
    if torch.cuda.is_bf16_supported():
        dtypes.append(torch.bfloat16)

    for dt in dtypes:
        run_accuracy_test(dt, device)
        run_speed_test(dt, device)
