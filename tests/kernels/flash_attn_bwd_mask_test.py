import math
import torch

from nanorlhf.kernels.flash_attn.bwd import flash_attn_bwd
from nanorlhf.kernels.flash_attn.fwd import flash_attn_fwd

torch.manual_seed(0)


def flash_attn_eager(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    causal: bool = True,
    softmax_scale: float | None = None,
):
    assert q.ndim == 4
    B, H, Lq, D = q.shape
    _, Hk, Lk, Dk = k.shape
    assert Hk == H and Dk == D
    assert v.shape == (B, H, Lk, D)

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(D)

    bh = B * H
    q_ = q.contiguous().view(bh, Lq, D)
    k_ = k.contiguous().view(bh, Lk, D)
    v_ = v.contiguous().view(bh, Lk, D)

    scores = torch.bmm(q_, k_.transpose(1, 2)) * softmax_scale

    if attention_mask is not None:
        assert attention_mask.ndim == 4
        Mb, Mh, Mq, Mk = attention_mask.shape
        assert Mb == B
        assert Mq == Lq or Mk == Lk

        mask_num_heads = Mh
        mask_expanded = torch.empty(
            (B, H, Lq, Lk),
            device=attention_mask.device,
            dtype=attention_mask.dtype,
        )
        for b in range(B):
            for h in range(H):
                src_h = min(h, mask_num_heads - 1)
                mask_expanded[b, h] = attention_mask[b, src_h]

        mask_ = mask_expanded.view(bh, Lq, Lk)
        scores = scores + mask_.to(scores.dtype)

    if causal:
        offset = Lk - Lq
        q_pos = (offset + torch.arange(Lq, device=q.device)).view(1, Lq, 1)
        kv_pos = torch.arange(Lk, device=q.device).view(1, 1, Lk)
        causal_mask = kv_pos > q_pos
        scores = scores.masked_fill(causal_mask, float("-inf"))

    attn = torch.softmax(scores, dim=-1)
    out = torch.bmm(attn, v_)
    out = out.view(B, H, Lq, D)
    return out


def compute_diff_stats(ref: torch.Tensor, test: torch.Tensor):
    ref_f = ref.float()
    test_f = test.float()
    diff = test_f - ref_f
    max_abs = diff.abs().max().item()
    denom = ref_f.abs().clamp_min(1e-6)
    max_rel = (diff.abs() / denom).max().item()
    rms_rel = torch.sqrt(((diff / denom) ** 2).mean()).item()
    return max_abs, max_rel, rms_rel


def print_tensor_stats(name: str, ref: torch.Tensor, test: torch.Tensor):
    max_abs, max_rel, rms_rel = compute_diff_stats(ref, test)
    print(f"    [{name}] max_abs = {max_abs:.6e}, " f"max_rel = {max_rel:.6e}, rms_rel = {rms_rel:.6e}")


def make_attention_mask(bsz, num_heads_mask, Lq, Lk, device, dtype):
    mask = torch.zeros(bsz, num_heads_mask, Lq, Lk, device=device, dtype=dtype)
    tri_mask = torch.triu(torch.ones(Lq, Lk, device=device, dtype=torch.bool), diagonal=1)
    tri_mask = tri_mask.unsqueeze(0).unsqueeze(0)
    mask[tri_mask.expand_as(mask)] = -1e9
    return mask


def test_flash_attn_bwd_case(
    dtype,
    batch_size,
    num_heads,
    seq_len_q,
    seq_len_k,
    dim,
    use_attention_mask: bool,
    causal: bool,
):
    device = "cuda"
    B, H, Lq, Lk, D = batch_size, num_heads, seq_len_q, seq_len_k, dim

    print(f"[dtype={dtype}, B={B}, H={H}, Lq={Lq}, Lk={Lk}, D={D}]")
    print(f"  case: attention_mask={use_attention_mask}, causal={causal}")

    q_base = torch.randn(B, H, Lq, D, device=device, dtype=dtype)
    k_base = torch.randn(B, H, Lk, D, device=device, dtype=dtype)
    v_base = torch.randn(B, H, Lk, D, device=device, dtype=dtype)
    do = torch.randn(B, H, Lq, D, device=device, dtype=dtype)

    if use_attention_mask:
        num_heads_mask = max(1, H // 2)
        attention_mask = make_attention_mask(B, num_heads_mask, Lq, Lk, device, dtype=torch.float32)
    else:
        attention_mask = None

    softmax_scale = 1.0 / math.sqrt(D)

    q_ref = q_base.detach().clone().requires_grad_(True)
    k_ref = k_base.detach().clone().requires_grad_(True)
    v_ref = v_base.detach().clone().requires_grad_(True)

    out_eager = flash_attn_eager(
        q_ref,
        k_ref,
        v_ref,
        attention_mask=attention_mask,
        causal=causal,
        softmax_scale=softmax_scale,
    )
    loss_eager = (out_eager * do).sum()
    loss_eager.backward()

    dq_eager = q_ref.grad.detach()
    dk_eager = k_ref.grad.detach()
    dv_eager = v_ref.grad.detach()

    out_triton, max_q, ez_sum = flash_attn_fwd(
        q_base,
        k_base,
        v_base,
        attention_mask=attention_mask,
        causal=causal,
        softmax_scale=softmax_scale,
    )

    dq_triton, dk_triton, dv_triton = flash_attn_bwd(
        q_base,
        k_base,
        v_base,
        do,
        max_q,
        ez_sum,
        attention_mask=attention_mask,
        causal=causal,
        softmax_scale=softmax_scale,
    )

    out_eager_flat = out_eager.view(B * H, Lq, D)
    out_triton_flat = out_triton.view(B * H, Lq, D)
    print(f"    out_eager.shape  = {tuple(out_eager_flat.shape)}")
    print(f"    out_triton.shape = {tuple(out_triton_flat.shape)}")
    print_tensor_stats("out_fwd", out_eager_flat, out_triton_flat)

    dq_eager_flat = dq_eager.view(B * H, Lq, D)
    dq_triton_flat = dq_triton.view(B * H, Lq, D)
    print_tensor_stats("dq", dq_eager_flat, dq_triton_flat)

    dk_eager_flat = dk_eager.view(B * H, Lk, D)
    dk_triton_flat = dk_triton.view(B * H, Lk, D)
    print_tensor_stats("dk", dk_eager_flat, dk_triton_flat)

    dv_eager_flat = dv_eager.view(B * H, Lk, D)
    dv_triton_flat = dv_triton.view(B * H, Lk, D)
    print_tensor_stats("dv", dv_eager_flat, dv_triton_flat)

    dk_eager_L2 = dk_eager_flat.float().pow(2).sum().sqrt().item()
    dv_eager_L2 = dv_eager_flat.float().pow(2).sum().sqrt().item()
    dk_triton_L2 = dk_triton_flat.float().pow(2).sum().sqrt().item()
    dv_triton_L2 = dv_triton_flat.float().pow(2).sum().sqrt().item()
    dk_eager_max = dk_eager_flat.float().abs().max().item()
    dv_eager_max = dv_eager_flat.float().abs().max().item()
    dk_triton_max = dk_triton_flat.float().abs().max().item()
    dv_triton_max = dv_triton_flat.float().abs().max().item()

    print(f"    [dk_eager] L2 = {dk_eager_L2:.6e}, max = {dk_eager_max:.6e}")
    print(f"    [dk_triton] L2 = {dk_triton_L2:.6e}, max = {dk_triton_max:.6e}")
    print(f"    [dv_eager] L2 = {dv_eager_L2:.6e}, max = {dv_eager_max:.6e}")
    print(f"    [dv_triton] L2 = {dv_triton_L2:.6e}, max = {dv_triton_max:.6e}")
    print()


def main():
    configs = [
        dict(batch_size=4, num_heads=8, seq_len_q=8, seq_len_k=8, dim=64),
        dict(batch_size=2, num_heads=4, seq_len_q=16, seq_len_k=16, dim=64),
        dict(batch_size=1, num_heads=8, seq_len_q=32, seq_len_k=32, dim=64),
    ]

    print("========== Backward REF vs Triton test: bf16 ==========")
    for cfg in configs:
        print("========================================")
        print(f"Config (torch.bfloat16): {cfg}")
        for use_mask in [False, True]:
            for causal in [False, True]:
                test_flash_attn_bwd_case(
                    dtype=torch.bfloat16,
                    use_attention_mask=use_mask,
                    causal=causal,
                    **cfg,
                )

    print("========== Backward REF vs Triton test: fp32 ==========")
    for cfg in configs:
        print("========================================")
        print(f"Config (torch.float32): {cfg}")
        for use_mask in [False, True]:
            for causal in [False, True]:
                test_flash_attn_bwd_case(
                    dtype=torch.float32,
                    use_attention_mask=use_mask,
                    causal=causal,
                    **cfg,
                )


if __name__ == "__main__":
    main()
