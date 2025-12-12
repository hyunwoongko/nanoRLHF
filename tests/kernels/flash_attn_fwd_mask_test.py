import torch

from nanorlhf.kernels.flash_attn.fwd import flash_attn_fwd


def flash_attn_eager(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attention_mask: torch.Tensor = None,
    causal: bool = False,
    softmax_scale: float = None,
):
    assert q.ndim == 4 and k.ndim == 4 and v.ndim == 4
    B, Hq, Lq, D = q.shape
    Bk, Hk, Lk, Dk = k.shape
    Bv, Hv, Lv, Dv = v.shape
    assert B == Bk == Bv
    assert Lk == Lv and D == Dk == Dv
    assert Hq == Hk == Hv

    if softmax_scale is None:
        softmax_scale = 1.0 / (D ** 0.5)

    q_f = q.to(torch.float32)
    k_f = k.to(torch.float32)
    v_f = v.to(torch.float32)

    scores = torch.matmul(q_f, k_f.transpose(-1, -2)) * softmax_scale

    if attention_mask is not None:
        assert attention_mask.ndim == 4
        assert attention_mask.shape[0] == B
        assert attention_mask.shape[-2] == Lq and attention_mask.shape[-1] == Lk

        if attention_mask.shape[1] == 1:
            attn_mask = attention_mask
        elif attention_mask.shape[1] == Hq:
            attn_mask = attention_mask
        else:
            raise ValueError(
                f"attention_mask.shape[1] must be 1 or H({Hq}), got {attention_mask.shape[1]}"
            )

        scores = scores + attn_mask.to(scores.dtype)

    if causal:
        offset = Lk - Lq
        q_pos = torch.arange(Lq, device=scores.device) + offset
        k_pos = torch.arange(Lk, device=scores.device)
        causal_mask = k_pos.view(1, 1, 1, Lk) > q_pos.view(1, 1, Lq, 1)
        scores = scores.masked_fill(causal_mask, float("-inf"))

    attn = torch.softmax(scores, dim=-1)
    out = torch.matmul(attn, v_f)
    return out.to(q.dtype)


def make_dummy_inputs(
    batch_size=4,
    num_heads=8,
    seq_len_q=8,
    seq_len_k=8,
    dim=128,
    dtype=torch.bfloat16,
    device="cuda",
    use_mask: bool = False,
):
    torch.manual_seed(0)
    device = torch.device(device)

    q = torch.randn(batch_size, num_heads, seq_len_q, dim, device=device, dtype=dtype)
    k = torch.randn(batch_size, num_heads, seq_len_k, dim, device=device, dtype=dtype)
    v = torch.randn(batch_size, num_heads, seq_len_k, dim, device=device, dtype=dtype)

    if not use_mask:
        return q, k, v, None

    minus_inf = torch.finfo(dtype).min
    attention_mask = torch.zeros(
        batch_size, 1, seq_len_q, seq_len_k, device=device, dtype=dtype
    )

    valid_lens = torch.randint(
        low=max(1, seq_len_k // 2),
        high=seq_len_k + 1,
        size=(batch_size,),
        device=device,
    )

    k_pos = torch.arange(seq_len_k, device=device).view(1, 1, 1, seq_len_k)  # [1,1,1,Lk]
    for b in range(batch_size):
        vlen = valid_lens[b].item()
        mask_b = (k_pos >= vlen)
        attention_mask[b:b+1] = attention_mask[b:b+1].masked_fill(mask_b, minus_inf)

    return q, k, v, attention_mask


def align_outputs_for_diff(out_eager, out_triton, batch_size, num_heads):
    print(f"    out_eager.shape  = {tuple(out_eager.shape)}")
    print(f"    out_triton.shape = {tuple(out_triton.shape)}")

    if out_eager.ndim == 4 and out_triton.ndim == 4:
        return out_eager, out_triton

    if out_eager.ndim == 4 and out_triton.ndim == 3:
        B, H, L, D = out_eager.shape
        assert B == batch_size and H == num_heads
        assert out_triton.shape == (B * H, L, D), \
            f"Expected triton ({B*H}, {L}, {D}), got {tuple(out_triton.shape)}"
        out_e = out_eager.view(B * H, L, D)
        return out_e, out_triton

    if out_eager.ndim == 3 and out_triton.ndim == 4:
        B, H, L, D = out_triton.shape
        assert B == batch_size and H == num_heads
        assert out_eager.shape == (B * H, L, D), \
            f"Expected eager ({B*H}, {L}, {D}), got {tuple(out_eager.shape)}"
        out_t = out_triton.view(B * H, L, D)
        return out_eager, out_t

    if out_eager.ndim == 3 and out_triton.ndim == 3:
        if out_eager.shape == out_triton.shape:
            return out_eager, out_triton

    raise RuntimeError(
        f"Cannot align shapes: eager {tuple(out_eager.shape)}, triton {tuple(out_triton.shape)}"
    )


def test_flash_attn_fwd_case(
    batch_size=4,
    num_heads=8,
    seq_len_q=8,
    seq_len_k=8,
    dim=128,
    dtype=torch.bfloat16,
    device="cuda",
    use_mask: bool = False,
    causal: bool = False,
):
    q, k, v, attention_mask = make_dummy_inputs(
        batch_size=batch_size,
        num_heads=num_heads,
        seq_len_q=seq_len_q,
        seq_len_k=seq_len_k,
        dim=dim,
        dtype=dtype,
        device=device,
        use_mask=use_mask,
    )

    out_eager = flash_attn_eager(
        q, k, v,
        attention_mask=attention_mask,
        causal=causal,
        softmax_scale=None,
    )

    out_triton, max_q, ez_sum = flash_attn_fwd(
        q, k, v,
        causal=causal,
        softmax_scale=None,
        attention_mask=attention_mask,
    )

    print(f"[dtype={dtype}, B={batch_size}, H={num_heads}, Lq={seq_len_q}, Lk={seq_len_k}, D={dim}]")
    print(f"  case: attention_mask={use_mask}, causal={causal}")
    out_eager_aligned, out_triton_aligned = align_outputs_for_diff(
        out_eager, out_triton, batch_size, num_heads
    )

    diff = (out_eager_aligned - out_triton_aligned).to(torch.float32)
    max_abs = diff.abs().max().item()

    denom = torch.maximum(
        out_eager_aligned.abs().to(torch.float32),
        out_triton_aligned.abs().to(torch.float32),
    )
    eps = 1e-6
    max_rel = (diff.abs() / torch.clamp(denom, min=eps)).max().item()

    print(f"    max_abs_diff = {max_abs:.6e}")
    print(f"    max_rel_diff = {max_rel:.6e}")
    print()

    return max_abs, max_rel


def test_flash_attn_all_cases_for_config(
    batch_size=4,
    num_heads=8,
    seq_len_q=8,
    seq_len_k=8,
    dim=128,
    dtype=torch.bfloat16,
    device="cuda",
):
    for use_mask in (False, True):
        for causal in (False, True):
            test_flash_attn_fwd_case(
                batch_size=batch_size,
                num_heads=num_heads,
                seq_len_q=seq_len_q,
                seq_len_k=seq_len_k,
                dim=dim,
                dtype=dtype,
                device=device,
                use_mask=use_mask,
                causal=causal,
            )


if __name__ == "__main__":
    configs = [
        dict(batch_size=4, num_heads=8, seq_len_q=8, seq_len_k=8, dim=128),
        dict(batch_size=2, num_heads=4, seq_len_q=16, seq_len_k=16, dim=64),
        dict(batch_size=1, num_heads=8, seq_len_q=32, seq_len_k=32, dim=128),
    ]

    for cfg in configs:
        print("========================================")
        print("Config (bf16):", cfg)
        test_flash_attn_all_cases_for_config(dtype=torch.bfloat16, **cfg)

    for cfg in configs:
        print("========================================")
        print("Config (fp32):", cfg)
        test_flash_attn_all_cases_for_config(dtype=torch.float32, **cfg)