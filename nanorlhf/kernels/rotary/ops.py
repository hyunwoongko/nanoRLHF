import torch
import triton
import triton.language as tl


@triton.jit
def apply_rotary_kernel(
    q_ptr,
    k_ptr,
    cos_ptr,
    sin_ptr,
    out_q_ptr,
    out_k_ptr,
    bsz, head_q, head_k, seq_len, dim,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_cb, stride_cs, stride_cd,
    stride_sb, stride_ss, stride_sd,
    stride_oqb, stride_oqh, stride_oqs, stride_oqd,
    stride_okb, stride_okh, stride_oks, stride_okd,
    block_size: tl.constexpr,
):
    pid = tl.program_id(0)

    half = dim // 2
    block_q = bsz * head_q * seq_len
    block_k = bsz * head_k * seq_len

    offs = tl.arange(0, block_size)
    mask = offs < half

    # compute query
    if pid < block_q:
        tmp = pid
        bs = head_q * seq_len
        b = tmp // bs
        tmp = tmp % bs
        h = tmp // seq_len
        s = tmp % seq_len

        base_q = b * stride_qb + h * stride_qh + s * stride_qs
        base_oq = b * stride_oqb + h * stride_oqh + s * stride_oqs

        base_c = b * stride_cb + s * stride_cs
        base_sin = b * stride_sb + s * stride_ss

        q1 = tl.load(q_ptr + base_q + offs * stride_qd, mask=mask, other=0.0)
        c1 = tl.load(cos_ptr + base_c + offs * stride_cd, mask=mask, other=0.0)
        s1 = tl.load(sin_ptr + base_sin + offs * stride_sd, mask=mask, other=0.0)

        q2 = tl.load(q_ptr + base_q + (offs + half) * stride_qd, mask=mask, other=0.0)
        c2 = tl.load(cos_ptr + base_c + (offs + half) * stride_cd, mask=mask, other=0.0)
        s2 = tl.load(sin_ptr + base_sin + (offs + half) * stride_sd, mask=mask, other=0.0)

        q1f = q1.to(tl.float32)
        q2f = q2.to(tl.float32)
        c1f = c1.to(tl.float32)
        s1f = s1.to(tl.float32)
        c2f = c2.to(tl.float32)
        s2f = s2.to(tl.float32)

        q1_new = q1f * c1f - q2f * s1f
        q2_new = q2f * c2f + q1f * s2f

        q1_new = q1_new.to(q1.dtype)
        q2_new = q2_new.to(q2.dtype)

        tl.store(out_q_ptr + base_oq + offs * stride_oqd, q1_new, mask=mask)
        tl.store(out_q_ptr + base_oq + (offs + half) * stride_oqd, q2_new, mask=mask)

    # compute key
    elif pid < block_q + block_k:
        kid = pid - block_q
        tmp = kid
        bs_k = head_k * seq_len
        b = tmp // bs_k
        tmp = tmp % bs_k
        h = tmp // seq_len
        s = tmp % seq_len

        base_k = b * stride_kb + h * stride_kh + s * stride_ks
        base_ok = b * stride_okb + h * stride_okh + s * stride_oks

        base_c = b * stride_cb + s * stride_cs
        base_sin = b * stride_sb + s * stride_ss

        k1 = tl.load(k_ptr + base_k + offs * stride_kd, mask=mask, other=0.0)
        c1 = tl.load(cos_ptr + base_c + offs * stride_cd, mask=mask, other=0.0)
        s1 = tl.load(sin_ptr + base_sin + offs * stride_sd, mask=mask, other=0.0)

        k2 = tl.load(k_ptr + base_k + (offs + half) * stride_kd, mask=mask, other=0.0)
        c2 = tl.load(cos_ptr + base_c + (offs + half) * stride_cd, mask=mask, other=0.0)
        s2 = tl.load(sin_ptr + base_sin + (offs + half) * stride_sd, mask=mask, other=0.0)

        k1f = k1.to(tl.float32)
        k2f = k2.to(tl.float32)
        c1f = c1.to(tl.float32)
        s1f = s1.to(tl.float32)
        c2f = c2.to(tl.float32)
        s2f = s2.to(tl.float32)

        k1_new = k1f * c1f - k2f * s1f
        k2_new = k2f * c2f + k1f * s2f

        k1_new = k1_new.to(k1.dtype)
        k2_new = k2_new.to(k2.dtype)

        tl.store(out_k_ptr + base_ok + offs * stride_okd, k1_new, mask=mask)
        tl.store(out_k_ptr + base_ok + (offs + half) * stride_okd, k2_new, mask=mask)


def apply_rotary_func(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids=None,  # deprecated
    unsqueeze_dim: int = 1,  # keep this for compatibility
):
    assert q.is_cuda and k.is_cuda
    assert q.shape[0] == k.shape[0]
    assert q.shape[2] == k.shape[2]
    assert q.shape[-1] == k.shape[-1]
    assert cos.shape == sin.shape

    bsz, head_q, seq_len, dim = q.shape
    head_k = k.shape[1]
    head_dim = dim
    assert head_dim % 2 == 0

    cos = cos.to(q.dtype)
    sin = sin.to(q.dtype)

    q_c = q.contiguous()
    k_c = k.contiguous()
    cos_c = cos.contiguous()
    sin_c = sin.contiguous()

    out_q = torch.empty_like(q_c)
    out_k = torch.empty_like(k_c)

    stride_qb, stride_qh, stride_qs, stride_qd = q_c.stride()
    stride_kb, stride_kh, stride_ks, stride_kd = k_c.stride()
    stride_cb, stride_cs, stride_cd = cos_c.stride()
    stride_sb, stride_ss, stride_sd = sin_c.stride()
    stride_oqb, stride_oqh, stride_oqs, stride_oqd = out_q.stride()
    stride_okb, stride_okh, stride_oks, stride_okd = out_k.stride()

    block_q = bsz * head_q * seq_len
    block_k = bsz * head_k * seq_len
    total_rows = block_q + block_k

    half = head_dim // 2
    block_size = 1 << (half - 1).bit_length()
    grid = ((total_rows + 1) // 1,)

    apply_rotary_kernel[grid](
        q_c,
        k_c,
        cos_c,
        sin_c,
        out_q,
        out_k,
        bsz, head_q, head_k, seq_len, dim,
        stride_qb, stride_qh, stride_qs, stride_qd,
        stride_kb, stride_kh, stride_ks, stride_kd,
        stride_cb, stride_cs, stride_cd,
        stride_sb, stride_ss, stride_sd,
        stride_oqb, stride_oqh, stride_oqs, stride_oqd,
        stride_okb, stride_okh, stride_oks, stride_okd,
        block_size=block_size,
    )

    return out_q.view_as(q), out_k.view_as(k)
