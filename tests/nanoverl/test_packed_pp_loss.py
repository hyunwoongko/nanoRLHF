import argparse
from typing import List

import torch
import torch.distributed as dist
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from nanorlhf.kernels import patch_kernel
from nanorlhf.nanotron import MPU, PipelineParallel
from nanorlhf.nanotron.distributed.mode import ParallelMode
from nanorlhf.nanoverl.dataset.sft_dataset import SFTDataset


def _pp_forward_collect_logits(pp_model, batch_kwargs, concat_dim: int) -> torch.Tensor:
    micro_logits: List[torch.Tensor] = []
    for out in pp_model(**batch_kwargs):
        micro_logits.append(out.logits)

    if not micro_logits:
        raise RuntimeError("No micro outputs produced by PipelineParallel forward.")

    if len(micro_logits) == 1:
        return micro_logits[0]
    return torch.cat(micro_logits, dim=concat_dim)


def compute_causal_lm_loss(logits: torch.Tensor, input_ids: torch.Tensor, loss_mask: torch.Tensor) -> torch.Tensor:
    if logits.dim() != 3:
        raise ValueError(f"logits must be 3D [B,T,V], got {tuple(logits.shape)}")
    if input_ids.dim() != 2:
        raise ValueError(f"input_ids must be 2D [B,T], got {tuple(input_ids.shape)}")
    if loss_mask.dim() != 2:
        raise ValueError(f"loss_mask must be 2D [B,T], got {tuple(loss_mask.shape)}")

    if logits.size(0) != input_ids.size(0) or logits.size(1) != input_ids.size(1):
        raise ValueError(
            f"Shape mismatch: logits {tuple(logits.shape)} vs input_ids {tuple(input_ids.shape)}"
        )
    if loss_mask.size(0) != input_ids.size(0) or loss_mask.size(1) != input_ids.size(1):
        raise ValueError(
            f"Shape mismatch: loss_mask {tuple(loss_mask.shape)} vs input_ids {tuple(input_ids.shape)}"
        )

    shift_logits = logits[:, :-1, :].float()
    shift_labels = input_ids[:, 1:]
    shift_mask = loss_mask[:, 1:].float()

    per_token = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
        reduction="none",
    ).reshape_as(shift_labels)

    loss_sum = (per_token * shift_mask).sum()
    denom = shift_mask.sum().clamp_min_(1.0)
    return loss_sum / denom


def count_valid_loss_tokens(loss_mask: torch.Tensor) -> torch.Tensor:
    return loss_mask[:, 1:].float().sum()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-0.6B-base")
    parser.add_argument("--pp", type=int, default=4)
    parser.add_argument("--micro_batch_size", type=int, default=1)
    parser.add_argument("--num_seqs", type=int, default=16)
    parser.add_argument("--start", type=int, default=0)
    args = parser.parse_args()

    mpu = MPU.from_torch(
        data_parallel_size=1,
        tensor_parallel_size=1,
        pipeline_parallel_size=args.pp,
        backend="nccl",
        seed=42,
    )

    is_rank0 = mpu.is_first_rank(ParallelMode.GLOBAL)
    world = mpu.get_world_size(ParallelMode.GLOBAL)
    device = torch.device("cuda", torch.cuda.current_device())

    data = SFTDataset("../../data/NuminaMath-CoT-Small-Hard-200k/preprocessed/valid.nano")
    samples = data[args.start : args.start + args.num_seqs]
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    model = AutoModelForCausalLM.from_pretrained(args.model_name, dtype=torch.bfloat16)
    model = PipelineParallel(model, mpu, micro_batch_size=args.micro_batch_size)
    model.parallelize()
    model.eval()
    model = patch_kernel(model)

    # packed inputs
    packed_input_ids = torch.tensor([samples["input_ids"]]).to(device)
    packed_position_ids = torch.tensor([samples["position_ids"]]).to(device)
    packed_loss_mask = torch.tensor([samples["loss_mask"]]).to(device)
    packed_cu = torch.tensor(samples["cu_seq_lens_q"]).to(device, dtype=torch.int64)

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    if pad_id is None:
        pad_id = 0

    n = int(packed_cu.numel() - 1)
    seq_input_ids = [packed_input_ids[0, packed_cu[i] : packed_cu[i + 1]] for i in range(n)]
    seq_loss_masks = [packed_loss_mask[0, packed_cu[i] : packed_cu[i + 1]] for i in range(n)]
    max_len = max(s.numel() for s in seq_input_ids)

    non_input_ids = torch.full((n, max_len), fill_value=pad_id, dtype=packed_input_ids.dtype, device=device)
    non_attn = torch.zeros((n, max_len), dtype=torch.long, device=device)
    non_loss_mask = torch.zeros((n, max_len), dtype=packed_loss_mask.dtype, device=device)

    for i, (ids, lm) in enumerate(zip(seq_input_ids, seq_loss_masks)):
        L = ids.numel()
        non_input_ids[i, :L] = ids
        non_attn[i, :L] = 1
        non_loss_mask[i, :L] = lm

    with torch.no_grad():
        packed_logits = _pp_forward_collect_logits(
            model,
            {
                "input_ids": packed_input_ids,
                "position_ids": packed_position_ids,
                "attention_mask": None,
                "cu_seq_lens_q": packed_cu,
                "cu_seq_lens_k": packed_cu,
            },
            concat_dim=1,
        )

        last_pos_packed = packed_cu[1:] - 1
        packed_last_logits = packed_logits[0, last_pos_packed, :]  # [n, V]

        normal_logits = _pp_forward_collect_logits(
            model,
            {
                "input_ids": non_input_ids,
                "attention_mask": non_attn,
            },
            concat_dim=0,
        )

        last_pos_normal = non_attn.sum(dim=1) - 1
        assert normal_logits.size(0) == n, (normal_logits.shape, n)
        assert (last_pos_normal >= 0).all()
        assert (last_pos_normal < normal_logits.size(1)).all()
        normal_last_logits = normal_logits[torch.arange(n, device=device), last_pos_normal, :]

        diff_last = (packed_last_logits - normal_last_logits).abs()
        max_abs_last = diff_last.max().to(torch.float32)
        mean_abs_last = diff_last.mean().to(torch.float32)

        # (2) loss compare with loss_mask 
        packed_loss = compute_causal_lm_loss(packed_logits, packed_input_ids, packed_loss_mask).to(torch.float32)
        normal_loss = compute_causal_lm_loss(normal_logits, non_input_ids, non_loss_mask).to(torch.float32)
        loss_diff = (packed_loss - normal_loss).abs()

        packed_denom = count_valid_loss_tokens(packed_loss_mask).to(torch.float32)
        normal_denom = count_valid_loss_tokens(non_loss_mask).to(torch.float32)

    # gather to rank0
    stats = torch.stack(
        [
            max_abs_last,
            mean_abs_last,
            packed_loss,
            normal_loss,
            loss_diff,
            packed_denom,
            normal_denom,
        ]
    )  # [7]

    gather_stats = [torch.empty_like(stats) for _ in range(world)] if is_rank0 else None
    dist.gather(stats, gather_list=gather_stats, dst=0)

    if is_rank0:
        print("=== (1) Packed vs Non-packed last-token logits diff (per rank) ===")
        for r, st in enumerate(gather_stats):
            print(f"[rank {r}] max_abs_diff={st[0].item():.6g}, mean_abs_diff={st[1].item():.6g}")

        print("\n=== (2) Packed vs Non-packed masked causal LM loss (per rank) ===")
        for r, st in enumerate(gather_stats):
            print(
                f"[rank {r}] packed_loss={st[2].item():.8f}, normal_loss={st[3].item():.8f}, "
                f"abs_diff={st[4].item():.6g}, denom_packed={st[5].item():.0f}, denom_normal={st[6].item():.0f}"
            )

    dist.barrier()


if __name__ == "__main__":
    main()