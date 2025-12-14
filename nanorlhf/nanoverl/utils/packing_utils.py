from typing import Dict, Any, Optional

import torch


def packed_collate_fn(batch):
    input_ids = []
    loss_mask = []
    position_ids = []
    cu_seq_lens = [0]

    for sample in batch:
        length = len(sample["input_ids"])
        input_ids.extend(sample["input_ids"])
        loss_mask.extend(sample["loss_mask"])
        position_ids.extend(range(length))
        cu_seq_lens.append(cu_seq_lens[-1] + length)

    input_ids = torch.tensor([input_ids], dtype=torch.long)
    loss_mask = torch.tensor([loss_mask], dtype=torch.long)
    position_ids = torch.tensor([position_ids], dtype=torch.long)

    labels = input_ids.clone()
    # inter sequence tokens must not contribute to the loss
    labels[position_ids == 0] = -100
    # apply the loss mask provided from the dataset
    labels[loss_mask == 0] = -100

    return {
        "input_ids": input_ids,
        "labels": labels,
        "position_ids": position_ids,
        "cu_seq_lens_q": torch.tensor(cu_seq_lens, dtype=torch.long),
        "cu_seq_lens_k": torch.tensor(cu_seq_lens, dtype=torch.long),
    }


def split_packed_batch(
    batch: Dict[str, Any],
    chunk_idx: int,
    num_chunks: int,
    cu_seq_lens: Optional[torch.Tensor] = None,
):
    if cu_seq_lens is None:
        if "position_ids" not in batch:
            raise KeyError("batch must contain 'position_ids' to split as micro batches")
        pos = batch["position_ids"]
        starts = (pos[0] == 0).nonzero(as_tuple=False).flatten()
        ends = torch.cat([starts[1:], torch.tensor([pos[0].numel()], device=pos.device)], dim=0)
        cu_seq_lens = torch.cat([torch.zeros(1, device=pos.device, dtype=ends.dtype), ends], dim=0)

    num_seqs = cu_seq_lens.numel() - 1
    chunk_size = num_seqs // num_chunks
    seq_start = chunk_idx * chunk_size
    seq_end = seq_start + chunk_size

    if seq_start >= num_seqs:
        raise IndexError("chunk_rank out of range")

    tok_start = cu_seq_lens[seq_start].item()
    tok_end = cu_seq_lens[seq_end].item()
    total_tokens = cu_seq_lens[-1].item()

    if not (0 <= tok_start <= tok_end <= total_tokens):
        raise ValueError("Invalid token slice")

    local_batch = {}
    for k, v in batch.items():
        if not torch.is_tensor(v):
            local_batch[k] = v
            continue

        if k in ("cu_seq_lens_q", "cu_seq_lens_k"):
            local = cu_seq_lens[seq_start : seq_end + 1].clone()
            local -= local[0]
            local_batch[k] = local
            continue

        if v.dim() == 2 and v.size(0) == 1 and v.size(1) == total_tokens:
            local_batch[k] = v[:, tok_start:tok_end].contiguous()
            continue

        local_batch[k] = v

    return local_batch
