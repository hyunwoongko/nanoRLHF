from typing import Dict, Any, List

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


def packed_distributed_sampler(
    batch: Dict[str, Any],
    data_parallel_rank: int,
    data_parallel_size: int,
):
    if data_parallel_size == 1:
        return batch

    cu_seq_lens = batch["cu_seq_lens_q"]
    num_seqs = int(cu_seq_lens.numel()) - 1
    if num_seqs <= 0:
        raise ValueError("No sequences found in the packed batch.")

    if num_seqs % data_parallel_size != 0:
        raise ValueError(
            f"Number of sequences {num_seqs} is not divisible by data parallel size {data_parallel_size}."
        )

    local_num_seqs = num_seqs // data_parallel_size
    seq_start = data_parallel_rank * local_num_seqs
    seq_end = seq_start + local_num_seqs

    tok_start = cu_seq_lens[seq_start].item()
    tok_end = cu_seq_lens[seq_end].item()
    total_tokens = cu_seq_lens[-1].item()

    if not (0 <= tok_start <= tok_end <= total_tokens):
        raise ValueError("Invalid token range computed for the data parallel split.")

    local_batch = {}
    for k, v in batch.items():
        if not torch.is_tensor(v):
            local_batch[k] = v
            continue

        if k in ("cu_seq_lens_q", "cu_seq_lens_k"):
            local_cu_seq_lens = v[seq_start : seq_end + 1].clone()
            local_cu_seq_lens = local_cu_seq_lens - local_cu_seq_lens[0]
            local_batch[k] = local_cu_seq_lens
            continue

        if v.dim() == 2 and v.size(0) == 1 and v.size(1) == total_tokens:
            local_batch[k] = v[:, tok_start:tok_end].contiguous()
            continue
        local_batch[k] = v
    return local_batch
