from typing import Dict, Any, Optional

import torch


def packed_collate_fn_for_sft(batch):
    """
    Collate function to pack a batch of sequences for supervised fine-tuning (SFT).

    Args:
        batch (List[Dict[str, Any]]): A list of samples, each containing 'input_ids' and 'loss_mask'.

    Returns:
        Dict[str, torch.Tensor]: A dictionary containing packed 'input_ids', 'labels', 'position_ids',
                                 'cu_seq_lens_q', and 'cu_seq_lens_k'.
    """
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


def packed_collate_fn_for_rl(batch):
    """
    Collate function to pack a batch of sequences for reinforcement learning (RL).

    Args:
        batch (List[Dict[str, Any]]): A list of samples, each containing 'input_ids' and 'reward_model'.

    Returns:
        Dict[str, torch.Tensor]: A dictionary containing packed 'input_ids', 'position_ids',
                                 'cu_seq_lens_q', 'cu_seq_lens_k', and 'reward_model'.
    """
    input_ids = []
    position_ids = []
    cu_seq_lens = [0]

    for sample in batch:
        length = len(sample["input_ids"])
        input_ids.extend(sample["input_ids"])
        position_ids.extend(range(length))
        cu_seq_lens.append(cu_seq_lens[-1] + length)

    input_ids = torch.tensor([input_ids], dtype=torch.long)
    position_ids = torch.tensor([position_ids], dtype=torch.long)

    return {
        "input_ids": input_ids,
        "position_ids": position_ids,
        "cu_seq_lens_q": torch.tensor(cu_seq_lens, dtype=torch.long),
        "cu_seq_lens_k": torch.tensor(cu_seq_lens, dtype=torch.long),
        "reward_model": [sample["reward_model"] for sample in batch],
    }


def split_packed_batch(
    batch: Dict[str, Any],
    chunk_idx: int,
    num_chunks: int,
    cu_seq_lens: Optional[torch.Tensor] = None,
):
    """
    Split a packed batch into micro-batches for data parallelism and gradient accumulation.

    Args:
        batch (Dict[str, Any]): The packed batch containing various tensors.
        chunk_idx (int): The index of the current chunk (micro-batch).
        num_chunks (int): The total number of chunks to split the batch into.
        cu_seq_lens (Optional[torch.Tensor]): Cumulative sequence lengths tensor.
            If None, it will be computed from position_ids.

    Returns:
        Dict[str, Any]: The local batch corresponding to the specified chunk index.
    """
    if cu_seq_lens is None:
        if "position_ids" not in batch:
            raise KeyError("batch must contain 'position_ids' to split as micro batches")
        position_ids = batch["position_ids"]
        starts = (position_ids[0] == 0).nonzero(as_tuple=False).flatten()
        ends = torch.cat([starts[1:], torch.tensor([position_ids[0].numel()], device=position_ids.device)], dim=0)
        cu_seq_lens = torch.cat([torch.zeros(1, device=position_ids.device, dtype=ends.dtype), ends], dim=0)

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
    for key, value in batch.items():
        if not torch.is_tensor(value):
            if isinstance(value, list) and len(value) == num_seqs:
                local_batch[key] = value[seq_start:seq_end]
            else:
                local_batch[key] = value
            continue

        if key in ("cu_seq_lens_q", "cu_seq_lens_k"):
            local_cu_seq_lens = value[seq_start : seq_end + 1].clone()
            local_cu_seq_lens = local_cu_seq_lens - local_cu_seq_lens[0]
            local_batch[key] = local_cu_seq_lens
            continue

        if value.dim() == 1 and value.numel() == total_tokens:
            local_batch[key] = value[tok_start:tok_end].contiguous()
            continue

        if value.dim() == 2 and value.size(0) == 1 and value.size(1) == total_tokens:
            local_batch[key] = value[:, tok_start:tok_end].contiguous()
            continue

        local_batch[key] = value

    return local_batch


def unpack_sequences(input_ids, position_ids, reward_model_list):
    """
    Unpack a packed sequence into individual sequences.

    Args:
        input_ids (torch.Tensor): The packed input IDs tensor of shape (1, total_length).
        position_ids (torch.Tensor): The packed position IDs tensor of shape (1, total_length).
        reward_model_list (Optional[List[Any]]): A list of reward models corresponding to each sequence.

    Returns:
        List[Dict[str, Any]]: A list of unpacked sequences, each represented as a dictionary.
    """
    assert input_ids.size(0) == 1
    starts = (position_ids[0] == 0).nonzero(as_tuple=False).flatten().tolist()
    ends = starts[1:] + [input_ids.numel()]
    unpacked_sequences = []

    for i, (start, end) in enumerate(zip(starts, ends)):
        if end <= start:
            continue
        reward_model = reward_model_list[i] if reward_model_list is not None else None
        unpacked_input_ids = input_ids[:, start:end]
        unpacked_position_ids = position_ids[:, start:end]
        unpacked_sequences.append(
            {
                "input_ids": unpacked_input_ids,
                "position_ids": unpacked_position_ids,
                "reward_model": reward_model,
            }
        )

    return unpacked_sequences


def repack_sequences(unpacked_sequences):
    """
    Repack individual sequences into a single packed sequence.

    Args:
        unpacked_sequences (List[Dict[str, Any]]): A list of unpacked sequences, each represented as a dictionary.

    Returns:
        Dict[str, torch.Tensor]: A dictionary containing packed 'input_ids', 'position_ids',
                                 and optionally 'loss_mask' and 'reward_model'.
    """
    packed_input_ids = []
    packed_position_ids = []
    packed_loss_mask = []
    packed_reward_model = []

    for unpacked_sequence in unpacked_sequences:
        packed_input_ids.append(unpacked_sequence["input_ids"])
        packed_position_ids.append(unpacked_sequence["position_ids"])
        if "reward_model" in unpacked_sequence:
            packed_reward_model.append(unpacked_sequence["reward_model"])
        if "loss_mask" in unpacked_sequence:
            packed_loss_mask.append(unpacked_sequence["loss_mask"])

    packed_input_ids = torch.cat(packed_input_ids, dim=-1)
    packed_position_ids = torch.cat(packed_position_ids, dim=-1)
    output_dict = {"input_ids": packed_input_ids, "position_ids": packed_position_ids}

    if len(packed_loss_mask) != 0:
        output_dict["loss_mask"] = torch.cat(packed_loss_mask, dim=-1)
    if len(packed_reward_model) != 0:
        output_dict["reward_model"] = packed_reward_model

    return output_dict
