import math
from collections.abc import Iterable
from typing import Any, Dict, List

import torch


def partition_layers(num_stages: int, num_layers: int) -> list[int]:
    """
    Partition `num_layers` into `num_stages` contiguous chunks.

    Args:
        num_stages (int): Number of pipeline stages.
        num_layers (int): Total number of layers to partition.

    Returns:
        list[int]: A list of length `num_stages + 1` where each entry indicates
            the starting layer index for each stage.

    Examples:
        >>> partition_layers(4, 24)
        [0, 6, 12, 18, 24]
    """
    partitions = [0] * (num_stages + 1)
    if num_layers <= num_stages:
        for p in range(num_stages + 1):
            partitions[p] = min(p, num_layers)
        return partitions
    chunk_size = math.floor(num_layers / num_stages)
    for p in range(num_stages):
        partitions[p] = min(p * chunk_size, num_layers)
    partitions[num_stages] = num_layers
    return partitions


def get_layer_owner(partition: List[int]) -> List[int]:
    """
    Expand the layer partition into a list mapping each layer index to its owning stage.

    Args:
        partition (List[int]): A list of length `num_stages + 1` indicating the starting
            layer index for each stage.

    Returns:
        List[int]: A list where the index represents the layer index and the value

    Examples:
        >>> get_layer_owner([0, 6, 12, 18, 24])
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3]
    """
    last_part = len(partition) - 1
    ext_partition = []
    for i in range(len(partition)):
        if i != last_part:
            num_layers = partition[i + 1] - partition[i]
            ext_partition += [i] * num_layers
    return ext_partition


def guess_batch_size(batches: Dict[str, Any]) -> int:
    """
    Infer the batch size from input batches.

    Args:
        batches (Dict[str, Any]): The input batches containing tensors.

    Returns:
        int: The inferred batch size.
    """
    for key in ["input_ids", "attention_mask", "labels"]:
        if batches.get(key, None) is not None:
            assert torch.is_tensor(
                batches[key]
            ), f"Expected tensor for key '{key}', but got {type(batches[key])}."
            return batches[key].size(0)

    types = {k: type(v).__qualname__ for k, v in batches.items()}
    raise ValueError(
        "At least, you must input one of `input_ids`, `attention_mask`, or `labels` as a tensor. "
        f"Please check your input batches: {types}"
    )


def zero_grads(inputs):
    """
    Recursively zero out gradients in the given inputs.

    Args:
        inputs (Any): The input structure containing tensors.

    Returns:
        Any: The input structure with gradients zeroed out.
    """

    def zero_grad(x):
        if torch.is_tensor(x):
            if x.is_leaf and x.grad is not None:
                x.grad.data.zero_()
        elif isinstance(x, dict):
            for v in x.values():
                zero_grad(v)
        elif isinstance(x, Iterable):
            for v in x:
                zero_grad(v)

    zero_grad(inputs)
    return inputs
