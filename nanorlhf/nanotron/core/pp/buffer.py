from dataclasses import dataclass, field, fields
from typing import List, Dict, Tuple, Iterator, Any

import torch


@dataclass
class PipelineBuffer:
    """
    A buffer slot for pipeline parallelism.

    Attributes:
        inputs (List[Dict[str, Any]]): The input tensors for the micro-batch.
        outputs (List[Dict[str, Any]]): The output tensors for the micro-batch.
        grads (List[Dict[str, Any]]): The gradient tensors for the micro-batch.
        embeddings (Dict[torch.Tensor, torch.Tensor]): Mapping of tied embedding tensors.
    """
    inputs: List[Dict[str, Any]] = field(default_factory=list)
    outputs: List[Dict[str, Any]] = field(default_factory=list)
    grads: List[Dict[str, Any]] = field(default_factory=list)
    embeddings: Dict[torch.Tensor, torch.Tensor] = field(default_factory=dict)

    def keys(self) -> Iterator[str]:
        """Dict-like keys iterator."""
        for f in fields(self):
            yield f.name

    def values(self) -> Iterator[Any]:
        """Dict-like values iterator."""
        for f in fields(self):
            yield getattr(self, f.name)

    def items(self) -> Iterator[Tuple[str, Any]]:
        """Dict-like keys and values iterator."""
        for f in fields(self):
            yield f.name, getattr(self, f.name)
