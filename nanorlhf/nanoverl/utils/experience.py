from dataclasses import dataclass
from typing import Optional, Union

import torch


@dataclass
class Experience:
    input_ids: torch.Tensor
    position_ids: torch.Tensor
    loss_mask: torch.Tensor

    old_logprobs: torch.Tensor
    ref_logprobs: torch.Tensor
    old_values: torch.Tensor

    rewards: Optional[torch.Tensor] = None
    advantages: Optional[torch.Tensor] = None
    returns: Optional[torch.Tensor] = None

    def to(
        self,
        device: Optional[Union[torch.device, str]] = None,
        non_blocking: bool = True,
        pin_memory: bool = False,
        detach: bool = False,
    ):
        """
        Move all tensor attributes to the specified device.

        Args:
            device (Optional[Union[torch.device, str]]): The target device to move tensors to.
            non_blocking (bool): Whether to use non-blocking transfers.
            pin_memory (bool): Whether to pin memory for CPU tensors.
            detach (bool): Whether to detach tensors from the computation graph.

        Returns:
            Experience: The updated Experience object with tensors moved to the specified device.
        """
        if device is not None and not isinstance(device, torch.device):
            device = torch.device(device)

        for name, value in vars(self).items():
            if not torch.is_tensor(value):
                continue

            t = value
            if detach:
                t = t.detach()
            if device is not None:
                t = t.to(device, non_blocking=non_blocking)
            if pin_memory:
                if t.device.type != "cpu":
                    raise ValueError(f"pin_memory=True requires CPU tensors, but {name} is on {t.device}")
                t = t.pin_memory()
            setattr(self, name, t)

        return self

    def to_dict(self):
        """
        Convert the Experience object to a dictionary, excluding None values.

        Returns:
            dict: A dictionary representation of the Experience object.
        """
        result = {}
        for name, value in vars(self).items():
            if value is None:
                continue
            result[name] = value
        return result
