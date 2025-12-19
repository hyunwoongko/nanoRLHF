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
        result = {}
        for name, value in vars(self).items():
            if value is None:
                continue
            result[name] = value
        return result
