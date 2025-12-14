import math

import torch
from torch.optim.lr_scheduler import LambdaLR

from nanorlhf.nanotron.core.dp.optim import ZeroOptimizer


def get_optimizer_param_groups(model, weight_decay: float):
    no_decay_ids = set()
    for module in model.modules():
        weight = getattr(module, "weight", None)
        bias = getattr(module, "bias", None)
        if isinstance(weight, torch.Tensor) and weight.dim() == 1:
            no_decay_ids.add(id(weight))
        elif isinstance(bias, torch.Tensor):
            no_decay_ids.add(id(bias))

    decay = []
    no_decay = []
    for param in model.parameters():
        if not param.requires_grad:
            continue
        if id(param) in no_decay_ids:
            no_decay.append(param)
        else:
            decay.append(param)

    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def get_scheduler(config, optimizer, total_steps):
    scheduler_name = config.optim.lr_scheduler
    if scheduler_name is None:
        return None

    if scheduler_name not in ("cosine", "linear"):
        raise ValueError(f"Unsupported lr_scheduler={scheduler_name}. Only 'cosine' and 'linear' are supported.")

    warmup_steps = int(total_steps * float(config.optim.lr_warmup_steps_ratio))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)

        progress = float(step - warmup_steps) / max(1, total_steps - warmup_steps)

        if scheduler_name == "linear":
            return max(0.0, 1.0 - progress)
        else:
            return 0.5 * (1.0 + math.cos(math.pi * progress))

    if isinstance(optimizer, ZeroOptimizer):
        return LambdaLR(optimizer.base, lr_lambda=lr_lambda)
    return LambdaLR(optimizer, lr_lambda=lr_lambda)