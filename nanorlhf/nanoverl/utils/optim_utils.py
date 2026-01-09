import math

import torch
from torch.optim.lr_scheduler import LambdaLR

from nanorlhf.nanotron.core.dp.optim import ZeroOptimizer


def get_optimizer_param_groups(model, weight_decay: float):
    """
    Create parameter groups for optimizer with and without weight decay.

    Args:
        model: The model whose parameters are to be grouped.
        weight_decay (float): The weight decay value to be applied to applicable parameters.

    Returns:
        A list of parameter groups for the optimizer.
    """
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
    """
    Create a learning rate scheduler based on the configuration.

    Args:
        config: Configuration object containing optimizer settings.
        optimizer: The optimizer for which the scheduler is to be created.
        total_steps: Total number of training steps.

    Returns:
        A learning rate scheduler instance or None if no scheduler is specified.
    """
    scheduler_name = config.optim.lr_scheduler
    if scheduler_name is None:
        return None

    if scheduler_name not in ("cosine", "linear"):
        raise ValueError(f"Unsupported lr_scheduler={scheduler_name}. Only 'cosine' and 'linear' are supported.")

    base_lr = float(config.optim.lr)
    min_lr = float(config.optim.min_lr) if config.optim.min_lr is not None else 0.0
    min_lr_ratio = 0.0 if min_lr <= 0.0 else min(1.0, min_lr / base_lr)
    warmup_steps = int(total_steps * float(config.optim.lr_warmup_steps_ratio))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)

        progress = float(step - warmup_steps) / max(1, total_steps - warmup_steps)

        if scheduler_name == "linear":
            raw = max(0.0, 1.0 - progress)
        else:
            raw = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * raw

    if isinstance(optimizer, ZeroOptimizer):
        return LambdaLR(optimizer.base, lr_lambda=lr_lambda)
    return LambdaLR(optimizer, lr_lambda=lr_lambda)