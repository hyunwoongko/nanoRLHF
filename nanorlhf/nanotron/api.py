from typing import Optional, Union, Tuple

import torch
from torch import nn

from nanorlhf.nanotron.core.dp.engine import DataParallelWrapper
from nanorlhf.nanotron.core.dp.optim import ZeroOptimizer
from nanorlhf.nanotron.core.pp.engine import PipelineParallelWrapper
from nanorlhf.nanotron.core.tp.engine import TensorParallelWrapper
from nanorlhf.nanotron.distributed.mode import ParallelMode
from nanorlhf.nanotron.distributed.mpu import MPU
from nanorlhf.nanotron.utils.wrapping import register_wrapper, NoParallelWrapper


def TensorParallel(  # noqa
    model: nn.Module,
    mpu: MPU,
    is_rollout: bool = False,
) -> nn.Module:
    """
    Register tensor parallel wrapper for the given model.

    Args:
        model (nn.Module): The model to be wrapped.
        mpu (MPU): The model parallel unit.
        is_rollout (bool): Whether to use rollout tensor parallel mode.

    Returns:
        nn.Module: The wrapped model.
    """
    mode = ParallelMode.ROLLOUT_TENSOR if is_rollout else ParallelMode.TENSOR
    if mpu.get_world_size(mode) <= 1:
        wrapper = NoParallelWrapper(model, mpu)
    else:
        wrapper = TensorParallelWrapper(model, mpu, mode)

    register_wrapper(module=model, mode=mode, wrapper=wrapper)
    return model


def PipelineParallel(  # noqa
    model: nn.Module,
    mpu: MPU,
    micro_batch_size: int = 1,
    gradient_checkpointing_enable: bool = False,
) -> nn.Module:
    """
    Register pipeline parallel wrapper for the given model.

    Args:
        model (nn.Module): The model to be wrapped.
        mpu (MPU): The model parallel unit.
        micro_batch_size (int): The micro batch size for pipeline parallelism.
        gradient_checkpointing_enable (bool): Whether to enable gradient checkpointing.

    Returns:
        nn.Module: The wrapped model.
    """
    if mpu.get_world_size(ParallelMode.PIPELINE) <= 1:
        wrapper = NoParallelWrapper(model, mpu)
    else:
        wrapper = PipelineParallelWrapper(
            model,
            mpu,
            micro_batch_size=micro_batch_size,
            gradient_checkpointing_enable=gradient_checkpointing_enable,
        )
    register_wrapper(module=model, mode=ParallelMode.PIPELINE, wrapper=wrapper)
    return model


def DataParallel(  # noqa
    model: nn.Module,
    mpu: MPU,
    optimizer: Optional[torch.optim.Optimizer] = None,
    zero_stage: int = 0,
    accum_steps: int = 1,
) -> Tuple[nn.Module, Optional[Union[torch.optim.Optimizer, ZeroOptimizer]]]:
    """
    Register data parallel wrapper for the given model.

    Args:
        model (nn.Module): The model to be wrapped.
        mpu (MPU): The model parallel unit.
        optimizer (Optional[torch.optim.Optimizer]): The optimizer to be wrapped. Default is None.
        zero_stage (int): The ZeRO optimization stage. Default is 0.
        accum_steps (int): The number of accumulation steps. Default is 1.

    Returns:
        Tuple[nn.Module, Optional[Union[torch.optim.Optimizer, ZeroOptimizer]]]: The wrapped model and optimizer.
    """
    if mpu.get_world_size(ParallelMode.DATA) <= 1:
        wrapper = NoParallelWrapper(model, mpu)
    else:
        wrapper = DataParallelWrapper(
            model,
            mpu,
            zero_stage=zero_stage,
            accum_steps=accum_steps,
        )
        if optimizer is not None:
            optimizer = wrapper.get_zero_optimizer(optimizer)
    register_wrapper(module=model, mode=ParallelMode.DATA, wrapper=wrapper)
    return model, optimizer
