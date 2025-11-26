from typing import Optional, Union, Tuple

import torch
from torch import nn

from nanorlhf.nanotron.core.dp.engine import DataParallelWrapper
from nanorlhf.nanotron.core.dp.optim import ZeroOptimizer
from nanorlhf.nanotron.core.pp.engine import PipelineParallelWrapper
from nanorlhf.nanotron.core.tp.engine import TensorParallelWrapper
from nanorlhf.nanotron.distributed.mode import ParallelMode
from nanorlhf.nanotron.distributed.mpu import MPU
from nanorlhf.nanotron.utils.wrapping import register_wrapper


def TensorParallel(model: nn.Module, mpu: MPU) -> nn.Module:  # noqa
    if mpu.get_world_size(ParallelMode.TENSOR) == 1:
        return model

    wrapper = TensorParallelWrapper(model, mpu)
    register_wrapper(module=model, mode=ParallelMode.TENSOR, wrapper=wrapper, mpu=mpu)
    return model


def PipelineParallel(model: nn.Module, mpu: MPU, micro_batch_size: int = 1) -> nn.Module:  # noqa
    if mpu.get_world_size(ParallelMode.PIPELINE) == 1:
        return model

    wrapper = PipelineParallelWrapper(model, mpu, micro_batch_size=micro_batch_size)
    register_wrapper(module=model, mode=ParallelMode.PIPELINE, wrapper=wrapper, mpu=mpu)
    return model


def DataParallel(  # noqa
    model: nn.Module,
    mpu: MPU,
    optimizer: Optional[torch.optim.Optimizer] = None,
    zero_stage: int = 0,
    accum_steps: int = 1,
) -> Tuple[nn.Module, Optional[Union[torch.optim.Optimizer, ZeroOptimizer]]]:
    if mpu.get_world_size(ParallelMode.DATA) == 1:
        return model, optimizer

    wrapper = DataParallelWrapper(model, mpu, zero_stage=zero_stage, accum_steps=accum_steps)
    register_wrapper(module=model, mode=ParallelMode.DATA, wrapper=wrapper, mpu=mpu)
    if optimizer is not None:
        optimizer = wrapper.get_zero_optimizer(optimizer)
    return model, optimizer
