from torch import nn

from nanorlhf.nanotron.core.pp.engine import PipelineParallelWrapper
from nanorlhf.nanotron.core.tp.engine import TensorParallelWrapper
from nanorlhf.nanotron.distributed.mode import ParallelMode
from nanorlhf.nanotron.distributed.mpu import MPU
from nanorlhf.nanotron.utils.wrapping import register_wrapper


def PipelineParallel(model: nn.Module, mpu: MPU, micro_batch_size: int = 1):  # noqa
    wrapper = PipelineParallelWrapper(model, mpu, micro_batch_size=micro_batch_size)
    register_wrapper(
        module=model,
        mode=ParallelMode.PIPELINE,
        wrapper=wrapper,
        mpu=mpu,
    )
    return model


def TensorParallel(model: nn.Module, mpu: MPU):  # noqa
    wrapper = TensorParallelWrapper(model, mpu)
    register_wrapper(
        module=model,
        mode=ParallelMode.TENSOR,
        wrapper=wrapper,
        mpu=mpu,
    )
    return model
