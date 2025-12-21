from typing import Any

import torch

from nanorlhf.nanotron.distributed.collectives import Collectives
from nanorlhf.nanotron.distributed.mode import ParallelMode
from nanorlhf.nanotron.distributed.mpu import MPU


class TPBroadcastFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx: Any, inputs: torch.Tensor, mpu: MPU, mode: ParallelMode):
        ctx.collectives = Collectives(mpu, mode=mode)
        return inputs

    @staticmethod
    def backward(ctx: Any, grad: torch.Tensor):  # noqa
        return ctx.collectives.all_reduce(grad), None, None


class TPAllReduceFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx: Any, inputs: torch.Tensor, mpu: MPU, mode: ParallelMode):
        collectives = Collectives(mpu, mode=mode)
        return collectives.all_reduce(inputs)

    @staticmethod
    def backward(ctx: Any, grad: torch.Tensor):  # noqa
        return grad, None, None


class TPAllGatherFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, inputs: torch.Tensor, dim: int, mpu: MPU, mode: ParallelMode):
        ctx.dim = dim
        ctx.collectives = Collectives(mpu, mode=mode)
        return ctx.collectives.all_gather(inputs, dim=dim)

    @staticmethod
    def backward(ctx: Any, grad: torch.Tensor):  # noqa
        return ctx.collectives.scatter(grad, dim=ctx.dim), None, None, None


class TPScatterFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, inputs: torch.Tensor, dim: int, mpu: MPU, mode: ParallelMode):
        ctx.dim = dim
        ctx.collectives = Collectives(mpu, mode=mode)
        return ctx.collectives.scatter(inputs, dim=dim)

    @staticmethod
    def backward(ctx: Any, grad: torch.Tensor):  # noqa
        return ctx.collectives.all_gather(grad, dim=ctx.dim), None, None, None


def tp_broadcast(inputs: torch.Tensor, mpu: MPU, mode: ParallelMode) -> torch.Tensor:
    return TPBroadcastFunction.apply(inputs, mpu, mode)


def tp_all_reduce(inputs: torch.Tensor, mpu: MPU, mode: ParallelMode) -> torch.Tensor:
    return TPAllReduceFunction.apply(inputs, mpu, mode)


def tp_all_gather(inputs: torch.Tensor, dim: int, mpu: MPU, mode: ParallelMode) -> torch.Tensor:
    return TPAllGatherFunction.apply(inputs, dim, mpu, mode)


def tp_scatter(inputs: torch.Tensor, dim: int, mpu: MPU, mode: ParallelMode) -> torch.Tensor:
    return TPScatterFunction.apply(inputs, dim, mpu, mode)
