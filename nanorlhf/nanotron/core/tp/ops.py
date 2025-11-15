from typing import Any

import torch

from nanorlhf.nanotron.distributed.collectives import Collectives
from nanorlhf.nanotron.distributed.mode import ParallelMode
from nanorlhf.nanotron.distributed.mpu import MPU


class TPBroadcastFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx: Any, inputs: torch.Tensor, mpu: MPU):
        ctx.collectives = Collectives(mpu, mode=ParallelMode.TENSOR)
        return inputs

    @staticmethod
    def backward(ctx: Any, grad: torch.Tensor):
        return ctx.collectives.all_reduce(grad), None


class TPAllReduceFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx: Any, inputs: torch.Tensor, mpu: MPU):
        collectives = Collectives(mpu, mode=ParallelMode.TENSOR)
        return collectives.all_reduce(inputs)

    @staticmethod
    def backward(ctx: Any, grad: torch.Tensor):
        return grad, None


class TPAllGatherFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, inputs: torch.Tensor, dim: int, mpu: MPU):
        ctx.dim = dim
        ctx.collectives = Collectives(mpu, mode=ParallelMode.TENSOR)
        return ctx.collectives.all_gather(inputs, dim=dim)

    @staticmethod
    def backward(ctx: Any, grad: torch.Tensor):
        return ctx.collectives.scatter(grad, dim=ctx.dim), None, None


class TPScatterFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, inputs: torch.Tensor, dim: int, mpu: MPU):
        ctx.dim = dim
        ctx.collectives = Collectives(mpu, mode=ParallelMode.TENSOR)
        return ctx.collectives.scatter(inputs, dim=dim)

    @staticmethod
    def backward(ctx: Any, grad: torch.Tensor):
        return ctx.collectives.all_gather(grad, dim=ctx.dim), None, None


def tp_broadcast(inputs: torch.Tensor, mpu: MPU) -> torch.Tensor:
    return TPBroadcastFunction.apply(inputs, mpu)


def tp_all_reduce(inputs: torch.Tensor, mpu: MPU) -> torch.Tensor:
    return TPAllReduceFunction.apply(inputs, mpu)


def tp_all_gather(inputs: torch.Tensor, dim: int, mpu: MPU) -> torch.Tensor:
    return TPAllGatherFunction.apply(inputs, dim, mpu)


def tp_scatter(inputs: torch.Tensor, dim: int, mpu: MPU) -> torch.Tensor:
    return TPScatterFunction.apply(inputs, dim, mpu)
