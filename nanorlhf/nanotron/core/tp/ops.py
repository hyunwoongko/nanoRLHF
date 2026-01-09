from typing import Any

import torch

from nanorlhf.nanotron.distributed.collectives import Collectives
from nanorlhf.nanotron.distributed.mode import ParallelMode
from nanorlhf.nanotron.distributed.mpu import MPU


class TPBroadcastFunction(torch.autograd.Function):
    """
    Tensor Parallel Broadcast Function with custom forward and backward methods.
    """

    @staticmethod
    def forward(ctx: Any, inputs: torch.Tensor, mpu: MPU, mode: ParallelMode):
        """
        Forward pass for tensor parallel broadcast.

        Args:
            ctx (Any): Context object to save information for backward computation.
            inputs (torch.Tensor): Input tensor to be broadcasted.
            mpu (MPU): Model parallel unit for distributed operations.
            mode (ParallelMode): Parallel mode for communication.
        """
        ctx.collectives = Collectives(mpu, mode=mode)
        return inputs

    @staticmethod
    def backward(ctx: Any, grad: torch.Tensor):  # noqa
        """
        Backward pass for tensor parallel broadcast.

        Args:
            ctx (Any): Context object containing saved information from forward pass.
            grad (torch.Tensor): Gradient tensor from the subsequent layer.
        """
        return ctx.collectives.all_reduce(grad), None, None


class TPAllReduceFunction(torch.autograd.Function):
    """
    Tensor Parallel All-Reduce Function with custom forward and backward methods.
    """

    @staticmethod
    def forward(ctx: Any, inputs: torch.Tensor, mpu: MPU, mode: ParallelMode):
        """
        Forward pass for tensor parallel all-reduce.

        Args:
            ctx (Any): Context object to save information for backward computation.
            inputs (torch.Tensor): Input tensor to be all-reduced.
            mpu (MPU): Model parallel unit for distributed operations.
            mode (ParallelMode): Parallel mode for communication.
        """
        collectives = Collectives(mpu, mode=mode)
        return collectives.all_reduce(inputs)

    @staticmethod
    def backward(ctx: Any, grad: torch.Tensor):  # noqa
        """
        Backward pass for tensor parallel all-reduce.

        Args:
            ctx (Any): Context object containing saved information from forward pass.
            grad (torch.Tensor): Gradient tensor from the subsequent layer.
        """
        return grad, None, None


class TPAllGatherFunction(torch.autograd.Function):
    """
    Tensor Parallel All-Gather Function with custom forward and backward methods.
    """

    @staticmethod
    def forward(ctx: Any, inputs: torch.Tensor, dim: int, mpu: MPU, mode: ParallelMode):
        """
        Forward pass for tensor parallel all-gather.

        Args:
            ctx (Any): Context object to save information for backward computation.
            inputs (torch.Tensor): Input tensor to be all-gathered.
            dim (int): Dimension along which to gather.
            mpu (MPU): Model parallel unit for distributed operations.
            mode (ParallelMode): Parallel mode for communication.
        """
        ctx.dim = dim
        ctx.collectives = Collectives(mpu, mode=mode)
        return ctx.collectives.all_gather(inputs, dim=dim)

    @staticmethod
    def backward(ctx: Any, grad: torch.Tensor):  # noqa
        """
        Backward pass for tensor parallel all-gather.

        Args:
            ctx (Any): Context object containing saved information from forward pass.
            grad (torch.Tensor): Gradient tensor from the subsequent layer.
        """
        return ctx.collectives.scatter(grad, dim=ctx.dim), None, None, None


class TPScatterFunction(torch.autograd.Function):
    """
    Tensor Parallel Scatter Function with custom forward and backward methods.
    """

    @staticmethod
    def forward(ctx: Any, inputs: torch.Tensor, dim: int, mpu: MPU, mode: ParallelMode):
        """
        Forward pass for tensor parallel scatter.

        Args:
            ctx (Any): Context object to save information for backward computation.
            inputs (torch.Tensor): Input tensor to be scattered.
            dim (int): Dimension along which to scatter.
            mpu (MPU): Model parallel unit for distributed operations.
            mode (ParallelMode): Parallel mode for communication.
        """
        ctx.dim = dim
        ctx.collectives = Collectives(mpu, mode=mode)
        return ctx.collectives.scatter(inputs, dim=dim)

    @staticmethod
    def backward(ctx: Any, grad: torch.Tensor):  # noqa
        """
        Backward pass for tensor parallel scatter.

        Args:
            ctx (Any): Context object containing saved information from forward pass.
            grad (torch.Tensor): Gradient tensor from the subsequent layer.
        """
        return ctx.collectives.all_gather(grad, dim=ctx.dim), None, None, None


def tp_broadcast(inputs: torch.Tensor, mpu: MPU, mode: ParallelMode) -> torch.Tensor:
    """
    Tensor Parallel Broadcast operation.

    Args:
        inputs (torch.Tensor): Input tensor to be broadcasted.
        mpu (MPU): Model parallel unit for distributed operations.
        mode (ParallelMode): Parallel mode for communication.

    Returns:
        torch.Tensor: Broadcasted tensor.
    """
    return TPBroadcastFunction.apply(inputs, mpu, mode)


def tp_all_reduce(inputs: torch.Tensor, mpu: MPU, mode: ParallelMode) -> torch.Tensor:
    """
    Tensor Parallel All-Reduce operation.

    Args:
        inputs (torch.Tensor): Input tensor to be all-reduced.
        mpu (MPU): Model parallel unit for distributed operations.
        mode (ParallelMode): Parallel mode for communication.

    Returns:
        torch.Tensor: All-reduced tensor.
    """
    return TPAllReduceFunction.apply(inputs, mpu, mode)


def tp_all_gather(inputs: torch.Tensor, dim: int, mpu: MPU, mode: ParallelMode) -> torch.Tensor:
    """
    Tensor Parallel All-Gather operation.

    Args:
        inputs (torch.Tensor): Input tensor to be all-gathered.
        dim (int): Dimension along which to gather.
        mpu (MPU): Model parallel unit for distributed operations.
        mode (ParallelMode): Parallel mode for communication.

    Returns:
        torch.Tensor: All-gathered tensor.
    """
    return TPAllGatherFunction.apply(inputs, dim, mpu, mode)


def tp_scatter(inputs: torch.Tensor, dim: int, mpu: MPU, mode: ParallelMode) -> torch.Tensor:
    """
    Tensor Parallel Scatter operation.

    Args:
        inputs (torch.Tensor): Input tensor to be scattered.
        dim (int): Dimension along which to scatter.
        mpu (MPU): Model parallel unit for distributed operations.
        mode (ParallelMode): Parallel mode for communication.

    Returns:
        torch.Tensor: Scattered tensor.
    """
    return TPScatterFunction.apply(inputs, dim, mpu, mode)
