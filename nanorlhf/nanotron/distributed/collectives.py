from typing import Optional, Tuple, Union

import torch
import torch.distributed as dist
from torch.distributed import ReduceOp

from nanorlhf.nanotron.distributed.mode import ParallelMode
from nanorlhf.nanotron.distributed.mpu import MPU


def _require_divisible(tensor: torch.Tensor, dim: int, world_size: int):
    assert tensor.size(dim) % world_size == 0, "tensor_size must be divisible by world size for tensor parallelism"


class Collectives:
    def __init__(self, mpu: MPU, mode: ParallelMode):
        self.mpu = mpu
        self.mode = mode
        self.world_size = mpu.get_world_size(mode)

    def _maybe_async_return(
        self, output: torch.Tensor, work: Optional[dist.Work], async_op: bool
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, dist.Work]]:
        return (output, work) if async_op else output

    def all_gather(
        self,
        tensor: torch.Tensor,
        dim: int,
        on_cpu: bool = False,
        async_op: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, dist.Work]]:
        if self.world_size == 1:
            return self._maybe_async_return(tensor, None, async_op)

        group = self.mpu.get_cpu_group(self.mode) if on_cpu else self.mpu.get_group(self.mode)
        shape = list(tensor.shape)

        shape[0], shape[dim] = shape[dim], shape[0]
        shape[0] *= self.world_size

        output = torch.empty(shape, dtype=tensor.dtype, device=tensor.device)
        tmp = list(torch.chunk(output, self.world_size, dim=0))
        work = dist.all_gather(
            tensor_list=tmp,
            tensor=tensor.transpose(0, dim).contiguous(),
            group=group,
            async_op=async_op,
        )
        output = output.transpose(0, dim)
        return self._maybe_async_return(output, work, async_op)

    def reduce_scatter(
        self,
        tensor: torch.Tensor,
        dim: int,
        op: ReduceOp = ReduceOp.SUM,
        on_cpu: bool = False,
        async_op: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, dist.Work]]:
        if self.world_size == 1:
            return self._maybe_async_return(tensor, None, async_op)

        group = self.mpu.get_cpu_group(self.mode) if on_cpu else self.mpu.get_group(self.mode)
        _require_divisible(tensor, dim, self.world_size)

        chunks = [c.contiguous() for c in torch.chunk(tensor, self.world_size, dim=dim)]
        output = torch.empty(chunks[0].shape, dtype=tensor.dtype, device=tensor.device)
        work = dist.reduce_scatter(
            output=output,
            input_list=chunks,
            op=op,
            group=group,
            async_op=async_op,
        )
        return self._maybe_async_return(output, work, async_op)

    def all_reduce(
        self,
        tensor: torch.Tensor,
        op: ReduceOp = ReduceOp.SUM,
        on_cpu: bool = False,
        async_op: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, dist.Work]]:
        if self.world_size == 1:
            return self._maybe_async_return(tensor, None, async_op)

        group = self.mpu.get_cpu_group(self.mode) if on_cpu else self.mpu.get_group(self.mode)
        output = tensor.contiguous()
        work = dist.all_reduce(output, op=op, group=group, async_op=async_op)
        return self._maybe_async_return(output, work, async_op)

    def broadcast(
        self,
        tensor: torch.Tensor,
        src: int,
        on_cpu: bool = False,
        async_op: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, dist.Work]]:
        if self.world_size == 1:
            return self._maybe_async_return(tensor, None, async_op)

        group = self.mpu.get_cpu_group(self.mode) if on_cpu else self.mpu.get_group(self.mode)
        output = tensor.contiguous()
        work = dist.broadcast(output, src=src, group=group, async_op=async_op)
        return self._maybe_async_return(output, work, async_op)

    def reduce(
        self,
        tensor: torch.Tensor,
        dst: int,
        op: ReduceOp = ReduceOp.SUM,
        on_cpu: bool = False,
        async_op: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, dist.Work]]:
        if self.world_size == 1:
            return self._maybe_async_return(tensor, None, async_op)

        group = self.mpu.get_cpu_group(self.mode) if on_cpu else self.mpu.get_group(self.mode)
        output = tensor.contiguous()
        work = dist.reduce(output, dst=dst, op=op, group=group, async_op=async_op)
        return self._maybe_async_return(output, work, async_op)

    def scatter(self, tensor: torch.Tensor, dim: int) -> torch.Tensor:
        if self.world_size == 1:
            return tensor

        _require_divisible(tensor, dim, self.world_size)
        rank = self.mpu.get_local_rank(self.mode)
        return torch.chunk(tensor, self.world_size, dim=dim)[rank].contiguous()
