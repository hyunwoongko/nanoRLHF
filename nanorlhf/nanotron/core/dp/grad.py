from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Sequence, Any, Optional

import torch
import torch.distributed as dist
from torch import nn

from nanorlhf.nanotron.distributed.mode import ParallelMode
from nanorlhf.nanotron.distributed.mpu import MPU


@dataclass
class Zero3ParamMeta:
    param: nn.Parameter
    shape: torch.Size
    numel: int


class ZeroGradReducer(ABC):
    """
    The abstract class for ZeRO gradient reducer.

    Args:
        model (nn.Module): The model whose gradients are to be reduced.
        mpu (MPU): The model parallel unit.
        zero_stage (int): The ZeRO optimization stage (0, 1, 2, or 3).
        accum_steps (int): The number of accumulation steps for gradient reduction.
    """

    def __init__(self, model: nn.Module, mpu: MPU, zero_stage: int, accum_steps: int = 1):
        assert zero_stage in (0, 1, 2, 3), f"Unsupported ZeRO stage: {zero_stage}"
        self.zero_stage = int(zero_stage)
        self.model = model
        self.group = mpu.get_group(ParallelMode.DATA)
        self.world_size = mpu.get_world_size(ParallelMode.DATA)
        self.rank = mpu.get_local_rank(ParallelMode.DATA)

        self.params: List[nn.Parameter] = [p for p in model.parameters() if p.requires_grad]
        self._accum_steps = max(int(accum_steps), 1)
        self._accum_counter = 0
        self._fwd_started = False

        self.attach_model_hooks()

    @abstractmethod
    def model_bwd_post_hook(self):
        raise NotImplementedError

    def attach_model_hooks(self):
        """
        Attach hooks to the model for gradient reduction after backward pass.
        """
        def on_model_fwd_pre(_m: nn.Module, _in: Sequence[Any]):
            self._fwd_started = True

        def on_model_bwd_post(_m: nn.Module, _gin, _gout):
            if not self._fwd_started:
                return
            self._fwd_started = False
            self._accum_counter += 1
            if (self._accum_counter % self._accum_steps) == 0:
                self.model_bwd_post_hook()

        self.model.register_forward_pre_hook(on_model_fwd_pre)
        self.model.register_full_backward_hook(on_model_bwd_post)


class ZeroGradReducerStage0(ZeroGradReducer):
    """
    ZeRO stage-0 gradient reducer with bucketed all-reduce.
    It is same with DDP.

    Args:
        model (nn.Module): The model whose gradients are to be reduced.
        mpu (MPU): The model parallel unit.
        accum_steps (int): The number of accumulation steps for gradient reduction.
    """

    DEFAULT_BUCKET_SIZE_BYTES = 25 * 1024 * 1024

    def __init__(self, model: nn.Module, mpu: MPU, accum_steps: int = 1):
        super().__init__(model, mpu, zero_stage=0, accum_steps=accum_steps)
        self.bucket_size_bytes = int(self.DEFAULT_BUCKET_SIZE_BYTES)
        self._buckets: List[dict] = []
        self.build_buckets()

    def build_buckets(self):
        if len(self.params) == 0:
            self._buckets = []
            return

        buckets: List[dict] = []

        cur_params: List[nn.Parameter] = []
        cur_offsets: List[int] = []
        cur_numel = 0
        cur_bytes = 0
        cur_dtype = self.params[0].dtype
        cur_device = self.params[0].device

        def flush_bucket():
            nonlocal cur_params, cur_offsets, cur_numel, cur_bytes, cur_dtype, cur_device
            if not cur_params:
                return
            buffer = torch.zeros(cur_numel, dtype=cur_dtype, device=cur_device)
            buckets.append(
                {
                    "params": list(cur_params),
                    "offsets": list(cur_offsets),
                    "buffer": buffer,
                }
            )
            cur_params.clear()
            cur_offsets.clear()
            cur_numel = 0
            cur_bytes = 0

        for param in self.params:
            if param is None or (not param.requires_grad):
                continue

            if param.dtype != cur_dtype or param.device != cur_device:
                flush_bucket()
                cur_dtype = param.dtype
                cur_device = param.device

            p_numel = param.numel()
            p_bytes = p_numel * param.element_size()

            if cur_params and (cur_bytes + p_bytes > self.bucket_size_bytes):
                flush_bucket()

            cur_offsets.append(cur_numel)
            cur_params.append(param)
            cur_numel += p_numel
            cur_bytes += p_bytes

        flush_bucket()
        self._buckets = buckets

    def model_bwd_post_hook(self):
        """
        Perform bucketed all-reduce of gradients after backward pass.
        """
        if self.world_size == 1:
            return
        if not self._buckets:
            return

        for bucket in self._buckets:
            params: List[nn.Parameter] = bucket["params"]
            offsets: List[int] = bucket["offsets"]
            buf: torch.Tensor = bucket["buffer"]

            buf.zero_()
            for p, offset in zip(params, offsets):
                if p.grad is None:
                    continue
                g = p.grad.detach()
                view = g.reshape(-1)
                buf[offset : offset + view.numel()].copy_(view)

            dist.all_reduce(buf, op=dist.ReduceOp.SUM, group=self.group)
            buf.div_(self.world_size)

            for p, offset in zip(params, offsets):
                if p.grad is None:
                    continue
                g = p.grad.detach()
                view = g.reshape(-1)
                view.copy_(buf[offset : offset + view.numel()])


class ZeroGradReducerStage1(ZeroGradReducerStage0):
    """
    ZeRO stage-1 gradient reducer, same as stage-0.
    """

    def __init__(self, model: nn.Module, mpu: MPU, accum_steps: int = 1):
        super().__init__(model, mpu, accum_steps=accum_steps)


class ZeroGradReducerStage2(ZeroGradReducer):
    """
    ZeRO stage-2 gradient reducer with parameter-wise reduction.
    """

    def __init__(self, model: nn.Module, mpu: MPU, accum_steps: int = 1):
        super().__init__(model, mpu, zero_stage=2, accum_steps=accum_steps)
        self.owners: List[int] = [idx % self.world_size for idx, _ in enumerate(self.params)]

    def model_bwd_post_hook(self):
        """
        Perform parameter-wise gradient reduction after backward pass.
        """
        if self.world_size == 1:
            return

        for idx, param in enumerate(self.params):
            if param.grad is None:
                continue

            owner_group_rank = self.owners[idx]
            owner_global_rank = dist.get_global_rank(self.group, owner_group_rank)
            dist.reduce(param.grad, dst=owner_global_rank, group=self.group)

            if self.rank == owner_group_rank:
                param.grad.div_(self.world_size)
            else:
                param.grad = None


class ZeroGradReducerStage3(ZeroGradReducer):
    """
    ZeRO stage-3 gradient reducer with flat parameter shard reduction.

    Args:
        model (nn.Module): The model whose gradients are to be reduced.
        mpu (MPU): The model parallel unit.
        flat_param (nn.Parameter): The flat parameter tensor containing all model parameters.
        param_metas (List[Zero3ParamMeta]): The metadata for each parameter in the model.
        total_numel (int): The total number of elements in all parameters.
        shard_size (int): The size of the parameter shard for each data parallel rank.
        accum_steps (int): The number of accumulation steps for gradient reduction.
    """

    def __init__(
        self,
        model: nn.Module,
        mpu: MPU,
        flat_param: nn.Parameter,
        param_metas: List[Zero3ParamMeta],
        total_numel: int,
        shard_size: int,
        accum_steps: int = 1,
    ):
        super().__init__(model, mpu, zero_stage=3, accum_steps=accum_steps)
        self.flat_param = flat_param
        self.param_metas = list(param_metas)
        self.total_numel = int(total_numel)
        self.shard_size = int(shard_size)

        self._offsets: List[int] = []
        offset = 0
        for meta in self.param_metas:
            self._offsets.append(offset)
            offset += meta.numel
        assert offset == self.total_numel, f"total_numel mismatch: {offset} vs {self.total_numel}"

        device = flat_param.device
        dtype = flat_param.dtype
        self._full_grad_buffer = torch.zeros(
            self.shard_size * self.world_size,
            device=device,
            dtype=dtype,
        )

    def model_bwd_post_hook(self):
        device = self.flat_param.device
        dtype = self.flat_param.dtype

        full_buf = self._full_grad_buffer
        full_buf.zero_()

        for meta, offset in zip(self.param_metas, self._offsets):
            param = meta.param
            if param.grad is None:
                continue
            grad = param.grad.detach().to(device=device, dtype=dtype).reshape(-1)
            assert grad.numel() == meta.numel, "grad numel != meta.numel"
            full_buf[offset : offset + meta.numel].copy_(grad)
            param.grad = None

        if self.flat_param.grad is None or self.flat_param.grad.numel() != self.shard_size:
            shard_grad = torch.zeros(self.shard_size, device=device, dtype=dtype)
            self.flat_param.grad = shard_grad
        else:
            shard_grad = self.flat_param.grad
            shard_grad.zero_()

        if hasattr(dist, "reduce_scatter_tensor"):
            dist.reduce_scatter_tensor(
                shard_grad,
                full_buf,
                op=dist.ReduceOp.SUM,
                group=self.group,
            )
        else:
            input_list = list(full_buf.chunk(self.world_size))
            dist.reduce_scatter(
                shard_grad,
                input_list,
                op=dist.ReduceOp.SUM,
                group=self.group,
            )

        shard_grad.div_(self.world_size)


def build_zero_grad_reducer(
    model: nn.Module,
    mpu: MPU,
    zero_stage: int,
    accum_steps: int = 1,
    flat_param: Optional[nn.Parameter] = None,
    param_metas: Optional[List[Zero3ParamMeta]] = None,
    total_numel: Optional[int] = None,
    shard_size: Optional[int] = None,
) -> ZeroGradReducer:
    """
    Build a ZeRO gradient reducer based on the specified ZeRO stage.

    Args:
        model (nn.Module): The model whose gradients are to be reduced.
        mpu (MPU): The model parallel unit.
        zero_stage (int): The ZeRO optimization stage (0, 1, 2, or 3).
        accum_steps (int): The number of accumulation steps for gradient reduction.
        flat_param (nn.Parameter, optional): The flat parameter tensor for stage-3.
        param_metas (List[Zero3ParamMeta], optional): The metadata for each parameter for stage-3.
        total_numel (int, optional): The total number of elements in all parameters for stage-3.
        shard_size (int, optional): The size of the parameter shard for each data parallel rank for stage-3.

    Returns:
        ZeroGradReducer: The constructed ZeRO gradient reducer.
    """
    if zero_stage == 0:
        return ZeroGradReducerStage0(model, mpu, accum_steps=accum_steps)
    if zero_stage == 1:
        return ZeroGradReducerStage1(model, mpu, accum_steps=accum_steps)
    if zero_stage == 2:
        return ZeroGradReducerStage2(model, mpu, accum_steps=accum_steps)
    if zero_stage == 3:
        if flat_param is None or param_metas is None or total_numel is None or shard_size is None:
            raise ValueError("Stage-3 reducer requires flat_param, param_metas, total_numel, shard_size")
        return ZeroGradReducerStage3(
            model,
            mpu,
            flat_param=flat_param,
            param_metas=param_metas,
            total_numel=total_numel,
            shard_size=shard_size,
            accum_steps=accum_steps,
        )
    raise ValueError(f"Unsupported ZeRO stage: {zero_stage}")
