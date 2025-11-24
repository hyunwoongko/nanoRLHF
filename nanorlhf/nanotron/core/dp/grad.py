from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Sequence, Any

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

        self._attach_model_hooks()

    @abstractmethod
    def _model_bwd_post_hook(self):
        raise NotImplementedError

    def _attach_model_hooks(self):
        def _on_model_fwd_pre(_m: nn.Module, _in: Sequence[Any]):
            self._fwd_started = True

        def _on_model_bwd_post(_m: nn.Module, _gin, _gout):
            if not self._fwd_started:
                return
            self._fwd_started = False
            self._accum_counter += 1
            if (self._accum_counter % self._accum_steps) == 0:
                self._model_bwd_post_hook()

        self.model.register_forward_pre_hook(_on_model_fwd_pre)
        self.model.register_full_backward_hook(_on_model_bwd_post)


class ZeroGradReducerStage0(ZeroGradReducer):

    def __init__(self, model: nn.Module, mpu: MPU, accum_steps: int = 1):
        super().__init__(model, mpu, zero_stage=0, accum_steps=accum_steps)

    def _model_bwd_post_hook(self):
        if self.world_size == 1:
            return
        for p in self.params:
            if p.grad is None:
                continue
            dist.all_reduce(p.grad, group=self.group)
            p.grad.div_(self.world_size)


class ZeroGradReducerStage1(ZeroGradReducerStage0):

    def __init__(self, model: nn.Module, mpu: MPU, accum_steps: int = 1):
        super().__init__(model, mpu, accum_steps=accum_steps)


class ZeroGradReducerStage2(ZeroGradReducer):

    def __init__(self, model: nn.Module, mpu: MPU, accum_steps: int = 1):
        super().__init__(model, mpu, zero_stage=2, accum_steps=accum_steps)
        self.owners = [idx % self.world_size for idx, _ in enumerate(self.params)]

    def _model_bwd_post_hook(self):
        if self.world_size == 1:
            return

        for idx, p in enumerate(self.params):
            if p.grad is None:
                continue

            owner_group_rank = self.owners[idx]
            owner_global_rank = dist.get_global_rank(self.group, owner_group_rank)
            dist.reduce(p.grad, dst=owner_global_rank, group=self.group)

            if self.rank == owner_group_rank:
                p.grad.div_(self.world_size)
            else:
                p.grad = None


class ZeroGradReducerStage3(ZeroGradReducer):

    def __init__(
        self,
        model: nn.Module,
        mpu: MPU,
        *,
        flat_param: nn.Parameter,
        param_metas,
        total_numel: int,
        shard_size: int,
        accum_steps: int = 1,
    ):
        super().__init__(model, mpu, zero_stage=3, accum_steps=accum_steps)
        self.flat_param = flat_param
        self.param_metas = list(param_metas)
        self.total_numel = int(total_numel)
        self.shard_size = int(shard_size)

    def _model_bwd_post_hook(self):
        device = self.flat_param.device
        dtype = self.flat_param.dtype

        flat_grad = torch.zeros(
            self.total_numel,
            dtype=dtype,
            device=device,
        )
        offset = 0
        for meta in self.param_metas:
            p = meta.param
            n = meta.numel
            if p.grad is not None:
                flat_grad[offset : offset + n] = p.grad.reshape(-1).to(device=device, dtype=dtype)
                p.grad = None
            offset += n

        padded_len = self.world_size * self.shard_size
        assert padded_len >= self.total_numel
        grad_padded = torch.zeros(
            padded_len,
            dtype=dtype,
            device=device,
        )
        grad_padded[: self.total_numel] = flat_grad
        chunks = list(grad_padded.chunk(self.world_size, dim=0))
        recv = torch.zeros(self.shard_size, dtype=dtype, device=device)
        dist.reduce_scatter(recv, chunks, group=self.group)
        recv.div_(self.world_size)
        self.flat_param.grad = recv


def build_zero_grad_reducer(
    model: nn.Module,
    mpu: MPU,
    zero_stage: int,
    accum_steps: int = 1,
    flat_param: nn.Parameter | None = None,
    param_metas=None,
    total_numel: int | None = None,
    shard_size: int | None = None,
) -> ZeroGradReducer:
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
