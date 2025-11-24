from typing import Any, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from torch import nn

from nanorlhf.nanotron.core.dp.grad import build_zero_grad_reducer, Zero3ParamMeta
from nanorlhf.nanotron.core.dp.optim import (
    ZeroOptimizerStage1,
    ZeroOptimizerStage2,
    ZeroOptimizerStage3,
    ZeroOptimizer,
)
from nanorlhf.nanotron.distributed.mode import ParallelMode
from nanorlhf.nanotron.distributed.mpu import MPU
from nanorlhf.nanotron.utils.wrapping import ParallelizationWrapper, tag_module


class DataParallelWrapper(ParallelizationWrapper):
    def __init__(self, model: nn.Module, mpu: MPU, zero_stage: int = 0, accum_steps: int = 1):
        super().__init__(model, mpu, parallelization_priority=2)
        if zero_stage not in (0, 1, 2, 3):
            raise ValueError(f"Unsupported ZeRO stage: {zero_stage}")
        self.zero_stage = zero_stage
        self.accum_steps = accum_steps
        self._zero3_param_metas: Optional[List[Zero3ParamMeta]] = None
        self._zero3_total_numel: Optional[int] = None
        self._zero3_flat_param: Optional[torch.Tensor] = None
        self._zero3_shard_size: Optional[int] = None
        self._zero3_gather_buffer = None
        self._zero_reducer = None
        self._zero3_hook_handle = None

    def _forward(self, *args, **kwargs) -> Any:
        return self.model_forward(*args, **kwargs)

    def _build_zero3_param_metas(self) -> Tuple[List[Zero3ParamMeta], int]:
        metas: List[Zero3ParamMeta] = []
        total = 0
        for p in self.model.parameters():
            if p is None or not p.requires_grad:
                continue
            n = p.numel()
            metas.append(Zero3ParamMeta(param=p, shape=p.shape, numel=n))
            total += n
        return metas, total

    def get_zero_optimizer(self, optimizer: torch.optim.Optimizer) -> Union[torch.optim.Optimizer, ZeroOptimizer]:
        pp_world_size = self.mpu.get_world_size(ParallelMode.PIPELINE)
        if self.zero_stage >= 2 and pp_world_size > 2:
            raise ValueError(
                f"ZeRO stage {self.zero_stage} is not supported when pipeline parallel size > 2 "
                f"(got pp_world_size={pp_world_size}). Use zero_stage <= 1 in this configuration."
            )
        first_param = next((p for p in self.model.parameters() if p.requires_grad), None)
        if first_param is None:
            raise ValueError("Model has no trainable parameters.")
        if self.zero_stage == 0:
            return optimizer
        if self.zero_stage == 1:
            return ZeroOptimizerStage1(optimizer, self.mpu, model=self.model)
        if self.zero_stage == 2:
            return ZeroOptimizerStage2(optimizer, self.mpu, model=self.model)
        if self.zero_stage == 3:
            metas, total_numel = self._build_zero3_param_metas()
            if not torch.cuda.is_available():
                raise ValueError("ZeRO stage 3 requires CUDA.")
            device = torch.device(torch.cuda.current_device())
            dtype = first_param.dtype
            opt3 = ZeroOptimizerStage3(
                optimizer,
                self.mpu,
                param_metas=metas,
                total_numel=total_numel,
                device=device,
                dtype=dtype,
            )
            self._zero3_param_metas = metas
            self._zero3_total_numel = total_numel
            self._zero3_flat_param = opt3.flat_param
            self._zero3_shard_size = opt3.shard_size
            return opt3
        raise ValueError(f"Unsupported ZeRO stage: {self.zero_stage}")

    def _parallelize(self):
        dp_rank = self.mpu.get_local_rank(ParallelMode.DATA)
        dp_world_size = self.mpu.get_world_size(ParallelMode.DATA)
        tag_module(self.model, ParallelMode.DATA, dp_rank)

        if dp_world_size == 1 and self.zero_stage == 0:
            return

        if self.zero_stage == 3:
            if (
                self._zero3_param_metas is None
                or self._zero3_total_numel is None
                or self._zero3_flat_param is None
                or self._zero3_shard_size is None
            ):
                raise ValueError("ZeRO stage 3 requires optimizer to be created through DataParallel first.")

            metas = self._zero3_param_metas
            total_numel = self._zero3_total_numel
            flat_param = self._zero3_flat_param
            shard_size = self._zero3_shard_size

            reducer = build_zero_grad_reducer(
                self.model,
                self.mpu,
                zero_stage=3,
                accum_steps=self.accum_steps,
                flat_param=flat_param,
                param_metas=metas,
                total_numel=total_numel,
                shard_size=shard_size,
            )

            group = self.mpu.get_group(ParallelMode.DATA)
            padded_len = dp_world_size * shard_size

            if self._zero3_gather_buffer is None:
                self._zero3_gather_buffer = torch.empty(
                    padded_len,
                    device=flat_param.device,
                    dtype=flat_param.dtype,
                )
            gather_buffer = self._zero3_gather_buffer

            def _zero3_fwd_pre(_m, _in):
                local = flat_param.data
                dist.all_gather_into_tensor(gather_buffer, local, group=group)
                flat_full = gather_buffer[:total_numel]
                offset = 0
                for meta in metas:
                    n = meta.numel
                    meta.param.data = flat_full[offset: offset + n].view(meta.shape)
                    offset += n

            self._zero3_hook_handle = self.model.register_forward_pre_hook(_zero3_fwd_pre)
        else:
            reducer = build_zero_grad_reducer(
                self.model,
                self.mpu,
                zero_stage=self.zero_stage,
                accum_steps=self.accum_steps,
            )

        setattr(self.model, "__nanotron_zero_reducer__", reducer)
        self._zero_reducer = reducer

    def _deparallelize(self):
        if self._zero3_hook_handle is not None:
            self._zero3_hook_handle.remove()
            self._zero3_hook_handle = None
        if hasattr(self.model, "__nanotron_zero_reducer__"):
            delattr(self.model, "__nanotron_zero_reducer__")
        if self._zero_reducer is not None and hasattr(self._zero_reducer, "close"):
            try:
                self._zero_reducer.close()
            except Exception:
                pass
        self._zero_reducer = None
        self._zero3_gather_buffer = None

