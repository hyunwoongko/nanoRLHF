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
    """
    Data parallel wrapper with ZeRO optimization support.

    Args:
        model (nn.Module): The model to be parallelized.
        mpu (MPU): The model parallel unit.
        zero_stage (int): The ZeRO optimization stage (0, 1, 2, or 3).
        accum_steps (int): The number of accumulation steps for gradient reduction.
    """
    def __init__(self, model: nn.Module, mpu: MPU, zero_stage: int = 0, accum_steps: int = 1):
        super().__init__(model, mpu, parallelization_priority=2)
        if zero_stage not in (0, 1, 2, 3):
            raise ValueError(f"Unsupported ZeRO stage: {zero_stage}")
        self.zero_stage = zero_stage
        self.accum_steps = accum_steps
        self.zero3_param_metas: Optional[List[Zero3ParamMeta]] = None
        self.zero3_total_numel: Optional[int] = None
        self.zero3_flat_param: Optional[torch.Tensor] = None
        self.zero3_shard_size: Optional[int] = None
        self.zero3_gather_buffer = None
        self.zero_reducer = None
        self.zero3_hook_handle = None

    def _forward(self, *args, **kwargs) -> Any:
        return self.model_forward(*args, **kwargs)

    def build_zero3_param_metas(self) -> Tuple[List[Zero3ParamMeta], int]:
        """
        Build ZeRO stage 3 parameter metadata.

        Returns:
            Tuple[List[Zero3ParamMeta], int]:
                A tuple containing the list of parameter metadata and the total number of elements.
        """
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
        """
        Wrap the given optimizer with ZeRO optimization according to the specified stage.

        Args:
            optimizer (torch.optim.Optimizer): The original optimizer.

        Returns:
            Union[torch.optim.Optimizer, ZeroOptimizer]: The wrapped optimizer with ZeRO support.
        """
        pp_world_size = self.mpu.get_world_size(ParallelMode.PIPELINE)
        tp_world_size = self.mpu.get_world_size(ParallelMode.TENSOR)

        if self.zero_stage == 2:
            if pp_world_size > 1:
                raise ValueError(
                    f"ZeRO stage {self.zero_stage} is not supported when pipeline parallel size > 1 "
                    f"(got pp_world_size={pp_world_size}). Use zero_stage <= 1 in this configuration."
                )
        elif self.zero_stage == 3:
            if pp_world_size > 1 or tp_world_size > 1:
                raise ValueError(
                    f"ZeRO stage 3 is not supported when tensor parallel > 1 or pipeline parallel size > 1 "
                    f"(got tp_world_size={tp_world_size}, pp_world_size={pp_world_size}). "
                    f"Use zero_stage <= 1 in this configuration."
                )

        first_param = next((p for p in self.model.parameters() if p.requires_grad), None)
        if first_param is None:
            raise ValueError("Model has no trainable parameters.")
        if self.zero_stage == 0:
            return optimizer
        elif self.zero_stage == 1:
            zero_opt = ZeroOptimizerStage1(optimizer, self.mpu, model=self.model)
        elif self.zero_stage == 2:
            zero_opt = ZeroOptimizerStage2(optimizer, self.mpu, model=self.model)
        elif self.zero_stage == 3:
            metas, total_numel = self.build_zero3_param_metas()
            if not torch.cuda.is_available():
                raise ValueError("ZeRO stage 3 requires CUDA.")
            device = torch.device(torch.cuda.current_device())
            dtype = first_param.dtype
            zero_opt = ZeroOptimizerStage3(
                optimizer,
                self.mpu,
                param_metas=metas,
                total_numel=total_numel,
                device=device,
                dtype=dtype,
            )
            self.zero3_param_metas = metas
            self.zero3_total_numel = total_numel
            self.zero3_flat_param = zero_opt.flat_param
            self.zero3_shard_size = zero_opt.shard_size
        else:
            raise ValueError(f"Unsupported ZeRO stage: {self.zero_stage}")

        setattr(self.model, "__nanotron_zero_optimizer__", zero_opt)
        return zero_opt

    def _parallelize(self):
        """
        Parallelize the model with ZeRO optimization support.
        """
        dp_rank = self.mpu.get_local_rank(ParallelMode.DATA)
        dp_world_size = self.mpu.get_world_size(ParallelMode.DATA)
        tag_module(self.model, ParallelMode.DATA, dp_rank)
        if dp_world_size == 1 and self.zero_stage == 0:
            return
        if self.zero_stage == 3:
            if (
                self.zero3_param_metas is None
                or self.zero3_total_numel is None
                or self.zero3_flat_param is None
                or self.zero3_shard_size is None
            ):
                raise ValueError("ZeRO stage 3 requires optimizer to be created through DataParallel first.")
            metas = self.zero3_param_metas
            total_numel = self.zero3_total_numel
            flat_param = self.zero3_flat_param
            shard_size = self.zero3_shard_size

            reducer = build_zero_grad_reducer(
                self.model,
                self.mpu,
                zero_stage=3,
                accum_steps=self.accum_steps,
                flat_param=flat_param,  # noqa
                param_metas=metas,
                total_numel=total_numel,
                shard_size=shard_size,
            )

            group = self.mpu.get_group(ParallelMode.DATA)
            dp_world_size = self.mpu.get_world_size(ParallelMode.DATA)
            full_gather_numel = dp_world_size * shard_size
            if self.zero3_gather_buffer is None or self.zero3_gather_buffer.numel() != full_gather_numel:
                self.zero3_gather_buffer = torch.empty(
                    full_gather_numel,
                    device=flat_param.device,
                    dtype=flat_param.dtype,
                )

            gather_buffer = self.zero3_gather_buffer

            def _zero3_fwd_pre(_m, _in):
                local = flat_param.data
                dist.all_gather_into_tensor(gather_buffer, local, group=group)
                flat_full = gather_buffer[:total_numel]
                offset = 0
                for meta in metas:
                    n = meta.numel
                    meta.param.data = flat_full[offset : offset + n].view(meta.shape)
                    offset += n

            self.zero3_hook_handle = self.model.register_forward_pre_hook(_zero3_fwd_pre)
        else:
            reducer = build_zero_grad_reducer(
                self.model,
                self.mpu,
                zero_stage=self.zero_stage,
                accum_steps=self.accum_steps,
            )
        setattr(self.model, "__nanotron_zero_reducer__", reducer)
        self.zero_reducer = reducer

    def _deparallelize(self):
        """
        Deparallelize the model and gather ZeRO stage 3 parameters if applicable.
        """
        if self.zero_stage == 3 and self.zero3_flat_param is not None:
            dp_group = self.mpu.get_group(ParallelMode.DATA)
            dp_world = self.mpu.get_world_size(ParallelMode.DATA)

            flat_local = self.zero3_flat_param.data
            device = flat_local.device
            dtype = flat_local.dtype

            metas = self.zero3_param_metas
            total = self.zero3_total_numel
            shard_size = self.zero3_shard_size

            if metas is None or total is None or shard_size is None:
                raise RuntimeError("ZeRO3 deparallelize: missing metadata")

            if dp_world > 1:
                gather_buf = torch.empty(
                    dp_world * shard_size,
                    device=device,
                    dtype=dtype,
                )
                dist.all_gather_into_tensor(gather_buf, flat_local, group=dp_group)
                flat_full = gather_buf[:total]
            else:
                flat_full = flat_local[:total]

            offset = 0
            for meta in metas:
                n = meta.numel
                chunk = flat_full[offset: offset + n].view(meta.shape)
                meta.param.data = chunk.clone()
                offset += n

            self.zero3_param_metas = None
            self.zero3_total_numel = None
            self.zero3_flat_param = None
            self.zero3_shard_size = None
            self.zero3_gather_buffer = None

        if self.zero3_hook_handle is not None:
            self.zero3_hook_handle.remove()
            self.zero3_hook_handle = None
        if hasattr(self.model, "__nanotron_zero_reducer__"):
            delattr(self.model, "__nanotron_zero_reducer__")
        if hasattr(self.model, "__nanotron_zero_optimizer__"):
            delattr(self.model, "__nanotron_zero_optimizer__")
        self.zero_reducer = None

