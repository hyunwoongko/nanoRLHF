from typing import Dict, List, Any, Optional

import torch
import torch.distributed as dist
from torch import nn

from nanorlhf.nanotron.distributed.mode import ParallelMode
from nanorlhf.nanotron.distributed.mpu import MPU


class ZeroOptimizer:

    def __init__(self, base_optim: torch.optim.Optimizer, mpu: MPU):
        self.base = base_optim
        self.mpu = mpu
        self.group = mpu.get_group(ParallelMode.DATA)
        self.world_size = mpu.get_world_size(ParallelMode.DATA)
        self.rank = mpu.get_local_rank(ParallelMode.DATA)

    @property
    def param_groups(self):
        return self.base.param_groups

    def zero_grad(self, set_to_none: bool = True):
        return self.base.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> Dict[str, Any]:
        return self.base.state_dict()

    def load_state_dict(self, state_dict: Dict[str, Any]):
        return self.base.load_state_dict(state_dict)

    def add_param_group(self, param_group: Dict[str, Any]):
        return self.base.add_param_group(param_group)

    def step(self, closure=None):
        raise NotImplementedError


def _build_owner_map_from_model(model: Optional[nn.Module], world_size: int):
    if model is None:
        return {}
    mapping = {}
    idx = 0
    for p in model.parameters():
        if p is None or not p.requires_grad:
            continue
        mapping[id(p)] = idx % world_size
        idx += 1
    return mapping


class ZeroOptimizerStage1(ZeroOptimizer):

    def __init__(self, base_optim: torch.optim.Optimizer, mpu: MPU, model: Optional[nn.Module] = None):
        super().__init__(base_optim, mpu)
        self._param_owner = _build_owner_map_from_model(model, self.world_size)
        self._owned_param_ids = set()

        if self._param_owner:
            for pg in self.base.param_groups:
                for p in pg["params"]:
                    if p is None:
                        continue
                    owner = self._param_owner.get(id(p), None)
                    if owner is not None and owner == self.rank:
                        self._owned_param_ids.add(id(p))
        else:
            for pg in self.base.param_groups:
                for idx, p in enumerate(pg["params"]):
                    if p is not None and (idx % self.world_size) == self.rank:
                        self._owned_param_ids.add(id(p))

    def _get_owner_group_rank(self, p: nn.Parameter, local_idx: int) -> int:
        if self._param_owner:
            return self._param_owner.get(id(p), local_idx % self.world_size)
        return local_idx % self.world_size

    @torch.no_grad()
    def step(self, closure=None):
        masked_params: List[nn.Parameter] = []
        masked_grads: List[torch.Tensor] = []

        for pg in self.base.param_groups:
            for p in pg["params"]:
                if p is None or p.grad is None:
                    continue
                if id(p) not in self._owned_param_ids:
                    masked_params.append(p)
                    masked_grads.append(p.grad)
                    p.grad = None

        loss = self.base.step(closure=closure) if closure is not None else self.base.step()

        for pg in self.base.param_groups:
            for local_idx, p in enumerate(pg["params"]):
                if p is None:
                    continue
                owner_group_rank = self._get_owner_group_rank(p, local_idx)
                owner_global_rank = dist.get_global_rank(self.group, owner_group_rank)
                dist.broadcast(p.data, src=owner_global_rank, group=self.group)

        for p, g in zip(masked_params, masked_grads):
            p.grad = g

        return loss


class ZeroOptimizerStage2(ZeroOptimizerStage1):

    def __init__(self, base_optim: torch.optim.Optimizer, mpu: MPU, model: Optional[nn.Module] = None):
        super().__init__(base_optim, mpu, model=model)


class ZeroOptimizerStage3(ZeroOptimizer):

    def __init__(
        self,
        base_optim: torch.optim.Optimizer,
        mpu: MPU,
        param_metas,
        total_numel: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        super().__init__(base_optim, mpu)

        self.param_metas = list(param_metas)
        self.total_numel = int(total_numel)
        self.shard_size = (self.total_numel + self.world_size - 1) // self.world_size
        self.shard_start = self.rank * self.shard_size
        self.shard_end = self.shard_start + self.shard_size

        flat_full = torch.zeros(self.total_numel, dtype=dtype, device=device)
        offset = 0
        for meta in self.param_metas:
            p = meta.param
            n = meta.numel
            flat_full[offset : offset + n] = p.data.detach().reshape(-1).to(device=device, dtype=dtype)
            offset += n

        local = torch.zeros(self.shard_size, dtype=dtype, device=device)
        if self.shard_start < self.total_numel:
            end = min(self.shard_end, self.total_numel)
            n = end - self.shard_start
            local[:n] = flat_full[self.shard_start : self.shard_start + n]

        self.flat_param = nn.Parameter(local, requires_grad=True)

        for pg in self.base.param_groups:
            pg["params"] = [self.flat_param]
        self.base.state = {}

        for meta in self.param_metas:
            meta.param.data = torch.empty(0, dtype=dtype, device=device)

    @torch.no_grad()
    def step(self, closure=None):
        return self.base.step(closure=closure) if closure is not None else self.base.step()
