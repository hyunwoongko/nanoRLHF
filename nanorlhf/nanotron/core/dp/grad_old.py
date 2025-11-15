from abc import ABC, abstractmethod
from collections import defaultdict
from typing import DefaultDict, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from torch import nn

from nanorlhf.nanotron.distributed.mode import ParallelMode
from nanorlhf.nanotron.distributed.mpu import MPU


class ZeroGradReducer(ABC):

    def __init__(self, model: nn.Module, mpu: MPU, zero_stage: int, accum_steps: int = 1):
        assert zero_stage in (0, 1, 2, 3), f"Unsupported ZeRO stage: {zero_stage}"
        self.model = model
        self.group = mpu.get_group(ParallelMode.DATA)
        self.world_size = mpu.get_world_size(ParallelMode.DATA)
        self.rank = mpu.get_local_rank(ParallelMode.DATA)

        self.zero_stage = zero_stage
        self.owner = lambda i: int(i % self.world_size)

        if zero_stage == 3:
            self.params = [p for p in model.parameters()]
        else:
            self.params = [p for p in model.parameters() if p.requires_grad]

        self.buckets: List[List[int]] = [[] for _ in range(self.world_size)]
        for idx in range(len(self.params)):
            self.buckets[self.owner(idx)].append(idx)

        self._p2idx = {id(p): i for i, p in enumerate(self.params)}
        self._owners = [self.owner(i) for i in range(len(self.params))]
        self._meta = [(tuple(p.shape), p.dtype, p.device) for p in self.params]

        self._accum_steps = max(int(accum_steps), 1)
        self._accum_counter = 0
        self._fwd_started = False

        self._attach_parameter_hooks_for_stage()
        self._attach_runtime_hooks()

    @abstractmethod
    def _attach_parameter_hooks_for_stage(self) -> None:
        ...

    @abstractmethod
    def _attach_runtime_hooks(self) -> None:
        def _on_fwd_pre_trigger(_m: nn.Module, _in):
            self._fwd_started = True

        def _on_bwd_post_trigger(_m: nn.Module, _gin, _gout):
            if not self._fwd_started:
                return
            self._fwd_started = False
            self._accum_counter += 1
            if (self._accum_counter % self._accum_steps) == 0:
                self._finalize_for_stage()

        self.model.register_forward_pre_hook(_on_fwd_pre_trigger)
        self.model.register_full_backward_hook(_on_bwd_post_trigger)

    @abstractmethod
    def _finalize_for_stage(self) -> None:
        ...

    def _dev_dtype(self) -> Tuple[torch.device, torch.dtype]:
        p = next(self.model.parameters())
        return p.device, p.dtype

    def _all_reduce_avg(self, t: torch.Tensor) -> torch.Tensor:
        if self.world_size == 1:
            return t
        dist.all_reduce(t, group=self.group)
        t.div_(self.world_size)
        return t

    def _reduce_scatter_avg(self, flat: torch.Tensor, shard_elems: int) -> torch.Tensor:
        if self.world_size == 1:
            return flat
        out = torch.empty(shard_elems, dtype=flat.dtype, device=flat.device)
        dist.reduce_scatter_tensor(out, flat, op=dist.ReduceOp.SUM, group=self.group)
        out.div_(self.world_size)
        return out

    def _bucket_sizes(self, grads: List[Optional[torch.Tensor]], zeros_if_none: bool) -> List[int]:
        sizes: List[int] = []
        for r in range(self.world_size):
            s = 0
            for idx in self.buckets[r]:
                p = self.params[idx]
                g = grads[idx] if idx < len(grads) else None
                s += g.numel() if g is not None else (p.numel() if zeros_if_none else 0)
            sizes.append(s)
        return sizes

    def _pack_flat_and_local_slices(
        self, grads: List[Optional[torch.Tensor]], max_shard: int
    ) -> Tuple[torch.Tensor, Dict[int, Tuple[int, int]]]:
        device, dtype = self._dev_dtype()
        parts: List[torch.Tensor] = []
        local: Dict[int, Tuple[int, int]] = {}
        for r in range(self.world_size):
            off = 0
            for idx in self.buckets[r]:
                p = self.params[idx]
                g = grads[idx]
                flat = (
                    (g if g is not None else torch.zeros_like(p, device=device, dtype=p.dtype)).contiguous().view(-1)
                )
                if r == self.rank:
                    local[idx] = (off, off + flat.numel())
                parts.append(flat)
                off += flat.numel()
            if off < max_shard:
                parts.append(torch.zeros(max_shard - off, device=device, dtype=dtype))
        flat = torch.cat(parts) if parts else torch.empty(0, device=device, dtype=dtype)
        return flat, local

    def _scatter_back_to_owners(
        self, shard: torch.Tensor, local: Dict[int, Tuple[int, int]], clear_non_owner: bool = True
    ) -> None:
        for idx, p in enumerate(self.params):
            if self.owner(idx) != self.rank:
                if clear_non_owner:
                    p.grad = None
                continue
            ls, le = local[idx]
            p.grad = shard[ls:le].view_as(p).contiguous()


class ZeroGradReducerStage0(ZeroGradReducer):
    def __init__(self, model: nn.Module, mpu: MPU, accum_steps: int = 1):
        super().__init__(model, mpu, zero_stage=0, accum_steps=accum_steps)

    def _attach_parameter_hooks_for_stage(self) -> None:
        for p in self.params:
            p.register_hook(lambda g: self._all_reduce_avg(g))


class ZeroGradReducerStage2(ZeroGradReducerStage0):
    def __init__(self, model: nn.Module, mpu: MPU, accum_steps: int = 1):
        ZeroGradReducer.__init__(self, model, mpu, zero_stage=2, accum_steps=accum_steps)
        self._attach_parameter_hooks_for_stage()

    def _attach_parameter_hooks_for_stage(self) -> None:
        self._captured: List[Optional[torch.Tensor]] = [None] * len(self.params)
        for i, p in enumerate(self.params):
            p.register_hook(lambda g, i=i: self._capture(i, g))

    def _finalize_for_stage(self) -> None:
        sizes = self._bucket_sizes(self._captured, zeros_if_none=True)
        max_shard = max(sizes) if sizes else 0
        if max_shard == 0:
            self._captured = [None] * len(self.params)
            return
        flat, local = self._pack_flat_and_local_slices(self._captured, max_shard)
        shard = self._reduce_scatter_avg(flat, max_shard)
        self._scatter_back_to_owners(shard, local, clear_non_owner=True)
        self._captured = [None] * len(self.params)

    def _capture(self, i: int, g: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        self._captured[i] = g
        return g


class ZeroGradReducerStage3(ZeroGradReducer):
    def __init__(self, model: nn.Module, mpu: MPU, accum_steps: int = 1):
        super().__init__(model, mpu, zero_stage=3, accum_steps=accum_steps)

    def _attach_runtime_hooks(self) -> None:
        mod_params: Dict[nn.Module, List[nn.Parameter]] = {}
        for m in self.model.modules():
            ps = [p for p in m.parameters(recurse=False)]
            if ps:
                mod_params[m] = ps
        refcnt: DefaultDict[int, int] = defaultdict(int)

        for idx, p in enumerate(self.params):
            if self.rank != self._owners[idx]:
                p.data = torch.empty(0, device=p.device, dtype=p.dtype)

        def on_fwd_pre_module(mod: nn.Module, _inputs):
            for p in mod_params.get(mod, []):
                idx = self._p2idx[id(p)]
                owner = self._owners[idx]
                shape, dtype, device = self._meta[idx]
                if self.rank != owner and p.data.numel() == 0:
                    p.data = torch.empty(shape, device=device, dtype=dtype)
                    dist.broadcast(p.data, src=owner, group=self.group)
                elif self.rank == owner and p.data.numel() == 0:
                    p.data = torch.empty(shape, device=device, dtype=dtype)
                refcnt[id(p)] += 1

        def on_bwd_full_module(mod: nn.Module, _gin, _gout):
            for p in mod_params.get(mod, []):
                idx = self._p2idx[id(p)]
                owner = self._owners[idx]
                shape, dtype, device = self._meta[idx]
                g = p.grad
                if g is None:
                    g = torch.zeros(shape, device=device, dtype=dtype)
                dist.reduce(g, dst=owner, op=dist.ReduceOp.SUM, group=self.group)

                if self.rank == owner:
                    p.grad = (g / self.world_size).contiguous()
                else:
                    p.grad = None

                refcnt[id(p)] -= 1
                if self.rank != owner and refcnt[id(p)] == 0:
                    p.data = torch.empty(0, device=device, dtype=dtype)

        for m in mod_params.keys():
            m.register_forward_pre_hook(on_fwd_pre_module)
            m.register_full_backward_hook(on_bwd_full_module)


def make_zero_grad_reducer(model: nn.Module, mpu: MPU, stage: int, accum_steps: int = 1) -> ZeroGradReducer:
    if stage == 0:
        return ZeroGradReducerStage0(model, mpu, accum_steps=accum_steps)
    if stage == 1:
        return ZeroGradReducerStage0(model, mpu, accum_steps=accum_steps)
    if stage == 2:
        return ZeroGradReducerStage2(model, mpu, accum_steps=accum_steps)
    if stage == 3:
        return ZeroGradReducerStage3(model, mpu, accum_steps=accum_steps)
    raise ValueError(f"Unsupported ZeRO stage: {stage}")
