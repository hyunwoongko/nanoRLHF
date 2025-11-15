import torch

from nanorlhf.nanotron.distributed.mode import ParallelMode
from nanorlhf.nanotron.distributed.mpu import MPU


class ZeroOptimizer(torch.optim.Optimizer):
    pass


class ZeroOptimizerStage1(ZeroOptimizer):
    def __init__(self, base_optim: torch.optim.Optimizer, mpu: MPU):
        self.base = base_optim
        self.group = mpu.get_group(ParallelMode.DATA)
        self.world_size = mpu.get_world_size(ParallelMode.DATA)
        self.rank = mpu.get_local_rank(ParallelMode.DATA)

        self._owned_params = []
        self._owned_param_ids = set()

        for pg in self.base.param_groups:
            params = list(pg["params"])
            for idx, p in enumerate(params):
                if p is not None and (idx % self.world_size) == self.rank:
                    self._owned_params.append(p)
                    self._owned_param_ids.add(id(p))

    @property
    def param_groups(self):
        return self.base.param_groups

    @property
    def state(self):
        return self.base.state

    def zero_grad(self, set_to_none: bool = True):
        return self.base.zero_grad(set_to_none=set_to_none)

    def add_param_group(self, param_group):
        return self.base.add_param_group(param_group)

    def step(self, closure=None):
        masked = []
        for pg in self.base.param_groups:
            for p in pg["params"]:
                if p is None or p.grad is None:
                    continue
                if id(p) not in self._owned_param_ids:
                    masked.append(p.grad)
                    p.grad = None

        loss = self.base.step(closure=closure) if closure is not None else self.base.step()

        for pg in self.base.param_groups:
            for idx, p in enumerate(pg["params"]):
                if p is None:
                    continue
                owner = idx % self.world_size
                torch.distributed.broadcast(p.data, src=owner, group=self.group)

        it = iter(masked)
        for pg in self.base.param_groups:
            for p in pg["params"]:
                if p is None:
                    continue
                if p.grad is None and id(p) not in self._owned_param_ids:
                    try:
                        p.grad = next(it)
                    except StopIteration:
                        break

        return loss


class ZeroOptimizerStage3(ZeroOptimizer):
    pass
