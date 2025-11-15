from abc import ABC

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

        if zero_stage == 3:
            self.params = [p for p in model.parameters()]
        else:
            self.params = [p for p in model.parameters() if p.requires_grad]

        self.pid2pidx = {}
        self.owners = []
        self.meta = []
        self.buckets = [[] for _ in range(self.world_size)]
        for idx, param in enumerate(self.params):
            owner = idx % self.world_size
            self.pid2pidx[id(param)] = idx
            self.owners.append(owner)
            self.buckets[owner].append(idx)

        self._accum_steps = max(int(accum_steps), 1)
        self._accum_counter = 0
        self._fwd_started = False
        self._attach_model_hooks()

    def _model_bwd_post_hook(self):
        pass

    def _attach_model_hooks(self):
        def _on_model_fwd_pre(_m: nn.Module, _in):
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
