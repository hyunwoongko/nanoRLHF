import copy
import os
from abc import ABC, abstractmethod
from collections import OrderedDict
from functools import partial
from typing import List, Union, Optional, Dict, Any, Callable

import torch
from torch import nn

from nanorlhf.nanotron.distributed.mode import ParallelMode
from nanorlhf.nanotron.distributed.mpu import MPU
from nanorlhf.nanotron.utils.checkpoint import save_parallelized, from_parallelized
from nanorlhf.nanotron.utils.tracing import ModelParallelTracer


def tag_param(t: torch.Tensor, mode: ParallelMode, local_rank: int):
    if t is None:
        return
    mapping = dict(getattr(t, "__nanotron_parallel__", {}))
    mapping[mode] = local_rank
    setattr(t, "__nanotron_parallel__", mapping)


def tag_module(module: nn.Module, mode: ParallelMode, local_rank: int):
    for p in module.parameters(recurse=False):
        tag_param(p, mode, local_rank)
    for b in module.buffers(recurse=False):
        tag_param(b, mode, local_rank)


def tag_modules(modules: List[nn.Module], mode: ParallelMode, local_rank: int):
    for module in modules:
        tag_module(module, mode, local_rank)


class ParallelizationWrapper(ABC):
    def __init__(self, model: nn.Module, mpu: MPU, parallelization_priority: int):
        self.mpu = mpu
        self.model = model
        self.model_forward = copy.copy(self.model.forward)
        self.parallelization_priority = parallelization_priority
        self.tracer = ModelParallelTracer(model)

        if hasattr(self.model, "__nanotron__mp_plan__"):
            self.mp_plan = self.model.__nanotron__mp_plan__
        else:
            self.mp_plan = self.tracer.trace()

    @abstractmethod
    def _forward(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def _parallelize(self):
        raise NotImplementedError

    @abstractmethod
    def _deparallelize(self):
        raise NotImplementedError

    def parallelize(self):
        if hasattr(self.model, "__nanotron_wrappers__"):
            self.model.__nanotron_wrappers__ = OrderedDict(
                sorted(
                    self.model.__nanotron_wrappers__.items(),
                    key=lambda item: item[1].parallelization_priority,
                    reverse=True,
                    # (mode, wrapper)
                )
            )
            setattr(self.model, "__nanotron_forward__", self.model_forward)
            for wrapper in self.model.__nanotron_wrappers__.values():
                if hasattr(wrapper, "_parallelize"):
                    wrapper._parallelize()
                    setattr(self.model, "forward", wrapper._forward)

        for parameter in self.model.parameters():
            if hasattr(parameter, "__nanotron_parallel__"):
                # sorting parallel groups to fix parallelization order
                parameter.__nanotron_parallel__ = OrderedDict(
                    sorted(
                        parameter.__nanotron_parallel__.items(),
                        key=lambda item: str(item[0]),
                        reverse=True,
                        # (mode, group)
                    )
                )
                device = self.mpu.ranks2device(parameter.__nanotron_parallel__)
                if device is not None:
                    parameter.data = parameter.to(f"cuda:{device % self.mpu.local_world_size}")
            else:
                parameter.data = parameter.to(torch.cuda.current_device())

        for buffer in self.model.buffers():
            if hasattr(buffer, "__nanotron_parallel__"):
                # sorting parallel groups to fix parallelization order
                buffer.__nanotron_parallel__ = OrderedDict(
                    sorted(
                        buffer.__nanotron_parallel__.items(),
                        key=lambda item: str(item[0]),
                        reverse=True,
                        # (mode, group)
                    )
                )
                device = self.mpu.ranks2device(buffer.__nanotron_parallel__)
                if device is not None:
                    buffer.data = buffer.to(f"cuda:{device % self.mpu.local_world_size}")
            else:
                buffer.data = buffer.to(torch.cuda.current_device())

        def save_parallelized_method(
            save_directory: Union[str, os.PathLike],
            save_config: bool = True,
            state_dict: Optional[Dict[str, Any]] = None,
            save_function: Callable = torch.save,
            merge_checkpoints: bool = False,
        ):
            return save_parallelized(
                self=self.model,
                mpu=self.mpu,
                save_directory=save_directory,
                save_config=save_config,
                state_dict=state_dict,
                save_function=save_function,
                merge_checkpoints=merge_checkpoints,
            )

        def from_parallelized_method(
            load_directory: Union[str, os.PathLike],
            strict: bool = False,
        ):
            return from_parallelized(
                self=self.model,
                mpu=self.mpu,
                load_directory=load_directory,
                strict=strict,
            )

        setattr(self.model, "save_parallelized", save_parallelized_method)
        setattr(self.model, "from_parallelized", from_parallelized_method)

    def deparallelize(self):
        if hasattr(self.model, "__nanotron_wrappers__"):
            self.model.__nanotron_wrappers__ = OrderedDict(
                sorted(
                    self.model.__nanotron_wrappers__.items(),
                    key=lambda item: item[1].parallelization_priority,
                    reverse=False,
                    # (mode, wrapper)
                )
            )
        if hasattr(self.model, "__nanotron_wrappers__"):
            for wrapper in self.model.__nanotron_wrappers__.values():
                if hasattr(wrapper, "_deparallelize"):
                    wrapper._deparallelize()

            if hasattr(self.model, "__nanotron_forward__"):
                setattr(self.model, "forward", self.model.__nanotron_forward__)
                delattr(self.model, "__nanotron_forward__")

        for parameter in self.model.parameters():
            parameter.data = parameter.data.to(torch.device("cpu"))
            if hasattr(parameter, "__nanotron_parallel__"):
                delattr(parameter, "__nanotron_parallel__")

        for buffer in self.model.buffers():
            buffer.data = buffer.data.to(torch.device("cpu"))
            if hasattr(buffer, "__nanotron_parallel__"):
                delattr(buffer, "__nanotron_parallel__")

        delattr(self.model, "save_parallelized")
        delattr(self.model, "from_parallelized")


def register_wrapper(
    module: nn.Module,
    mode: ParallelMode,
    wrapper: ParallelizationWrapper,
    mpu: MPU,
):
    if hasattr(module, "__nanotron_wrappers__"):
        module.__nanotron_wrappers__[mode] = wrapper
    else:
        setattr(module, "__nanotron_wrappers__", {mode: wrapper})

    if not hasattr(module, "__nanotron_mpu__"):
        setattr(module, "__nanotron_mpu__", mpu)

    setattr(module, "__nanotron__mp_plan__", wrapper.mp_plan)
    setattr(module, "parallelize", wrapper.parallelize)
    setattr(module, "deparallelize", wrapper.deparallelize)
