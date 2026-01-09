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
    """
    Tag a tensor with its parallel mode and local rank.

    Args:
        t (torch.Tensor): The tensor to tag.
        mode (ParallelMode): The parallel mode to tag the tensor with.
        local_rank (int): The local rank within the parallel group.
    """
    if t is None:
        return
    mapping = dict(getattr(t, "__nanotron_parallel__", {}))
    mapping[mode] = local_rank
    setattr(t, "__nanotron_parallel__", mapping)


def tag_module(module: nn.Module, mode: ParallelMode, local_rank: int):
    """
    Tag all parameters and buffers of a module with its parallel mode and local rank.

    Args:
        module (nn.Module): The module to tag.
        mode (ParallelMode): The parallel mode to tag the module with.
        local_rank (int): The local rank within the parallel group.
    """
    for p in module.parameters(recurse=False):
        tag_param(p, mode, local_rank)
    for b in module.buffers(recurse=False):
        tag_param(b, mode, local_rank)


def tag_modules(modules: List[nn.Module], mode: ParallelMode, local_rank: int):
    """
    Tag a list of modules with their parallel mode and local rank.

    Args:
        modules (List[nn.Module]): The list of modules to tag.
        mode (ParallelMode): The parallel mode to tag the modules with.
        local_rank (int): The local rank within the parallel group.
    """
    for module in modules:
        tag_module(module, mode, local_rank)


def restrict_embedding_resizing(model):
    """
    Restrict resizing of token embeddings after model parallelization.

    Args:
        model (nn.Module): The model to restrict embedding resizing for.
    """

    def resize_token_embeddings(new_num_tokens: Optional[int] = None, **kwargs):
        raise RuntimeError(
            "you can't use ``model.resize_token_embeddings()`` after calling `model.parallelize()`\n"
            "please resize token embedding size before parallelization."
        )

    setattr(model, "__nanotron_resize_token_embeddings__", model.resize_token_embeddings)
    setattr(model, "resize_token_embeddings", partial(resize_token_embeddings, self=model))
    return model


def restore_embedding_resizing(model):
    """
    Restore the original resizing of token embeddings after deparallelization.

    Args:
        model (nn.Module): The model to restore embedding resizing for.
    """

    if hasattr(model, "__nanotron_resize_token_embeddings__"):
        setattr(model, "resize_token_embeddings", model.__nanotron_resize_token_embeddings__)
        delattr(model, "__nanotron_resize_token_embeddings__")
    return model


class ParallelizationWrapper(ABC):
    """
    The abstract class for parallelization wrappers.

    Args:
        model (nn.Module): The model to be parallelized.
        mpu (MPU): The model parallel unit for distributed operations.
        parallelization_priority (int): The priority of this wrapper during parallelization.
    """
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
        """
        Parallelize the model by applying the wrapper and moving parameters/buffers to appropriate devices.
        """
        if hasattr(self.model, "__nanotron_wrappers__"):
            # sorting wrappers to fix parallelization order
            # higher priority wrappers are applied first
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

                if hasattr(wrapper, "convert_tensor_to_micro_loss"):
                    setattr(
                        self.model,
                        "convert_tensor_to_micro_loss",
                        wrapper.convert_tensor_to_micro_loss,
                    )

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
        restrict_embedding_resizing(self.model)

    def deparallelize(self):
        """
        Deparallelize the model by removing wrappers and moving parameters/buffers back to CPU.
        """
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
        restore_embedding_resizing(self.model)

        delattr(self.model, "__nanotron__mp_plan__")
        delattr(self.model, "__nanotron_wrappers__")
        if hasattr(self.model, "convert_tensor_to_micro_loss"):
            delattr(self.model, "convert_tensor_to_micro_loss")


class NoParallelWrapper(ParallelizationWrapper):
    """
    A no-op parallelization wrapper that does not modify the model.

    Args:
        model (nn.Module): The model to be parallelized.
        mpu (MPU): The model parallel unit for distributed operations.
    """
    def __init__(self, model: nn.Module, mpu: MPU):
        super().__init__(model, mpu, parallelization_priority=99)

    def _forward(self, *args, **kwargs):
        return self.model_forward(*args, **kwargs)

    def _parallelize(self):
        pass

    def _deparallelize(self):
        pass


def register_wrapper(module: nn.Module, mode: ParallelMode, wrapper: ParallelizationWrapper):
    """
    Register a parallelization wrapper to a module.

    Args:
        module (nn.Module): The module to register the wrapper to.
        mode (ParallelMode): The parallel mode associated with the wrapper.
        wrapper (ParallelizationWrapper): The parallelization wrapper to register.
    """

    if hasattr(module, "__nanotron_wrappers__"):
        module.__nanotron_wrappers__[mode] = wrapper
    else:
        setattr(module, "__nanotron_wrappers__", {mode: wrapper})

    setattr(module, "__nanotron__mp_plan__", wrapper.mp_plan)
    setattr(module, "parallelize", wrapper.parallelize)
    setattr(module, "deparallelize", wrapper.deparallelize)
