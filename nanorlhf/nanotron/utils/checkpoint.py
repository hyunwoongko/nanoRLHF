import os
from logging import getLogger
from typing import Optional, Union, Callable, Dict, Any

import torch
import torch.distributed as dist
from torch import nn

from nanorlhf.nanotron.distributed.mode import ParallelMode
from nanorlhf.nanotron.distributed.mpu import MPU

logger = getLogger(__name__)


def get_any_param_dtype(model: nn.Module) -> torch.dtype:
    """
    Get the data type of any parameter in the model.

    Args:
        model (nn.Module): The model to inspect.
    """
    for param in model.parameters():
        if param is not None:
            return param.dtype
    return torch.float32


def load_zero3_flat_param(model, state_dict):
    """
    Load ZeRO-3 flat parameter from state_dict into the model.

    Args:
        model (nn.Module): The model to load parameters into.
        state_dict (Dict[str, Any]): The state dictionary containing the flat parameter.
    """
    reducer = getattr(model, "__nanotron_zero_reducer__", None)
    if reducer is None:
        raise RuntimeError("from_parallelized(ZeRO-3): __nanotron_zero_reducer__ not found on model")

    if "flat_param" in state_dict:
        flat_param = reducer.flat_param
        loaded_flat = state_dict["flat_param"].to(
            device=flat_param.device,
            dtype=flat_param.dtype,
        )

        if loaded_flat.numel() != flat_param.numel():
            raise RuntimeError(
                f"from_parallelized(ZeRO-3 shard): size mismatch for flat_param: "
                f"ckpt numel={loaded_flat.numel()} vs local numel={flat_param.numel()}"
            )

        flat_param.data.copy_(loaded_flat)
    else:
        raise RuntimeError(
            "from_parallelized(ZeRO-3): unrecognized checkpoint format, 'flat_param' key not found "
            "for ZeRO-3 checkpoint."
        )


def save_parallelized_with_merge(
    self,
    mpu: MPU,
    save_directory: Union[str, os.PathLike],
    save_config: bool = True,
    state_dict: Optional[Dict[str, Any]] = None,
    save_function: Callable = torch.save,
):
    """
    Save a parallelized model by merging its parameters into a single state dictionary.

    Args:
        self (nn.Module): The model to save.
        mpu (MPU): The model parallel unit for managing process groups.
        save_directory (Union[str, os.PathLike]): The directory to save the model.
        save_config (bool): Whether to save the model configuration.
        state_dict (Optional[Dict[str, Any]]): The state dictionary to save. If None, use the model's state_dict().
        save_function (Callable): The function to use for saving the state dictionary.
    """
    from nanorlhf.nanotron.api import TensorParallel, PipelineParallel, DataParallel

    tp_rank = mpu.get_local_rank(ParallelMode.TENSOR)
    pp_rank = mpu.get_local_rank(ParallelMode.PIPELINE)
    dp_rank = mpu.get_local_rank(ParallelMode.DATA)

    zero_stg = 0
    if hasattr(self, "__nanotron_zero_reducer__"):
        zero_stg = self.__nanotron_zero_reducer__.zero_stage

    model_to_save = self.__class__(self.config).eval()

    if state_dict is None:
        state_dict = self.state_dict()

    if hasattr(self, "__nanotron_wrappers__"):
        for parallel_mode, wrapper in self.__nanotron_wrappers__.items():
            if parallel_mode == ParallelMode.TENSOR:
                model_to_save = TensorParallel(model_to_save, mpu)
            elif parallel_mode == ParallelMode.PIPELINE:
                model_to_save = PipelineParallel(
                    model_to_save,
                    mpu,
                    micro_batch_size=wrapper.micro_batch_size,
                )
            elif parallel_mode == ParallelMode.DATA:
                optimizer = (
                    self.__nanotron_zero_optimizer__.base if hasattr(self, "__nanotron_zero_optimizer__") else None
                )
                model_to_save, _ = DataParallel(
                    model_to_save,
                    mpu,
                    optimizer=optimizer,
                    zero_stage=wrapper.zero_stage,
                    accum_steps=wrapper.accum_steps,
                )

        model_to_save.parallelize()

        if zero_stg != 3:
            model_to_save.load_state_dict(state_dict)
        else:
            reducer = getattr(self, "__nanotron_zero_reducer__", None)
            if reducer is None or not hasattr(reducer, "flat_param"):
                raise RuntimeError("save_parallelized(ZeRO-3): __nanotron_zero_reducer__.flat_param not found")
            flat_param = reducer.flat_param.detach().cpu()
            state_dict_zero3 = {"flat_param": flat_param}
            load_zero3_flat_param(model_to_save, state_dict_zero3)

        model_to_save.deparallelize()
        state_dict_merged = model_to_save.state_dict()
    else:
        raise RuntimeError("save_parallelized_with_merge: __nanotron_wrappers__ not found on model")

    if hasattr(model_to_save, "config"):
        dtype = get_any_param_dtype(model_to_save)
        model_to_save.config.torch_dtype = str(dtype).split(".")[-1]
        model_to_save.config.architectures = [model_to_save.__class__.__name__]
        if save_config and tp_rank == 0 and pp_rank == 0 and dp_rank == 0:
            model_to_save.config.save_pretrained(save_directory)

    if getattr(model_to_save, "_keys_to_ignore_on_save", None) is not None:
        state_dict_merged = {
            k: v for k, v in state_dict_merged.items() if k not in model_to_save._keys_to_ignore_on_save
        }

    weights_name = "pytorch_model.bin"
    output_model_file = os.path.join(save_directory, weights_name)

    if tp_rank == 0 and pp_rank == 0 and dp_rank == 0:
        save_function(state_dict_merged, output_model_file)
        logger.info(f"[merge] Model weights saved in {output_model_file}")


def save_parallelized_without_merge(
    self,
    mpu: MPU,
    save_directory: Union[str, os.PathLike],
    save_config: bool = True,
    state_dict: Optional[Dict[str, Any]] = None,
    save_function: Callable = torch.save,
):
    """
    Save a parallelized model without merging its parameters.

    Args:
        self (nn.Module): The model to save.
        mpu (MPU): The model parallel unit for managing process groups.
        save_directory (Union[str, os.PathLike]): The directory to save the model.
        save_config (bool): Whether to save the model configuration.
        state_dict (Optional[Dict[str, Any]]): The state dictionary to save. If None, use the model's state_dict().
        save_function (Callable): The function to use for saving the state dictionary.
    """
    tp_rank = mpu.get_local_rank(ParallelMode.TENSOR)
    pp_rank = mpu.get_local_rank(ParallelMode.PIPELINE)
    dp_rank = mpu.get_local_rank(ParallelMode.DATA)

    tp_world_size = mpu.get_world_size(ParallelMode.TENSOR)
    pp_world_size = mpu.get_world_size(ParallelMode.PIPELINE)

    zero_stage = 0
    if hasattr(self, "__nanotron_zero_reducer__"):
        zero_stage = self.__nanotron_zero_reducer__.zero_stage

    if hasattr(self, "config"):
        dtype = get_any_param_dtype(self)
        self.config.torch_dtype = str(dtype).split(".")[-1]
        self.config.architectures = [self.__class__.__name__]
        if save_config and tp_rank == 0 and pp_rank == 0 and dp_rank == 0:
            self.config.save_pretrained(save_directory)

    if state_dict is None:
        state_dict = self.state_dict()

    if getattr(self, "_keys_to_ignore_on_save", None) is not None:
        state_dict = {k: v for k, v in state_dict.items() if k not in self._keys_to_ignore_on_save}

    output_model_file = None

    if zero_stage == 3:
        reducer = getattr(self, "__nanotron_zero_reducer__", None)
        if reducer is None or not hasattr(reducer, "flat_param"):
            raise RuntimeError("save_parallelized(ZeRO-3): __nanotron_zero_reducer__.flat_param not found")
        flat_param = reducer.flat_param.detach().cpu()
        shard_state = {"flat_param": flat_param}
        output_model_file = os.path.join(save_directory, f"pytorch_model_zero3_dp{dp_rank}.bin")
        save_function(shard_state, output_model_file)

    elif tp_world_size > 1 or pp_world_size > 1:
        output_model_file = os.path.join(save_directory, f"pytorch_model_tp{tp_rank}_pp{pp_rank}.bin")
        save_function(state_dict, output_model_file)

    elif dp_rank == 0:
        output_model_file = os.path.join(save_directory, "pytorch_model.bin")
        save_function(state_dict, output_model_file)

    if output_model_file is not None:
        logger.info(f"[shard] Model weights saved in {output_model_file}")


def raise_if_rollout_model(self):
    """
    Raise an error if the model is a rollout model.

    Args:
        self (nn.Module): The model to check.
    """
    found_rollout_group = False
    if hasattr(self, "__nanotron_wrappers__"):
        for parallel_mode, wrapper in self.__nanotron_wrappers__.items():
            if parallel_mode in [ParallelMode.ROLLOUT_TENSOR, ParallelMode.ROLLOUT_DATA]:
                found_rollout_group = True
                break

    if found_rollout_group:
        raise RuntimeError(
            "There's no reason to save/load a rollout model checkpoint to/from specific path."
            "Just use the original model path or name. 😅"
        )


def save_parallelized(
    self,
    mpu: MPU,
    save_directory: Union[str, os.PathLike],
    save_config: bool = True,
    state_dict: Optional[Dict[str, Any]] = None,
    save_function: Callable = torch.save,
    merge_checkpoints: bool = False,
):
    """
    Save a parallelized model.

    Args:
        self (nn.Module): The model to save.
        mpu (MPU): The model parallel unit for managing process groups.
        save_directory (Union[str, os.PathLike]): The directory to save the model.
        save_config (bool): Whether to save the model configuration.
        state_dict (Optional[Dict[str, Any]]): The state dictionary to save. If None, use the model's state_dict().
        save_function (Callable): The function to use for saving the state dictionary.
        merge_checkpoints (bool): Whether to merge checkpoints before saving.
    """
    with torch.no_grad():
        if os.path.isfile(save_directory):
            raise ValueError(f"Provided path ({save_directory}) should be a directory, not a file")

        os.makedirs(save_directory, exist_ok=True)
        raise_if_rollout_model(self)

        if merge_checkpoints:
            save_parallelized_with_merge(
                self,
                mpu=mpu,
                save_directory=save_directory,
                save_config=save_config,
                state_dict=state_dict,
                save_function=save_function,
            )
        else:
            save_parallelized_without_merge(
                self,
                mpu=mpu,
                save_directory=save_directory,
                save_config=save_config,
                state_dict=state_dict,
                save_function=save_function,
            )

    if dist.is_initialized():
        dist.barrier(group=mpu.get_group(ParallelMode.TENSOR))
        dist.barrier(group=mpu.get_group(ParallelMode.PIPELINE))
        dist.barrier(group=mpu.get_group(ParallelMode.DATA))


def from_parallelized(
    self,
    mpu: MPU,
    load_directory: Union[str, os.PathLike],
    strict: bool = False,
):
    """
    Load a parallelized model from a checkpoint.

    Args:
        self (nn.Module): The model to load parameters into.
        mpu (MPU): The model parallel unit for managing process groups.
        load_directory (Union[str, os.PathLike]): The directory to load the model from.
        strict (bool): Whether to strictly enforce that the keys in state_dict match the model's keys.
    """
    with torch.no_grad():
        if not os.path.isdir(load_directory):
            raise NotADirectoryError(f"directory named {load_directory} is not valid.")
        raise_if_rollout_model(self)

        tp_rank = mpu.get_local_rank(ParallelMode.TENSOR)
        pp_rank = mpu.get_local_rank(ParallelMode.PIPELINE)
        dp_rank = mpu.get_local_rank(ParallelMode.DATA)

        tp_world_size = mpu.get_world_size(ParallelMode.TENSOR)
        pp_world_size = mpu.get_world_size(ParallelMode.PIPELINE)
        dp_world_size = mpu.get_world_size(ParallelMode.DATA)

        zero_stage = 0
        if hasattr(self, "__nanotron_zero_reducer__"):
            zero_stage = getattr(self.__nanotron_zero_reducer__, "zero_stage", 0)

        if tp_world_size > 1 or pp_world_size > 1:
            weights_name = f"pytorch_model_tp{tp_rank}_pp{pp_rank}.bin"
        elif dp_world_size > 1 and zero_stage == 3:
            weights_name = f"pytorch_model_zero3_dp{dp_rank}.bin"
        else:
            weights_name = "pytorch_model.bin"

        ckpt_path = os.path.join(load_directory, weights_name)

        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")

        state_dict = torch.load(ckpt_path, map_location="cpu")

        if getattr(self, "_keys_to_ignore_on_save", None) is not None:
            state_dict = {k: v for k, v in state_dict.items() if k not in self._keys_to_ignore_on_save}

        if zero_stage != 3:
            self.load_state_dict(state_dict, strict=strict)
        else:
            load_zero3_flat_param(self, state_dict)

        logger.info("All model parameters loaded successfully from checkpoint.")
