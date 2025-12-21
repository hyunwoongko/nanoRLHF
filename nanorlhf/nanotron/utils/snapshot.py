import inspect
from dataclasses import dataclass
from typing import Dict, Any, Optional

import torch
from torch import nn


class EarlyStop(RuntimeError):
    """Exception to signal early stopping of a pipeline stage."""


@dataclass
class ModuleSnapshot:
    """
    A `ModuleSnapshot` represents the inputs and outputs of a module.

    Attributes:
        input_param_name (str): The name of the hidden state parameter.
        output_tensor (torch.Tensor): The hidden state tensor from the module.
        kwargs (Dict[str, Any]): Additional keyword arguments passed to the module except the hidden state.
    """

    input_param_name: str
    output_tensor: torch.Tensor
    kwargs: Dict[str, Any]


def to_kwargs(fn, args, kwargs):
    sig = inspect.signature(fn)
    bound = sig.bind_partial(*args, **kwargs)
    bound.apply_defaults()
    bound.arguments.pop("self", None)
    return bound.arguments


class ModuleSnapshotGenerator:
    """
    `ModuleSnapshotGenerator` wraps a module's forward method to generate
    a snapshot of its inputs and outputs.

    Args:
        module (nn.Module): The module to generate a snapshot for.
    """

    def __init__(self, module: nn.Module):
        self.module = module
        self.snapshot: Optional[ModuleSnapshot] = None
        self._module_orig_forward = module.forward

    def pick_hidden_state_param(
        self,
        bound_args: Dict[str, Any],
        out_tensor: torch.Tensor,
    ) -> Optional[str]:
        """
        Identify the hidden state parameter from the bound arguments.

        Args:
            bound_args (Dict[str, Any]): The bound arguments of the forward method.
            out_tensor (torch.Tensor): The output tensor from the module.

        Returns:
            Optional[str]: The name of the hidden state parameter, or `None` if not found.
        """
        if "hidden_states" in bound_args:
            return "hidden_states"
        for name, arg in bound_args.items():
            if torch.is_tensor(arg) and arg.shape == out_tensor.shape and arg.dtype == out_tensor.dtype:
                return name
        return None

    def wrapper(self, *args, **kwargs) -> Any:
        """
        Wrapper for the module's forward method to capture inputs and outputs.

        Returns:
            Any: The output of the original forward method.
        """
        output = self._module_orig_forward(*args, **kwargs)
        hidden_states = output[0] if isinstance(output, (tuple, list)) else output
        if not torch.is_tensor(hidden_states):
            raise RuntimeError("`ModuleSnapshotGenerator` expected the first output to be a tensor.")

        _kwargs = to_kwargs(self._module_orig_forward, args, kwargs)
        param_name = self.pick_hidden_state_param(_kwargs, hidden_states)
        if param_name is None:
            raise RuntimeError("`ModuleSnapshotGenerator` could not identify the hidden state parameter.")

        self.snapshot = ModuleSnapshot(
            input_param_name=param_name,
            output_tensor=hidden_states,
            kwargs={k: v for k, v in _kwargs.items() if k != param_name},
        )
        raise EarlyStop(
            "Early stopping after capturing module snapshot to avoid unnecessary computation."
        )

    def install(self):
        """Install the wrapper to the module's forward method."""
        self.module.forward = self.wrapper

    def uninstall(self):
        """Uninstall the wrapper and restore the original forward method."""
        self.module.forward = self._module_orig_forward

    def generate(self, model: nn.Module, inputs: Dict[str, Any]):
        """
        Generate a snapshot by running the model with the provided inputs.

        Args:
            model (nn.Module): The model to run.
            inputs (Dict[str, Any]): The inputs to the model.

        Returns:
            ModuleSnapshot: The captured snapshot of the module's inputs and outputs.
        """
        try:
            self.install()
            _ = model.__nanotron_forward__(**inputs)
        except EarlyStop:
            pass
        finally:
            self.uninstall()

        return self.snapshot
