import functools
from contextlib import contextmanager

import torch
from torch import Tensor

from nanorlhf.nanotron.distributed.mpu import ParallelMode


class SeedManager:
    """
    Manages random seeds for different parallel modes in a distributed training setup.

    Attributes:
        _current_mode (ParallelMode): The current parallel mode.
        _seeds (dict): A dictionary mapping ParallelMode to their respective seeds.
        _seed_states (dict): A dictionary mapping ParallelMode to their respective RNG states.

    Discussion:
        Q. Why manage seeds per parallel mode?
            In distributed training, different parallel modes (e.g., data parallelism, model parallelism)
            may require different random seeds to ensure reproducibility and proper randomness.
            This manager allows for easy switching between modes while preserving their RNG states.
    """

    def __init__(self):
        self._current_mode = None
        self._seeds = dict()
        self._seed_states = dict()

    @property
    def current_mode(self):
        """
        Get the current parallel mode.

        Returns:
            ParallelMode: The current parallel mode, or None if not set.
        """
        return self._current_mode

    @property
    def seeds(self):
        """
        Get the dictionary of seeds for all parallel modes.

        Returns:
            dict: A dictionary mapping ParallelMode to their respective seeds.
        """
        return self._seeds

    @property
    def seed_states(self):
        """
        Get the dictionary of RNG states for all parallel modes.

        Returns:
            dict: A dictionary mapping ParallelMode to their respective RNG states.
        """
        return self._seed_states

    def set_state(self, mode: ParallelMode, state: Tensor):
        """
        Set the RNG state for a specific parallel mode.

        Args:
            mode (ParallelMode): The parallel mode to set the state for.
            state (Tensor): The RNG state tensor to set.
        """
        assert (
            mode in self._seed_states
        ), f"Parallel mode {mode} is not found in the seed manager"
        self._seed_states[mode] = state

    def set_mode(self, mode: ParallelMode):
        """
        Switch to a different parallel mode, saving and restoring RNG states as needed.

        Args:
            mode (ParallelMode): The parallel mode to switch to.
        """
        if self.current_mode:
            # save the current state for current mode
            self._seed_states[self._current_mode] = torch.cuda.get_rng_state()

        # set the new state for new mode
        self._current_mode = mode
        torch.cuda.set_rng_state(self._seed_states[mode])

    def add_seed(self, mode: ParallelMode, seed: int, overwrite: bool = False):
        """
        Add a new seed for a specific parallel mode.

        Args:
            mode (ParallelMode): The parallel mode to add the seed for.
            seed (int): The seed value to set.
            overwrite (bool): Whether to overwrite an existing seed for the mode. Default is False.
        """
        assert isinstance(mode, ParallelMode), "A valid ParallelMode must be provided"
        if overwrite is False:
            assert mode not in self._seed_states, f"The seed for {mode} has been added"
        elif mode in self._seed_states:
            print(f"Warnning: {mode} seed has been overwritten.", flush=True)

        current_state = torch.cuda.get_rng_state()
        torch.cuda.manual_seed(seed)
        self._seed_states[mode] = torch.cuda.get_rng_state()
        self._seeds[mode] = seed
        torch.cuda.set_rng_state(current_state)

    def reset(self):
        """
        Reset the seed manager, clearing all modes, seeds, and states.
        """
        self._current_mode = None
        self._seeds = dict()
        self._seed_states = dict()


# Global seed manager instance
GLOBAL_SEED_MANAGER = SeedManager()


# Helper functions to interact with the global seed manager
def get_seeds():
    """
    Get the dictionary of seeds for all parallel modes.

    Returns:
        dict: A dictionary mapping ParallelMode to their respective seeds.
    """
    return GLOBAL_SEED_MANAGER.seeds


def get_states(copy: bool = False):
    """
    Get the dictionary of RNG states for all parallel modes.

    Args:
        copy (bool): Whether to return a copy of the states. Default is False.

    Returns:
        dict: A dictionary mapping ParallelMode to their respective RNG states.
    """
    states = GLOBAL_SEED_MANAGER.seed_states

    if copy:
        new_states = dict()

        for mode, state in states.items():
            new_states[mode] = state.clone()
        return new_states
    else:
        return GLOBAL_SEED_MANAGER.seed_states


def get_current_mode():
    """
    Get the current parallel mode.

    Returns:
        ParallelMode: The current parallel mode, or None if not set.
    """
    return GLOBAL_SEED_MANAGER.current_mode


def add_seed(mode: ParallelMode, seed: int, overwrite: bool = False):
    """
    Add a new seed for a specific parallel mode.

    Args:
        mode (ParallelMode): The parallel mode to add the seed for.
        seed (int): The seed value to set.
        overwrite (bool): Whether to overwrite an existing seed for the mode. Default is False.
    """
    GLOBAL_SEED_MANAGER.add_seed(mode, seed, overwrite)


def set_mode(mode: ParallelMode):
    """
    Switch to a different parallel mode, saving and restoring RNG states as needed.

    Args:
        mode (ParallelMode): The parallel mode to switch to.
    """
    GLOBAL_SEED_MANAGER.set_mode(mode)


def set_seed_states(mode: ParallelMode, state: Tensor):
    """
    Set the RNG state for a specific parallel mode.

    Args:
        mode (ParallelMode): The parallel mode to set the state for.
        state (Tensor): The RNG state tensor to set.
    """
    GLOBAL_SEED_MANAGER.set_state(mode, state)


def sync_states():
    """
    Sync the current RNG state with the seed manager for the current parallel mode.
    """
    current_mode = get_current_mode()
    current_states = torch.cuda.get_rng_state()
    set_seed_states(current_mode, current_states)


@contextmanager
def seed(mode: ParallelMode):
    """
    Context manager to temporarily switch to a different parallel mode.

    Args:
        mode (ParallelMode): The parallel mode to switch to temporarily.
    """
    prev = GLOBAL_SEED_MANAGER.current_mode
    GLOBAL_SEED_MANAGER.set_mode(mode)
    try:
        yield
    finally:
        if prev is not None:
            GLOBAL_SEED_MANAGER.set_mode(prev)


def with_seed(func, mode: ParallelMode):
    """
    Decorator to execute a function within a specific parallel mode context.

    Args:
        func (callable): The function to decorate.
        mode (ParallelMode): The parallel mode to switch to during function execution.

    Returns:
        callable: The decorated function that runs within the specified parallel mode.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # switch mode
        current_mode = GLOBAL_SEED_MANAGER.current_mode
        GLOBAL_SEED_MANAGER.set_mode(mode)

        # exec func
        output = func(*args, **kwargs)

        # recover state
        GLOBAL_SEED_MANAGER.set_mode(current_mode)

        return output

    return wrapper


def reset_seeds():
    """
    Reset the global seed manager, clearing all modes, seeds, and states.
    """
    GLOBAL_SEED_MANAGER.reset()
