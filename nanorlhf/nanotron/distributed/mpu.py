import os
import random
from typing import Optional, List, Tuple

import numpy as np
import torch
import torch.distributed as dist

from nanorlhf.nanotron.distributed.initializers import (
    DataParallelGroupInitializer,
    TensorParallelGroupInitializer,
    PipelineParallelGroupInitializer,
    TiedEmbeddingGroupInitializer,
)
from nanorlhf.nanotron.distributed.mode import ParallelMode
from nanorlhf.nanotron.distributed.seed import add_seed, set_mode


class MPU:
    """
    MPU is a model parallel unit that handles the distribution of model parameters.

    Examples:
        >>> from nanorlhf.nanotron.distributed.mpu import MPU, ParallelMode

        >>> # Initialize from torch.distributed.launch
        >>> mpu = MPU.from_torch(
        ...     data_parallel_size=1,
        ...     pipeline_parallel_size=1,
        ...     tensor_parallel_size=1,
        ... )

        >>> # Initialize from SLURM launcher
        >>> mpu = MPU.from_slurm(
        ...     host="MY_HOST",
        ...     port=1234,
        ...     data_parallel_size=1,
        ...     pipeline_parallel_size=1,
        ...     tensor_parallel_size=1,
        ... )

        >>> # Initialize from OpenMPI launcher
        >>> mpu = MPU.from_openmpi(
        ...     host="MY_HOST",
        ...     port=1234,
        ...     data_parallel_size=1,
        ...     pipeline_parallel_size=1,
        ...     tensor_parallel_size=1,
        ... )

        >>> # parallel_context world size
        >>> mpu.get_world_size(ParallelMode.DATA)

        >>> # get local size
        >>> mpu.get_local_rank(ParallelMode.DATA)

        >>> # get group
        >>> mpu.get_group(ParallelMode.DATA)

        >>> # get cpu group (gloo backend)
        >>> mpu.get_cpu_group(ParallelMode.DATA)

        >>> # get whole ranks in group
        >>> mpu.get_ranks_in_group(ParallelMode.DATA)

        >>> # get next global rank
        >>> mpu.get_next_global_rank(ParallelMode.DATA)

        >>> # get prev global rank
        >>> mpu.get_prev_global_rank(ParallelMode.DATA)

        Discussion:
            Q. How model and data parallelism are organized?
                Let's say we have a total of 16 GPUs denoted g0, ... g15,
                and we use 2 GPUs to parallelize the model tensors,
                and 4 GPUs to parallelize the model pipeline.

                The present method will create 8 tensor parallel groups,
                and 4 pipeline parallel groups and 8 data parallel groups as:

                - width: 4 pipeline parallel group
                    [g0, g2, g4, g6], [g1, g3, g5, g7], [g8, g10, g12, g14], [g9, g11, g13, g15]
                - height: 8 tensor parallel group
                    [g0, g1], [g2, g3], [g4, g5], [g6, g7], [g8, g9], [g10, g11], [g12, g13], [g14, g15]
                - depth: 8 data parallel group
                    [g0, g8], [g1, g9], [g2, g10], [g3, g11], [g4, g12], [g5, g13], [g6, g14], [g7, g15]

                                [g08, g10, g12, g14]
                              /  |              /  |
                             [g00, g02, g04, g06]  |
                             |   |             |   |
                3D parallel  |  [g09, g11, g13, g15]
                             |  /              |  /
                             [g01, g03, g05, g07]

                             +-----+  +----------+  +----------+  +----------+  +----------+  +-----+
                      model  | g00 |  |   g00    |  |   g02    |  |   g04    |  |   g06    |  | g06 |
                data         +-----+  +----------+  +----------+  +----------+  +----------+  +-----+  ===> forward
                      model  | g01 |  |   g01    |  |   g03    |  |   g05    |  |   g07    |  | g07 |
                             +-----+  +----------+  +----------+  +----------+  +----------+  +-----+
                            embedding   pipeline      pipeline      pipeline      pipeline   embedding

                             +-----+  +----------+  +----------+  +----------+  +----------+  +-----+
                      model  | g08 |  |   g08    |  |   g10    |  |   g12    |  |   g14    |  | g14 |
                data         +-----+  +----------+  +----------+  +----------+  +----------+  +-----+  ===> forward
                      model  | g09 |  |   g09    |  |   g11    |  |   g13    |  |   g15    |  | g15 |
                             +-----+  +----------+  +----------+  +----------+  +----------+  +-----+
                            embedding   pipeline      pipeline      pipeline      pipeline   embedding
    """

    @classmethod
    def from_torch(
        cls,
        data_parallel_size: int = 1,
        pipeline_parallel_size: int = 1,
        tensor_parallel_size: int = 1,
        rollout_data_parallel_size: int = 0,
        rollout_tensor_parallel_size: int = 0,
        backend: str = "nccl",
        seed: int = 42,
    ):
        """
        Initialize parallel context from `torch.distributed.launch`.

        Args:
            data_parallel_size (int): data parallel size
            pipeline_parallel_size (int): pipeline parallel size
            tensor_parallel_size (int): tensor parallel size
            rollout_data_parallel_size (int): rollout data parallel size
            rollout_tensor_parallel_size (int): rollout tensor parallel size
            backend (str): distributed backend
            seed (int): random seed value

        Returns:
            ParallelContext: parallel context object

        Examples:
            >>> # Initialize from torch.distributed.launch
            >>> mpu = MPU.from_torch(
            ...     data_parallel_size=1,
            ...     pipeline_parallel_size=1,
            ...     tensor_parallel_size=1,
            ... )
        """
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
        host = os.environ["MASTER_ADDR"]
        port = int(os.environ["MASTER_PORT"])

        return cls(
            rank=rank,
            local_rank=local_rank,
            world_size=world_size,
            local_world_size=local_world_size,
            host=host,
            port=port,
            data_parallel_size=data_parallel_size,
            pipeline_parallel_size=pipeline_parallel_size,
            tensor_parallel_size=tensor_parallel_size,
            rollout_data_parallel_size=rollout_data_parallel_size,
            rollout_tensor_parallel_size=rollout_tensor_parallel_size,
            backend=backend,
            seed=seed,
        )

    @classmethod
    def from_slurm(
        cls,
        host: str,
        port: int,
        data_parallel_size: int = 1,
        pipeline_parallel_size: int = 1,
        tensor_parallel_size: int = 1,
        rollout_data_parallel_size: int = 0,
        rollout_tensor_parallel_size: int = 0,
        backend: str = "nccl",
        seed: int = 42,
        local_rank: Optional[int] = None,
    ):
        """
        Initialize parallel context from SLURM launcher.

        Args:
            host (str): host server
            port (int): communication port
            data_parallel_size (int): data parallel size
            pipeline_parallel_size (int): pipeline parallel size
            tensor_parallel_size (int): tensor parallel size
            rollout_data_parallel_size (int): rollout data parallel size
            rollout_tensor_parallel_size (int): rollout tensor parallel size
            backend (str): distributed backend
            seed (int): random seed value
            local_rank (Optional[int]): local rank

        Returns:
            ParallelContext: parallel context object

        Examples:
            >>> # Initialize from SLURM launcher
            >>> mpu = MPU.from_slurm(
            ...     host="MY_HOST",
            ...     port=1234,
            ...     data_parallel_size=1,
            ...     pipeline_parallel_size=1,
            ...     tensor_parallel_size=1,
            ... )
        """
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NPROCS"])
        local_world_size = int(os.environ["SLURM_GPUS_ON_NODE"])

        return cls(
            rank=rank,
            local_rank=local_rank,
            world_size=world_size,
            local_world_size=local_world_size,
            host=host,
            port=port,
            data_parallel_size=data_parallel_size,
            pipeline_parallel_size=pipeline_parallel_size,
            tensor_parallel_size=tensor_parallel_size,
            rollout_data_parallel_size=rollout_data_parallel_size,
            rollout_tensor_parallel_size=rollout_tensor_parallel_size,
            backend=backend,
            seed=seed,
        )

    @classmethod
    def from_openmpi(
        cls,
        host: str,
        port: int,
        data_parallel_size: int = 1,
        pipeline_parallel_size: int = 1,
        tensor_parallel_size: int = 1,
        rollout_data_parallel_size: int = 0,
        rollout_tensor_parallel_size: int = 0,
        backend: str = "nccl",
        seed: int = 42,
    ):
        """
        Initialize parallel context from OpenMPI launcher.

        Args:
            host (str): host server
            port (int): communication port
            data_parallel_size (int): data parallel size
            pipeline_parallel_size (int): pipeline parallel size
            tensor_parallel_size (int): tensor parallel size
            rollout_data_parallel_size (int): rollout data parallel size
            rollout_tensor_parallel_size (int): rollout tensor parallel size
            backend (str): distributed backend
            seed (int): random seed value

        Returns:
            ParallelContext: parallel context object

        Examples:
            >>> # Initialize from OpenMPI launcher
            >>> mpu = MPU.from_openmpi(
            ...     host="MY_HOST",
            ...     port=1234,
            ...     data_parallel_size=1,
            ...     pipeline_parallel_size=1,
            ...     tensor_parallel_size=1,
            ... )
        """
        rank = int(os.environ["OMPI_COMM_WORLD_RANK"])
        local_rank = int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"])
        world_size = int(os.environ["OMPI_COMM_WORLD_SIZE"])
        local_world_size = int(os.environ["OMPI_COMM_WORLD_LOCAL_SIZE"])

        return cls(
            rank=rank,
            local_rank=local_rank,
            world_size=world_size,
            local_world_size=local_world_size,
            host=host,
            port=port,
            data_parallel_size=data_parallel_size,
            pipeline_parallel_size=pipeline_parallel_size,
            tensor_parallel_size=tensor_parallel_size,
            rollout_data_parallel_size=rollout_data_parallel_size,
            rollout_tensor_parallel_size=rollout_tensor_parallel_size,
            backend=backend,
            seed=seed,
        )

    def __init__(
        self,
        rank: int,
        local_rank: Optional[int],
        world_size: int,
        local_world_size: int,
        host: str,
        port: int,
        data_parallel_size: int,
        pipeline_parallel_size: int,
        tensor_parallel_size: int,
        rollout_data_parallel_size: int,
        rollout_tensor_parallel_size: int,
        backend: str,
        seed: int,
    ):
        self.actor_world_size = data_parallel_size * pipeline_parallel_size * tensor_parallel_size
        self.rollout_world_size = rollout_data_parallel_size * rollout_tensor_parallel_size
        assert world_size == self.actor_world_size + self.rollout_world_size, (
            "world_size must be equal to actor_world_size + rollout_world_size. "
            "actor_world_size = data_parallel_size * pipeline_parallel_size * tensor_parallel_size = "
            f"{self.actor_world_size}, "
            "rollout_world_size = rollout_data_parallel_size * rollout_tensor_parallel_size = "
            f"{self.rollout_world_size}, "
            f"but got world_size = {world_size}."
        )

        self.global_ranks = {}
        self.local_ranks = {}
        self.world_sizes = {}
        self.groups = {}
        self.cpu_groups = {}
        self.ranks_in_group = {}
        self.ranks_to_device = {}

        self.data_parallel_size = data_parallel_size
        self.pipeline_parallel_size = pipeline_parallel_size
        self.tensor_parallel_size = tensor_parallel_size
        self.rollout_data_parallel_size = rollout_data_parallel_size
        self.rollout_tensor_parallel_size = rollout_tensor_parallel_size
        self.local_world_size = local_world_size

        for mode in ParallelMode:
            self.add_world_size(mode, 0)
            self.add_local_rank(mode, 0)
            self.add_group(mode, None)
            self.add_cpu_group(mode, None)
            self.add_ranks_in_group(mode, [])

        self.init_global_dist(rank, world_size, backend, host, port)

        if torch.cuda.is_available():
            self.set_device(local_rank)

        self.init_parallel_groups()
        self.set_seed(seed)
        self.seed = seed
        self.make_ranks_to_devices()

    # sanity check
    @staticmethod
    def check_parallel_mode(mode: ParallelMode) -> None:
        """
        Check if the given parallel mode is valid.

        Args:
            mode (ParallelMode): The parallel mode to check.
        """
        if not isinstance(mode, ParallelMode):
            raise ValueError(f"Invalid parallel mode: {mode}. Expected one of {[m.value for m in ParallelMode]}.")

    # world sizes
    def get_world_size(self, mode: ParallelMode) -> int:
        """
        Get the world size for the given parallel mode.

        Args:
            mode (ParallelMode): The parallel mode.

        Returns:
            int: The world size for the given parallel mode.

        Examples:
            >>> mpu = ...
            >>> mpu.get_world_size(ParallelMode.DATA)
            4
        """
        self.check_parallel_mode(mode)
        return self.world_sizes[mode]

    def add_world_size(self, mode: ParallelMode, world_size: int):
        """
        Add the world size for the given parallel mode.

        Args:
            mode (ParallelMode): The parallel mode.
            world_size (int): The world size.

        Examples:
            >>> mpu = ...
            >>> mpu.add_world_size(ParallelMode.DATA, 4)
        """
        self.check_parallel_mode(mode)
        self.world_sizes[mode] = world_size

    # local ranks
    def get_local_rank(self, mode: ParallelMode) -> int:
        """
        Get the local rank for the given parallel mode.

        Args:
            mode (ParallelMode): The parallel mode.

        Returns:
            int: The local rank for the given parallel mode.

        Examples:
            >>> mpu = ...
            >>> mpu.get_local_rank(ParallelMode.DATA)
            0
        """
        self.check_parallel_mode(mode)
        return self.local_ranks[mode]

    def add_local_rank(self, mode: ParallelMode, local_rank: int):
        """
        Add the local rank for the given parallel mode.

        Args:
            mode (ParallelMode): The parallel mode.
            local_rank (int): The local rank.

        Examples:
            >>> mpu = ...
            >>> mpu.add_local_rank(ParallelMode.DATA, 0)
        """
        self.check_parallel_mode(mode)
        self.local_ranks[mode] = local_rank

    def get_local_ranks(self):
        """
        Get the local ranks for all parallel modes.

        Returns:
            dict: A dictionary mapping parallel mode to local rank.

        Examples:
            >>> mpu = ...
            >>> mpu.get_local_ranks()
            {
                ParallelMode.GLOBAL: 0,
                ParallelMode.DATA: 0,
                ParallelMode.TENSOR: 0,
                ParallelMode.PIPELINE: 0,
            }
        """
        return self.local_ranks

    # global ranks
    def get_global_rank(self) -> int:
        """
        Get the global rank for the given parallel mode.

        Returns:
            int: The global rank for the given parallel mode.

        Examples:
            >>> mpu = ...
            >>> mpu.get_global_rank(ParallelMode.DATA)
            0
        """
        return self.global_ranks[ParallelMode.GLOBAL]

    def add_global_rank(self, mode: ParallelMode, global_rank: int):
        """
        Add the global rank for the given parallel mode.

        Args:
            mode (ParallelMode): The parallel mode.
            global_rank (int): The global rank.

        Examples:
            >>> mpu = ...
            >>> mpu.add_global_rank(ParallelMode.DATA, 0)
        """
        self.check_parallel_mode(mode)
        self.global_ranks[mode] = global_rank

    def get_global_ranks(self):
        """
        Get the global ranks for all parallel modes.

        Returns:
            dict: A dictionary mapping parallel mode to global rank.

        Examples:
            >>> mpu = ...
                >>> mpu.get_global_ranks()
                {
                    ParallelMode.GLOBAL: 0,
                    ParallelMode.DATA: 0,
                    ParallelMode.TENSOR: 0,
                    ParallelMode.PIPELINE: 0,
                }
        """
        return self.global_ranks

    def get_next_global_rank(self, mode: ParallelMode) -> int:
        """
        Get next global rank by given parallel mode

        Args:
            mode (ParallelMode): ParallelMode object

        Returns:
            int: The next global rank by given parallel mode

        Examples:
            >>> mpu = ...
            >>> mpu.get_next_global_rank(ParallelMode.DATA)
        """
        self.check_parallel_mode(mode)

        local_rank = self.get_local_rank(mode)
        world_size = self.get_world_size(mode)
        ranks_in_group = self.get_ranks_in_group(mode)

        return ranks_in_group[(local_rank + 1) % world_size]

    def get_prev_global_rank(self, mode: ParallelMode) -> int:
        """
        Get previous global rank by given parallel mode

        Args:
            mode (ParallelMode): ParallelMode object

        Returns:
            int: The previous global rank by given parallel mode

        Examples:
            >>> mpu = ...
            >>> mpu.get_prev_global_rank(ParallelMode.DATA)
        """
        self.check_parallel_mode(mode)

        local_rank = self.get_local_rank(mode)
        world_size = self.get_world_size(mode)
        ranks_in_group = self.get_ranks_in_group(mode)

        return ranks_in_group[(local_rank - 1 + world_size) % world_size]

    def is_first_rank(self, mode: ParallelMode):
        """
        Check if the current rank is the first rank in the given parallel mode.

        Args:
            mode (ParallelMode): The parallel mode.

        Returns:
            bool: True if the current rank is the first rank in the given parallel mode, False otherwise.

        Examples:
            >>> mpu = ...
            >>> mpu.is_first_rank(ParallelMode.DATA)
            True
        """
        self.check_parallel_mode(mode)
        return self.get_local_rank(mode) == 0

    def is_last_rank(self, mode: ParallelMode):
        """
        Check if the current rank is the last rank in the given parallel mode.

        Args:
            mode (ParallelMode): The parallel mode.

        Returns:
            bool: True if the current rank is the last rank in the given parallel mode, False otherwise.

        Examples:
            >>> mpu = ...
            >>> mpu.is_last_rank(ParallelMode.DATA)
            False
        """
        self.check_parallel_mode(mode)
        return self.get_local_rank(mode) == self.get_world_size(mode) - 1

    # groups
    def get_group(self, mode: ParallelMode) -> Optional[dist.ProcessGroup]:
        """
        Get the process group for the given parallel mode.

        Args:
            mode (ParallelMode): The parallel mode.

        Returns:
            torch.distributed.ProcessGroup: The process group for the given parallel mode.

        Examples:
            >>> mpu = ...
            >>> mpu.get_group(ParallelMode.DATA)
            ProcessGroupNCCL
        """
        self.check_parallel_mode(mode)
        return self.groups.get(mode, None)

    def add_group(self, mode: ParallelMode, group: Optional[torch.distributed.ProcessGroup]):
        """
        Add the process group for the given parallel mode.

        Args:
            mode (ParallelMode): The parallel mode.
            group (torch.distributed.ProcessGroup): The process group.

        Examples:
            >>> process_group = ...
            >>> mpu = ...
            >>> mpu.add_group(ParallelMode.DATA, process_group)
        """
        self.check_parallel_mode(mode)
        self.groups[mode] = group

    def get_cpu_group(self, mode: ParallelMode) -> Optional[dist.ProcessGroup]:
        """
        Get the CPU process group for the given parallel mode.

        Args:
            mode (ParallelMode): The parallel mode.

        Returns:
            torch.distributed.ProcessGroup: The CPU process group for the given parallel mode.

        Examples:
            >>> mpu = ...
            >>> mpu.get_cpu_group(ParallelMode.DATA)
            ProcessGroupGloo
        """
        self.check_parallel_mode(mode)
        return self.cpu_groups.get(mode, None)

    def add_cpu_group(self, mode: ParallelMode, group: Optional[torch.distributed.ProcessGroup]):
        """
        Add the CPU process group for the given parallel mode.

        Args:
            mode (ParallelMode): The parallel mode.
            group (torch.distributed.ProcessGroup): The CPU process group.

        Examples:
            >>> process_group = ...
            >>> mpu = ...
            >>> mpu.add_cpu_group(ParallelMode.DATA, process_group)
        """
        self.check_parallel_mode(mode)
        self.cpu_groups[mode] = group

    # ranks in group
    def get_ranks_in_group(self, mode: ParallelMode) -> List[int]:
        """
        Get the ranks in the process group for the given parallel mode.

        Args:
            mode (ParallelMode): The parallel mode.

        Returns:
            List[int]: The ranks in the process group for the given parallel mode.

        Examples:
            >>> mpu = ...
            >>> mpu.get_ranks_in_group(ParallelMode.DATA)
            [0, 4, 8, 12]
        """
        self.check_parallel_mode(mode)
        return self.ranks_in_group[mode]

    def add_ranks_in_group(self, mode: ParallelMode, ranks: List[int]):
        """
        Add the ranks in the process group for the given parallel mode.

        Args:
            mode (ParallelMode): The parallel mode.
            ranks (List[int]): The ranks in the process group.

        Examples:
            >>> mpu = ...
            >>> mpu.add_ranks_in_group(ParallelMode.DATA, [0, 4, 8, 12])
        """
        self.check_parallel_mode(mode)
        self.ranks_in_group[mode] = ranks

    def make_ranks_to_devices(self):
        """
        Make a mapping from (fixed-ordered modes -> local ranks) to global device (rank).
        Ensures all ranks use the SAME order & length, so all_gather never mismatches.
        """
        ordered_modes = [
            ParallelMode.GLOBAL,
            ParallelMode.DATA,
            ParallelMode.TENSOR,
            ParallelMode.PIPELINE,
            ParallelMode.TIED_EMBEDDING,
            ParallelMode.ROLLOUT_DATA,
            ParallelMode.ROLLOUT_TENSOR,
        ]

        vals = []
        for mode in ordered_modes:
            local_rank = self.local_ranks.get(mode, 0)
            vals.append(local_rank)
        rank_tensor = torch.tensor(vals, dtype=torch.long, device="cuda")

        world = self.get_world_size(ParallelMode.GLOBAL)
        gather_list = [torch.empty_like(rank_tensor) for _ in range(world)]
        dist.all_gather(gather_list, rank_tensor)

        self.ranks_to_device.clear()
        for global_rank, rt in enumerate(gather_list):
            modes_and_ranks = tuple((mode, int(val)) for mode, val in zip(ordered_modes, rt.tolist()))
            self.ranks_to_device[modes_and_ranks] = global_rank

    def ranks2device(self, ranks: dict) -> Optional[int]:
        """
        Get the device (global rank) for the given local ranks in different parallel modes.

        Args:
            ranks (dict): A dictionary mapping parallel mode to local rank.

        Examples:
            ranks:
                {
                    <ParallelMode.TENSOR: 'tensor'>: 1
                    <ParallelMode.DATA: 'data'>: 0
                }

            self._ranks_to_device:
            {
                (
                    (<ParallelMode.GLOBAL: 'global'>, 0),
                    (<ParallelMode.DATA: 'data'>, 0),
                    (<ParallelMode.TENSOR: 'tensor'>, 0),
                ): 0,
                (
                    (<ParallelMode.GLOBAL: 'global'>, 1),
                    (<ParallelMode.DATA: 'data'>, 0),
                    (<ParallelMode.TENSOR: 'tensor'>, 1),
                ): 1,
                ...
            }

            return device: 1
        """
        ordered_modes = [
            ParallelMode.GLOBAL,
            ParallelMode.DATA,
            ParallelMode.TENSOR,
            ParallelMode.PIPELINE,
            ParallelMode.TIED_EMBEDDING,
            ParallelMode.ROLLOUT_DATA,
            ParallelMode.ROLLOUT_TENSOR,
        ]

        key = []
        for mode in ordered_modes:
            if mode in ranks:
                key.append((mode, ranks[mode]))
            else:
                key.append((mode, self.local_ranks.get(mode, 0)))

        key = tuple(key)
        return self.ranks_to_device.get(key, None)

    # init distributed group
    def init_global_dist(
        self,
        rank: int,
        world_size: int,
        backend: str,
        host: str,
        port: int,
    ):
        """
        Initialize the global distributed process group.

        Args:
            rank (int): The global rank of the current process.
            world_size (int): The total number of processes.
            backend (str): The backend to use. One of 'nccl', 'gloo', 'mpi'.
            host (str): The master node's hostname or IP address.
            port (int): The master node's port.

        Examples:
            >>> mpu = ...
            >>> mpu.init_global_dist(
            ...     rank=0,
            ...     world_size=4,
            ...     backend='nccl',
            ...     host='localhost',
            ...     port=12345,
            ... )
        """
        if not dist.is_initialized():
            init_method = f"tcp://{host}:{port}"
            dist.init_process_group(
                rank=rank,
                world_size=world_size,
                backend=backend,
                init_method=init_method,
            )

        ranks = list(range(world_size))
        cpu_group = dist.new_group(ranks, backend="gloo") if dist.get_backend() != "gloo" else None
        self.register_dist(rank, world_size, None, cpu_group, ranks, ParallelMode.GLOBAL)
        self.add_global_rank(ParallelMode.GLOBAL, rank)

    def register_dist(
        self,
        local_rank: int,
        group_world_size: int,
        process_group: Optional[dist.ProcessGroup],
        cpu_group: Optional[dist.ProcessGroup],
        ranks_in_group: List[int],
        mode: ParallelMode,
    ):
        """
        Register distributed setting by give parallel mode

        Args:
            local_rank (int): local rank
            group_world_size (int): group world size
            process_group (Optional[dist.ProcessGroup]): process group
            cpu_group (Optional[dist.ProcessGroup]): cpu process group
            ranks_in_group (List[int]): whole ranks in the group
            mode (ParallelMode): ParallelMode object
        """
        self.add_local_rank(mode, local_rank)
        self.add_world_size(mode, group_world_size)
        self.add_group(mode, process_group)
        self.add_cpu_group(mode, cpu_group)
        self.add_ranks_in_group(mode, ranks_in_group)

    def init_parallel_groups(self):
        """
        Initialize all parallel process groups: data, model, tensor, pipeline.
        """
        rank = self.get_global_rank()
        world_size = self.get_world_size(ParallelMode.GLOBAL)

        actor_initializer_params = {
            "rank": rank,
            "world_size": world_size,
            "data_parallel_size": self.data_parallel_size,
            "pipeline_parallel_size": self.pipeline_parallel_size,
            "tensor_parallel_size": self.tensor_parallel_size,
            "group_world_size": self.actor_world_size,
            "group_rank_offset": 0,
        }
        rollout_initializer_param = {
            "rank": rank,
            "world_size": world_size,
            "data_parallel_size": self.rollout_data_parallel_size,
            "pipeline_parallel_size": 1,
            "tensor_parallel_size": self.rollout_tensor_parallel_size,
            "group_world_size": self.rollout_world_size,
            "group_rank_offset": self.actor_world_size,
        }

        initializer_results = []
        if self.actor_world_size != 0:
            initializer_results.append(DataParallelGroupInitializer(**actor_initializer_params).init_dist_group())
            initializer_results.append(TensorParallelGroupInitializer(**actor_initializer_params).init_dist_group())
            initializer_results.append(PipelineParallelGroupInitializer(**actor_initializer_params).init_dist_group())
            initializer_results.append(TiedEmbeddingGroupInitializer(**actor_initializer_params).init_dist_group())

        if self.rollout_world_size != 0:
            initializer_results.append(
                DataParallelGroupInitializer(**rollout_initializer_param).init_dist_group(ParallelMode.ROLLOUT_DATA)
            )
            initializer_results.append(
                TensorParallelGroupInitializer(**rollout_initializer_param).init_dist_group(
                    ParallelMode.ROLLOUT_TENSOR
                )
            )

        for initializer_result in initializer_results:
            if initializer_result is None:
                continue
            elif isinstance(initializer_result, list):
                for res in initializer_result:
                    self.register_dist(**res)
            else:
                self.register_dist(**initializer_result)

    def is_initialized(self, mode: ParallelMode) -> bool:
        """
        Check if the process group for the given parallel mode is initialized.

        Args:
            mode (ParallelMode): The parallel mode.

        Returns:
            bool: True if the process group for the given parallel mode is initialized, False otherwise.

        Examples:
            >>> mpu = ...
            >>> mpu.is_initialized(ParallelMode.DATA)
            True
        """
        self.check_parallel_mode(mode)
        return mode in self.groups

    def destroy(self):
        """Destroy all the parallel groups"""
        for mode, group in self.groups.items():
            if mode is not ParallelMode.GLOBAL and group is not None:
                dist.destroy_process_group(group)

        dist.destroy_process_group()
        self.groups.clear()

    def set_device(self, device_ordinal: Optional[int] = None):
        """
        Set the current device to the given device ordinal.

        Args:
            device_ordinal (Optional[int]): The device ordinal. If None, use the local rank of the global parallel mode.

        Examples:
            >>> mpu = ...
            >>> mpu.set_device(0)
        """
        global_rank = self.get_global_rank()
        if device_ordinal is None:
            devices_per_node = torch.cuda.device_count()
            device_ordinal = global_rank % devices_per_node
        torch.cuda.set_device(device_ordinal)

    def set_seed(self, seed: int):
        """
        Set the random seed for all parallel modes.

        Args:
            seed (int): The random seed.

        Examples:
            >>> mpu = ...
            >>> mpu.set_seed(42)

        Discussion:
            Q. How are the seeds set for different parallel modes?
                - Data parallel mode:     All ranks in the data parallel group use the same seed.
                - Tensor parallel mode:   Ranks in the tensor parallel group use different seeds,
                                          offset by their local rank and pipeline stage.
                - Pipeline parallel mode: Ranks in the pipeline parallel group use different seeds,

            Q. Why is it important to set different seeds for different parallel modes?
                Setting different seeds helps ensure that operations that rely on randomness
                (e.g., dropout, weight initialization) produce different results across
                different parallel groups, which can improve model performance and convergence.
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if not torch.cuda.is_available():
            return

        # actor slice seeds
        if self.is_initialized(ParallelMode.DATA):
            add_seed(ParallelMode.DATA, seed)
        if self.is_initialized(ParallelMode.TENSOR):
            pipeline_offset = self.local_ranks.get(ParallelMode.PIPELINE, 0)
            tp_rank = self.get_local_rank(ParallelMode.TENSOR)
            tp_rank_with_offset = tp_rank + pipeline_offset * 1024
            add_seed(ParallelMode.TENSOR, seed + tp_rank_with_offset)

        # rollout slice seeds
        if self.is_initialized(ParallelMode.ROLLOUT_DATA) or self.is_initialized(ParallelMode.ROLLOUT_TENSOR):
            rollout_local = self.get_global_rank() - self.actor_world_size
            if 0 <= rollout_local < self.rollout_world_size:
                rollout_tp = self.rollout_tensor_parallel_size
                replica_id = rollout_local // rollout_tp
                tp_rank = rollout_local % rollout_tp
                add_seed(ParallelMode.ROLLOUT_DATA, seed + replica_id * 4096)
                add_seed(ParallelMode.ROLLOUT_TENSOR, seed + replica_id * 4096 + tp_rank)

        if self.is_initialized(ParallelMode.DATA):
            set_mode(ParallelMode.DATA)
        elif self.is_initialized(ParallelMode.ROLLOUT_DATA):
            set_mode(ParallelMode.ROLLOUT_DATA)
        else:
            set_mode(ParallelMode.GLOBAL)

    @staticmethod
    def get_local_ranks_from_global_rank(
        global_rank: int,
        data_parallel_size: int,
        tensor_parallel_size: int,
        pipeline_parallel_size: int,
        global_rank_offset: int = 0,
    ) -> Tuple[int, int, int]:
        """
        Get local ranks in data, tensor, and pipeline parallel groups from the given global rank.

        Args:
            global_rank (int): The global rank.
            data_parallel_size (int): The data parallel size.
            tensor_parallel_size (int): The tensor parallel size.
            pipeline_parallel_size (int): The pipeline parallel size.
            global_rank_offset (int): The global rank offset.

        Returns:
            Tuple[int, int, int]: A tuple containing the local ranks in data, tensor, and pipeline parallel groups.
        """
        local_rank_in_slice = global_rank - global_rank_offset
        model_parallel_size = tensor_parallel_size * pipeline_parallel_size
        world_size_in_slice = data_parallel_size * model_parallel_size

        if not (0 <= local_rank_in_slice < world_size_in_slice):
            raise ValueError(
                f"global_rank={global_rank} is out of slice range. "
                f"offset={global_rank_offset}, slice_world_size={world_size_in_slice}."
            )

        data_parallel_rank = local_rank_in_slice // model_parallel_size
        model_parallel_rank = local_rank_in_slice % model_parallel_size
        tensor_parallel_rank = model_parallel_rank % tensor_parallel_size
        pipeline_parallel_rank = model_parallel_rank // tensor_parallel_size
        return data_parallel_rank, tensor_parallel_rank, pipeline_parallel_rank
