import os
import random
from typing import Optional, List

import numpy as np
import torch
import torch.distributed as dist

from nanorlhf.nanotron.distributed.initializers import (
    DataParallelGroupInitializer,
    ModelParallelGroupInitializer,
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
        backend: str = "nccl",
        seed: int = 42,
    ):
        """
        Initialize parallel context from `torch.distributed.launch`.

        Args:
            data_parallel_size (int): data parallel size
            pipeline_parallel_size (int): pipeline parallel size
            tensor_parallel_size (int): tensor parallel size
            backend (str): distributed backend
            seed (int): random seed value

        Returns:
            ParallelContext: parallel context object

        Examples:
            >>> # Initialize from torch.distributed.launch
            >>> mpu = MPU.from_torch(
            ...     data_parallel_size=1,
            ...     sequence_parallel_size=1,
            ...     expert_parallel_size=1,
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
            ...     sequence_parallel_size=1,
            ...     expert_parallel_size=1,
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
            ...     sequence_parallel_size=1,
            ...     expert_parallel_size=1,
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
        backend: str,
        seed: int,
    ):
        assert (
            world_size
            == data_parallel_size * pipeline_parallel_size * tensor_parallel_size
        ), (
            f"Expected the world size `{world_size}` to "
            f"data parallel size ({data_parallel_size}) * "
            f"pipeline parallel size ({pipeline_parallel_size}) * "
            f"tensor parallel size ({tensor_parallel_size}), "
            f"but got `{data_parallel_size * pipeline_parallel_size * tensor_parallel_size}`."
        )

        self._global_ranks = {}
        self._local_ranks = {}
        self._world_sizes = {}
        self._groups = {}
        self._cpu_groups = {}
        self._ranks_in_group = {}
        self._ranks_to_device = {}

        self.data_parallel_size = data_parallel_size
        self.pipeline_parallel_size = pipeline_parallel_size
        self.tensor_parallel_size = tensor_parallel_size

        self.local_world_size = local_world_size
        self.init_global_dist(rank, world_size, backend, host, port)
        self.init_parallel_groups()

        if torch.cuda.is_available():
            self.set_device(local_rank)

        self.set_seed(seed)
        self.seed = seed
        self.make_ranks_to_devices()

    # sanity check
    @staticmethod
    def _check_parallel_mode(mode: ParallelMode) -> None:
        if not isinstance(mode, ParallelMode):
            raise ValueError(
                f"Invalid parallel mode: {mode}. Expected one of {[m.value for m in ParallelMode]}."
            )

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
        self._check_parallel_mode(mode)
        return self._world_sizes[mode]

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
        self._check_parallel_mode(mode)
        self._world_sizes[mode] = world_size

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
        self._check_parallel_mode(mode)
        return self._local_ranks[mode]

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
        self._check_parallel_mode(mode)
        self._local_ranks[mode] = local_rank

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
                ParallelMode.MODEL: 0,
                ParallelMode.TENSOR: 0,
                ParallelMode.PIPELINE: 0,
            }
        """
        return self._local_ranks

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
        return self._global_ranks[ParallelMode.GLOBAL]

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
        self._check_parallel_mode(mode)
        self._global_ranks[mode] = global_rank

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
                    ParallelMode.MODEL: 0,
                    ParallelMode.TENSOR: 0,
                    ParallelMode.PIPELINE: 0,
                }
        """
        return self._global_ranks

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
        self._check_parallel_mode(mode)

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
        self._check_parallel_mode(mode)

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
        self._check_parallel_mode(mode)
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
        self._check_parallel_mode(mode)
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
        self._check_parallel_mode(mode)
        return self._groups.get(mode, None)

    def add_group(self, mode: ParallelMode, group: torch.distributed.ProcessGroup):
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
        self._check_parallel_mode(mode)
        self._groups[mode] = group

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
        self._check_parallel_mode(mode)
        return self._cpu_groups.get(mode, None)

    def add_cpu_group(self, mode: ParallelMode, group: torch.distributed.ProcessGroup):
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
        self._check_parallel_mode(mode)
        self._cpu_groups[mode] = group

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
        self._check_parallel_mode(mode)
        return self._ranks_in_group[mode]

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
        self._check_parallel_mode(mode)
        self._ranks_in_group[mode] = ranks

    def make_ranks_to_devices(self):
        """
        Make a mapping from (fixed-ordered modes -> local ranks) to global device (rank).
        Ensures all ranks use the SAME order & length, so all_gather never mismatches.
        """
        ordered_modes = [
            ParallelMode.GLOBAL,
            ParallelMode.DATA,
            ParallelMode.MODEL,
            ParallelMode.TENSOR,
            ParallelMode.PIPELINE,
            ParallelMode.TIED_EMBEDDING,
        ]

        vals = []
        for mode in ordered_modes:
            vals.append(self._local_ranks.get(mode, 0))
        rank_tensor = torch.tensor(vals, dtype=torch.long, device="cuda")

        world = self.get_world_size(ParallelMode.GLOBAL)
        gather_list = [torch.empty_like(rank_tensor) for _ in range(world)]
        dist.all_gather(gather_list, rank_tensor)

        self._ranks_to_device.clear()
        for global_rank, rt in enumerate(gather_list):
            modes_and_ranks = tuple((mode, int(val)) for mode, val in zip(ordered_modes, rt.tolist()))
            self._ranks_to_device[modes_and_ranks] = global_rank

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
                    (<ParallelMode.MODEL: 'model'>, 0),
                    (<ParallelMode.TENSOR: 'tensor'>, 0),
                ): 0,
                (
                    (<ParallelMode.GLOBAL: 'global'>, 1),
                    (<ParallelMode.DATA: 'data'>, 0),
                    (<ParallelMode.MODEL: 'model'>, 1),
                    (<ParallelMode.TENSOR: 'tensor'>, 1),
                ): 1,
                ...
            }

            return device: 1
        """
        ordered_modes = [
            ParallelMode.GLOBAL,
            ParallelMode.DATA,
            ParallelMode.MODEL,
            ParallelMode.TENSOR,
            ParallelMode.PIPELINE,
            ParallelMode.TIED_EMBEDDING,
        ]

        key = []
        for mode in ordered_modes:
            if mode in ranks:
                key.append((mode, ranks[mode]))
            else:
                key.append((mode, self._local_ranks.get(mode, 0)))

        key = tuple(key)
        return self._ranks_to_device.get(key, None)

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

        Returns:
            None

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
        init_method = f"tcp://{host}:{port}"
        dist.init_process_group(
            rank=rank, world_size=world_size, backend=backend, init_method=init_method
        )

        ranks = list(range(world_size))
        cpu_group = (
            dist.new_group(ranks, backend="gloo")
            if dist.get_backend() != "gloo"
            else None
        )
        self._register_dist(
            rank, world_size, None, cpu_group, ranks, ParallelMode.GLOBAL
        )
        self.add_global_rank(ParallelMode.GLOBAL, rank)

    def _register_dist(
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

        initializer_param = {
            "rank": rank,
            "world_size": world_size,
            "data_parallel_size": self.data_parallel_size,
            "pipeline_parallel_size": self.pipeline_parallel_size,
            "tensor_parallel_size": self.tensor_parallel_size,
        }

        initializer_results = [
            DataParallelGroupInitializer(**initializer_param).init_dist_group(),
            ModelParallelGroupInitializer(**initializer_param).init_dist_group(),
            TensorParallelGroupInitializer(**initializer_param).init_dist_group(),
            PipelineParallelGroupInitializer(**initializer_param).init_dist_group(),
            TiedEmbeddingGroupInitializer(**initializer_param).init_dist_group(),
        ]

        for initializer_result in initializer_results:
            if isinstance(initializer_result, list):
                for res in initializer_result:
                    self._register_dist(**res)
            else:
                self._register_dist(**initializer_result)

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
        self._check_parallel_mode(mode)
        return mode in self._groups

    def destroy(self):
        """Destroy all the parallel groups"""
        for mode, group in self._groups.items():
            if mode is not ParallelMode.GLOBAL:
                dist.destroy_process_group(group)

        dist.destroy_process_group()
        self._groups.clear()

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

        if torch.cuda.is_available():
            # create random seed for different parallel modes
            # data parallel seed are kept the same
            parallel_seed = seed
            add_seed(ParallelMode.DATA, parallel_seed)

            # model parallel seeds are different across ranks
            pipeline_offset = self._local_ranks.get(ParallelMode.PIPELINE, 0)

            # add seed for data parallel and tensor parallel only
            if self.is_initialized(ParallelMode.TENSOR):
                tp_rank = self.get_local_rank(ParallelMode.TENSOR)

                # 100 is only to increase the diff in seeds between pipeline stages
                tp_rank_with_offset = tp_rank + pipeline_offset * 1024
                tp_seed = seed + tp_rank_with_offset
                add_seed(ParallelMode.TENSOR, tp_seed)

            set_mode(ParallelMode.DATA)
