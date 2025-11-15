from abc import ABC, abstractmethod

import torch.distributed as dist

from nanorlhf.nanotron.distributed.mode import ParallelMode


class ProcessGroupInitializer(ABC):
    """
    The abstract class for process group initialization.

    Args:
        rank (int): The rank of current process
        world_size (int): Size of whole communication world
        data_parallel_size (int): Size of data parallelization
        pipeline_parallel_size (int): Size of pipeline parallelization
        tensor_parallel_size (int): Size of tensor parallelization
    """

    def __init__(
        self,
        rank: int,
        world_size: int,
        data_parallel_size: int,
        pipeline_parallel_size: int,
        tensor_parallel_size: int,
    ):
        self.rank = rank
        self.world_size = world_size
        self.data_parallel_size = data_parallel_size
        self.pipeline_parallel_size = pipeline_parallel_size
        self.tensor_parallel_size = tensor_parallel_size

    @abstractmethod
    def init_dist_group(self):
        """Initialize the process group."""
        raise NotImplementedError


class DataParallelGroupInitializer(ProcessGroupInitializer):
    """
    An initializer for data parallel process group.

    Discussion:
        Q. How are data parallel groups formed?
              - Ranks are divided into groups of size `data_parallel_size`.
              - Each group contains ranks that are spaced evenly across the entire world.
              - For example, with `world_size=8` and `data_parallel_size=2`, the groups would be:
                 - Group 1: Ranks [0, 4]
                 - Group 2: Ranks [1, 5]
                 - Group 3: Ranks [2, 6]
                 - Group 4: Ranks [3, 7]
              - This ensures that each data parallel group has a representative
                from each segment of the overall rank distribution.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_data_parallel_group = self.world_size // self.data_parallel_size

    def init_dist_group(self):
        """
        Initialize the data parallel process group.

        Returns:
            dict: A dictionary containing:
                - local_rank (int): The rank of the current process within its data parallel group.
                - group_world_size (int): The total number of processes in the data parallel group.
                - process_group (ProcessGroup): The process group for data parallel communication.
                - cpu_group (ProcessGroup): A CPU-based process group for data parallel communication.
                - ranks_in_group (list[int]): List of global ranks in the data parallel group.
                - mode (ParallelMode): The parallel mode, set to ParallelMode.DATA.
        """
        local_rank = None
        ranks_in_group = None
        process_group = None
        cpu_group = None
        group_world_size = None
        mode = ParallelMode.DATA

        for i in range(self.num_data_parallel_group):
            ranks = [
                i + j * self.num_data_parallel_group
                for j in range(self.data_parallel_size)
            ]
            group = dist.new_group(ranks)
            group_cpu = (
                dist.new_group(ranks, backend="gloo")
                if dist.get_backend() != "gloo"
                else group
            )

            if self.rank in ranks:
                local_rank = ranks.index(self.rank)
                group_world_size = len(ranks)
                process_group = group
                cpu_group = group_cpu
                ranks_in_group = ranks

        return {
            "local_rank": local_rank,
            "group_world_size": group_world_size,
            "process_group": process_group,
            "cpu_group": cpu_group,
            "ranks_in_group": ranks_in_group,
            "mode": mode,
        }


class ModelParallelGroupInitializer(ProcessGroupInitializer):
    """
    An initializer for model parallel process group.

    Discussion:
        Q. How are model parallel groups formed?
              - Ranks are divided into groups of size `model_parallel_size`,
                where `model_parallel_size = tensor_parallel_size * pipeline_parallel_size`.
              - Each group contains ranks that are contiguous in the overall rank ordering.
              - For example, with `world_size=8`, `tensor_parallel_size=2`, and `pipeline_parallel_size=2`,
                the groups would be:
                 - Group 1: Ranks [0, 1, 2, 3]
                 - Group 2: Ranks [4, 5, 6, 7]
              - This ensures that each model parallel group consists of ranks that are
                closely related in terms of their assigned tasks.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_parallel_size = (
            self.tensor_parallel_size * self.pipeline_parallel_size
        )
        self.num_group = self.world_size // self.model_parallel_size

    def init_dist_group(self):
        """
        Initialize the model parallel process group.

        Returns:
            dict: A dictionary containing:
                - local_rank (int): The rank of the current process within its model parallel group.
                - group_world_size (int): The total number of processes in the model parallel group.
                - process_group (ProcessGroup): The process group for model parallel communication.
                - cpu_group (ProcessGroup): A CPU-based process group for model parallel communication.
                - ranks_in_group (list[int]): List of global ranks in the model parallel group.
                - mode (ParallelMode): The parallel mode, set to ParallelMode.MODEL.
        """
        local_rank = None
        ranks_in_group = None
        process_group = None
        cpu_group = None
        group_world_size = None
        mode = ParallelMode.MODEL

        for i in range(self.num_group):
            ranks = [
                i * self.model_parallel_size + j
                for j in range(self.model_parallel_size)
            ]

            group = dist.new_group(ranks)
            group_cpu = (
                dist.new_group(ranks, backend="gloo")
                if dist.get_backend() != "gloo"
                else group
            )

            if self.rank in ranks:
                local_rank = ranks.index(self.rank)
                group_world_size = len(ranks)
                process_group = group
                cpu_group = group_cpu
                ranks_in_group = ranks

        return {
            "local_rank": local_rank,
            "group_world_size": group_world_size,
            "process_group": process_group,
            "cpu_group": cpu_group,
            "ranks_in_group": ranks_in_group,
            "mode": mode,
        }


class PipelineParallelGroupInitializer(ProcessGroupInitializer):
    """
    An initializer for pipeline parallel process group.

    Discussion:
        Q. How are pipeline parallel groups formed?
              - Ranks are first divided into data parallel groups of size `data_parallel_size`.
              - Each data parallel group is then subdivided into pipeline stages of size `pipeline_stage_size`.
              - Each pipeline stage contains ranks that are spaced evenly within the data parallel group.
              - For example, with `world_size=8`, `data_parallel_size=4`, and `pipeline_parallel_size=2`,
                the groups would be:
                 - Data Parallel Group 1: Ranks [0, 1, 2, 3]
                   - Pipeline Stage 1: Ranks [0, 2]
                   - Pipeline Stage 2: Ranks [1, 3]
                 - Data Parallel Group 2: Ranks [4, 5, 6, 7]
                   - Pipeline Stage 1: Ranks [4, 6]
                   - Pipeline Stage 2: Ranks [5, 7]
              - This ensures that each pipeline parallel group consists of ranks that are
                appropriately distributed across the data parallel groups.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_group_size = self.world_size // self.data_parallel_size
        assert (
            self.data_group_size % self.pipeline_parallel_size == 0
        ), f"Invalid config: (world={self.world_size}, dp={self.data_parallel_size}, pp={self.pipeline_parallel_size})"
        self.pipeline_stage_size = self.data_group_size // self.pipeline_parallel_size

    def init_dist_group(self):
        """
        Initialize the pipeline parallel process group.

        Returns:
            list[dict]: A list of dictionaries, each containing:
                - local_rank (int): The rank of the current process within its pipeline parallel group.
                - group_world_size (int): The total number of processes in the pipeline parallel group.
                - process_group (ProcessGroup): The process group for pipeline parallel communication.
                - cpu_group (ProcessGroup): A CPU-based process group for pipeline parallel communication.
                - ranks_in_group (list[int]): List of global ranks in the pipeline parallel group.
                - mode (ParallelMode): The parallel mode, set to ParallelMode.PIPELINE.
        """
        dist_settings = list()
        for i in range(self.data_parallel_size):
            for j in range(self.pipeline_stage_size):
                pipe_ranks = list(
                    range(
                        i * self.data_group_size + j,
                        (i + 1) * self.data_group_size,
                        self.pipeline_stage_size,
                    )
                )
                group_size = len(pipe_ranks)
                group = dist.new_group(pipe_ranks)
                group_cpu = (
                    dist.new_group(pipe_ranks, backend="gloo")
                    if dist.get_backend() != "gloo"
                    else group
                )

                if self.rank in pipe_ranks:
                    local_rank = pipe_ranks.index(self.rank)
                    group_world_size = group_size
                    process_group = group
                    cpu_group = group_cpu
                    ranks_in_group = pipe_ranks
                    dist_settings.append(
                        {
                            "local_rank": local_rank,
                            "group_world_size": group_world_size,
                            "process_group": process_group,
                            "cpu_group": cpu_group,
                            "ranks_in_group": ranks_in_group,
                            "mode": ParallelMode.PIPELINE,
                        }
                    )

        return dist_settings


class TiedEmbeddingGroupInitializer(ProcessGroupInitializer):
    """
    Make groups of [first_rank, last_rank] within each pipeline-parallel group (per data-parallel slice).
    If pipeline size == 1, the group is the single rank.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_group_size = self.world_size // self.data_parallel_size
        assert self.data_group_size % self.pipeline_parallel_size == 0
        self.pipeline_stage_size = self.data_group_size // self.pipeline_parallel_size

    def init_dist_group(self):
        dist_settings = list()
        for i in range(self.data_parallel_size):
            for j in range(self.pipeline_stage_size):
                pipe_ranks = list(
                    range(
                        i * self.data_group_size + j,
                        (i + 1) * self.data_group_size,
                        self.pipeline_stage_size,
                    )
                )
                if len(pipe_ranks) == 1:
                    embedding_ranks = pipe_ranks
                else:
                    embedding_ranks = [pipe_ranks[0], pipe_ranks[-1]]

                group = dist.new_group(embedding_ranks)
                group_cpu = (
                    dist.new_group(embedding_ranks, backend="gloo")
                    if dist.get_backend() != "gloo" else group
                )
                if self.rank in embedding_ranks:
                    local_rank = embedding_ranks.index(self.rank)
                    dist_settings.append({
                        "local_rank": local_rank,
                        "group_world_size": len(embedding_ranks),
                        "process_group": group,
                        "cpu_group": group_cpu,
                        "ranks_in_group": embedding_ranks,
                        "mode": ParallelMode.TIED_EMBEDDING,
                    })
        return dist_settings


class TensorParallelGroupInitializer(ProcessGroupInitializer):
    """
    An initializer for tensor parallel process group.

    Discussion:
        Q. How are tensor parallel groups formed?
              - Ranks are divided into groups of size `tensor_parallel_size`.
              - Each group contains ranks that are contiguous in the overall rank ordering.
              - For example, with `world_size=8` and `tensor_parallel_size=2`, the groups would be:
                 - Group 1: Ranks [0, 1]
                 - Group 2: Ranks [2, 3]
                 - Group 3: Ranks [4, 5]
                 - Group 4: Ranks [6, 7]
              - This ensures that each tensor parallel group consists of ranks that are
                closely related in terms of their assigned tasks.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_tensor_parallel_group = self.world_size // self.tensor_parallel_size

    def init_dist_group(self):
        """
        Initialize the tensor parallel process group.

        Returns:
            dict: A dictionary containing:
                - local_rank (int): The rank of the current process within its tensor parallel group.
                - group_world_size (int): The total number of processes in the tensor parallel group.
                - process_group (ProcessGroup): The process group for tensor parallel communication.
                - cpu_group (ProcessGroup): A CPU-based process group for tensor parallel communication.
                - ranks_in_group (list[int]): List of global ranks in the tensor parallel group.
                - mode (ParallelMode): The parallel mode, set to ParallelMode.TENSOR.
        """
        local_rank = None
        ranks_in_group = None
        process_group = None
        cpu_group = None
        group_world_size = None
        mode = ParallelMode.TENSOR

        for i in range(self.num_tensor_parallel_group):
            ranks = [
                i * self.tensor_parallel_size + j
                for j in range(self.tensor_parallel_size)
            ]
            group = dist.new_group(ranks)
            group_cpu = (
                dist.new_group(ranks, backend="gloo")
                if dist.get_backend() != "gloo"
                else group
            )

            if self.rank in ranks:
                local_rank = ranks.index(self.rank)
                group_world_size = len(ranks)
                process_group = group
                cpu_group = group_cpu
                ranks_in_group = ranks

        return {
            "local_rank": local_rank,
            "group_world_size": group_world_size,
            "process_group": process_group,
            "cpu_group": cpu_group,
            "ranks_in_group": ranks_in_group,
            "mode": mode,
        }
