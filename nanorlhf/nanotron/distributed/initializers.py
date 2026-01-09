from abc import ABC, abstractmethod

import torch.distributed as dist

from nanorlhf.nanotron.distributed.mode import ParallelMode


class ProcessGroupInitializer(ABC):
    """
    The abstract class for process group initialization.

    Args:
        rank (int): Global rank (in GLOBAL group)
        world_size (int): Global world size
        data_parallel_size (int): DP size (for this slice)
        pipeline_parallel_size (int): PP size (for this slice)
        tensor_parallel_size (int): TP size (for this slice)
        group_world_size (int): Sub-world size for this initializer (actor slice or rollout slice)
        group_rank_offset (int): Global rank offset of this slice
    """

    def __init__(
        self,
        rank: int,
        world_size: int,
        data_parallel_size: int,
        pipeline_parallel_size: int,
        tensor_parallel_size: int,
        group_world_size: int,
        group_rank_offset: int = 0,
    ):
        self.rank = rank
        self.world_size = world_size

        self.group_world_size = group_world_size
        self.group_rank_offset = group_rank_offset

        self.data_parallel_size = data_parallel_size
        self.pipeline_parallel_size = pipeline_parallel_size
        self.tensor_parallel_size = tensor_parallel_size

    def to_global_ranks(self, local_ranks):
        return [self.group_rank_offset + r for r in local_ranks]

    @abstractmethod
    def init_dist_group(self, mode: ParallelMode):
        raise NotImplementedError


class DataParallelGroupInitializer(ProcessGroupInitializer):
    """
    Data parallel process group initializer.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.group_world_size % self.data_parallel_size == 0, (
            f"Invalid DP config for slice: group_world_size={self.group_world_size}, "
            f"data_parallel_size={self.data_parallel_size}"
        )
        self.num_data_parallel_group = self.group_world_size // self.data_parallel_size

    def init_dist_group(self, mode: ParallelMode = ParallelMode.DATA):
        local_rank = None
        ranks_in_group = None
        process_group = None
        cpu_group = None
        group_world_size = None

        for i in range(self.num_data_parallel_group):
            local_ranks = [i + j * self.num_data_parallel_group for j in range(self.data_parallel_size)]
            ranks = self.to_global_ranks(local_ranks)

            group = dist.new_group(ranks)
            group_cpu = dist.new_group(ranks, backend="gloo") if dist.get_backend() != "gloo" else group

            if self.rank in ranks:
                local_rank = ranks.index(self.rank)
                group_world_size = len(ranks)
                process_group = group
                cpu_group = group_cpu
                ranks_in_group = ranks

        if local_rank is None:
            return None

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
    Pipeline parallel process group initializer.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert (
            self.group_world_size % self.data_parallel_size == 0
        ), f"Invalid PP config for slice: group_world_size={self.group_world_size}, dp={self.data_parallel_size}"
        self.data_group_size = self.group_world_size // self.data_parallel_size
        assert (
            self.data_group_size % self.pipeline_parallel_size == 0
        ), f"Invalid PP config for slice: data_group_size={self.data_group_size}, pp={self.pipeline_parallel_size}"
        self.pipeline_stage_size = self.data_group_size // self.pipeline_parallel_size

    def init_dist_group(self, mode: ParallelMode = ParallelMode.PIPELINE):
        dist_settings = []
        for i in range(self.data_parallel_size):
            for j in range(self.pipeline_stage_size):
                local_pipe_ranks = list(
                    range(
                        i * self.data_group_size + j,
                        (i + 1) * self.data_group_size,
                        self.pipeline_stage_size,
                    )
                )
                pipe_ranks = self.to_global_ranks(local_pipe_ranks)

                group = dist.new_group(pipe_ranks)
                group_cpu = dist.new_group(pipe_ranks, backend="gloo") if dist.get_backend() != "gloo" else group

                if self.rank in pipe_ranks:
                    dist_settings.append(
                        {
                            "local_rank": pipe_ranks.index(self.rank),
                            "group_world_size": len(pipe_ranks),
                            "process_group": group,
                            "cpu_group": group_cpu,
                            "ranks_in_group": pipe_ranks,
                            "mode": mode,
                        }
                    )
        return dist_settings


class TensorParallelGroupInitializer(ProcessGroupInitializer):
    """
    Tensor parallel process group initializer.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert (
            self.group_world_size % self.tensor_parallel_size == 0
        ), f"Invalid TP config for slice: group_world_size={self.group_world_size}, tp={self.tensor_parallel_size}"
        self.num_tensor_parallel_group = self.group_world_size // self.tensor_parallel_size

    def init_dist_group(self, mode: ParallelMode = ParallelMode.TENSOR):
        local_rank = None
        ranks_in_group = None
        process_group = None
        cpu_group = None
        group_world_size = None

        for i in range(self.num_tensor_parallel_group):
            local_ranks = [i * self.tensor_parallel_size + j for j in range(self.tensor_parallel_size)]
            ranks = self.to_global_ranks(local_ranks)

            group = dist.new_group(ranks)
            group_cpu = dist.new_group(ranks, backend="gloo") if dist.get_backend() != "gloo" else group

            if self.rank in ranks:
                local_rank = ranks.index(self.rank)
                group_world_size = len(ranks)
                process_group = group
                cpu_group = group_cpu
                ranks_in_group = ranks

        if local_rank is None:
            return None

        return {
            "local_rank": local_rank,
            "group_world_size": group_world_size,
            "process_group": process_group,
            "cpu_group": cpu_group,
            "ranks_in_group": ranks_in_group,
            "mode": mode,
        }


class TiedEmbeddingGroupInitializer(ProcessGroupInitializer):
    """
    Tied embedding process group initializer.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.group_world_size % self.data_parallel_size == 0
        self.data_group_size = self.group_world_size // self.data_parallel_size
        assert self.data_group_size % self.pipeline_parallel_size == 0
        self.pipeline_stage_size = self.data_group_size // self.pipeline_parallel_size

    def init_dist_group(self, mode: ParallelMode = ParallelMode.TIED_EMBEDDING):
        dist_settings = []
        for i in range(self.data_parallel_size):
            for j in range(self.pipeline_stage_size):
                local_pipe_ranks = list(
                    range(
                        i * self.data_group_size + j,
                        (i + 1) * self.data_group_size,
                        self.pipeline_stage_size,
                    )
                )
                pipe_ranks = self.to_global_ranks(local_pipe_ranks)

                if len(pipe_ranks) == 1:
                    embedding_ranks = pipe_ranks
                else:
                    embedding_ranks = [pipe_ranks[0], pipe_ranks[-1]]

                group = dist.new_group(embedding_ranks)
                group_cpu = dist.new_group(embedding_ranks, backend="gloo") if dist.get_backend() != "gloo" else group

                if self.rank in embedding_ranks:
                    dist_settings.append(
                        {
                            "local_rank": embedding_ranks.index(self.rank),
                            "group_world_size": len(embedding_ranks),
                            "process_group": group,
                            "cpu_group": group_cpu,
                            "ranks_in_group": embedding_ranks,
                            "mode": mode,
                        }
                    )
        return dist_settings
