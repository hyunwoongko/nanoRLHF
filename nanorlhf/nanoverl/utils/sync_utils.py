import torch
import torch.distributed as dist

from nanorlhf.nanotron import ParallelMode
from nanorlhf.nanotron.core.pp.utils import partition_layers
from nanorlhf.nanotron.utils.wrapping import tag_module


class ParameterSyncManager:
    """
    Manages synchronization of model parameters between actor and rollout models

    Args:
        model: The model whose parameters are to be synchronized.
        mpu: Model Parallel Unit managing parallelism.
        config: Configuration object containing parallelism settings.
        is_rollout (bool): Flag indicating if the current model is a rollout model.
    """
    def __init__(self, model, mpu, config, is_rollout: bool):
        self.model = model
        self.mpu = mpu
        self.is_rollout = is_rollout

        self.actor_data_parallel_size = config.actor.data_parallel_size
        self.actor_tensor_parallel_size = config.actor.tensor_parallel_size
        self.actor_pipeline_parallel_size = config.actor.pipeline_parallel_size
        self.rollout_data_parallel_size = config.rollout.data_parallel_size
        self.rollout_tensor_parallel_size = config.rollout.tensor_parallel_size

        if self.actor_tensor_parallel_size != self.rollout_tensor_parallel_size:
            raise RuntimeError(
                "Actor and rollout tensor parallel size must match, but got "
                f"actor_tensor_parallel_size={self.actor_tensor_parallel_size}, "
                f"rollout_tensor_parallel_size={self.rollout_tensor_parallel_size}"
            )

        self.sync_groups = {}
        self.build_actor_rollout_sync_groups()

        if self.is_rollout and self.actor_pipeline_parallel_size > 1:
            self.tag_pipeline_ranks_for_rollout()

        self.tensor_parallel_mode = ParallelMode.ROLLOUT_TENSOR if self.is_rollout else ParallelMode.TENSOR
        self.stage_objects = self.scan_stage_objects()

    def get_mp_plan(self):
        """
        Retrieve the model parallelization plan from the model.

        Returns:
            The model parallelization plan.
        """
        if hasattr(self.model, "__nanotron__mp_plan__"):
            return self.model.__nanotron__mp_plan__
        raise RuntimeError("mp_plan not found. parallelize() must have been called.")

    def build_actor_rollout_sync_groups(self):
        """
        Build synchronization groups between actor and rollout models.
        """
        for tensor_parallel_rank in range(self.actor_tensor_parallel_size):
            for pipeline_parallel_rank in range(self.actor_pipeline_parallel_size):
                actor_sender_global_rank = (
                    pipeline_parallel_rank * self.actor_tensor_parallel_size + tensor_parallel_rank
                )
                rollout_global_ranks = [
                    self.mpu.actor_world_size
                    + rollout_data_parallel_rank * self.rollout_tensor_parallel_size
                    + tensor_parallel_rank
                    for rollout_data_parallel_rank in range(self.rollout_data_parallel_size)
                ]
                ranks = [actor_sender_global_rank] + rollout_global_ranks
                self.sync_groups[(tensor_parallel_rank, pipeline_parallel_rank)] = dist.new_group(ranks=ranks)

    def get_tag_rank(self, tensor: torch.Tensor, mode: ParallelMode):
        """
        Get the parallel rank tag for a tensor.

        Args:
            tensor (torch.Tensor): The tensor to check.
            mode (ParallelMode): The parallel mode to check for.

        Returns:
            The parallel rank tag for the tensor in the specified mode.
        """
        mapping = getattr(tensor, "__nanotron_parallel__", None)
        if not mapping:
            return 0
        return mapping.get(mode, 0)

    def tag_pipeline_ranks_for_rollout(self):
        """
        Tag modules with pipeline parallel ranks for rollout model.
        """
        mp_plan = self.get_mp_plan()
        module_list = mp_plan.main_module_list

        partitions = partition_layers(self.actor_pipeline_parallel_size, len(module_list))
        embeddings, pre_modules, post_modules, heads = mp_plan.extract_modules()

        for pipeline_parallel_rank in range(self.actor_pipeline_parallel_size):
            local_start = partitions[pipeline_parallel_rank]
            local_end = partitions[pipeline_parallel_rank + 1]

            if pipeline_parallel_rank == 0:
                for module in embeddings + pre_modules:
                    tag_module(module, ParallelMode.PIPELINE, pipeline_parallel_rank)

            for module in module_list[local_start:local_end]:
                tag_module(module, ParallelMode.PIPELINE, pipeline_parallel_rank)

            if pipeline_parallel_rank == (self.actor_pipeline_parallel_size - 1):
                for module in post_modules + heads:
                    tag_module(module, ParallelMode.PIPELINE, pipeline_parallel_rank)

    def modules_for_pipeline_stage(self, pipeline_parallel_rank: int):
        """
        Get the list of modules for a specific pipeline stage.

        Args:
            pipeline_parallel_rank (int): The pipeline parallel rank of the stage.

        Returns:
            List of modules for the specified pipeline stage.
        """
        mp_plan = self.get_mp_plan()
        module_list = mp_plan.main_module_list
        partitions = partition_layers(self.actor_pipeline_parallel_size, len(module_list))
        embeddings, pre_modules, post_modules, heads = mp_plan.extract_modules()

        local_start = partitions[pipeline_parallel_rank]
        local_end = partitions[pipeline_parallel_rank + 1]

        modules = []
        if pipeline_parallel_rank == 0:
            modules.extend(embeddings)
            modules.extend(pre_modules)

        modules.extend(module_list[local_start:local_end])

        if pipeline_parallel_rank == (self.actor_pipeline_parallel_size - 1):
            modules.extend(post_modules)
            modules.extend(heads)

        return modules

    def scan_stage_objects(self):
        """
        Scan and collect tensors for each pipeline stage and tensor parallel rank.

        Returns:
            A dictionary mapping (tensor_parallel_rank, pipeline_parallel_rank) to a list of tensors.
        """
        stage_objects = {
            (tensor_parallel_rank, pipeline_parallel_rank): []
            for tensor_parallel_rank in range(self.actor_tensor_parallel_size)
            for pipeline_parallel_rank in range(self.actor_pipeline_parallel_size)
        }

        for pipeline_parallel_rank in range(self.actor_pipeline_parallel_size):
            modules = self.modules_for_pipeline_stage(pipeline_parallel_rank)

            ordered = []
            seen = set()

            for module in modules:
                for _, param in sorted(module.named_parameters(recurse=True), key=lambda x: x[0]):
                    if param is None:
                        continue
                    oid = id(param)
                    if oid in seen:
                        continue
                    seen.add(oid)
                    ordered.append(param)

                for _, buf in sorted(module.named_buffers(recurse=True), key=lambda x: x[0]):
                    if buf is None:
                        continue
                    oid = id(buf)
                    if oid in seen:
                        continue
                    seen.add(oid)
                    ordered.append(buf)

            for obj in ordered:
                tensor_parallel_rank = self.get_tag_rank(obj, self.tensor_parallel_mode)
                key = (tensor_parallel_rank, pipeline_parallel_rank)
                if key in stage_objects:
                    stage_objects[key].append(obj.data if hasattr(obj, "data") else obj)

        return stage_objects

    def broadcast(self, tensor: torch.Tensor, src: int, group: dist.ProcessGroup):
        """
        Broadcast a tensor from the source rank to all other ranks in the group.

        Args:
            tensor (torch.Tensor): The tensor to broadcast.
            src (int): The source rank from which to broadcast.
            group (dist.ProcessGroup): The process group for broadcasting.
        """
        # make a tensor bfloat16 because policy model is fp32 and rollout model is bfloat16.
        temp = tensor.bfloat16().contiguous()
        dist.broadcast(temp, src=src, group=group)

    def sync_actor_to_rollout(self):
        """
        Synchronize parameters from the actor model to the rollout model.

        Returns:
            A dictionary containing synchronization statistics.
        """
        num_tensors_synced = 0

        if self.is_rollout:
            current_data_parallel_rank = self.mpu.get_local_rank(ParallelMode.ROLLOUT_DATA)
            current_tensor_parallel_rank = self.mpu.get_local_rank(ParallelMode.ROLLOUT_TENSOR)
            current_pipeline_parallel_rank = 0
        else:
            current_data_parallel_rank = self.mpu.get_local_rank(ParallelMode.DATA)
            current_tensor_parallel_rank = self.mpu.get_local_rank(ParallelMode.TENSOR)
            current_pipeline_parallel_rank = self.mpu.get_local_rank(ParallelMode.PIPELINE)

        for tensor_parallel_rank in range(self.actor_tensor_parallel_size):
            for pipeline_parallel_rank in range(self.actor_pipeline_parallel_size):
                if self.is_rollout:
                    if tensor_parallel_rank != current_tensor_parallel_rank:
                        continue
                else:
                    if current_data_parallel_rank != 0:
                        continue
                    if tensor_parallel_rank != current_tensor_parallel_rank:
                        continue
                    if pipeline_parallel_rank != current_pipeline_parallel_rank:
                        continue

                group = self.sync_groups[(tensor_parallel_rank, pipeline_parallel_rank)]
                src = pipeline_parallel_rank * self.actor_tensor_parallel_size + tensor_parallel_rank
                tensors = self.stage_objects[(tensor_parallel_rank, pipeline_parallel_rank)]

                count = torch.tensor([len(tensors)], device=torch.cuda.current_device(), dtype=torch.int32)
                self.broadcast(count, src=src, group=group)
                count = int(count.item())

                if count != len(tensors):
                    raise RuntimeError(f"sync tensor count mismatch: expected={count}, got={len(tensors)}")

                for tensor in tensors:
                    self.broadcast(tensor, src=src, group=group)

                num_tensors_synced += len(tensors)

        torch.cuda.synchronize()

        return {
            "rank": dist.get_rank(),
            "is_rollout": self.is_rollout,
            "num_tensors_synced": num_tensors_synced,
        }
