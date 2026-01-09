from nanorlhf import nanoray
from nanorlhf.nanotron import MPU
from nanorlhf.nanoverl.utils.packing_utils import split_packed_batch


class SFTWorkerGroup:
    """
    Worker group that manages supervised fine-tuning (SFT) workers.

    Args:
        config: Configuration object containing model settings.
        workers: List of remote worker instances.
    """

    def __init__(self, config, workers):
        self.config = config
        self.workers = workers

        self.global_world_size = (
            self.config.model.data_parallel_size
            * self.config.model.tensor_parallel_size
            * self.config.model.pipeline_parallel_size
        )

        self.data_parallel_ranks = []
        for global_rank in range(self.global_world_size):
            dp_rank, _, _ = MPU.get_local_ranks_from_global_rank(
                global_rank,
                self.config.model.data_parallel_size,
                self.config.model.tensor_parallel_size,
                self.config.model.pipeline_parallel_size,
            )
            self.data_parallel_ranks.append(dp_rank)

    def step(self, input_batch, train: bool):
        """
        Perform a training or evaluation step on the SFT workers.

        Args:
            input_batch: The input batch to be processed.
            train (bool): Whether to perform a training step (True) or evaluation step (False).

        Returns:
            The output from the first worker after processing the input batch.
        """
        per_data_parallel_batches = []
        for data_parallel_rank in range(self.config.model.data_parallel_size):
            data_parallel_batch = split_packed_batch(
                input_batch, chunk_idx=data_parallel_rank, num_chunks=self.config.model.data_parallel_size
            )
            per_data_parallel_batches.append(data_parallel_batch)

        object_refs = []
        for global_rank in range(self.global_world_size):
            data_parallel_rank = self.data_parallel_ranks[global_rank]
            input_batch = per_data_parallel_batches[data_parallel_rank]
            object_ref = self.workers[global_rank].step.remote(input_batch, train, blocking=False)
            object_refs.append(object_ref)
        return nanoray.get(object_refs)[0]

    def save_parallelized(self, global_step):
        """
        Save the state of all SFT workers in a parallelized manner.

        Args:
            global_step: The current global training step.
        """
        experiment_dir = (
            f"{self.config.training.default_local_dir}"
            f"/{self.config.training.project_name}"
            f"/{self.config.training.experiment_name}"
        )
        save_dir = f"{experiment_dir}/step_{global_step}"
        object_refs = []
        for model in self.workers:
            object_ref = model.save_parallelized.remote(save_dir, blocking=False)
            object_refs.append(object_ref)
        outputs = nanoray.get(object_refs)

        if all(out["ok"] for out in outputs):
            with open(f"{experiment_dir}/latest_checkpointed_iteration.txt", "w") as f:
                f.write(str(global_step))
            print(f"\n[SAVE] Saved checkpoint at step {global_step} to {save_dir}")
        else:
            print(f"\n[SAVE] Failed to save checkpoint at step {global_step} to {save_dir}")
