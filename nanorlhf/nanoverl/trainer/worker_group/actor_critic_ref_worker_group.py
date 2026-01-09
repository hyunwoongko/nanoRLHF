from nanorlhf import nanoray
from nanorlhf.nanotron import MPU
from nanorlhf.nanoverl.utils.packing_utils import split_packed_batch


class ActorCriticRefWorkerGroup:
    """
    Worker group that manages actor and critic reference workers for reinforcement learning.

    Args:
        config: Configuration object containing actor settings.
        workers: List of remote worker instances.
    """

    def __init__(self, config, workers):
        self.config = config
        self.workers = workers
        self.actor_world_size = (
            self.config.actor.data_parallel_size
            * self.config.actor.tensor_parallel_size
            * self.config.actor.pipeline_parallel_size
        )

        self.actor_data_parallel_ranks = []
        self.actor_sender_local_ranks = []
        for actor_local_rank in range(self.actor_world_size):
            dp_rank, _, _ = MPU.get_local_ranks_from_global_rank(
                actor_local_rank,
                self.config.actor.data_parallel_size,
                self.config.actor.tensor_parallel_size,
                self.config.actor.pipeline_parallel_size,
            )
            self.actor_data_parallel_ranks.append(dp_rank)
            if dp_rank == 0:
                self.actor_sender_local_ranks.append(actor_local_rank)

    def make_experience(self, total_tokens_repacked, reward_scores):
        """
        Create experience data on actor workers.

        Args:
            total_tokens_repacked: The repacked total tokens batch.
            reward_scores: The computed reward scores.

        Returns:
            The experience data created by the actor workers.
        """
        per_data_parallel_batches = []
        for data_parallel_rank in range(self.config.actor.data_parallel_size):
            data_parallel_batch = split_packed_batch(
                total_tokens_repacked, chunk_idx=data_parallel_rank, num_chunks=self.config.actor.data_parallel_size
            )
            per_data_parallel_batches.append(data_parallel_batch)

        reward_scores_per_data_parallel = []
        offset = 0
        for data_parallel_batch in per_data_parallel_batches:
            num_sequences = int((data_parallel_batch["position_ids"] == 0).sum().item())
            reward_scores_per_data_parallel.append(reward_scores[offset : offset + num_sequences])
            offset += num_sequences
        assert offset == len(reward_scores), f"reward_scores len mismatch: used={offset}, total={len(reward_scores)}"

        object_refs = []
        for actor_local_rank in range(self.actor_world_size):
            data_parallel_rank = self.actor_data_parallel_ranks[actor_local_rank]
            total_tokens_repacked = per_data_parallel_batches[data_parallel_rank]
            reward_scores = reward_scores_per_data_parallel[data_parallel_rank]
            object_ref = self.workers[actor_local_rank].make_experience.remote(
                total_tokens_repacked, reward_scores, blocking=False
            )
            object_refs.append(object_ref)
        return nanoray.get(object_refs)[0]

    def step(self):
        """
        Perform a training step on actor workers.

        Returns:
            The result of the training step from the actor workers.
        """
        object_refs = []
        for actor_local_rank in range(self.actor_world_size):
            object_ref = self.workers[actor_local_rank].step.remote(blocking=False)
            object_refs.append(object_ref)
        return nanoray.get(object_refs)[0]

    def save_parallelized(self, global_step):
        """
        Save the parallelized model checkpoints on all workers.

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

    def sync_actor_to_rollout(self):
        """
        Synchronize parameters from actor models to rollout models.

        Returns:
            List of object references for the synchronization tasks.
        """
        object_refs = []
        for actor_local_rank_dp0 in self.actor_sender_local_ranks:
            object_ref = self.workers[actor_local_rank_dp0].sync_actor_to_rollout.remote(blocking=False)
            object_refs.append(object_ref)
        return object_refs
