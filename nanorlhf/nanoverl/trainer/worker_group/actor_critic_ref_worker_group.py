from nanorlhf import nanoray
from nanorlhf.nanotron import MPU
from nanorlhf.nanoverl.utils.packing_utils import split_packed_batch


class ActorCriticRefWorkerGroup:

    def __init__(self, config, workers):
        self.config = config
        self.workers = workers
        self.actor_world_size = (
            self.config.actor.data_parallel_size
            * self.config.actor.tensor_parallel_size
            * self.config.actor.pipeline_parallel_size
        )

        self.actor_data_parallel_ranks = []
        for actor_local_rank in range(self.actor_world_size):
            dp_rank, _, _ = MPU.get_local_ranks_from_global_rank(
                actor_local_rank,
                self.config.actor.data_parallel_size,
                self.config.actor.tensor_parallel_size,
                self.config.actor.pipeline_parallel_size,
            )
            self.actor_data_parallel_ranks.append(dp_rank)

    def make_experience(self, total_tokens_repacked, reward_scores):
        per_data_parallel_batches = []
        for data_parallel_rank in range(self.config.actor.data_parallel_size):
            data_parallel_batch = split_packed_batch(
                total_tokens_repacked, chunk_idx=data_parallel_rank, num_chunks=self.config.actor.data_parallel_size
            )
            per_data_parallel_batches.append(data_parallel_batch)

        reward_scores_per_data_parallel = []
        offset = 0
        for data_parallel_batch in per_data_parallel_batches:
            n_seq = int(
                ((data_parallel_batch["position_ids"] == 0) & (data_parallel_batch["loss_mask"] == 0)).sum().item()
            )
            reward_scores_per_data_parallel.append(reward_scores[offset : offset + n_seq])
            offset += n_seq
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
