from argparse import ArgumentParser

from torch.utils.data import DataLoader
from tqdm import tqdm

from nanorlhf import nanoray
from nanorlhf.nanoray import Bundle, PlacementStrategy, NANORAY_BASE_PORT
from nanorlhf.nanoverl.configs.rl_config import RLConfig
from nanorlhf.nanoverl.dataset.rl_dataset import RLDataset
from nanorlhf.nanoverl.reward.reward_manager import RewardManager
from nanorlhf.nanoverl.trainer.base_trainer import BaseTrainer
from nanorlhf.nanoverl.trainer.worker.actor_critic_ref_worker import ActorCriticRefWorker
from nanorlhf.nanoverl.trainer.worker.rollout_worker import RolloutWorker
from nanorlhf.nanoverl.trainer.worker_group.actor_critic_ref_worker_group import ActorCriticRefWorkerGroup
from nanorlhf.nanoverl.trainer.worker_group.rollout_worker_group import RolloutWorkerGroup
from nanorlhf.nanoverl.utils.packing_utils import packed_collate_fn_for_rl


class RLTrainer(BaseTrainer):
    """
    Reinforcement Learning Trainer that orchestrates the training process using actor-critic architecture.

    Args:
        config (str): Path to the RL configuration YAML file.
    """
    def __init__(self, config: str):
        super().__init__(config=RLConfig.from_yaml(config))
        self.train_dataloader = self.load_dataloader(self.config, split="train")
        self.valid_dataloader = self.load_dataloader(self.config, split="valid")
        self.total_steps = self.config.training.total_epochs * len(self.train_dataloader)

        self.actor_world_size = (
            self.config.actor.data_parallel_size
            * self.config.actor.tensor_parallel_size
            * self.config.actor.pipeline_parallel_size
        )
        self.rollout_world_size = self.config.rollout.data_parallel_size * self.config.rollout.tensor_parallel_size
        self.global_world_size = self.actor_world_size + self.rollout_world_size
        self.init_ray(self.config)
        self.actor_pg, self.rollout_pg = self.create_placement_groups()
        self.reward_manager = RewardManager(self.config)

        actor_workers, rollout_workers = self.spawn_workers(self.config, self.total_steps)
        self.actor_critic_ref_worker_group = ActorCriticRefWorkerGroup(self.config, actor_workers)
        self.rollout_worker_group = RolloutWorkerGroup(self.config, rollout_workers)

    def load_dataloader(self, config, split: str):
        """
        Load the dataloader for the specified split.

        Args:
            config: Configuration object containing data settings.
            split (str): The data split to load ('train' or 'valid').
        """
        assert split in ["train", "valid"], "split must be 'train' or 'valid'"
        file_path = config.data.train_data if split == "train" else config.data.valid_data
        dataset = RLDataset(file_path, max_prompt_length=config.rollout.max_prompt_len)

        if split == "train":
            batch_size = config.data.train_batch_size
        else:
            batch_size = config.data.valid_batch_size

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
            num_workers=config.data.num_workers,
            collate_fn=packed_collate_fn_for_rl,
        )

    def init_ray(self, config):
        """
        Initialize the nanoray distributed framework with the specified configuration.

        Args:
            config: Configuration object containing actor settings.
        """
        nodes = {}
        for rank in range(self.actor_world_size):
            nodes[f"actor-global_rank={rank}"] = nanoray.NodeConfig(
                cpus=1.0,
                gpus=1.0,
                rpc=True,
                host=config.actor.host,
                port=NANORAY_BASE_PORT + rank,
            )
        for rank in range(self.rollout_world_size):
            rank = rank + self.actor_world_size
            nodes[f"rollout-global_rank={rank}"] = nanoray.NodeConfig(
                cpus=1.0,
                gpus=1.0,
                rpc=True,
                host=config.actor.host,
                port=NANORAY_BASE_PORT + rank,
            )

        session = nanoray.init(nodes, default_node_id=f"actor-global_rank=0")
        node_ids = list(session.workers.keys())
        if len(node_ids) < self.global_world_size:
            raise RuntimeError(
                "`nanoray` was initialized with fewer nodes than `global_world_size`; "
                "please provide at least one NodeConfig per global rank."
            )

    def create_placement_groups(self):
        """
        Create placement groups for actor and rollout workers.

        Returns:
            Tuple containing actor and rollout placement groups.
        """
        actor_pg = nanoray.create_placement_group(
            bundles=[Bundle(cpus=1.0, gpus=1.0) for _ in range(self.actor_world_size)],
            strategy=PlacementStrategy.SPREAD,
        )
        rollout_pg = nanoray.create_placement_group(
            bundles=[Bundle(cpus=1.0, gpus=1.0) for _ in range(self.rollout_world_size)],
            strategy=PlacementStrategy.SPREAD,
        )
        return actor_pg, rollout_pg

    def spawn_workers(self, config, total_steps: int):
        """
        Spawn actor and rollout workers.

        Args:
            config: Configuration object containing actor and rollout settings.
            total_steps (int): Total number of training steps for scheduler initialization.

        Returns:
            Tuple containing lists of actor and rollout worker references.
        """
        actor_refs = []
        for actor_local_rank in range(self.actor_world_size):
            actor_ref = ActorCriticRefWorker.options(
                placement_group=self.actor_pg, bundle_index=actor_local_rank
            ).remote(config=config, rank=actor_local_rank, total_steps=total_steps, blocking=False)
            actor_refs.append(actor_ref)

        rollout_refs = []
        for rollout_dp_rank in range(config.rollout.data_parallel_size):
            for rollout_tp_rank in range(config.rollout.tensor_parallel_size):
                rollout_local_rank = rollout_dp_rank * config.rollout.tensor_parallel_size + rollout_tp_rank
                global_rank = self.actor_world_size + rollout_local_rank
                rollout_ref = RolloutWorker.options(
                    placement_group=self.rollout_pg, bundle_index=rollout_local_rank
                ).remote(config=config, rank=global_rank, blocking=False)
                rollout_refs.append(rollout_ref)

        models = nanoray.get(actor_refs + rollout_refs)

        rollouts = []
        for rollout_dp_rank in range(config.rollout.data_parallel_size):
            tensor_parallel_workers = []
            for rollout_tp_rank in range(config.rollout.tensor_parallel_size):
                rollout_local_rank = rollout_dp_rank * config.rollout.tensor_parallel_size + rollout_tp_rank
                global_rank = self.actor_world_size + rollout_local_rank
                tensor_parallel_workers.append(models[global_rank])
            rollouts.append(tensor_parallel_workers)

        actors = models[: self.actor_world_size]
        return actors, rollouts

    def sync_actor_to_rollout(self):
        """
        Synchronize parameters from the actor model to the rollout model.

        Returns:
            List of synchronization information from each worker.
        """
        actor_object_refs = self.actor_critic_ref_worker_group.sync_actor_to_rollout()
        rollout_object_refs = self.rollout_worker_group.sync_actor_to_rollout()
        return nanoray.get(actor_object_refs + rollout_object_refs)

    def create_continuous_iterator(self):
        """
        Create a continuous iterator over the training dataloader.

        Yields:
            Tuple containing the batch, progress bar, and epoch index.
        """
        for epoch in range(self.config.training.total_epochs):
            pbar = tqdm(
                self.train_dataloader,
                desc=f"Epoch {epoch + 1}/{self.config.training.total_epochs}",
                dynamic_ncols=True,
            )
            for batch in pbar:
                yield batch, pbar, epoch

    def async_generate_next_batch(self, iterator):
        """
        Asynchronously generate the next batch using the rollout worker group.

        Args:
            iterator: An iterator that yields batches.

        Returns:
            Tuple containing the rollout future, batch, progress bar, and epoch index.
        """
        try:
            batch, pbar, epoch = next(iterator)
        except StopIteration:
            return None

        pbar.set_postfix(global_step=self.global_step, status="synchronizing_actor_to_rollout")
        sync_info = self.sync_actor_to_rollout()

        self.log(
            {f"sync/num_tensors_synced_rank_{n}": sync_info[n]["num_tensors_synced"] for n in range(len(sync_info))}
        )

        rollout_future = self.rollout_worker_group.async_generate(batch)
        return rollout_future, batch, pbar, epoch

    def fit(self):
        """
        Start the training loop for reinforcement learning.
        """
        continuous_iterator = self.create_continuous_iterator()
        batch_data_future = self.async_generate_next_batch(continuous_iterator)
        mean_reward = 0

        while batch_data_future is not None:
            self.global_step += 1
            rollout_future, batch, pbar, epoch = batch_data_future

            pbar.set_postfix(global_step=self.global_step, status="waiting_for_rollout", reward=mean_reward)
            total_tokens_repacked, prompt_tokens_unpacked, response_tokens_unpacked = rollout_future.result()

            if self.global_step % self.config.training.test_freq != 0:
                batch_data_future = self.async_generate_next_batch(continuous_iterator)

            pbar.set_postfix(global_step=self.global_step, status="computing_rewards", reward=mean_reward)
            reward_scores = self.reward_manager.compute_score(prompt_tokens_unpacked, response_tokens_unpacked)
            mean_reward = sum(reward_scores) / len(reward_scores)

            pbar.set_postfix(global_step=self.global_step, status="making_experiences", reward=mean_reward)
            experience_info = self.actor_critic_ref_worker_group.make_experience(total_tokens_repacked, reward_scores)

            pbar.set_postfix(global_step=self.global_step, status="training_policies", reward=mean_reward)
            train_step_output = self.actor_critic_ref_worker_group.step()

            self.log(
                {
                    "train/epoch": epoch,
                    "train/global_step": self.global_step,
                    "train/reward": mean_reward,
                    **{f"train/{k}": v for k, v in experience_info.items()},
                    **{f"train/{k}": v for k, v in train_step_output.items()},
                }
            )

            if self.global_step % self.config.training.test_freq == 0:
                valid_reward_scores = []
                pbar = tqdm(self.valid_dataloader, desc="Validation", dynamic_ncols=True)
                for valid_batch in pbar:
                    pbar.set_postfix(global_step=self.global_step, status="generating_validation_responses")
                    _, prompt_tokens_unpacked, response_tokens_unpacked = self.rollout_worker_group.generate(
                        valid_batch
                    )

                    pbar.set_postfix(global_step=self.global_step, status="computing_validation_rewards")
                    reward_scores = self.reward_manager.compute_score(prompt_tokens_unpacked, response_tokens_unpacked)
                    valid_reward_scores.extend(reward_scores)

                mean_valid_reward = sum(valid_reward_scores) / len(valid_reward_scores)
                batch_data_future = self.async_generate_next_batch(continuous_iterator)

                self.log(
                    {
                        "valid/reward": mean_valid_reward,
                        "valid/epoch": epoch,
                        "valid/global_step": self.global_step,
                    }
                )
                print(f"\n[Validation] step {self.global_step}, reward: {mean_valid_reward:.6f}")

            if self.global_step % self.config.training.save_freq == 0:
                self.actor_critic_ref_worker_group.save_parallelized(self.global_step)

        self.actor_critic_ref_worker_group.save_parallelized(self.global_step)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the RL config yaml file.")
    trainer = RLTrainer(parser.parse_args().config)
    trainer.fit()
