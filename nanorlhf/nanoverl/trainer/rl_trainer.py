from argparse import ArgumentParser
from typing import List

from torch.utils.data import DataLoader
from tqdm import tqdm

from nanorlhf import nanoray
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
        self.node_ids = self.init_ray(self.config)
        self.reward_manager = RewardManager(self.config)

        actor_workers, rollout_workers = self.spawn_workers(self.config, self.node_ids, self.total_steps)
        self.actor_critic_ref_worker_group = ActorCriticRefWorkerGroup(self.config, actor_workers)
        self.rollout_worker_group = RolloutWorkerGroup(self.config, rollout_workers)

    def load_dataloader(self, config, split: str):
        assert split in ["train", "valid"], "split must be 'train' or 'valid'"
        file_path = config.data.train_data if split == "train" else config.data.valid_data
        dataset = RLDataset(file_path, max_prompt_length=config.rollout.max_prompt_len)

        if split == "train":
            batch_size = config.data.train_batch_size
            shuffle = drop_last = True
        else:
            batch_size = config.data.valid_batch_size
            shuffle = drop_last = False

            if config.actor.pipeline_parallel_size > 1:
                valid_micro_batch_size = config.data.valid_micro_batch_size_per_gpu
                assert len(dataset) % valid_micro_batch_size == 0, (
                    "For pipeline parallel validation, because we don't drop the last incomplete batch, "
                    "the dataset size must be divisible by the `valid_micro_batch_size`. "
                    f"valid dataset size: {len(dataset)}, valid micro batch size: {valid_micro_batch_size}."
                )

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=config.data.num_workers,
            pin_memory=True,
            drop_last=drop_last,
            collate_fn=packed_collate_fn_for_rl,
        )

    def init_ray(self, config):
        nodes = {}
        base_port = 9200

        for rank in range(self.actor_world_size):
            nodes[f"actor-global_rank={rank}"] = nanoray.NodeConfig(
                cpus=4.0,
                gpus=1.0,
                rpc=True,
                host=config.actor.host,
                port=base_port + rank,
            )

        for rank in range(self.rollout_world_size):
            rank = rank + self.actor_world_size
            nodes[f"rollout-global_rank={rank}"] = nanoray.NodeConfig(
                cpus=4.0,
                gpus=1.0,
                rpc=True,
                host=config.actor.host,
                port=base_port + rank,
            )

        session = nanoray.init(nodes, default_node_id=f"actor-global_rank=0")
        node_ids = list(session._workers.keys())
        if len(node_ids) < self.global_world_size:
            raise RuntimeError(
                "`nanoray` was initialized with fewer nodes than `global_world_size`; "
                "please provide at least one NodeConfig per global rank."
            )

        return node_ids

    def spawn_workers(self, config, node_ids: List[str], total_steps: int):
        actor_refs = []
        for actor_local_rank in range(self.actor_world_size):
            node_id = node_ids[actor_local_rank % len(node_ids)]
            actor_ref = ActorCriticRefWorker.options(pinned_node_id=node_id).remote(
                config=config,
                rank=actor_local_rank,
                total_steps=total_steps,
                blocking=False,
            )
            actor_refs.append(actor_ref)

        rollout_refs = []
        for rollout_dp_rank in range(config.rollout.data_parallel_size):
            for rollout_tp_rank in range(config.rollout.tensor_parallel_size):
                rollout_local_rank = rollout_dp_rank * config.rollout.tensor_parallel_size + rollout_tp_rank
                global_rank = self.actor_world_size + rollout_local_rank
                node_id = node_ids[global_rank % len(node_ids)]
                rollout_ref = RolloutWorker.options(pinned_node_id=node_id).remote(
                    config=config, rank=global_rank, blocking=False
                )
                rollout_refs.append(rollout_ref)

        models = nanoray.get(actor_refs + rollout_refs)

        rollouts = []
        for rollout_dp_rank in range(config.rollout.data_parallel_size):
            tensor_parallel_workers = []
            for rollout_tp_rank in range(config.rollout.tensor_parallel_size):
                rollout_local_rank = rollout_dp_rank * config.rollout.tensor_parallel_size + rollout_tp_rank
                global_rank = self.actor_world_size + rollout_local_rank
                tensor_parallel_worker = models[global_rank]
                tensor_parallel_workers.append(tensor_parallel_worker)
            rollouts.append(tensor_parallel_workers)

        actors = models[: self.actor_world_size]
        return actors, rollouts

    def fit(self):
        for epoch in range(self.config.training.total_epochs):
            pbar = tqdm(
                self.train_dataloader,
                desc=f"Epoch {epoch + 1}/{self.config.training.total_epochs}",
                dynamic_ncols=True,
            )
            for batch in pbar:
                self.global_step += 1
                pbar.set_postfix(global_step=self.global_step, status="generating_responses")
                total_tokens_repacked, response_tokens_unpacked = self.rollout_worker_group.generate(batch)

                pbar.set_postfix(global_step=self.global_step, status="computing_rewards")
                reward_scores = self.reward_manager.compute_score(response_tokens_unpacked)
                mean_reward = sum(reward_scores) / len(reward_scores)

                pbar.set_postfix(global_step=self.global_step, status="making_experiences", reward=mean_reward)
                experience_info = self.actor_critic_ref_worker_group.make_experience(
                    total_tokens_repacked, reward_scores
                )
                pbar.set_postfix(global_step=self.global_step, status="training_policies", reward=mean_reward)
                train_step_output = self.actor_critic_ref_worker_group.step()

                self.log(
                    {
                        "train/loss": train_step_output["loss_total"],
                        "train/policy_loss": train_step_output["loss_policy"],
                        "train/value_loss": train_step_output["loss_value"],
                        "train/epoch": epoch,
                        "train/global_step": self.global_step,
                        "train/num_sequences": experience_info["num_sequences"],
                        "train/num_total_tokens": experience_info["num_total_tokens"],
                        "train/num_response_tokens": experience_info["num_response_tokens"],
                        "train/approx_kl": experience_info["approx_kl"],
                        "train/mean_logprobs": experience_info["mean_logprobs"],
                        "train/reward": mean_reward,
                        "train/actor_lr": train_step_output["actor_lr"],
                        "train/critic_lr": train_step_output["critic_lr"],
                        "train/skipped": 1 if train_step_output["skipped"] else 0,
                        "train/num_micro_batches": train_step_output["num_micro_batches"],
                        "train/num_updates": train_step_output["num_updates"],
                    }
                )

                if self.global_step % self.config.training.test_freq == 0:
                    total_valid_reward_scores = []
                    pbar = tqdm(self.valid_dataloader, desc="Validation", dynamic_ncols=True)
                    for valid_batch in pbar:
                        pbar.set_postfix(global_step=self.global_step, status="generating_validation_responses")
                        _, response_tokens_unpacked = self.rollout_worker_group.generate(valid_batch)

                        pbar.set_postfix(global_step=self.global_step, status="computing_validation_rewards")
                        valid_reward_scores = self.reward_manager.compute_score(response_tokens_unpacked)
                        total_valid_reward_scores.extend(valid_reward_scores)
                    mean_valid_reward = sum(total_valid_reward_scores) / len(total_valid_reward_scores)

                    self.log(
                        {
                            "valid/reward": mean_valid_reward,
                            "valid/epoch": epoch,
                            "valid/global_step": self.global_step,
                        }
                    )
                    print(f"\n[Validation] step {self.global_step}, reward: {mean_valid_reward:.6f}")

                if self.global_step % self.config.training.save_freq == 0:
                    # Periodic save during training
                    self.actor_critic_ref_worker_group.save_parallelized(self.global_step)

        # Final save after training
        self.actor_critic_ref_worker_group.save_parallelized(self.global_step)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the RL config yaml file.")
    trainer = RLTrainer(parser.parse_args().config)
    trainer.fit()
