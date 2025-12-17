from argparse import ArgumentParser
from dataclasses import asdict
from typing import List

import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm

from nanorlhf import nanoray
from nanorlhf.nanotron import MPU
from nanorlhf.nanoverl.configs.rl_config import RLConfig
from nanorlhf.nanoverl.dataset.rl_dataset import RLDataset
from nanorlhf.nanoverl.utils.packing_utils import packed_collate_fn_for_rl, split_packed_batch
from nanorlhf.nanoverl.utils.rollout_utils import RolloutManager
from nanorlhf.nanoverl.worker.actor_critic_ref_worker import ActorCriticRefWorker
from nanorlhf.nanoverl.worker.rollout_worker import RolloutWorker


class RLTrainer:
    def __init__(self, config: str):
        self.config = RLConfig.from_yaml(config)
        self.train_dataloader = self.load_dataloader(self.config, split="train")
        self.valid_dataloader = self.load_dataloader(self.config, split="valid")
        self.total_steps = self.config.training.total_epochs * len(self.train_dataloader)
        self.global_step = 0
        self.actor_world_size = (
            self.config.actor.data_parallel_size
            * self.config.actor.tensor_parallel_size
            * self.config.actor.pipeline_parallel_size
        )
        self.rollout_world_size = self.config.rollout.data_parallel_size * self.config.rollout.tensor_parallel_size
        self.global_world_size = self.actor_world_size + self.rollout_world_size

        self.actor_data_parallel_ranks = []
        for actor_local_rank in range(self.actor_world_size):
            dp_rank, _, _ = MPU.get_local_ranks_from_global_rank(
                actor_local_rank,
                self.config.actor.data_parallel_size,
                self.config.actor.tensor_parallel_size,
                self.config.actor.pipeline_parallel_size,
            )
            self.actor_data_parallel_ranks.append(dp_rank)

        self.node_ids = self.init_ray(self.config)
        self.actors, rollouts = self.spawn_all_workers(self.config, self.node_ids, self.total_steps)
        self.rollout_manager = RolloutManager(self.config, rollouts)
        self.maybe_init_logger()

    def maybe_init_logger(self):
        if not self.config.training.wandb:
            return

        wandb.init(
            project=self.config.training.project_name,
            name=self.config.training.experiment_name,
            config=asdict(self.config),
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

    def spawn_all_workers(self, config, node_ids: List[str], total_steps: int):
        actor_world_size = (
            config.actor.data_parallel_size * config.actor.tensor_parallel_size * config.actor.pipeline_parallel_size
        )
        actor_refs = []
        for actor_local_rank in range(actor_world_size):
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
                global_rank = actor_world_size + rollout_local_rank
                node_id = node_ids[global_rank % len(node_ids)]
                rollout_ref = RolloutWorker.options(pinned_node_id=node_id).remote(
                    config=config, rank=global_rank, blocking=False
                )
                rollout_refs.append(rollout_ref)

        models = nanoray.get(actor_refs + rollout_refs)

        rollouts: List[List[RolloutWorker]] = []
        for rollout_dp_rank in range(config.rollout.data_parallel_size):
            tensor_parallel_workers: List[RolloutWorker] = []
            for rollout_tp_rank in range(config.rollout.tensor_parallel_size):
                rollout_local_rank = rollout_dp_rank * config.rollout.tensor_parallel_size + rollout_tp_rank
                global_rank = actor_world_size + rollout_local_rank
                tensor_parallel_worker = models[global_rank]
                tensor_parallel_workers.append(tensor_parallel_worker)
            rollouts.append(tensor_parallel_workers)

        actors = models[:actor_world_size]
        return actors, rollouts

    def load_dataloader(self, config, split: str):
        assert split in ["train", "valid"], "split must be 'train' or 'valid'"
        file_path = config.data.train_data if split == "train" else config.data.valid_data
        dataset = RLDataset(file_path)

        if split == "train":
            batch_size = config.data.train_batch_size
            shuffle = drop_last = True
        else:
            batch_size = config.data.valid_batch_size
            shuffle = drop_last = False

            if config.actor.pipeline_parallel_size > 1:
                valid_micro_batch_size = config.data.valid_micro_batch_size
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

    def make_experience(self, rollout_outputs):
        per_data_parallel_batches = []
        for data_parallel_rank in range(self.config.actor.data_parallel_size):
            data_parallel_batch = split_packed_batch(
                rollout_outputs, chunk_idx=data_parallel_rank, num_chunks=self.config.actor.data_parallel_size
            )
            per_data_parallel_batches.append(data_parallel_batch)

        object_refs = []
        for actor_local_rank in range(self.actor_world_size):
            data_parallel_rank = self.actor_data_parallel_ranks[actor_local_rank]
            rollout_outputs = per_data_parallel_batches[data_parallel_rank]
            object_ref = self.actors[actor_local_rank].make_experience.remote(rollout_outputs, blocking=False)
            object_refs.append(object_ref)
        return nanoray.get(object_refs)[0]

    def fit(self):
        for epoch in range(self.config.training.total_epochs):
            pbar = tqdm(
                self.train_dataloader,
                desc=f"Epoch {epoch + 1}/{self.config.training.total_epochs}",
                dynamic_ncols=True,
            )
            for batch in pbar:
                self.global_step += 1
                rollout_outputs = self.rollout_manager.rollout(batch)
                experience_info = self.make_experience(rollout_outputs)
                print(experience_info)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the RL config yaml file.")
    trainer = RLTrainer(parser.parse_args().config)
    trainer.fit()
