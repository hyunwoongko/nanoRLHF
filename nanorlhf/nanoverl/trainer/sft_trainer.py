from argparse import ArgumentParser
from dataclasses import asdict
from typing import Dict, Any

import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm

from nanorlhf import nanoray
from nanorlhf.nanotron import MPU
from nanorlhf.nanoverl.dataset.sft_dataset import SFTDataset
from nanorlhf.nanoverl.utils.packing_utils import packed_collate_fn, split_packed_batch
from nanorlhf.nanoverl.configs.sft_config import SFTConfig
from nanorlhf.nanoverl.worker.actor_rollout_ref import ActorRolloutRef


class SFTTrainer:
    def __init__(self, config_path: str):
        self.config = SFTConfig.from_yaml(config_path)
        self.train_dataloader = self.load_dataloader(self.config, split="train")
        self.valid_dataloader = self.load_dataloader(self.config, split="valid")
        self.total_steps = self.config.training.total_epochs * len(self.train_dataloader)

        if self.config.data.train_batch_size % self.config.data.train_micro_batch_size != 0:
            raise ValueError(
                "`train_batch_size` must be divisible by `train_micro_batch_size`. "
                f"Got train_batch_size={self.config.data.train_batch_size} and "
                f"train_micro_batch_size={self.config.data.train_micro_batch_size}."
            )
        if self.config.data.valid_batch_size % self.config.data.valid_micro_batch_size != 0:
            raise ValueError(
                "`valid_batch_size` must be divisible by `valid_micro_batch_size`. "
                f"Got valid_batch_size={self.config.data.valid_batch_size} and "
                f"valid_micro_batch_size={self.config.data.valid_micro_batch_size}."
            )

        self.global_world_size = (
            self.config.model.data_parallel_size
            * self.config.model.tensor_parallel_size
            * self.config.model.pipeline_parallel_size
        )

        self.data_parallel_ranks = []
        for rank in range(self.global_world_size):
            dp_rank, _, _ = MPU.get_rank_from_global_rank(
                rank,
                self.config.model.data_parallel_size,
                self.config.model.tensor_parallel_size,
                self.config.model.pipeline_parallel_size,
            )
            self.data_parallel_ranks.append(dp_rank)

        self.global_step = 0
        self.node_ids = self.init_ray(self.config)
        self.models = self.create_model(self.config)
        self.maybe_init_logger()

    def maybe_init_logger(self):
        if not self.config.training.wandb:
            return

        wandb.init(
            project=self.config.training.project_name,
            name=self.config.training.experiment_name,
            config=asdict(self.config),
        )

    def log(self, metrics: Dict[str, Any]):
        if self.config.training.wandb:
            wandb.log(metrics, step=self.global_step)

    def load_dataloader(self, config, split: str):
        assert split in ["train", "valid"], "split must be 'train' or 'valid'"
        file_path = config.data.train_data if split == "train" else config.data.valid_data
        dataset = SFTDataset(file_path)

        if split == "train":
            batch_size = config.data.train_batch_size
            shuffle = drop_last = True
        else:
            batch_size = config.data.valid_batch_size
            shuffle = drop_last = False

            if config.model.pipeline_parallel_size > 1:
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
            collate_fn=packed_collate_fn,
        )

    def init_ray(self, config):
        nodes = {}
        base_port = 9200
        if self.global_world_size > 1:
            for rank in range(self.global_world_size):
                nodes[f"node-{rank + 1}"] = nanoray.NodeConfig(
                    cpus=4.0,
                    gpus=1.0,
                    rpc=True,
                    host=config.model.host,
                    port=base_port + rank,
                )
        else:
            nodes["node-1"] = nanoray.NodeConfig(
                cpus=4.0,
                gpus=1.0,
                rpc=False,
                host=config.model.host,
                port=base_port,
            )

        session = nanoray.init(nodes, default_node_id="node-1")
        node_ids = list(session._workers.keys())
        if len(node_ids) < self.global_world_size:
            raise RuntimeError(
                "`nanoray` was initialized with fewer nodes than `global_world_size`; "
                "please provide at least one NodeConfig per global rank."
            )

        return node_ids

    def create_model(self, config):
        object_refs = []
        for global_rank in range(self.global_world_size):
            node_id = self.node_ids[global_rank % len(self.node_ids)]
            object_ref = ActorRolloutRef.options(pinned_node_id=node_id).remote(
                config,
                rank=global_rank,
                total_steps=self.total_steps,
                initialize_ref=False,
                initialize_rollout=False,
                blocking=False,
            )
            object_refs.append(object_ref)
        return nanoray.get(object_refs)

    def step(self, input_batch, train: bool):
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
            object_ref = self.models[global_rank].step.remote(input_batch, train, blocking=False)
            object_refs.append(object_ref)
        return nanoray.get(object_refs)[0]

    def save_parallelized(self):
        save_dir = f"{self.config.training.default_local_dir}/global_step_{self.global_step}"
        object_refs = []
        for model in self.models:
            object_ref = model.save_parallelized.remote(save_dir, blocking=False)
            object_refs.append(object_ref)
        nanoray.get(object_refs)

        with open(f"{self.config.training.default_local_dir}/latest_checkpointed_iteration.txt", "w") as f:
            f.write(str(self.global_step))
        print(f"\n[SAVE] Saved checkpoint at step {self.global_step} to {save_dir}")

    def fit(self):
        for epoch in range(self.config.training.total_epochs):
            pbar = tqdm(
                self.train_dataloader,
                desc=f"Epoch {epoch + 1}/{self.config.training.total_epochs}",
                dynamic_ncols=True,
            )
            for batch in pbar:
                self.global_step += 1

                output = self.step(batch, train=True)
                train_loss, lr = output["loss"], output["lr"]
                pbar.set_postfix(loss=f"{train_loss:.6f}", lr=f"{lr:.6e}", global_step=self.global_step)
                self.log(
                    {
                        "train/loss": train_loss,
                        "train/epoch": epoch,
                        "train/lr": lr,
                        "train/global_step": self.global_step,
                    }
                )

                if self.global_step % self.config.training.test_freq == 0:
                    valid_losses = []
                    for valid_batch in tqdm(self.valid_dataloader, desc="Validation", dynamic_ncols=True):
                        valid_output = self.step(valid_batch, train=False)
                        valid_losses.append(valid_output["loss"])
                    mean_valid_loss = sum(valid_losses) / max(len(valid_losses), 1)
                    self.log(
                        {
                            "valid/loss": mean_valid_loss,
                            "valid/epoch": epoch,
                            "valid/global_step": self.global_step,
                        }
                    )
                    print(f"\n[Validation] step {self.global_step}, loss: {mean_valid_loss:.6f}")

                if self.global_step % self.config.training.save_freq == 0:
                    # Periodic save during training
                    self.save_parallelized()

        # Final save after training
        self.save_parallelized()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the SFT config yaml file.")
    trainer = SFTTrainer(parser.parse_args().config)
    trainer.fit()
