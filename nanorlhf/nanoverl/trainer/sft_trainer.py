from argparse import ArgumentParser

from torch.utils.data import DataLoader
from tqdm import tqdm

from nanorlhf import nanoray
from nanorlhf.nanoray.api.initialization import NANORAY_BASE_PORT
from nanorlhf.nanoverl.configs.sft_config import SFTConfig
from nanorlhf.nanoverl.dataset.sft_dataset import SFTDataset
from nanorlhf.nanoverl.trainer.base_trainer import BaseTrainer
from nanorlhf.nanoverl.trainer.worker.sft_worker import SFTWorker
from nanorlhf.nanoverl.trainer.worker_group.sft_worker_group import SFTWorkerGroup
from nanorlhf.nanoverl.utils.packing_utils import packed_collate_fn_for_sft


class SFTTrainer(BaseTrainer):
    def __init__(self, config: str):
        super().__init__(config=SFTConfig.from_yaml(config))
        self.train_dataloader = self.load_dataloader(self.config, split="train")
        self.valid_dataloader = self.load_dataloader(self.config, split="valid")
        self.total_steps = self.config.training.total_epochs * len(self.train_dataloader)

        self.global_world_size = (
            self.config.model.data_parallel_size
            * self.config.model.tensor_parallel_size
            * self.config.model.pipeline_parallel_size
        )
        self.node_ids = self.init_ray(self.config)

        sft_workers = self.spawn_workers(self.config)
        self.sft_worker_group = SFTWorkerGroup(self.config, sft_workers)

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
            collate_fn=packed_collate_fn_for_sft,
        )

    def init_ray(self, config):
        nodes = {}
        if self.global_world_size > 1:
            for rank in range(self.global_world_size):
                nodes[f"node-{rank}"] = nanoray.NodeConfig(
                    cpus=4.0,
                    gpus=1.0,
                    rpc=True,
                    host=config.model.host,
                    port=NANORAY_BASE_PORT + rank,
                )
        else:
            nodes["node-0"] = nanoray.NodeConfig(
                cpus=4.0, gpus=1.0, rpc=False, host=config.model.host, port=NANORAY_BASE_PORT
            )

        session = nanoray.init(nodes, default_node_id="node-0")
        node_ids = list(session._workers.keys())
        if len(node_ids) < self.global_world_size:
            raise RuntimeError(
                "`nanoray` was initialized with fewer nodes than `global_world_size`; "
                "please provide at least one NodeConfig per global rank."
            )

        return node_ids

    def spawn_workers(self, config):
        object_refs = []
        for global_rank in range(self.global_world_size):
            node_id = self.node_ids[global_rank % len(self.node_ids)]
            object_ref = SFTWorker.options(pinned_node_id=node_id).remote(
                config, rank=global_rank, total_steps=self.total_steps, blocking=False
            )
            object_refs.append(object_ref)
        return nanoray.get(object_refs)

    def fit(self):
        for epoch in range(self.config.training.total_epochs):
            pbar = tqdm(
                self.train_dataloader,
                desc=f"Epoch {epoch + 1}/{self.config.training.total_epochs}",
                dynamic_ncols=True,
            )
            for batch in pbar:
                self.global_step += 1

                output = self.sft_worker_group.step(batch, train=True)
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
                        valid_output = self.sft_worker_group.step(valid_batch, train=False)
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
                    self.sft_worker_group.save_parallelized(self.global_step)

        # Final save after training
        self.sft_worker_group.save_parallelized(self.global_step)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the SFT config yaml file.")
    trainer = SFTTrainer(parser.parse_args().config)
    trainer.fit()
