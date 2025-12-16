from argparse import ArgumentParser
from dataclasses import asdict
from typing import Dict, Any

import wandb
from torch.utils.data import DataLoader

from nanorlhf.nanoverl.configs.rl_config import RLConfig
from nanorlhf.nanoverl.dataset.rl_dataset import RLDataset
from nanorlhf.nanoverl.utils.packing_utils import packed_collate_fn_for_rl


class RLTrainer:
    def __init__(self, config: str):
        self.config = RLConfig.from_yaml(config)
        self.train_dataloader = self.load_dataloader(self.config, split="train")
        self.valid_dataloader = self.load_dataloader(self.config, split="valid")
        self.total_steps = self.config.training.total_epochs * len(self.train_dataloader)
        self.global_step = 0

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


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the RL config yaml file.")
    trainer = RLTrainer(parser.parse_args().config)
