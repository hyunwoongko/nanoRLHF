from dataclasses import asdict
from typing import Dict, Any

import wandb


class BaseTrainer:
    def __init__(self, config):
        self.config = config
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
