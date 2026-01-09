from dataclasses import asdict
from pprint import pprint
from typing import Dict, Any

import wandb


class BaseTrainer:
    """
    Base class for trainers.

    Args:
        config: Configuration object containing training settings.
    """
    def __init__(self, config):
        self.config = config
        pprint("Training configuration:")
        pprint(asdict(self.config))
        self.global_step = 0
        self.maybe_init_logger()

    def maybe_init_logger(self):
        """
        Initializes the Weights & Biases (wandb) logger if enabled in the configuration.
        """
        if not self.config.training.wandb:
            return

        wandb.init(
            project=self.config.training.project_name,
            name=self.config.training.experiment_name,
            config=asdict(self.config),
        )

    def log(self, metrics: Dict[str, Any]):
        """
        Logs metrics to Weights & Biases (wandb) if enabled.

        Args:
            metrics (Dict[str, Any]): A dictionary of metrics to log.
        """
        if self.config.training.wandb:
            wandb.log(metrics, step=self.global_step)
