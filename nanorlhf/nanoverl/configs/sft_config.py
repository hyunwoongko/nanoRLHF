from dataclasses import dataclass, field, asdict
from typing import Optional

import torch
import yaml


@dataclass
class DataConfig:
    train_batch_size: int = 256
    valid_batch_size: int = 200
    train_micro_batch_size_per_gpu: int = 64
    valid_micro_batch_size_per_gpu: int = 50
    train_data: Optional[str] = None
    valid_data: Optional[str] = None
    num_workers: int = 8


@dataclass
class ModelConfig:
    model_name_or_path: str = "Qwen/Qwen3-0.6B-base"
    tokenizer_name_or_path: str = "Qwen/Qwen3-0.6B"
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    data_parallel_size: int = 1
    zero_stage: int = 0
    host: str = "127.0.0.1"
    port: int = 23333
    backend: str = "nccl"
    seed: int = 42
    gradient_checkpointing_enable: bool = True


@dataclass
class OptimConfig:
    lr: float = 5e-6
    min_lr: float = 5e-7
    lr_warmup_steps_ratio: float = 0.1
    lr_scheduler: str = "cosine"
    beta1: float = 0.9
    beta2: float = 0.95
    clip_grad: float = 1.0
    weight_decay: float = 1e-3


@dataclass
class TrainingConfig:
    default_local_dir: str = "./checkpoints"
    project_name: str = "project"
    experiment_name: str = "experiment"
    total_epochs: int = 3
    wandb: bool = True
    seed: int = 42
    save_freq: int = 300
    test_freq: int = 300


@dataclass
class SFTConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def __post_init__(self):
        global_world_size = (
            self.model.data_parallel_size * self.model.tensor_parallel_size * self.model.pipeline_parallel_size
        )
        assert global_world_size <= torch.cuda.device_count(), (
            "Currently nanoRLHF doesn't support multi-node training. "
            f"Please set data_parallel_size * tensor_parallel_size * pipeline_parallel_size <= "
            f"{torch.cuda.device_count()}, but got {global_world_size}."
        )

        if self.data.train_batch_size % self.data.train_micro_batch_size_per_gpu != 0:
            raise ValueError(
                "`train_batch_size` must be divisible by `train_micro_batch_size_per_gpu`. "
                f"Got train_batch_size={self.data.train_batch_size} and "
                f"train_micro_batch_size_per_gpu={self.data.train_micro_batch_size_per_gpu}."
            )

        if self.data.valid_batch_size % self.data.valid_micro_batch_size_per_gpu != 0:
            raise ValueError(
                "`valid_batch_size` must be divisible by `valid_micro_batch_size_per_gpu`. "
                f"Got valid_batch_size={self.data.valid_batch_size} and "
                f"valid_micro_batch_size_per_gpu={self.data.valid_micro_batch_size_per_gpu}."
            )

    @classmethod
    def from_yaml(cls, file_path: str) -> "SFTConfig":
        with open(file_path, "r") as f:
            config_dict = yaml.safe_load(f) or {}

        data_config = DataConfig(**config_dict.get("data", {}))
        model_config = ModelConfig(**config_dict.get("model", {}))
        optim_config = OptimConfig(**config_dict.get("optim", {}))
        training_config = TrainingConfig(**config_dict.get("training", {}))

        if model_config.tokenizer_name_or_path is None:
            model_config.tokenizer_name_or_path = model_config.model_name_or_path

        return cls(
            data=data_config,
            model=model_config,
            optim=optim_config,
            training=training_config,
        )

    def to_yaml(self, file_path: str):
        data_dict = {
            "data": asdict(self.data),
            "model": asdict(self.model),
            "optim": asdict(self.optim),
            "training": asdict(self.training),
        }
        with open(file_path, "w") as f:
            yaml.dump(data_dict, f)
