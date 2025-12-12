from dataclasses import dataclass, field
from typing import Optional

import yaml


@dataclass
class DataConfig:
    train_batch_size: int = 128
    valid_batch_size: int = 128
    train_files: Optional[str] = None
    valid_files: Optional[str] = None
    messages_key: str = "messages"
    tools_key: str = "tools"
    enable_thinking_key: str = "enable_thinking"
    max_length: int = 8192
    truncation: str = "error"


@dataclass
class ModelConfig:
    partial_pretrain: str = "Qwen/Qwen3-4B"
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    data_parallel_size: int = 1
    zero_stage: int = 0
    enable_gradient_checkpointing: bool = True


@dataclass
class OptimConfig:
    lr: float = 5e-6
    beta1: float = 0.9
    beta2: float = 0.95
    lr_warmup_steps_ratio: float = 0.1
    lr_scheduler: str = "cosine"
    clip_grad: float = 1.0


@dataclass
class TrainingConfig:
    default_local_dir: str = "./checkpoints"
    project_name: str = "project"
    experiment_name: str = "experiment"
    total_epochs: int = 3
    logger: str = "wandb"
    seed: int = 42
    save_freq: int = 300
    test_freq: int = 300
    n_gpus_per_node: int = 8


@dataclass
class SFTConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    @classmethod
    def from_yaml(cls, file_path: str) -> "SFTConfig":
        with open(file_path, "r") as f:
            config_dict = yaml.safe_load(f) or {}

        data_config = DataConfig(**config_dict.get("data", {}))
        model_config = ModelConfig(**config_dict.get("model", {}))
        optim_config = OptimConfig(**config_dict.get("optim", {}))
        training_config = TrainingConfig(**config_dict.get("training", {}))

        return cls(
            data=data_config,
            model=model_config,
            optim=optim_config,
            training=training_config,
        )