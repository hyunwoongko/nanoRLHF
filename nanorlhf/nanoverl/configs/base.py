from dataclasses import dataclass
from typing import Optional


@dataclass
class DataConfig:
    train_batch_size: int = 128
    valid_batch_size: int = 128
    train_micro_batch_size: int = 8
    valid_micro_batch_size: int = 1
    train_data: Optional[str] = None
    valid_data: Optional[str] = None
    num_workers: int = 8


@dataclass
class ModelConfig:
    model_name_or_path: str = "Qwen/Qwen3-4B-base"
    tokenizer_name_or_path: str = "Qwen/Qwen3-4B"
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

