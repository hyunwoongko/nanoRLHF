from dataclasses import dataclass, field, asdict
from typing import Optional

import yaml
import torch


@dataclass
class DataConfig:
    train_batch_size: int = 256
    valid_batch_size: int = 200
    ppo_mini_batch_size_per_gpu: int = 32
    ppo_micro_batch_size_per_gpu: int = 8
    ppo_epochs: int = 1
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
class RolloutConfig:
    model_name_or_path: str = "Qwen/Qwen3-0.6B-base"
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 1024
    max_model_len: int = 2048
    max_prompt_len: int = 512
    max_response_len: int = 1536
    gpu_memory_utilization: float = 0.9
    kvcache_block_size: int = 256
    tensor_parallel_size: int = 1
    data_parallel_size: int = 1
    enforce_eager: bool = False
    temperature: float = 1.0


@dataclass
class RewardConfig:
    path: str = "nanorlhf.nanoverl.reward.custom_reward_fn"
    name: str = "compute_score"


@dataclass
class OptimConfig:
    lr: float = 5e-6
    min_lr: float = 0.0
    lr_warmup_steps_ratio: float = 0.1
    lr_scheduler: str = "cosine"
    beta1: float = 0.9
    beta2: float = 0.95
    clip_grad: float = 1.0
    weight_decay: float = 1e-3


@dataclass
class AlgorithmConfig:
    gamma: float = 1.0
    lam: float = 1.0
    use_kl_in_reward: bool = False
    kl_loss_coef: float = 0.01
    vf_loss_coef: float = 0.1
    clip_ratio_high: float = 0.2
    clip_ratio_low: float = 0.2
    clip_ratio_value: float = 0.2


@dataclass
class TrainingConfig:
    default_local_dir: str = "./checkpoints"
    project_name: str = "project"
    experiment_name: str = "experiment"
    total_epochs: int = 1
    wandb: bool = True
    seed: int = 42
    save_freq: int = 50
    test_freq: int = 50


@dataclass
class RLConfig:
    data: DataConfig = field(default_factory=DataConfig)
    actor: ModelConfig = field(default_factory=ModelConfig)
    ref: ModelConfig = field(default_factory=ModelConfig)
    rollout: RolloutConfig = field(default_factory=RolloutConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def __post_init__(self):
        actor_world_size = (
            self.actor.data_parallel_size * self.actor.tensor_parallel_size * self.actor.pipeline_parallel_size
        )
        rollout_world_size = self.rollout.data_parallel_size * self.rollout.tensor_parallel_size

        if actor_world_size + rollout_world_size > torch.cuda.device_count():
            raise ValueError(
                f"Currently nanoRLHF doesn't support multi-node training. "
                f"The sum of actor.nproc_per_node and rollout.nproc_per_node "
                f"must be less than or equal to the number of GPUs on a single node ({torch.cuda.device_count()}). "
            )

        if self.rollout.tensor_parallel_size != self.actor.tensor_parallel_size:
            raise ValueError(
                "Actor and Rollout must use the same tensor model parallel size to reduce "
                "parameter synchronization overhead. "
                f"Therefore, rollout.tensor_parallel_size must equal actor.tensor_parallel_size "
                f"({self.actor.tensor_parallel_size}).\n\n"
                f"But got rollout.tensor_parallel_size={self.rollout.tensor_parallel_size}."
            )

        if self.data.train_batch_size % self.data.ppo_micro_batch_size_per_gpu != 0:
            raise ValueError(
                "`train_batch_size` must be divisible by `ppo_micro_batch_size_per_gpu`. "
                f"Got train_batch_size={self.data.train_batch_size} and "
                f"ppo_micro_batch_size_per_gpu={self.data.ppo_micro_batch_size_per_gpu}."
            )

        assert self.data.train_batch_size % self.rollout.data_parallel_size == 0, (
            "train_batch_size must be divisible by rollout.data_parallel_size to avoid silently dropping sequences.\n"
            f"Got train_batch_size={self.data.train_batch_size}, rollout_dp={self.rollout.data_parallel_size}, "
            f"remainder={self.data.train_batch_size % self.rollout.data_parallel_size}.\n"
            "Fix: set train_batch_size = k * rollout_dp."
        )

        assert self.data.valid_batch_size % self.rollout.data_parallel_size == 0, (
            "valid_batch_size must be divisible by rollout.data_parallel_size to avoid silently dropping sequences.\n"
            f"Got valid_batch_size={self.data.valid_batch_size}, rollout_dp={self.rollout.data_parallel_size}, "
            f"remainder={self.data.valid_batch_size % self.rollout.data_parallel_size}.\n"
            "Fix: set valid_batch_size = k * rollout_dp."
        )
        assert self.rollout.max_prompt_len + self.rollout.max_response_len <= self.rollout.max_model_len, (
            "The sum of max_prompt_len and max_response_len must be less than or equal to max_model_len.\n"
            f"Got max_prompt_len={self.rollout.max_prompt_len}, "
            f"max_response_len={self.rollout.max_response_len}, "
            f"max_model_len={self.rollout.max_model_len}.\n"
            "Fix: set max_model_len >= max_prompt_len + max_response_len."
        )

        assert self.data.train_batch_size % self.data.ppo_mini_batch_size_per_gpu == 0, (
            "train_batch_size must be divisible by ppo_mini_batch_size to avoid silently dropping sequences.\n"
            f"Got train_batch_size={self.data.train_batch_size}, ppo_mini_batch_size={self.data.ppo_mini_batch_size_per_gpu}."
        )

        assert self.data.ppo_mini_batch_size_per_gpu % self.data.ppo_micro_batch_size_per_gpu == 0, (
            "ppo_mini_batch_size_per_gpu must be divisible by ppo_micro_batch_size_per_gpu "
            "to avoid silently dropping sequences.\n"
            f"Got ppo_mini_batch_size_per_gpu={self.data.ppo_mini_batch_size_per_gpu}, "
            f"ppo_micro_batch_size_per_gpu={self.data.ppo_micro_batch_size_per_gpu}."
        )

    @classmethod
    def from_yaml(cls, file_path: str) -> "RLConfig":
        with open(file_path, "r") as f:
            config_dict = yaml.safe_load(f) or {}

        data_config = DataConfig(**config_dict.get("data", {}))
        actor_config = ModelConfig(**config_dict.get("actor", {}))
        rollout_config = RolloutConfig(**config_dict.get("rollout", {}))
        reward_config = RewardConfig(**config_dict.get("reward", {}))
        algorithm_config = AlgorithmConfig(**config_dict.get("algorithm", {}))
        optim_config = OptimConfig(**config_dict.get("optim", {}))
        training_config = TrainingConfig(**config_dict.get("training", {}))

        if actor_config.tokenizer_name_or_path is None:
            actor_config.tokenizer_name_or_path = actor_config.model_name_or_path

        return cls(
            data=data_config,
            actor=actor_config,
            rollout=rollout_config,
            reward=reward_config,
            algorithm=algorithm_config,
            optim=optim_config,
            training=training_config,
        )

    def to_yaml(self, file_path: str):
        data_dict = {
            "data": asdict(self.data),
            "actor": asdict(self.actor),
            "ref": asdict(self.ref),
            "rollout": asdict(self.rollout),
            "reward": asdict(self.reward),
            "algorithm": asdict(self.algorithm),
            "optim": asdict(self.optim),
            "training": asdict(self.training),
        }
        with open(file_path, "w") as f:
            yaml.dump(data_dict, f)
