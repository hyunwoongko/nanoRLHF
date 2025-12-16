from dataclasses import dataclass, field, asdict
from typing import Optional

import yaml
import torch


@dataclass
class DataConfig:
    train_batch_size: int = 256
    valid_batch_size: int = 200
    train_micro_batch_size: int = 4
    valid_micro_batch_size: int = 2
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
    nproc_per_node: int = 1


@dataclass
class RolloutConfig:
    model_name_or_path: str = "Qwen/Qwen3-4B-base"
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 1024
    max_model_len: int = 2048
    n: int = 4
    gpu_memory_utilization: float = 0.4
    kvcache_block_size: int = 256
    nproc_per_node: int = 1
    tensor_parallel_size: int = 1
    data_parallel_size: int = 1


@dataclass
class RewardConfig:
    path: str = None
    name: str = "compute_score"


@dataclass
class OptimConfig:
    lr: float = 5e-7
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
    adv_estimator: str = "gae"
    use_kl_in_reward: bool = False
    kl_loss_coef: float = 0.1
    clip_ratio_high: float = 0.2
    clip_ratio_low: float = 0.2


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
        if self.actor.nproc_per_node + self.rollout.nproc_per_node > torch.cuda.device_count():
            raise ValueError(
                f"Currently nanoRLHF doesn't support multi-node training. "
                f"The sum of actor.nproc_per_node and rollout.nproc_per_node "
                f"must be less than or equal to the number of GPUs on a single node ({torch.cuda.device_count()}). "
            )

        actor_world_size = (
            self.actor.data_parallel_size * self.actor.tensor_parallel_size * self.actor.pipeline_parallel_size
        )
        if actor_world_size > self.actor.nproc_per_node:
            raise ValueError(
                "Currently nanoRLHF doesn't support multi-node training. "
                f"Please set actor.data_parallel_size * actor.tensor_parallel_size * "
                f"actor.pipeline_parallel_size <= self.actor.nproc_per_node={self.actor.nproc_per_node}, "
                f"but got {actor_world_size}."
            )

        min_rollout_world_size = self.actor.tensor_parallel_size
        if (
            min_rollout_world_size > self.rollout.nproc_per_node
            or self.rollout.nproc_per_node % min_rollout_world_size != 0
        ):
            raise ValueError(
                "Actor and Rollout must use the same tensor model parallel size to reduce "
                "parameter synchronization overhead. "
                "nanoRLHF fixes rollout tensor parallel size to actor.tensor_parallel_size.\n\n"
                f"Therefore, rollout.nproc_per_node must satisfy:\n"
                f"  - rollout.nproc_per_node >= actor.tensor_parallel_size ({self.actor.tensor_parallel_size})\n"
                f"  - rollout.nproc_per_node % actor.tensor_parallel_size == 0\n\n"
                "This is required to form an integer number of rollout replicas, where:\n"
                "  num_rollout_replicas = rollout.nproc_per_node // actor.tensor_parallel_size\n\n"
                f"But got rollout.nproc_per_node={self.rollout.nproc_per_node}, "
                f"actor.tensor_parallel_size={self.actor.tensor_parallel_size}."
            )

        if self.algorithm.adv_estimator not in ["gae", "grpo", "gspo"]:
            raise ValueError(
                f"Unsupported advantage estimator: {self.algorithm.adv_estimator}. "
                "Supported options are: 'gae', 'grpo', 'gspo'."
            )

        if self.data.train_batch_size % self.data.train_micro_batch_size != 0:
            raise ValueError(
                "`train_batch_size` must be divisible by `train_micro_batch_size`. "
                f"Got train_batch_size={self.data.train_batch_size} and "
                f"train_micro_batch_size={self.data.train_micro_batch_size}."
            )

        if self.data.valid_batch_size % self.data.valid_micro_batch_size != 0:
            raise ValueError(
                "`valid_batch_size` must be divisible by `valid_micro_batch_size`. "
                f"Got valid_batch_size={self.data.valid_batch_size} and "
                f"valid_micro_batch_size={self.data.valid_micro_batch_size}."
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
