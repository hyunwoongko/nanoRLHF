from dataclasses import dataclass, field, asdict

import yaml

from nanorlhf.nanoverl.configs.base import DataConfig, ModelConfig, OptimConfig, TrainingConfig
from nanorlhf.nanovllm.utils.config import NanoVLLMConfig


@dataclass
class RLConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    rollout: NanoVLLMConfig = field(default_factory=NanoVLLMConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
 
    @classmethod
    def from_yaml(cls, file_path: str) -> "RLConfig":
        with open(file_path, "r") as f:
            config_dict = yaml.safe_load(f) or {}

        data_config = DataConfig(**config_dict.get("data", {}))
        model_config = ModelConfig(**config_dict.get("model", {}))
        optim_config = OptimConfig(**config_dict.get("optim", {}))
        training_config = TrainingConfig(**config_dict.get("training", {}))
        rollout_config = NanoVLLMConfig(**config_dict.get("rollout", {}))

        return cls(
            data=data_config,
            model=model_config,
            rollout=rollout_config,
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
