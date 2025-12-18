from typing import List

import torch

from nanorlhf import nanoray
from nanorlhf.nanovllm.core.model_runner import ModelRunner
from nanorlhf.nanovllm.core.sequence import Sequence
from nanorlhf.nanovllm.utils.config import NanoVLLMConfig


@nanoray.actor
class RolloutWorker:
    def __init__(self, config, rank: int):
        self.config = config
        self.rank = rank

        rollout_config = NanoVLLMConfig(
            model=config.rollout.model_name_or_path,
            max_num_batched_tokens=config.rollout.max_num_batched_tokens,
            max_num_seqs=config.rollout.max_num_seqs,
            max_model_len=config.rollout.max_model_len,
            gpu_memory_utilization=config.rollout.gpu_memory_utilization,
            kvcache_block_size=config.rollout.kvcache_block_size,
            tensor_parallel_size=config.rollout.tensor_parallel_size,
            data_parallel_size=config.rollout.data_parallel_size,
            host=config.actor.host,
            port=config.actor.port,
            backend=config.actor.backend,
            seed=config.actor.seed,
            enforce_eager=config.rollout.enforce_eager,
        )
        self.runner = ModelRunner.cls(rollout_config, rank, actor_config=config.actor)

    def get_rollout_config(self):
        return self.runner.get_config()

    @torch.inference_mode()
    def generate(self, sequences: List[Sequence], is_prefill: bool) -> List[int]:
        return self.runner.run(sequences, is_prefill)
