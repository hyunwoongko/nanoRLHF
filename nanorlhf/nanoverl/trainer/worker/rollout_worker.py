from typing import List

import torch

from nanorlhf import nanoray
from nanorlhf.nanoverl.utils.sync_utils import ParameterSyncManager
from nanorlhf.nanovllm.core.model_runner import ModelRunner
from nanorlhf.nanovllm.core.sequence import Sequence
from nanorlhf.nanovllm.utils.config import NanoVLLMConfig


@nanoray.remote
class RolloutWorker:
    """
    Rollout worker that handles model inference for rollouts.

    Args:
        config: Configuration object containing rollout and actor settings.
        rank (int): The rank of the worker in a distributed setup.
    """
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
        self.parameter_sync_manager = ParameterSyncManager(
            self.runner.model, self.runner.mpu, self.config, is_rollout=True
        )

    def get_rollout_config(self):
        """
        Get the rollout configuration.

        Returns:
            NanoVLLMConfig: The rollout configuration.
        """
        return self.runner.get_config()

    @torch.inference_mode()
    def generate(self, sequences: List[Sequence], is_prefill: bool) -> List[int]:
        """
        Generate tokens for the given sequences.

        Args:
            sequences (List[Sequence]): List of sequences to generate tokens for.
            is_prefill (bool): Whether the generation is in prefill mode.
        """
        return self.runner.run(sequences, is_prefill)

    def sync_actor_to_rollout(self):
        """
        Synchronize parameters from the actor model to the rollout model.
        """
        return self.parameter_sync_manager.sync_actor_to_rollout()
