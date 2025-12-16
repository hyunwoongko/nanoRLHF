from dataclasses import dataclass
from typing import Optional, Dict, Any, List

import torch
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer

from nanorlhf import nanoray
from nanorlhf.kernels import patch_kernel
from nanorlhf.nanotron import MPU, TensorParallel, PipelineParallel, DataParallel
from nanorlhf.nanoverl.utils.optim_utils import get_optimizer_param_groups, get_scheduler


@dataclass
class Experience:
    input_ids: torch.Tensor
    position_ids: torch.Tensor
    loss_mask: torch.Tensor

    actor_logprobs_old: torch.Tensor
    ref_logprobs: torch.Tensor
    values_old: Optional[torch.Tensor]

    rewards: Optional[torch.Tensor] = None
    advantages: Optional[torch.Tensor] = None
    returns: Optional[torch.Tensor] = None
    reward_model: Optional[List[Dict[str, Any]]] = None


def initialize_model(config, rank, mpu: MPU = None, role: str = "actor"):
    assert role in ["actor", "ref", "critic"], "role must be one of ['actor', 'ref', 'critic']"

    model = AutoModelForCausalLM.from_pretrained(
        config.actor.model_name_or_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    if role in ["actor", "critic"]:
        optimizer = AdamW(
            get_optimizer_param_groups(model, float(config.optim.weight_decay)),
            lr=float(config.optim.lr),
            weight_decay=float(config.optim.weight_decay),
            betas=(float(config.optim.beta1), float(config.optim.beta2)),
        )
        model.train()
        if config.actor.gradient_checkpointing_enable:
            if config.actor.pipeline_parallel_size == 1:
                # pipeline parallel engine controls grad checkpointing itself.
                model.gradient_checkpointing_enable()
    else:
        optimizer = None
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)

    actor_world_size = (
        config.actor.data_parallel_size * config.actor.tensor_parallel_size * config.actor.pipeline_parallel_size
    )
    rollout_world_size = config.rollout.data_parallel_size * config.rollout.tensor_parallel_size
    global_world_size = actor_world_size + rollout_world_size

    assert global_world_size <= torch.cuda.device_count()

    if global_world_size > 1:
        assert rank < actor_world_size, "rank must be < dp*tp*pp"
        if mpu is None:
            mpu = MPU(
                rank=rank,
                local_rank=rank,
                world_size=global_world_size,
                local_world_size=global_world_size,
                host=config.actor.host,
                port=config.actor.port,
                data_parallel_size=config.actor.data_parallel_size,
                pipeline_parallel_size=config.actor.pipeline_parallel_size,
                tensor_parallel_size=config.actor.tensor_parallel_size,
                rollout_data_parallel_size=config.rollout.data_parallel_size,
                rollout_tensor_parallel_size=config.rollout.tensor_parallel_size,
                backend=config.actor.backend,
                seed=config.actor.seed,
            )

        model = TensorParallel(
            model,
            mpu=mpu,
        )
        model = PipelineParallel(
            model,
            mpu=mpu,
            micro_batch_size=config.data.train_micro_batch_size,
            gradient_checkpointing_enable=config.actor.gradient_checkpointing_enable,
        )
        model, optimizer = DataParallel(
            model,
            mpu=mpu,
            optimizer=optimizer,
            zero_stage=config.actor.zero_stage if role in ["actor", "critic"] else 0,
            accum_steps=config.data.train_batch_size // config.data.train_micro_batch_size,
        )
        model.parallelize()
    else:
        model.cuda()

    model = patch_kernel(model)
    return model, optimizer, mpu


@nanoray.actor
class ActorCriticRefWorker:
    def __init__(self, config, rank, total_steps: int):
        self.config = config
        self.rank = rank
        self.tokenizer = AutoTokenizer.from_pretrained(config.actor.tokenizer_name_or_path, trust_remote_code=True)

        self.actor, self.actor_optimizer, self.mpu = initialize_model(config, rank, role="actor")
        self.actor_scheduler = get_scheduler(config, self.actor_optimizer, total_steps)
        self.ref, _, _ = initialize_model(config, rank, mpu=self.mpu, role="ref")

        if config.algorithm.adv_estimator not in ["grpo", "gspo"]:
            self.critic, self.critic_optimizer, _ = initialize_model(config, rank, mpu=self.mpu, role="critic")
            self.critic_scheduler = get_scheduler(config, self.critic_optimizer, total_steps)

    @torch.inference_mode()
    def compute_logprobs(self, model, input_ids, position_ids):
        # TODO
        pass
