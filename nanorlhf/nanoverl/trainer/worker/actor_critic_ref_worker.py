from collections import deque
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

import torch
from torch._C._distributed_c10d import ReduceOp
from torch.nn import functional as F
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoModelForTokenClassification

from nanorlhf import nanoray
from nanorlhf.kernels import patch_kernel
from nanorlhf.nanotron import MPU, TensorParallel, PipelineParallel, DataParallel, ParallelMode
from nanorlhf.nanotron.distributed.collectives import Collectives
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


def initialize_model(config, rank, mpu: MPU = None, role: str = "actor"):
    assert role in ["actor", "ref", "critic"], "role must be one of ['actor', 'ref', 'critic']"

    if role == "critic":
        model = AutoModelForTokenClassification.from_pretrained(
            config.actor.model_name_or_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            num_labels=1,
        )
        # turn off dropout
        model.dropout = torch.nn.Identity()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            config.actor.model_name_or_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )

    if role == "ref":
        optimizer = None
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
    else:
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

    actor_world_size = (
        config.actor.data_parallel_size * config.actor.tensor_parallel_size * config.actor.pipeline_parallel_size
    )
    rollout_world_size = config.rollout.data_parallel_size * config.rollout.tensor_parallel_size
    global_world_size = actor_world_size + rollout_world_size

    assert global_world_size <= torch.cuda.device_count(), "Currently nanoRLHF doesn't support multi-node training"

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
            zero_stage=0 if role == "ref" else config.actor.zero_stage,
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
        self.experience_buffer = deque(maxlen=self.config.data.experience_staleness + 1)

        self.actor, self.actor_optimizer, self.mpu = initialize_model(config, rank, role="actor")
        self.actor_scheduler = get_scheduler(config, self.actor_optimizer, total_steps)
        self.ref, _, _ = initialize_model(config, rank, mpu=self.mpu, role="ref")

        if config.algorithm.adv_estimator not in ["grpo", "gspo"]:
            self.critic, self.critic_optimizer, _ = initialize_model(config, rank, mpu=self.mpu, role="critic")
            self.critic_scheduler = get_scheduler(config, self.critic_optimizer, total_steps)

    @torch.inference_mode()
    def compute_token_logprobs(self, model, input_ids: torch.Tensor, position_ids: torch.Tensor):
        assert input_ids.dtype == torch.long
        assert input_ids.dim() == 2
        batch_size, sequence_length = input_ids.shape
        if sequence_length <= 1:
            return torch.zeros((batch_size, sequence_length), device=input_ids.device, dtype=torch.float32)

        out = model(input_ids, position_ids=position_ids, attention_mask=None, use_cache=False)
        logits = out.logits
        logits = logits[:, :-1, :]
        targets = input_ids[:, 1:]

        vocab_global = model.config.vocab_size
        if int(targets.max().item()) >= int(vocab_global) or int(targets.min().item()) < 0:
            raise ValueError(
                f"Found token id outside global vocab: "
                f"min={int(targets.min().item())}, max={int(targets.max().item())}, vocab_size={int(vocab_global)}"
            )

        tensor_parallel_size = self.mpu.get_world_size(ParallelMode.TENSOR)
        tensor_parallel_rank = self.mpu.get_local_rank(ParallelMode.TENSOR)
        collectives = Collectives(self.mpu, mode=ParallelMode.TENSOR)

        if tensor_parallel_size <= 1:
            logprobs = F.log_softmax(logits.float(), dim=-1)
            token_logprobs = logprobs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
            full = torch.zeros((batch_size, sequence_length), device=input_ids.device, dtype=torch.float32)
            full[:, 1:] = token_logprobs
            full = full.masked_fill(position_ids == 0, 0.0)
            return full

        local_vocab_size = logits.size(-1)
        vocab_start_idx = tensor_parallel_rank * local_vocab_size
        vocab_end_idx = vocab_start_idx + local_vocab_size

        max_local_logits = logits.float().amax(dim=-1)
        max_global_logits = max_local_logits.clone()
        collectives.all_reduce(max_global_logits, op=ReduceOp.MAX)

        exp_local = torch.exp(logits.float() - max_global_logits.unsqueeze(-1))
        local_sumexp = exp_local.sum(dim=-1)
        global_sumexp = local_sumexp.clone()
        collectives.all_reduce(global_sumexp, op=ReduceOp.SUM)

        local_shard_condition = (targets >= vocab_start_idx) & (targets < vocab_end_idx)
        local_vocab_idx = torch.where(local_shard_condition, targets - vocab_start_idx, torch.zeros_like(targets))
        local_selected = logits.float().gather(-1, local_vocab_idx.unsqueeze(-1)).squeeze(-1)
        local_selected = local_selected * local_shard_condition.float()

        log_denom = torch.log(global_sumexp + 1e-12) + max_global_logits
        global_selected = local_selected.clone()
        collectives.all_reduce(global_selected, op=ReduceOp.SUM)
        token_logprobs = global_selected - log_denom

        full = torch.zeros((batch_size, sequence_length), device=input_ids.device, dtype=torch.float32)
        full[:, 1:] = token_logprobs
        full = full.masked_fill(position_ids == 0, 0.0)
        return full

    def compute_values(self, input_ids, position_ids, shift_for_actions=True):
        outputs = self.critic(input_ids, position_ids=position_ids, attention_mask=None, use_cache=False)
        raw_values = outputs.logits.squeeze(-1).float()

        # alignment to match token_logprobs convention
        if shift_for_actions:
            values = torch.zeros_like(raw_values)
            if raw_values.size(1) > 1:
                values[:, 1:] = raw_values[:, :-1]
        else:
            values = raw_values

        # position_id==0 should not contribute / should be stable.
        if position_ids is not None:
            values = values.masked_fill(position_ids == 0, 0.0)

        return values

    def make_experience(self, input_batch, reward_scores):
        input_ids = input_batch["input_ids"].to(torch.cuda.current_device())
        position_ids = input_batch["position_ids"].to(torch.cuda.current_device())
        loss_mask = input_batch["loss_mask"].to(torch.cuda.current_device())

        actor_lobprobs_old = self.compute_token_logprobs(self.actor, input_ids, position_ids)
        ref_logprobs = self.compute_token_logprobs(self.ref, input_ids, position_ids)

        values_old = None
        if self.config.algorithm.adv_estimator not in ["grpo", "gspo"]:
            values_old = self.compute_values(input_ids, position_ids, shift_for_actions=True)
            values_old = values_old.cpu()

        experience = Experience(
            input_ids=input_ids.cpu(),
            position_ids=position_ids.cpu(),
            loss_mask=loss_mask.cpu(),
            actor_logprobs_old=actor_lobprobs_old.cpu(),
            ref_logprobs=ref_logprobs.cpu(),
            values_old=values_old,
        )

        response_mask = loss_mask.bool()
        num_response_tokens = int(response_mask.sum().item())
        num_total_tokens = int(loss_mask.numel())
        num_sequences = int(((position_ids == 0) & (loss_mask == 0)).sum().item())

        assert len(reward_scores) == num_sequences
        reward_scores = [float(x) for x in reward_scores]
        position_ids_cpu = experience.position_ids
        loss_mask_cpu = experience.loss_mask

        rewards = torch.zeros_like(loss_mask_cpu, dtype=torch.float32)
        starts = ((position_ids_cpu[0] == 0) & (loss_mask_cpu[0] == 0)).nonzero(as_tuple=False).flatten().tolist()
        ends = starts[1:] + [position_ids_cpu.size(1)]

        for i, (start, end) in enumerate(zip(starts, ends)):
            is_response = (loss_mask_cpu[0, start:end] == 1)
            if not bool(is_response.any()):
                continue
            last_local = int(is_response.nonzero(as_tuple=False).flatten()[-1].item())
            last_idx = start + last_local
            rewards[0, last_idx] += reward_scores[i]

        experience.rewards = rewards
        self.experience_buffer.append(experience)

        if num_response_tokens > 0:
            approx_kl = float((actor_lobprobs_old - ref_logprobs)[response_mask].mean().item())
            mean_logprobs = float(actor_lobprobs_old[response_mask].mean().item())
        else:
            approx_kl = 0.0
            mean_logprobs = 0.0

        return {
            "num_total_tokens": num_total_tokens,
            "num_response_tokens": num_response_tokens,
            "num_sequences": num_sequences,
            "approx_kl": approx_kl,
            "mean_logprobs": mean_logprobs,
        }
