from collections import deque
from typing import Optional

import torch
import torch.distributed as dist
from torch.nn import functional as F
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoModelForTokenClassification, AutoTokenizer

from nanorlhf import nanoray
from nanorlhf.kernels import patch_kernel
from nanorlhf.nanotron import MPU, TensorParallel, PipelineParallel, DataParallel, ParallelMode
from nanorlhf.nanotron.core.tp.loss import VocabParallelCrossEntropyFunction
from nanorlhf.nanoverl.utils.experience import Experience
from nanorlhf.nanoverl.utils.optim_utils import get_optimizer_param_groups, get_scheduler
from nanorlhf.nanoverl.utils.packing_utils import split_packed_batch


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
            micro_batch_size=config.data.train_micro_batch_size_per_gpu,
            gradient_checkpointing_enable=config.actor.gradient_checkpointing_enable,
        )
        accum_steps = max(
            1,
            config.data.train_batch_size
            // (config.actor.data_parallel_size * config.data.train_micro_batch_size_per_gpu),
        )
        model, optimizer = DataParallel(
            model,
            mpu=mpu,
            optimizer=optimizer,
            zero_stage=0 if role == "ref" else config.actor.zero_stage,
            accum_steps=accum_steps,
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

        # Data is already tokenized so we don't use tokenizer here, but for saving it in the checkpoint path together.
        self.tokenizer = AutoTokenizer.from_pretrained(config.actor.tokenizer_name_or_path, trust_remote_code=True)

        self.actor, self.actor_optimizer, self.mpu = initialize_model(config, rank, role="actor")
        self.actor_scheduler = get_scheduler(config, self.actor_optimizer, total_steps)
        self.ref, _, _ = initialize_model(config, rank, mpu=self.mpu, role="ref")
        self.critic, self.critic_optimizer, _ = initialize_model(config, rank, mpu=self.mpu, role="critic")
        self.critic_scheduler = get_scheduler(config, self.critic_optimizer, total_steps)

    def compute_token_logprobs(
        self,
        model,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        loss_mask: torch.Tensor,
        enable_grad: bool,
        logits: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        assert input_ids.dtype == torch.long
        assert input_ids.dim() == 2
        batch_size, sequence_length = input_ids.shape

        if sequence_length <= 1:
            return torch.zeros((batch_size, sequence_length), device=input_ids.device, dtype=torch.float32)

        if logits is None:
            with torch.set_grad_enabled(enable_grad):
                outputs = model(input_ids, position_ids=position_ids, attention_mask=None, use_cache=False)

                if self.config.actor.pipeline_parallel_size > 1:
                    logits = torch.cat([out.logits for out in outputs], dim=1).contiguous()
                else:
                    logits = outputs.logits

        logits_shifted = logits[:, :-1, :]
        targets = input_ids[:, 1:]

        if self.mpu.get_world_size(ParallelMode.TENSOR) <= 1:
            logprobs = F.log_softmax(logits_shifted.float(), dim=-1)
            token_logprobs = logprobs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        else:
            nll = VocabParallelCrossEntropyFunction.apply(logits_shifted, targets, self.mpu, ParallelMode.TENSOR)
            token_logprobs = (-nll).float()

        full = torch.zeros((batch_size, sequence_length), device=input_ids.device, dtype=torch.float32)
        full[:, 1:] = token_logprobs

        # inter sequence tokens must not contribute to the loss
        full = full.masked_fill(position_ids == 0, 0.0)
        # apply the loss mask provided from the dataset
        full = full * loss_mask.to(dtype=full.dtype, device=full.device)
        return full

    def compute_values(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        shift_for_actions: bool = True,
        enable_grad: bool = False,
        logits: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if logits is None:
            with torch.set_grad_enabled(enable_grad):
                outputs = self.critic(input_ids, position_ids=position_ids, attention_mask=None, use_cache=False)

            if self.config.actor.pipeline_parallel_size > 1:
                logits = torch.cat([out.logits for out in outputs], dim=1).contiguous()
            else:
                logits = outputs.logits

        raw_values = logits.squeeze(-1).float()

        if shift_for_actions:
            values = torch.zeros_like(raw_values)
            if raw_values.size(1) > 1:
                values[:, 1:] = raw_values[:, :-1]
        else:
            values = raw_values

        # inter sequence tokens must not contribute to the loss
        values = values.masked_fill(position_ids == 0, 0.0)
        # don't need to apply loss mask because they will be masked in loss computation later
        return values

    def assign_sequence_rewards_to_tokens(self, experience, reward_scores, num_sequences):
        assert len(reward_scores) == num_sequences
        reward_scores = [float(x) for x in reward_scores]
        position_ids = experience.position_ids
        loss_mask = experience.loss_mask

        rewards = torch.zeros_like(loss_mask, dtype=torch.float32)
        starts = ((position_ids[0] == 0) & (loss_mask[0] == 0)).nonzero(as_tuple=False).flatten().tolist()
        ends = starts[1:] + [position_ids.size(1)]

        for i, (start, end) in enumerate(zip(starts, ends)):
            is_response = loss_mask[0, start:end] == 1
            if not bool(is_response.any()):
                continue
            last_local = int(is_response.nonzero(as_tuple=False).flatten()[-1].item())
            last_idx = start + last_local
            rewards[0, last_idx] += reward_scores[i]

        experience.rewards = rewards
        return experience

    def compute_returns_and_advantages(self, experience):
        gamma = float(self.config.algorithm.gamma)
        lam = float(self.config.algorithm.lam)
        kl_loss_coef = float(self.config.algorithm.kl_loss_coef)
        use_kl_in_reward = bool(self.config.algorithm.use_kl_in_reward)

        loss_mask = experience.loss_mask[0].to(torch.bool)
        position_ids = experience.position_ids[0]
        rewards = experience.rewards[0].float()
        values = experience.old_values[0].float()

        if use_kl_in_reward and kl_loss_coef != 0.0:
            kl = (experience.old_logprobs[0] - experience.ref_logprobs[0]).float()
            rewards = rewards - kl_loss_coef * kl

        starts = ((position_ids == 0) & (~loss_mask)).nonzero(as_tuple=False).flatten().tolist()
        ends = starts[1:] + [position_ids.numel()]

        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)

        for start, end in zip(starts, ends):
            gae = 0.0
            for t in range(end - 1, start - 1, -1):
                if not bool(loss_mask[t]):
                    gae = 0.0
                    continue

                v_t = float(values[t].item())
                v_next = float(values[t + 1].item()) if t + 1 < end else 0.0
                r_t = float(rewards[t].item())

                delta = r_t + gamma * v_next - v_t
                gae = delta + gamma * lam * gae
                advantages[t] = gae
                returns[t] = gae + v_t

        experience.advantages = advantages.unsqueeze(0)
        experience.returns = returns.unsqueeze(0)
        return experience

    @torch.inference_mode()
    def make_experience(self, input_batch, reward_scores):
        device = torch.cuda.current_device()
        input_ids = input_batch["input_ids"].to(device, non_blocking=True)
        position_ids = input_batch["position_ids"].to(device, non_blocking=True)
        loss_mask = input_batch["loss_mask"].to(device, non_blocking=True)

        num_sequences = int(((position_ids == 0) & (loss_mask == 0)).sum().item())
        micro_batch_size = self.config.data.train_micro_batch_size_per_gpu
        num_micro_batches = num_sequences // micro_batch_size

        input_batch_in_cuda = {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "loss_mask": loss_mask,
        }

        micro_batches = [
            split_packed_batch(input_batch_in_cuda, micro_idx, num_micro_batches)
            for micro_idx in range(num_micro_batches)
        ]

        if self.config.actor.pipeline_parallel_size > 1:
            actor_micro_iterator = self.actor(
                input_ids,
                position_ids=position_ids,
                attention_mask=None,
                use_cache=False,
            )
            ref_micro_iterator = self.ref(
                input_ids,
                position_ids=position_ids,
                attention_mask=None,
                use_cache=False,
            )
            critic_micro_iterator = self.critic(
                input_ids,
                position_ids=position_ids,
                attention_mask=None,
                use_cache=False,
            )
            micro_batch_iterator = enumerate(zip(actor_micro_iterator, ref_micro_iterator, critic_micro_iterator))
        else:
            micro_batch_iterator = enumerate(micro_batches)

        micro_old_logprobs_list = []
        micro_ref_logprobs_list = []
        micro_old_values_list = []

        for micro_idx, micro_input_or_output in micro_batch_iterator:
            if self.config.actor.pipeline_parallel_size > 1:
                actor_outputs, ref_outputs, critic_outputs = micro_input_or_output
                actor_logits, ref_logits, critic_logits = (
                    actor_outputs.logits,
                    ref_outputs.logits,
                    critic_outputs.logits,
                )
            else:
                actor_logits, ref_logits, critic_logits = None, None, None

            micro_batch = micro_batches[micro_idx]
            micro_old_logprobs = self.compute_token_logprobs(
                self.actor,
                micro_batch["input_ids"],
                micro_batch["position_ids"],
                micro_batch["loss_mask"],
                enable_grad=False,
                logits=actor_logits,
            )
            micro_ref_logprobs = self.compute_token_logprobs(
                self.ref,
                micro_batch["input_ids"],
                micro_batch["position_ids"],
                micro_batch["loss_mask"],
                enable_grad=False,
                logits=ref_logits,
            )
            micro_old_values = self.compute_values(
                micro_batch["input_ids"],
                micro_batch["position_ids"],
                shift_for_actions=True,
                enable_grad=False,
                logits=critic_logits,
            )
            micro_old_logprobs_list.append(micro_old_logprobs)
            micro_ref_logprobs_list.append(micro_ref_logprobs)
            micro_old_values_list.append(micro_old_values)

        old_logprobs = torch.cat(micro_old_logprobs_list, dim=1)
        ref_logprobs = torch.cat(micro_ref_logprobs_list, dim=1)
        old_values = torch.cat(micro_old_values_list, dim=1)

        experience = Experience(
            input_ids=input_ids,
            position_ids=position_ids,
            loss_mask=loss_mask,
            old_logprobs=old_logprobs,
            ref_logprobs=ref_logprobs,
            old_values=old_values,
        )

        response_mask = loss_mask.bool()
        num_response_tokens = int(response_mask.sum().item())
        num_total_tokens = int(loss_mask.numel())
        num_sequences = int(((position_ids == 0) & (loss_mask == 0)).sum().item())

        experience = self.assign_sequence_rewards_to_tokens(experience, reward_scores, num_sequences)
        experience = self.compute_returns_and_advantages(experience)
        self.experience_buffer.append(experience.to("cpu", pin_memory=True, detach=True))

        if num_response_tokens > 0:
            approx_kl = float((old_logprobs - ref_logprobs)[response_mask].mean().item())
            mean_logprobs = float(old_logprobs[response_mask].mean().item())
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

    def step(self):
        if len(self.experience_buffer) == 0:
            return {"skipped": True}

        device = torch.cuda.current_device()
        experience = self.experience_buffer.popleft().to(device)
        starts = ((experience.position_ids[0] == 0) & (experience.loss_mask[0] == 0)).nonzero(as_tuple=False).flatten()
        num_sequences = int(starts.numel())

        micro_batch_size = self.config.data.train_micro_batch_size_per_gpu
        num_of_micro_batches = num_sequences // micro_batch_size

        experience_dict = experience.to_dict()
        micro_batches = [
            split_packed_batch(experience_dict, micro_idx, num_of_micro_batches)
            for micro_idx in range(num_of_micro_batches)
        ]

        self.actor_optimizer.zero_grad(set_to_none=True)
        self.critic_optimizer.zero_grad(set_to_none=True)

        clip_min = 1.0 - float(self.config.algorithm.clip_ratio_low)
        clip_max = 1.0 + float(self.config.algorithm.clip_ratio_high)
        kl_loss_coef = float(self.config.algorithm.kl_loss_coef)

        sum_of_total_losses = torch.zeros((), device=device, dtype=torch.float32)
        sum_of_policy_losses = torch.zeros((), device=device, dtype=torch.float32)
        sum_of_value_losses = torch.zeros((), device=device, dtype=torch.float32)

        sum_of_valid_tokens = [micro_batch["loss_mask"].sum().to(device).float() for micro_batch in micro_batches]
        sum_of_valid_tokens = torch.stack(sum_of_valid_tokens).sum().clamp_min(1.0)
        num_of_updates = 0

        if self.config.actor.pipeline_parallel_size > 1:
            actor_micro_iterator = self.actor(
                experience.input_ids, position_ids=experience.position_ids, attention_mask=None, use_cache=False
            )
            critic_micro_iterator = self.critic(
                experience.input_ids, position_ids=experience.position_ids, attention_mask=None, use_cache=False
            )
            micro_batch_iterator = enumerate(zip(actor_micro_iterator, critic_micro_iterator))
        else:
            micro_batch_iterator = enumerate(micro_batches)

        for micro_idx, micro_input_or_output in micro_batch_iterator:
            micro_batch = micro_batches[micro_idx]

            micro_loss_mask = micro_batch["loss_mask"].to(torch.bool)
            num_of_micro_valid_tokens = micro_loss_mask.sum().to(device).float()
            if num_of_micro_valid_tokens.item() == 0:
                continue

            if self.config.actor.pipeline_parallel_size > 1:
                actor_outputs, critic_outputs = micro_input_or_output
                actor_logits, critic_logits = actor_outputs.logits, critic_outputs.logits
            else:
                actor_logits, critic_logits = None, None

            new_logprobs = self.compute_token_logprobs(
                self.actor,
                micro_batch["input_ids"],
                micro_batch["position_ids"],
                micro_batch["loss_mask"],
                enable_grad=True,
                logits=actor_logits,
            ).float()

            new_values = self.compute_values(
                micro_batch["input_ids"],
                micro_batch["position_ids"],
                shift_for_actions=True,
                enable_grad=True,
                logits=critic_logits,
            ).float()

            log_ratio = new_logprobs - micro_batch["old_logprobs"].float()
            ratio = torch.exp(log_ratio)
            ratio_clipped = torch.clamp(ratio, min=clip_min, max=clip_max)

            pg_loss_1 = -ratio * micro_batch["advantages"].float()
            pg_loss_2 = -ratio_clipped * micro_batch["advantages"].float()
            policy_loss = torch.maximum(pg_loss_1, pg_loss_2)[micro_loss_mask].mean()

            if (not self.config.algorithm.use_kl_in_reward) and kl_loss_coef != 0.0:
                micro_ref_logprobs = micro_batch["ref_logprobs"].float()
                kl = (new_logprobs - micro_ref_logprobs)[micro_loss_mask].mean()
                policy_loss = policy_loss + kl_loss_coef * kl

            value_diff = new_values - micro_batch["returns"].float()
            value_loss = (value_diff**2)[micro_loss_mask].mean()

            contribution = num_of_micro_valid_tokens / sum_of_valid_tokens
            total_loss = policy_loss + value_loss

            if self.config.actor.pipeline_parallel_size > 1:
                policy_loss = self.actor.convert_tensor_to_micro_loss(policy_loss, micro_idx)
                (policy_loss * contribution).backward()
                value_loss = self.critic.convert_tensor_to_micro_loss(value_loss, micro_idx)
                (value_loss * contribution).backward()
            else:
                (total_loss * contribution).backward()

            sum_of_total_losses += total_loss.detach() * num_of_micro_valid_tokens
            sum_of_policy_losses += policy_loss.detach() * num_of_micro_valid_tokens
            sum_of_value_losses += value_loss.detach() * num_of_micro_valid_tokens
            num_of_updates += 1

        if num_of_updates == 0:
            self.actor_optimizer.zero_grad(set_to_none=True)
            self.critic_optimizer.zero_grad(set_to_none=True)
            return {"skipped": True}

        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.optim.clip_grad)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.optim.clip_grad)

        self.actor_optimizer.step()
        self.critic_optimizer.step()

        if self.actor_scheduler is not None:
            self.actor_scheduler.step()
        if self.critic_scheduler is not None:
            self.critic_scheduler.step()

        if self.config.actor.data_parallel_size > 1:
            dist.all_reduce(sum_of_total_losses, op=dist.ReduceOp.SUM, group=self.mpu.get_group(ParallelMode.DATA))
            dist.all_reduce(sum_of_policy_losses, op=dist.ReduceOp.SUM, group=self.mpu.get_group(ParallelMode.DATA))
            dist.all_reduce(sum_of_value_losses, op=dist.ReduceOp.SUM, group=self.mpu.get_group(ParallelMode.DATA))
            dist.all_reduce(sum_of_valid_tokens, op=dist.ReduceOp.SUM, group=self.mpu.get_group(ParallelMode.DATA))

        total_loss = (sum_of_total_losses / sum_of_valid_tokens.clamp_min(1.0)).item()
        policy_loss = (sum_of_policy_losses / sum_of_valid_tokens.clamp_min(1.0)).item()
        value_loss = (sum_of_value_losses / sum_of_valid_tokens.clamp_min(1.0)).item()

        actor_lr = self.actor_optimizer.param_groups[0]["lr"]
        critic_lr = self.critic_optimizer.param_groups[0]["lr"]

        return {
            "skipped": False,
            "num_sequences": num_sequences,
            "num_micro_batches": num_of_micro_batches,
            "num_updates": num_of_updates,
            "loss_total": float(total_loss),
            "loss_policy": float(policy_loss),
            "loss_value": float(value_loss),
            "actor_lr": float(actor_lr),
            "critic_lr": float(critic_lr),
        }

    def save_parallelized(self, save_dir: str):
        self.actor.save_parallelized(save_dir)
        if self.mpu is None or self.mpu.get_global_rank() == 0:
            self.tokenizer.save_pretrained(save_dir)
        return {"ok": True, "save_dir": save_dir}
