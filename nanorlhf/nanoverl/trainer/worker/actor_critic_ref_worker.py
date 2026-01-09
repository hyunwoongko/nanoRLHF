from collections import deque
from typing import Optional

import torch
from torch.nn import functional as F
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoModelForTokenClassification, AutoTokenizer

from nanorlhf import nanoray
from nanorlhf.kernels import patch_kernel
from nanorlhf.nanotron import MPU, TensorParallel, PipelineParallel, DataParallel, ParallelMode
from nanorlhf.nanotron.core.tp.loss import VocabParallelCrossEntropyFunction
from nanorlhf.nanoverl.utils.experience import Experience
from nanorlhf.nanoverl.utils.metric_utils import (
    MetricsAccumulator,
    compute_explained_variance,
    accumulate_ppo_micro_metrics,
)
from nanorlhf.nanoverl.utils.optim_utils import get_optimizer_param_groups, get_scheduler
from nanorlhf.nanoverl.utils.packing_utils import split_packed_batch
from nanorlhf.nanoverl.utils.sync_utils import ParameterSyncManager
from nanorlhf.nanovllm.utils.config import MIN_TEMPERATURE


def initialize_model(config, rank, mpu: MPU = None, role: str = "actor"):
    """
    Initialize the model, optimizer, and model parallel unit (MPU) based on the specified role.

    Args:
        config: Configuration object containing model and training settings.
        rank (int): The global rank of the current process.
        mpu (MPU, optional): pre-initialized model parallel unit. Defaults to None.
        role (str): The role of the model to initialize. Must be one of 'actor', 'ref', or 'critic'.

    Returns:
        tuple: A tuple containing the initialized model, optimizer (or None for 'ref'), and MPU.
    """
    assert role in ["actor", "ref", "critic"], "role must be one of ['actor', 'ref', 'critic']"

    if role == "critic":
        model = AutoModelForTokenClassification.from_pretrained(
            config.actor.model_name_or_path,
            trust_remote_code=True,
            torch_dtype=torch.float32,
            num_labels=1,
        )
        # turn off dropout
        model.dropout = torch.nn.Identity()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            config.actor.model_name_or_path,
            trust_remote_code=True,
            torch_dtype=torch.float32,
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
        micro_batch_size=config.data.ppo_micro_batch_size_per_gpu,
        gradient_checkpointing_enable=config.actor.gradient_checkpointing_enable,
    )
    model, optimizer = DataParallel(
        model,
        mpu=mpu,
        optimizer=optimizer,
        zero_stage=0 if role == "ref" else config.actor.zero_stage,
        accum_steps=max(1, config.data.ppo_mini_batch_size_per_gpu // config.data.ppo_micro_batch_size_per_gpu),
    )
    model.parallelize()
    model = patch_kernel(model)
    return model, optimizer, mpu


@nanoray.remote
class ActorCriticRefWorker:
    """
    The Actor-Critic-Reference worker for PPO training.

    Args:
        config: Configuration object containing model and training settings.
        rank (int): The global rank of the current process.
        total_steps (int): Total number of training steps.
    """
    def __init__(self, config, rank, total_steps: int):
        self.config = config
        self.rank = rank
        self.experience_buffer = deque(maxlen=2)
        self.update_idx_for_rng = 0
        self.rollout_temperature = float(config.rollout.temperature)

        # Data is already tokenized so we don't use tokenizer here, but for saving it in the checkpoint path together.
        self.tokenizer = AutoTokenizer.from_pretrained(config.actor.tokenizer_name_or_path, trust_remote_code=True)

        self.actor, self.actor_optimizer, self.mpu = initialize_model(config, rank, role="actor")
        self.ref, _, _ = initialize_model(config, rank, mpu=self.mpu, role="ref")
        self.critic, self.critic_optimizer, _ = initialize_model(config, rank, mpu=self.mpu, role="critic")

        ppo_epochs = int(config.data.ppo_epochs)
        num_sequences_per_rank = int(config.data.train_batch_size // config.actor.data_parallel_size)
        num_mini_batches_per_update = int(num_sequences_per_rank // config.data.ppo_mini_batch_size_per_gpu)
        total_optimizer_steps = int(total_steps) * ppo_epochs * num_mini_batches_per_update

        self.actor_scheduler = get_scheduler(config, self.actor_optimizer, total_optimizer_steps)
        self.critic_scheduler = get_scheduler(config, self.critic_optimizer, total_optimizer_steps)
        self.parameter_sync_manager = ParameterSyncManager(self.actor, self.mpu, self.config, is_rollout=False)
        self.metrics = MetricsAccumulator(device=torch.device(torch.cuda.current_device()))

    def whiten_advantages(self, advantages, loss_mask, eps=1e-8):
        """
        Whitens the advantages using only the valid tokens indicated by the loss_mask.

        Args:
            advantages (torch.Tensor): The advantages tensor of shape (batch_size, sequence_length).
            loss_mask (torch.Tensor): The loss mask tensor of shape (batch_size, sequence_length).
            eps (float): A small value to avoid division by zero.

        Returns:
            torch.Tensor: The whitened advantages tensor of the same shape as input advantages.
        """
        assert advantages.dim() == 2 and loss_mask.dim() == 2
        loss_mask = loss_mask.to(dtype=torch.bool, device=advantages.device)

        if not bool(loss_mask.any()):
            return torch.zeros_like(advantages)

        masked_advantages = advantages[loss_mask]
        mean = masked_advantages.mean()
        var = masked_advantages.var(unbiased=True)
        std = torch.sqrt(var + eps)

        whitened = (advantages - mean) / std
        whitened = torch.where(loss_mask, whitened, torch.zeros_like(whitened))
        return whitened

    def compute_kl_per_token(self, actor_logprobs, ref_logprobs):
        """
        Compute the KL divergence per token between the actor and reference log probabilities.

        Args:
            actor_logprobs (torch.Tensor): Log probabilities from the actor model.
            ref_logprobs (torch.Tensor): Log probabilities from the reference model.

        Returns:
            torch.Tensor: KL divergence per token.
        """
        # This follows the k3 implementation of kl divergence approximation:
        # http://joschu.net/blog/kl-approx.html
        kl_per_token = (ref_logprobs - actor_logprobs).float()
        return torch.expm1(kl_per_token) - kl_per_token

    def compute_token_logprobs(
        self,
        model,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        loss_mask: torch.Tensor,
        enable_grad: bool,
        logits: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Compute the log probabilities of tokens given the model and input.

        Args:
            model: The language model to compute log probabilities.
            input_ids (torch.Tensor): Input token IDs of shape (batch_size, sequence_length).
            position_ids (torch.Tensor): Position IDs of shape (batch_size, sequence_length).
            loss_mask (torch.Tensor): Loss mask of shape (batch_size, sequence_length).
            enable_grad (bool): Whether to enable gradient computation.
            logits (torch.Tensor, optional): Precomputed logits to use instead of model forward pass.
            temperature (float): Temperature for scaling logits.

        Returns:
            torch.Tensor: Log probabilities of shape (batch_size, sequence_length).
        """
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

        if temperature <= 0.0:
            temperature = MIN_TEMPERATURE
        logits_shifted = logits_shifted / temperature

        if self.mpu.get_world_size(ParallelMode.TENSOR) <= 1:
            logprobs = F.log_softmax(logits_shifted.float(), dim=-1)
            token_logprobs = logprobs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        else:
            nll = VocabParallelCrossEntropyFunction.apply(logits_shifted, targets, self.mpu, ParallelMode.TENSOR)
            token_logprobs = (-nll).float()

        full = torch.zeros((batch_size, sequence_length), device=input_ids.device, dtype=torch.float32)
        full[:, 1:] = token_logprobs

        # inter sequence tokens must not contribute to the loss.
        full = full.masked_fill(position_ids == 0, 0.0)
        # apply the loss mask provided from the dataset.
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
        """
        Compute the state values given the critic model and input.

        Args:
            input_ids (torch.Tensor): Input token IDs of shape (batch_size, sequence_length).
            position_ids (torch.Tensor): Position IDs of shape (batch_size, sequence_length).
            shift_for_actions (bool): Whether to shift the values for action alignment.
            enable_grad (bool): Whether to enable gradient computation.
            logits (torch.Tensor, optional): Precomputed logits to use instead of model forward pass.

        Returns:
            torch.Tensor: State values of shape (batch_size, sequence_length).
        """
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

        # inter sequence tokens must not contribute to the loss.
        values = values.masked_fill(position_ids == 0, 0.0)
        # don't need to apply loss mask because they will be masked in loss computation later.
        return values

    def assign_sequence_rewards_to_tokens(self, experience, reward_scores, num_sequences):
        """
        Assign sequence-level rewards to the last token of the response in each sequence.

        Args:
            experience (Experience): The experience object containing input data.
            reward_scores (list): List of reward scores for each sequence.
            num_sequences (int): Number of sequences in the batch.

        Returns:
            Experience: The updated experience object with assigned rewards.
        """
        assert len(reward_scores) == num_sequences
        reward_scores = [float(x) for x in reward_scores]
        position_ids = experience.position_ids
        loss_mask = experience.loss_mask

        rewards = torch.zeros_like(loss_mask, dtype=torch.float32)
        starts = (position_ids[0] == 0).nonzero(as_tuple=False).flatten().tolist()
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
        """
        Compute returns and advantages using GAE-Lambda.

        Args:
            experience (Experience): The experience object containing input data.

        Returns:
            Experience: The updated experience object with computed returns and advantages.
        """
        gamma = float(self.config.algorithm.gamma)
        lam = float(self.config.algorithm.lam)
        kl_loss_coef = float(self.config.algorithm.kl_loss_coef)
        use_kl_in_reward = bool(self.config.algorithm.use_kl_in_reward)

        loss_mask = experience.loss_mask[0].to(torch.bool)
        position_ids = experience.position_ids[0]
        rewards = experience.rewards[0].float()
        values = experience.old_values[0].float()

        if use_kl_in_reward and kl_loss_coef != 0.0:
            kl = self.compute_kl_per_token(experience.old_logprobs[0], experience.ref_logprobs[0])
            rewards = rewards - (kl_loss_coef * kl * loss_mask.to(kl.dtype))

        starts = (position_ids == 0).nonzero(as_tuple=False).flatten().tolist()
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

        advantages = self.whiten_advantages(
            advantages=advantages.unsqueeze(0),
            loss_mask=experience.loss_mask,
        )

        experience.advantages = advantages
        experience.returns = returns.unsqueeze(0)
        return experience

    @torch.inference_mode()
    def make_experience(self, input_batch, reward_scores):
        """
        Generate experience from the input batch and reward scores.

        Args:
            input_batch (dict): A batch of input data containing 'input_ids', 'position_ids', and 'loss_mask'.
            reward_scores (list): List of reward scores for each sequence in the batch.

        Returns:
            dict: A dictionary containing rollout statistics.
        """
        device = torch.cuda.current_device()
        input_ids = input_batch["input_ids"].to(device, non_blocking=True)
        position_ids = input_batch["position_ids"].to(device, non_blocking=True)
        loss_mask = input_batch["loss_mask"].to(device, non_blocking=True)

        num_sequences = int((position_ids == 0).sum().item())
        micro_batch_size = self.config.data.ppo_micro_batch_size_per_gpu
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

        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
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
                    temperature=self.rollout_temperature,
                )
                micro_ref_logprobs = self.compute_token_logprobs(
                    self.ref,
                    micro_batch["input_ids"],
                    micro_batch["position_ids"],
                    micro_batch["loss_mask"],
                    enable_grad=False,
                    logits=ref_logits,
                    temperature=self.rollout_temperature,
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
        num_sequences = int((position_ids == 0).sum().item())

        experience = self.assign_sequence_rewards_to_tokens(experience, reward_scores, num_sequences)
        experience = self.compute_returns_and_advantages(experience)
        self.experience_buffer.append(experience.to("cpu", pin_memory=True, detach=True))

        if num_response_tokens > 0:
            rollout_approx_kl = float(
                self.compute_kl_per_token(old_logprobs, ref_logprobs)[response_mask].mean().item()
            )
            rollout_mean_logprob = float(old_logprobs[response_mask].mean().item())
        else:
            rollout_approx_kl = 0.0
            rollout_mean_logprob = 0.0

        return {
            "num_total_tokens": num_total_tokens,
            "num_response_tokens": num_response_tokens,
            "num_sequences": num_sequences,
            "rollout_approx_kl": rollout_approx_kl,
            "rollout_mean_logprob": rollout_mean_logprob,
        }

    def step(self):
        """
        Perform a single PPO update step using the experience in the buffer.

        Returns:
            dict: A dictionary containing PPO training metrics.
        """
        self.update_idx_for_rng += 1
        device = torch.cuda.current_device()
        experience = self.experience_buffer.popleft().to(device)

        batch_starts = (experience.position_ids[0] == 0).nonzero(as_tuple=False).flatten()
        num_sequences = int(batch_starts.numel())

        micro_batch_size = int(self.config.data.ppo_micro_batch_size_per_gpu)
        mini_batch_size = int(self.config.data.ppo_mini_batch_size_per_gpu)
        ppo_epochs = int(self.config.data.ppo_epochs)

        num_mini_batches = num_sequences // mini_batch_size
        experience_dict = experience.to_dict()
        mini_batches = [split_packed_batch(experience_dict, idx, num_mini_batches) for idx in range(num_mini_batches)]

        clip_min = 1.0 - float(self.config.algorithm.clip_ratio_low)
        clip_max = 1.0 + float(self.config.algorithm.clip_ratio_high)
        clip_value = float(self.config.algorithm.clip_ratio_value)
        kl_loss_coef = float(self.config.algorithm.kl_loss_coef)
        vf_loss_coef = float(self.config.algorithm.vf_loss_coef)

        self.metrics.reset()
        total_optimizer_steps = 0
        total_micro_updates = 0

        # https://github.com/huggingface/trl/blob/6a718789814bf1b653cbe213fdabb1d6ea31989f/trl/experimental/ppo/ppo_trainer.py#L788
        # we need triple for-loops: ppo-epoch -> mini-batch -> micro-batch
        for ppo_epoch_idx in range(ppo_epochs):
            # shuffle the mini-batches for some randomness
            rng = torch.Generator(device="cpu")
            rng.manual_seed(int(self.config.actor.seed) + (100 * self.update_idx_for_rng) + (10000 * ppo_epoch_idx))
            mini_perm = torch.randperm(num_mini_batches, generator=rng).tolist()

            for random_mini_idx in mini_perm:
                mini_batch = mini_batches[random_mini_idx]
                mini_starts = (mini_batch["position_ids"][0] == 0).nonzero(as_tuple=False).flatten()
                mini_num_sequences = int(mini_starts.numel())

                num_micro_batches = mini_num_sequences // micro_batch_size
                micro_batches = [
                    split_packed_batch(mini_batch, idx, num_micro_batches) for idx in range(num_micro_batches)
                ]

                self.actor_optimizer.zero_grad(set_to_none=True)
                self.critic_optimizer.zero_grad(set_to_none=True)

                micro_valid_tokens_list = [mb["loss_mask"].sum().to(device).float() for mb in micro_batches]
                mini_valid_tokens = torch.stack(micro_valid_tokens_list).sum().clamp_min(1.0)
                mini_micro_updates = 0

                if self.config.actor.pipeline_parallel_size > 1:
                    actor_micro_iterator = self.actor(
                        mini_batch["input_ids"],
                        position_ids=mini_batch["position_ids"],
                        attention_mask=None,
                        use_cache=False,
                    )
                    critic_micro_iterator = self.critic(
                        mini_batch["input_ids"],
                        position_ids=mini_batch["position_ids"],
                        attention_mask=None,
                        use_cache=False,
                    )
                    micro_batch_iterator = enumerate(zip(actor_micro_iterator, critic_micro_iterator))
                else:
                    micro_batch_iterator = enumerate(micro_batches)

                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    for micro_idx, micro_input_or_output in micro_batch_iterator:
                        micro_batch = micro_batches[micro_idx]

                        micro_loss_mask = micro_batch["loss_mask"].to(torch.bool)
                        micro_valid_tokens = micro_loss_mask.sum().to(device).float()
                        if micro_valid_tokens.item() == 0:
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
                            temperature=self.rollout_temperature,
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

                        advantages = micro_batch["advantages"].float()
                        pg_loss_1 = -ratio * advantages
                        pg_loss_2 = -ratio_clipped * advantages
                        policy_loss = torch.maximum(pg_loss_1, pg_loss_2)[micro_loss_mask].mean()

                        ref_logprobs = micro_batch["ref_logprobs"].float()
                        if (not self.config.algorithm.use_kl_in_reward) and kl_loss_coef != 0.0:
                            kl = self.compute_kl_per_token(new_logprobs, ref_logprobs)[micro_loss_mask].mean()
                            policy_loss = policy_loss + (kl_loss_coef * kl)

                        old_values = micro_batch["old_values"].float()
                        returns = micro_batch["returns"].float()

                        if clip_value > 0.0:
                            new_values_clipped = old_values + (new_values - old_values).clamp(-clip_value, clip_value)
                            value_loss_1 = (new_values - returns) ** 2
                            value_loss_2 = (new_values_clipped - returns) ** 2
                            value_loss = 0.5 * torch.maximum(value_loss_1, value_loss_2)[micro_loss_mask].mean()
                        else:
                            value_loss = 0.5 * ((new_values - returns) ** 2)[micro_loss_mask].mean()

                        value_loss = value_loss * vf_loss_coef
                        total_loss = policy_loss + value_loss if vf_loss_coef != 0 else policy_loss
                        contribution = micro_valid_tokens / mini_valid_tokens

                        if self.config.actor.pipeline_parallel_size > 1:
                            policy_loss_pp = self.actor.convert_tensor_to_micro_loss(policy_loss, micro_idx)
                            (policy_loss_pp * contribution).backward()

                            if vf_loss_coef != 0:
                                value_loss_pp = self.critic.convert_tensor_to_micro_loss(value_loss, micro_idx)
                                (value_loss_pp * contribution).backward()
                        else:
                            (total_loss * contribution).backward()

                        accumulate_ppo_micro_metrics(
                            metrics=self.metrics,
                            total_loss=total_loss,
                            policy_loss=policy_loss,
                            value_loss=value_loss,
                            ratio=ratio,
                            ratio_clipped=ratio_clipped,
                            new_logprobs=new_logprobs,
                            ref_logprobs=ref_logprobs,
                            new_values=new_values,
                            returns=returns,
                            advantages=advantages,
                            micro_loss_mask=micro_loss_mask,
                            micro_valid_tokens=micro_valid_tokens,
                            compute_kl_per_token_fn=self.compute_kl_per_token,
                        )
                        mini_micro_updates += 1

                if mini_micro_updates == 0:
                    self.actor_optimizer.zero_grad(set_to_none=True)
                    self.critic_optimizer.zero_grad(set_to_none=True)
                    continue

                actor_grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.actor.parameters(),
                    self.config.optim.clip_grad,
                )
                critic_grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.critic.parameters(),
                    self.config.optim.clip_grad,
                )

                self.actor_optimizer.step()
                self.critic_optimizer.step()

                if self.actor_scheduler is not None:
                    self.actor_scheduler.step()
                if self.critic_scheduler is not None:
                    self.critic_scheduler.step()

                self.metrics.add_many_scalars(
                    with_sq=False,
                    actor_grad_norm=actor_grad_norm,
                    critic_grad_norm=critic_grad_norm,
                )

                total_optimizer_steps += 1
                total_micro_updates += mini_micro_updates

        if self.config.actor.data_parallel_size > 1:
            self.metrics.all_reduce(group=self.mpu.get_group(ParallelMode.DATA))

        output = self.metrics.to_dict()
        output.update(
            {
                "num_optimizer_steps": int(total_optimizer_steps),
                "num_micro_updates": int(total_micro_updates),
                "actor_lr": float(self.actor_optimizer.param_groups[0]["lr"]),
                "critic_lr": float(self.critic_optimizer.param_groups[0]["lr"]),
                "explained_variance": compute_explained_variance(output),
            }
        )
        return output

    def save_parallelized(self, save_dir: str):
        """
        Save the parallelized actor model and tokenizer to the specified directory.

        Args:
            save_dir (str): The directory to save the model and tokenizer.

        Returns:
            dict: A dictionary indicating the save status.
        """
        self.actor.save_parallelized(save_dir)
        if self.mpu is None or self.mpu.get_global_rank() == 0:
            self.tokenizer.save_pretrained(save_dir)
        return {"ok": True, "save_dir": save_dir}

    def sync_actor_to_rollout(self):
        """
        Synchronize the actor model parameters to the rollout workers.
        """
        return self.parameter_sync_manager.sync_actor_to_rollout()
