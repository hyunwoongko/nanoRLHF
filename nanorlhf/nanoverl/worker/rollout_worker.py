import torch

from nanorlhf import nanoray
from nanorlhf.nanoverl.utils.packing_utils import unpack_sequences, repack_sequences
from nanorlhf.nanovllm import SamplingParams
from nanorlhf.nanovllm.core.model_runner import ModelRunner
from nanorlhf.nanovllm.core.scheduler import Scheduler
from nanorlhf.nanovllm.core.sequence import Sequence
from nanorlhf.nanovllm.utils.config import NanoVLLMConfig


@nanoray.actor
class RolloutWorker:
    def __init__(self, config, rank, total_steps: int):
        self.config = config
        self.rank = rank
        self.total_steps = total_steps

        rollout_config = NanoVLLMConfig(
            model=config.rollout.model_name_or_path,
            max_num_batched_tokens=config.rollout.max_num_batched_tokens,
            max_num_seqs=config.rollout.max_num_seqs,
            max_model_len=config.rollout.max_model_len,
            gpu_memory_utilization=config.rollout.gpu_memory_utilization,
            kvcache_block_size=config.rollout.kvcache_block_size,
            tensor_parallel_size=config.actor.tensor_parallel_size,
            host=config.actor.host,
            port=config.actor.port,
            backend=config.actor.backend,
            seed=config.actor.seed,
        )
        self.rollout_runner = ModelRunner.cls(rollout_config, rank)
        self.rollout_scheduler = Scheduler(self.rollout_runner.get_config())
        self.rollout_sampling_params = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            max_tokens=config.rollout.max_model_len,
        )

    @torch.inference_mode()
    def rollout_sequences(self, input_batch: dict):
        device = input_batch["input_ids"].device
        unpacked_sequences = unpack_sequences(
            input_ids=input_batch["input_ids"],
            position_ids=input_batch["position_ids"],
            reward_model_list=input_batch["reward_model"],
        )
        for unpacked_sequence in unpacked_sequences:
            rollout_sequence = Sequence(
                token_ids=unpacked_sequence["input_ids"],
                sampling_params=self.rollout_sampling_params,
            )
            self.rollout_scheduler.add(rollout_sequence)

        generated_outputs = {}
        while not self.rollout_scheduler.is_finished():
            sequences, is_prefill = self.rollout_scheduler.schedule()
            response_ids = self.rollout_runner.run(sequences, is_prefill)
            self.rollout_scheduler.postprocess(sequences, response_ids)
            for seq in sequences:
                if seq.is_finished and seq.seq_id not in generated_outputs:
                    generated_outputs[seq.seq_id] = seq.completion_token_ids

        generated_outputs = [generated_outputs[seq_id] for seq_id in sorted(generated_outputs.keys())]
        total_tokens_batch, response_tokens_batch = [], []
        for sequence_id in range(len(unpacked_sequences)):
            response_ids = torch.tensor(generated_outputs[sequence_id], device=device, dtype=torch.long).unsqueeze(0)
            response_position_ids = torch.arange(response_ids.numel(), device=device, dtype=torch.long).unsqueeze(0)
            response_tokens = {
                "input_ids": response_ids,
                "position_ids": response_position_ids,
                "loss_mask": torch.ones_like(response_ids, device=device),
            }
            response_tokens_batch.append(response_tokens)

            prompt_tokens = unpacked_sequences[sequence_id]
            prompt_tokens["loss_mask"] = torch.zeros_like(prompt_tokens["input_ids"], device=device)
            total_tokens = repack_sequences([prompt_tokens, response_tokens])
            total_tokens_batch.append(total_tokens)

        return total_tokens_batch, response_tokens_batch
