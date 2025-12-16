import torch

from nanorlhf import nanoray
from nanorlhf.nanoverl.utils.packing_utils import split_packed_batch, unpack_sequences, repack_sequences
from nanorlhf.nanovllm import SamplingParams
from nanorlhf.nanovllm.core.scheduler import Scheduler
from nanorlhf.nanovllm.core.sequence import Sequence


class RolloutManager:

    def __init__(self, config, rollouts):
        self.config = config
        self.workers = rollouts

        self.tensor_parallel_size = int(config.rollout.tensor_parallel_size)
        self.data_parallel_size = int(config.rollout.data_parallel_size)
        self.global_world_size = self.tensor_parallel_size * self.data_parallel_size

        self.schedulers = self.create_schedulers()
        self.sampling_params = SamplingParams(temperature=0.0, max_tokens=config.rollout.max_model_len)

    def create_schedulers(self):
        schedulers = []
        for data_parallel_rank in range(self.data_parallel_size):
            worker = self.workers[data_parallel_rank][0]
            scheduler = Scheduler(nanoray.get(worker.get_rollout_config.remote(blocking=True)))
            schedulers.append(scheduler)
        return schedulers

    def rollout(self, batches):
        # full batches -> data parallel batches
        data_parallel_batches = self.split_data_parallel_batch(batches)

        # data parallel batches -> data parallel unpacked batches
        data_parallel_unpacked_batches = []
        for data_parallel_batch in data_parallel_batches:
            unpacked_batch = unpack_sequences(
                input_ids=data_parallel_batch["input_ids"],
                position_ids=data_parallel_batch["position_ids"],
                reward_model_list=data_parallel_batch["reward_model"],
            )
            data_parallel_unpacked_batches.append(unpacked_batch)

        # data parallel unpacked batches -> add requests
        data_parallel_outputs = []
        for data_parallel_rank, unpacked_batch in enumerate(data_parallel_unpacked_batches):
            output_batch = self.add_request(data_parallel_rank, unpacked_batch)
            data_parallel_outputs.append(output_batch)

        # run model until all sequences are finished
        self.run_model()

        # packed data parallel outputs -> repacked outputs
        return self.repack_outputs(data_parallel_unpacked_batches, data_parallel_outputs)

    def split_data_parallel_batch(self, batch):
        assert "cu_seq_lens_q" in batch
        cu_seq_lens_q = batch["cu_seq_lens_q"]
        data_parallel_chunks = []
        for data_parallel_rank in range(self.data_parallel_size):
            data_parallel_chunk = split_packed_batch(
                batch=batch,
                chunk_idx=data_parallel_rank,
                num_chunks=self.data_parallel_size,
                cu_seq_lens=cu_seq_lens_q,
            )
            data_parallel_chunks.append(data_parallel_chunk)
        return data_parallel_chunks

    def add_request(self, data_parallel_rank, unpacked_prompts):
        scheduler = self.schedulers[data_parallel_rank]
        sequences = []
        for prompt in unpacked_prompts:
            token_ids = prompt["input_ids"][0].tolist()
            if len(token_ids) == 0:
                raise ValueError(f"Got empty prompt after unpack_sequences()\ntoken_ids: {token_ids}")
            sequence = Sequence(token_ids, sampling_params=self.sampling_params)
            scheduler.add(sequence)
            sequences.append(sequence)
        return sequences

    def run_model(self):
        while not all(scheduler.is_finished() for scheduler in self.schedulers):
            for data_parallel_rank in range(self.data_parallel_size):
                scheduler = self.schedulers[data_parallel_rank]
                if scheduler.is_finished():
                    continue

                sequences, is_prefill = scheduler.schedule()
                object_refs = []
                for tensor_parallel_rank in range(self.tensor_parallel_size):
                    runner = self.workers[data_parallel_rank][tensor_parallel_rank]
                    object_ref = runner.run.remote(sequences, is_prefill, blocking=False)
                    object_refs.append(object_ref)
                token_ids = nanoray.get(object_refs)[0]
                print(f"[R{data_parallel_rank}]: {token_ids}")
                scheduler.postprocess(sequences, token_ids)

    def repack_outputs(self, data_parallel_unpacked_batches, data_parallel_outputs):
        total_tokens_all, response_tokens_all = [], []

        for unpacked_batch, outputs in zip(data_parallel_unpacked_batches, data_parallel_outputs):
            for prompt, output in zip(unpacked_batch, outputs):
                response_ids = torch.tensor(output.completion_token_ids, dtype=torch.long).unsqueeze(0)
                response_position_ids = torch.arange(response_ids.numel(), dtype=torch.long).unsqueeze(0)
                response_tokens = {
                    "input_ids": response_ids,
                    "position_ids": response_position_ids,
                    "loss_mask": torch.ones_like(response_ids),
                }
                response_tokens_all.append(response_tokens)

                prompt_tokens = {
                    "input_ids": prompt["input_ids"],
                    "position_ids": prompt["position_ids"],
                    "loss_mask": torch.zeros_like(prompt["input_ids"]),
                    "reward_model": prompt["reward_model"],
                }
                total_tokens = repack_sequences([prompt_tokens, response_tokens])
                total_tokens_all.append(total_tokens)
        return total_tokens_all, response_tokens_all
