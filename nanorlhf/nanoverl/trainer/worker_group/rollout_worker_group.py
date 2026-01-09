from concurrent.futures import ThreadPoolExecutor

import torch

from nanorlhf import nanoray
from nanorlhf.nanoverl.utils.packing_utils import split_packed_batch, unpack_sequences, repack_sequences
from nanorlhf.nanovllm import SamplingParams
from nanorlhf.nanovllm.core.scheduler import Scheduler
from nanorlhf.nanovllm.core.sequence import Sequence


class RolloutWorkerGroup:
    """
    Worker group that manages rollout workers for model inference during rollouts.

    Args:
        config: Configuration object containing rollout settings.
        workers: List of remote worker instances.
    """

    def __init__(self, config, workers):
        self.config = config
        self.workers = workers

        self.tensor_parallel_size = int(config.rollout.tensor_parallel_size)
        self.data_parallel_size = int(config.rollout.data_parallel_size)
        self.global_world_size = self.tensor_parallel_size * self.data_parallel_size

        self.async_executor = ThreadPoolExecutor(max_workers=1)
        self.schedulers = self.create_schedulers()
        self.sampling_params = SamplingParams(
            top_p=1.0,
            temperature=float(config.rollout.temperature),
            max_tokens=int(config.rollout.max_response_len),
        )

    def create_schedulers(self):
        """
        Create schedulers for each data parallel rank.

        Returns:
            List[Scheduler]: List of schedulers for each data parallel rank.
        """
        schedulers = []
        for data_parallel_rank in range(self.data_parallel_size):
            worker = self.workers[data_parallel_rank][0]
            scheduler = Scheduler(nanoray.get(worker.get_rollout_config.remote(blocking=True)))
            schedulers.append(scheduler)
        return schedulers

    def generate(self, batches):
        """
        Generate tokens for the given batches.

        Args:
            batches: The input batches for generation.

        Returns:
            total_tokens_repacked: The repacked total tokens batch.
            prompt_tokens_unpacked: The unpacked prompt tokens.
            response_tokens_unpacked: The unpacked response tokens.
        """
        # full batches -> data parallel batches
        data_parallel_batches = self.split_data_parallel_batch(batches)

        # data parallel batches -> data parallel unpacked batches
        data_parallel_unpacked_batches = []
        for data_parallel_rank, data_parallel_batch_per_rank in enumerate(data_parallel_batches):
            unpacked_batch_per_rank = unpack_sequences(
                input_ids=data_parallel_batch_per_rank["input_ids"],
                position_ids=data_parallel_batch_per_rank["position_ids"],
                reward_model_list=data_parallel_batch_per_rank["reward_model"],
            )
            data_parallel_unpacked_batches.append(unpacked_batch_per_rank)

        # data parallel unpacked batches -> add requests
        data_parallel_outputs = []
        for data_parallel_rank, unpacked_batch_per_rank in enumerate(data_parallel_unpacked_batches):
            output_batch = self.add_request(data_parallel_rank, unpacked_batch_per_rank)
            data_parallel_outputs.append(output_batch)

        # run model until all sequences are finished
        self.run_model()

        # packed data parallel outputs -> repacked outputs
        total_tokens_repacked, prompt_tokens_unpacked, response_tokens_unpacked = self.repack_outputs(
            data_parallel_unpacked_batches, data_parallel_outputs
        )
        return total_tokens_repacked, prompt_tokens_unpacked, response_tokens_unpacked

    def async_generate(self, batches):
        """
        Asynchronously generate tokens for the given batches.

        Args:
            batches: The input batches for generation.
        """
        return self.async_executor.submit(self.generate, batches)

    def split_data_parallel_batch(self, batch):
        """
        Split the input batch into data parallel chunks.

        Args:
            batch: The input batch to be split.

        Returns:
            List[dict]: List of data parallel chunks.
        """
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
            batch_size_per_dp_rank = int((data_parallel_chunk["position_ids"][0] == 0).sum().item())
            assert len(data_parallel_chunk["reward_model"]) == batch_size_per_dp_rank, (
                f"[dp_rank={data_parallel_rank}] reward_model len={len(data_parallel_chunk['reward_model'])} "
                f"!= num_seqs_local={batch_size_per_dp_rank}"
            )
            data_parallel_chunks.append(data_parallel_chunk)
        return data_parallel_chunks

    def add_request(self, data_parallel_rank, unpacked_prompts):
        """
        Add generation requests to the scheduler for the given data parallel rank.

        Args:
            data_parallel_rank: The data parallel rank.
            unpacked_prompts: The unpacked prompts for the given rank.

        Returns:
            List[Sequence]: List of sequences added to the scheduler.
        """
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
        """
        Run the model until all sequences are finished.
        """
        while not all(scheduler.is_finished() for scheduler in self.schedulers):
            launches = []  # (dp_rank, sequences, object_refs)
            flat_object_refs = []

            for data_parallel_rank in range(self.data_parallel_size):
                scheduler = self.schedulers[data_parallel_rank]
                if scheduler.is_finished():
                    continue

                sequences, is_prefill = scheduler.schedule()
                object_refs = []
                for tensor_parallel_rank in range(self.tensor_parallel_size):
                    runner = self.workers[data_parallel_rank][tensor_parallel_rank]
                    object_ref = runner.generate.remote(sequences, is_prefill, blocking=False)
                    object_refs.append(object_ref)

                launches.append((data_parallel_rank, sequences, object_refs))
                flat_object_refs.extend(object_refs)

            if not launches:
                continue

            flat_results = nanoray.get(flat_object_refs)
            cu_tp_size = 0
            for data_parallel_rank, sequences, object_refs in launches:
                tp_size = len(object_refs)
                results = flat_results[cu_tp_size : cu_tp_size + tp_size]
                cu_tp_size += tp_size
                scheduler = self.schedulers[data_parallel_rank]
                scheduler.postprocess(sequences, results[0])

    def repack_outputs(self, data_parallel_unpacked_batches, data_parallel_outputs):
        """
        Repack the outputs from data parallel unpacked batches and outputs.

        Args:
            data_parallel_unpacked_batches: The unpacked batches per data parallel rank.
            data_parallel_outputs: The outputs per data parallel rank.

        Returns:
            total_tokens_repacked: The repacked total tokens batch.
            prompt_tokens_unpacked: The unpacked prompt tokens.
            response_tokens_unpacked: The unpacked response tokens.
        """
        total_tokens_repacked, prompt_tokens_unpacked, response_tokens_unpacked = [], [], []
        for unpacked_batch_per_rank, outputs_per_rank in zip(data_parallel_unpacked_batches, data_parallel_outputs):
            for prompt, output in zip(unpacked_batch_per_rank, outputs_per_rank):
                prompt_tokens = {
                    "input_ids": prompt["input_ids"],
                    "position_ids": prompt["position_ids"],
                    "loss_mask": torch.zeros_like(prompt["input_ids"]),
                }
                prompt_tokens_unpacked.append(prompt_tokens)

                response_ids = torch.tensor(output.completion_token_ids, dtype=torch.long).unsqueeze(0)
                response_position_ids = torch.arange(response_ids.numel(), dtype=torch.long).unsqueeze(0)
                response_tokens = {
                    "input_ids": response_ids,
                    "position_ids": response_position_ids + prompt["position_ids"].size(-1),
                    "loss_mask": torch.ones_like(response_ids),
                    "reward_model": prompt["reward_model"],
                }
                response_tokens_unpacked.append(response_tokens)

                total_tokens = repack_sequences([prompt_tokens, response_tokens])
                total_tokens_repacked.append(total_tokens)

        total_tokens_repacked = repack_sequences(total_tokens_repacked)
        return total_tokens_repacked, prompt_tokens_unpacked, response_tokens_unpacked

    def sync_actor_to_rollout(self):
        """
        Synchronize parameters from the actor model to the rollout model across all workers.

        Returns:
            List of object references for the synchronization tasks.
        """
        object_refs = []
        for data_parallel_rank in range(self.data_parallel_size):
            for tensor_parallel_rank in range(self.tensor_parallel_size):
                worker = self.workers[data_parallel_rank][tensor_parallel_rank]
                object_ref = worker.sync_actor_to_rollout.remote(blocking=False)
                object_refs.append(object_ref)
        return object_refs
