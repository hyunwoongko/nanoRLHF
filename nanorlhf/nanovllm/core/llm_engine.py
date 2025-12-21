from dataclasses import fields
from time import perf_counter
from typing import List

from tqdm import tqdm
from transformers import AutoTokenizer

from nanorlhf import nanoray
from nanorlhf.nanoray.api.initialization import NANORAY_BASE_PORT
from nanorlhf.nanovllm.core.model_runner import ModelRunner
from nanorlhf.nanovllm.core.scheduler import Scheduler
from nanorlhf.nanovllm.core.sequence import Sequence
from nanorlhf.nanovllm.utils.config import NanoVLLMConfig


class LLMEngine:
    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(NanoVLLMConfig)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = NanoVLLMConfig(model, **config_kwargs)

        self.config = config
        self.tensor_parallel_size = config.tensor_parallel_size
        self.data_parallel_size = config.data_parallel_size
        self.global_world_size = self.tensor_parallel_size * self.data_parallel_size

        self.tokenizer = AutoTokenizer.from_pretrained(config.model)
        config.eos = self.tokenizer.eos_token_id

        self.node_ids = self.init_ray(config)
        self.model_runners = self.create_model(config)

        model_runner_config = self.model_runners[0][0].get_config.remote(blocking=True)  # noqa
        self.schedulers = [Scheduler(nanoray.get(model_runner_config)) for _ in range(self.data_parallel_size)]
        self.round_robin_counter = 0

    def init_ray(self, config):
        nodes = {}
        if self.global_world_size > 1:
            for global_rank in range(self.global_world_size):
                nodes[f"node-{global_rank}"] = nanoray.NodeConfig(
                    cpus=4.0, gpus=1.0, rpc=True, host=config.host, port=NANORAY_BASE_PORT + global_rank
                )
        else:
            nodes["node-0"] = nanoray.NodeConfig(
                cpus=4.0, gpus=1.0, rpc=False, host=config.host, port=NANORAY_BASE_PORT
            )

        session = nanoray.init(nodes, default_node_id="node-0")
        return list(session.workers.keys())

    def create_model(self, config: NanoVLLMConfig):
        object_refs = []
        for data_parallel_rank in range(self.data_parallel_size):
            for tensor_parallel_rank in range(self.tensor_parallel_size):
                global_rank = data_parallel_rank * self.tensor_parallel_size + tensor_parallel_rank
                node_id = self.node_ids[global_rank % len(self.node_ids)]
                object_ref = ModelRunner.options(pinned_node_id=node_id).remote(
                    config, rank=global_rank, actor_config=None, blocking=False
                )
                object_refs.append(object_ref)

        resolved = nanoray.get(object_refs)
        runners: List[List[ModelRunner]] = []
        for data_parallel_rank in range(self.data_parallel_size):
            tensor_parallel_runners: List[ModelRunner] = []
            for tensor_parallel_rank in range(self.tensor_parallel_size):
                global_rank = data_parallel_rank * self.tensor_parallel_size + tensor_parallel_rank
                tensor_parallel_runners.append(resolved[global_rank])
            runners.append(tensor_parallel_runners)
        return runners

    def run_model(self, data_parallel_rank, sequences, is_prefill):
        object_refs = []
        for tensor_parallel_rank in range(self.tensor_parallel_size):
            runner = self.model_runners[data_parallel_rank][tensor_parallel_rank]
            object_refs.append(runner.run.remote(sequences, is_prefill, blocking=False))
        results = nanoray.get(object_refs)
        tokens = results[0]
        return tokens

    def add_request(self, prompt, sampling_params):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)

        sequence = Sequence(prompt, sampling_params)
        data_parallel_rank = self.round_robin_counter
        self.round_robin_counter = (self.round_robin_counter + 1) % self.data_parallel_size
        self.schedulers[data_parallel_rank].add(sequence)

    def step(self):
        all_outputs = []
        total_num_tokens = 0

        for data_parallel_rank in range(self.data_parallel_size):
            scheduler = self.schedulers[data_parallel_rank]
            if scheduler.is_finished():
                continue

            sequences, is_prefill = scheduler.schedule()
            token_ids = self.run_model(data_parallel_rank, sequences, is_prefill)
            scheduler.postprocess(sequences, token_ids)

            outputs = [
                (sequence.sequence_id, sequence.completion_token_ids, sequence.finish_reason)
                for sequence in sequences
                if sequence.is_finished
            ]
            all_outputs.extend(outputs)

            num_tokens = sum(len(sequence) for sequence in sequences) if is_prefill else -len(sequences)
            total_num_tokens += num_tokens

        return all_outputs, total_num_tokens

    def is_finished(self):
        return all(s.is_finished() for s in self.schedulers)

    def generate(self, prompts, sampling_params, use_tqdm=True):
        if use_tqdm:
            pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)

        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)

        for prompt, sampling_param in zip(prompts, sampling_params):
            self.add_request(prompt, sampling_param)

        outputs = {}
        prefill_throughput = decode_throughput = 0.0

        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()

            if use_tqdm:
                dt = perf_counter() - t
                if num_tokens > 0:
                    prefill_throughput = num_tokens / dt
                elif num_tokens < 0:
                    decode_throughput = -num_tokens / dt

                pbar.set_postfix(
                    {"Prefill": f"{int(prefill_throughput)}tok/s", "Decode": f"{int(decode_throughput)}tok/s"}
                )

            for seq_id, token_ids, finish_reason in output:
                outputs[seq_id] = (token_ids, finish_reason)
                if use_tqdm:
                    pbar.update(1)

        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [
            {
                "text": self.tokenizer.decode(token_ids),
                "token_ids": token_ids,
                "finish_reason": str(finish_reason),
            }
            for token_ids, finish_reason in outputs
        ]

        if use_tqdm:
            pbar.close()
        return outputs
