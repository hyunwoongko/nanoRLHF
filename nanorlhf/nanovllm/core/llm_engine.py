from dataclasses import fields
from time import perf_counter
from typing import List

from tqdm import tqdm
from transformers import AutoTokenizer

from nanorlhf import nanoray
from nanorlhf.nanoray import Bundle, PlacementStrategy, NANORAY_BASE_PORT
from nanorlhf.nanovllm.core.model_runner import ModelRunner
from nanorlhf.nanovllm.core.scheduler import Scheduler
from nanorlhf.nanovllm.core.sequence import Sequence
from nanorlhf.nanovllm.utils.config import NanoVLLMConfig


class LLMEngine:
    """
    LLMEngine is responsible for managing the distributed LLM inference using NanoRay.

    Args:
        model (str): The model name or path.
        **kwargs: Additional configuration parameters for NanoVLLMConfig.
    """
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

        self.init_ray(config)
        self.pg = self.create_placement_groups()
        self.model_runners = self.create_model(config)

        model_runner_config = self.model_runners[0][0].get_config.remote(blocking=True)  # noqa
        self.schedulers = [Scheduler(nanoray.get(model_runner_config)) for _ in range(self.data_parallel_size)]
        self.round_robin_counter = 0

    def init_ray(self, config):
        """
        Initialize the NanoRay session with the appropriate node configurations.

        Args:
            config (NanoVLLMConfig): The configuration for NanoVLLM.
        """
        nodes = {}
        if self.global_world_size > 1:
            for global_rank in range(self.global_world_size):
                nodes[f"global_rank={global_rank}"] = nanoray.NodeConfig(
                    cpus=1.0,
                    gpus=1.0,
                    rpc=True,
                    host=config.host,
                    port=NANORAY_BASE_PORT + global_rank,
                )
        else:
            nodes["global_rank=0"] = nanoray.NodeConfig(
                cpus=1.0,
                gpus=1.0,
                rpc=False,
                host=config.host,
                port=NANORAY_BASE_PORT,
            )

        session = nanoray.init(nodes, default_node_id="global_rank=0")
        node_ids = list(session.workers.keys())
        if len(node_ids) < self.global_world_size:
            raise RuntimeError(
                "`nanoray` was initialized with fewer nodes than `global_world_size`; "
                "please provide at least one NodeConfig per global rank."
            )

    def create_placement_groups(self):
        """
        Create placement groups for the model runners.

        Returns:
            PlacementGroup: The created placement group.
        """
        return nanoray.create_placement_group(
            bundles=[Bundle(cpus=1.0, gpus=1.0) for _ in range(self.global_world_size)],
            strategy=PlacementStrategy.SPREAD,
        )

    def create_model(self, config: NanoVLLMConfig):
        """
        Create model runners for each data parallel and tensor parallel rank.

        Args:
            config (NanoVLLMConfig): The configuration for NanoVLLM.

        Returns:
            List[List[ModelRunner]]: A 2D list of model runners indexed by data parallel and tensor parallel ranks.
        """
        object_refs = []
        for data_parallel_rank in range(self.data_parallel_size):
            for tensor_parallel_rank in range(self.tensor_parallel_size):
                global_rank = data_parallel_rank * self.tensor_parallel_size + tensor_parallel_rank
                object_ref = ModelRunner.options(placement_group=self.pg, bundle_index=global_rank).remote(
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
        """
        Run the model for the given data parallel rank and sequences.

        Args:
            data_parallel_rank (int): The data parallel rank.
            sequences (List[Sequence]): The list of sequences to process.
            is_prefill (bool): Whether the current step is a prefill step.

        Returns:
            List[int]: The generated token IDs.
        """
        object_refs = []
        for tensor_parallel_rank in range(self.tensor_parallel_size):
            runner = self.model_runners[data_parallel_rank][tensor_parallel_rank]
            object_refs.append(runner.run.remote(sequences, is_prefill, blocking=False))
        results = nanoray.get(object_refs)
        tokens = results[0]
        return tokens

    def add_request(self, prompt, sampling_params):
        """
        Add a new generation request to the engine.

        Args:
            prompt (str or List[int]): The input prompt as a string or list of token IDs.
            sampling_params: The sampling parameters for generation.
        """
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)

        sequence = Sequence(prompt, sampling_params)
        data_parallel_rank = self.round_robin_counter
        self.round_robin_counter = (self.round_robin_counter + 1) % self.data_parallel_size
        self.schedulers[data_parallel_rank].add(sequence)

    def step(self):
        """
        Perform a single generation step across all data parallel ranks.

        Returns:
            Tuple[List[Tuple[int, List[int], str]], int]:
                A tuple containing the list of outputs and the total number of tokens processed.
        """
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
        """
        Check if all schedulers have finished processing.

        Returns:
            bool: True if all schedulers are finished, False otherwise.
        """
        return all(s.is_finished() for s in self.schedulers)

    def generate(self, prompts, sampling_params, use_tqdm=True):
        """
        Generate text for a list of prompts.

        Args:
            prompts (List[str or List[int]]): The list of input prompts.
            sampling_params (Union[SamplingParams, List[SamplingParams]]): The sampling parameters.
            use_tqdm (bool): Whether to display a progress bar.

        Returns:
            List[Dict]: A list of generation results containing text, token IDs, and finish reasons.
        """
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
