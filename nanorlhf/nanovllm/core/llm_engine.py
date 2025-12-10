from dataclasses import fields
from time import perf_counter

from tqdm import tqdm
from transformers import AutoTokenizer

from nanorlhf import nanoray
from nanorlhf.nanovllm.core.model_runner import ModelRunner
from nanorlhf.nanovllm.core.scheduler import Scheduler
from nanorlhf.nanovllm.core.sequence import Sequence
from nanorlhf.nanovllm.utils.config import Config


class LLMEngine:
    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)

        self.tokenizer = AutoTokenizer.from_pretrained(config.model)
        config.eos = self.tokenizer.eos_token_id

        self.node_ids = self.init_ray(config)
        self.model_runners = self.create_model(config)

        # `num_kvcache_blocks` is computed lazily in ModelRunner,
        # so pass it to Scheduler after remote creation of ModelRunner.
        model_runner_config = self.model_runners[0].get_config.remote(blocking=True)
        self.scheduler = Scheduler(nanoray.get(model_runner_config))

    def init_ray(self, config):
        nodes = {}
        base_port = 9200
        if config.tensor_parallel_size > 1:
            for rank in range(config.tensor_parallel_size):
                nodes[f"node-{rank + 1}"] = nanoray.NodeConfig(
                    cpus=4.0,
                    gpus=1.0,
                    rpc=True,
                    host=config.host,
                    port=base_port + rank,
                )
        else:
            nodes["node-1"] = nanoray.NodeConfig(
                cpus=4.0,
                gpus=1.0,
                rpc=False,
                host=config.host,
                port=base_port,
            )

        session = nanoray.init(nodes, default_node_id="node-1")
        node_ids = list(session._workers.keys())

        if len(node_ids) < config.tensor_parallel_size:
            raise RuntimeError(
                "`nanoray` was initialized with fewer nodes than `tensor_parallel_size`; "
                "please provide at least one NodeConfig per tensor-parallel rank."
            )

        return node_ids

    def create_model(self, config: Config):
        refs = []
        for rank in range(config.tensor_parallel_size):
            node_id = self.node_ids[rank % len(self.node_ids)]
            ref = ModelRunner.options(pinned_node_id=node_id).remote(config, rank=rank, blocking=False)
            if ref is not None:
                refs.append(ref)

        while len(refs) < config.tensor_parallel_size:
            produced = nanoray.drain()
            if not produced:
                raise RuntimeError("Failed to launch all ModelRunner actors; no progress during drain.")
            refs.extend(produced)

        return [nanoray.get(r) for r in refs[: config.tensor_parallel_size]]

    def run_model(self, seqs, is_prefill):
        refs = []
        for runner in self.model_runners:
            ref = runner.run.remote(seqs, is_prefill, blocking=False)
            if ref is not None:
                refs.append(ref)

        while len(refs) < len(self.model_runners):
            produced = nanoray.drain()
            if not produced:
                raise RuntimeError("Model forward calls could not be placed; no progress during drain.")
            refs.extend(produced)

        results = [nanoray.get(r) for r in refs[: len(self.model_runners)]]
        return results[0]

    def add_request(self, prompt, sampling_params):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)

    def step(self):
        seqs, is_prefill = self.scheduler.schedule()
        token_ids = self.run_model(seqs, is_prefill)
        self.scheduler.postprocess(seqs, token_ids)
        outputs = [(seq.seq_id, seq.completion_token_ids, seq.finish_reason) for seq in seqs if seq.is_finished]
        num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -len(seqs)
        return outputs, num_tokens

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(self, prompts, sampling_params, use_tqdm=True):
        if use_tqdm:
            pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)

        outputs = {}
        prefill_throughput = decode_throughput = 0.0
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()
            if use_tqdm:
                if num_tokens > 0:
                    prefill_throughput = num_tokens / (perf_counter() - t)
                else:
                    decode_throughput = -num_tokens / (perf_counter() - t)

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
